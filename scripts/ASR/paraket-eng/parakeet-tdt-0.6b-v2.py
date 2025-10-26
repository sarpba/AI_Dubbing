# ==============================================================================
#      NVIDIA PARAKEET‑TDT TRANSCRIPTION SCRIPT  (V4.1-PROJECT-AWARE-INPLACE)
# ==============================================================================
#  🔊  Hosszú hangfájl‑transzkripció szó‑alapú időbélyegekkel (NVIDIA NeMo)
# ------------------------------------------------------------------------------
# Mit javít a v4.1?
#   • ✨ **Egyszerűsített kimenet**: A transzkripció eredményeként kapott `.json`
#     fájlok közvetlenül a bemeneti hangfájlok mellé, ugyanabba a
#     `2_separated_audio_speech` mappába kerülnek.
#   • ✨ Projekt-alapú működés (v4.0 funkció): A `--project-name` paraméterrel
#     megadható a feldolgozandó projekt. A szkript a `config.json` alapján
#     automatikusan megtalálja a megfelelő mappát.
#   • ✨ Maximális mondathossz, időbélyeg-finomítás, GPU-monitor, auto-chunk
#     funkciók változatlanok.
# ==============================================================================

import torch
import json
import argparse
from pathlib import Path
import numpy as np
import tempfile
import os
import sys
import warnings
import datetime as _dt

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

try:
    import nemo.collections.asr as nemo_asr
    import librosa
    import soundfile as sf
    from omegaconf import open_dict
    print("NVIDIA NeMo, librosa, soundfile és OmegaConf sikeresen betöltve.")
except ImportError as e:
    print(f"Hiba az importálás során: {e}")
    print("Kérlek, bizonyosodj meg róla, hogy a 'nemo' conda környezet aktív és a szükséges csomagok telepítve vannak.")
    exit(1)

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
SAMPLE_RATE = 16000
DEFAULT_CHUNK_S = 30
DEFAULT_MAX_PAUSE_S = 0.6 #2.0 volt
DEFAULT_PADDING_S = 0.2
DEFAULT_MAX_SEGMENT_S = 11.5
TARGET_GPU_UTIL = 0.9
MIN_CHUNK_S = 10
MAX_CHUNK_S = 180
CALIB_DUMMY_SEC = 120

# -----------------------------------------------------------------------------
#   MEMÓRIA‑SEGÉD – GPU monitorozás
# -----------------------------------------------------------------------------

def get_free_gpu_memory(device: torch.device):
    try:
        free, total = torch.cuda.mem_get_info(device.index)
        return free, total
    except Exception:
        return None, None

def log_gpu_memory(device: torch.device, ctx: str = ""):
    if device.type != "cuda":
        return
    free, total = get_free_gpu_memory(device)
    alloc = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    print(
        f"[GPU {ts}]{' ' + ctx if ctx else ''} / alloc: {alloc/1e9:.2f} GB, "
        f"reserved: {reserved/1e9:.2f} GB, free: {free/1e9 if free else 0:.2f} GB, "
        f"total: {total/1e9 if total else 0:.2f} GB, peak: {peak/1e9:.2f} GB"
    )

# -----------------------------------------------------------------------------
#   TIMESTAMP‑ENABLED MODEL HELPER
# -----------------------------------------------------------------------------

def enable_word_timestamps(asr_model):
    try:
        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.preserve_alignments = True
            decoding_cfg.compute_timestamps = True
            decoding_cfg.word_seperator = " "
        asr_model.change_decoding_strategy(decoding_cfg)
        print("  - Word‑timestamp konfiguráció engedélyezve.")
    except Exception as e:
        warnings.warn(f"Nem sikerült módosítani a dekóder konfigurációt: {e}")

# -----------------------------------------------------------------------------
#   AUTO‑CHUNK KALIBRÁCIÓ
# -----------------------------------------------------------------------------

def calibrate_chunk_size(asr_model, device: torch.device):
    free_mem, total_mem = get_free_gpu_memory(device)
    if free_mem is None:
        return DEFAULT_CHUNK_S
    print(f"  - GPU memória: {free_mem/1e9:.1f} GB szabad / {total_mem/1e9:.1f} GB összesen.")
    log_gpu_memory(device, "(before calibration)")
    tmp_wav = None
    try:
        dummy_audio = np.random.uniform(-0.2, 0.2, SAMPLE_RATE * CALIB_DUMMY_SEC).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            tmp_wav = fp.name
            sf.write(tmp_wav, dummy_audio, SAMPLE_RATE)
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        asr_model.transcribe([tmp_wav], batch_size=1, timestamps=True)
        after = torch.cuda.memory_allocated(device)
        delta = max(after - before, 1)
        bytes_per_sec = delta / CALIB_DUMMY_SEC
        est_sec = int((free_mem * TARGET_GPU_UTIL) / bytes_per_sec)
        est_sec = max(MIN_CHUNK_S, min(est_sec, MAX_CHUNK_S))
        print(f"  - Kalibrált chunk‑méret ≈ {est_sec}s (≈ {bytes_per_sec/1e6:.0f} MB/s)")
        log_gpu_memory(device, "(after calibration)")
        return est_sec
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        warnings.warn("Kalibráció közben OOM – marad a fix 30 s.")
        return DEFAULT_CHUNK_S
    except Exception as e:
        warnings.warn(f"Kalibráció nem sikerült: {e}")
        return DEFAULT_CHUNK_S
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)

# -----------------------------------------------------------------------------
#   MONDATSZEGMENTÁLÁS ÉS FINOMÍTÁS
# -----------------------------------------------------------------------------

def adjust_word_timestamps(word_segments, padding_s: float):
    if not word_segments or padding_s <= 0: return word_segments
    adjusted_segments = [word.copy() for word in word_segments]
    for i in range(len(adjusted_segments) - 1):
        gap = adjusted_segments[i+1]['start'] - adjusted_segments[i]['end']
        if gap > (padding_s * 2):
            adjustment = min(padding_s, gap / 2.0)
            adjusted_segments[i]['end'] += adjustment
            adjusted_segments[i+1]['start'] -= adjustment
    adjusted_segments[0]['start'] = max(0.0, adjusted_segments[0]['start'] - padding_s)
    adjusted_segments[-1]['end'] += padding_s
    for word in adjusted_segments:
        word['start'] = round(word['start'], 3)
        word['end'] = round(word['end'], 3)
    return adjusted_segments

def _create_segment_from_words(words):
    if not words: return None
    text = " ".join(w['word'] for w in words)
    text = text.replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return {"start": words[0]['start'], "end": words[-1]['end'], "text": text, "words": words}

def sentence_segments_from_words(words, max_pause_s: float):
    segs, current = [], []
    start_time = None
    for w in words:
        if current and (w["start"] - current[-1]["end"] > max_pause_s):
            if new_seg := _create_segment_from_words(current): segs.append(new_seg)
            current, start_time = [], None
        if start_time is None: start_time = w["start"]
        current.append(w)
        token = w["word"].strip()
        if token in {".", "!", "?"} or token[-1] in {".", "!", "?"}:
            if new_seg := _create_segment_from_words(current): segs.append(new_seg)
            current, start_time = [], None
    if new_seg := _create_segment_from_words(current): segs.append(new_seg)
    return segs

def split_long_segments(segments, max_duration_s: float):
    if max_duration_s <= 0: return segments
    final_segments = []
    for segment in segments:
        words_to_process = segment['words']
        while words_to_process:
            segment_duration = words_to_process[-1]['end'] - words_to_process[0]['start']
            if segment_duration <= max_duration_s:
                if new_seg := _create_segment_from_words(words_to_process): final_segments.append(new_seg)
                break
            else:
                candidate_words = [w for w in words_to_process if w['end'] - words_to_process[0]['start'] <= max_duration_s]
                if not candidate_words: candidate_words = words_to_process[:1]

                best_split_idx = len(candidate_words) - 1
                if len(candidate_words) > 1:
                    max_gap = -1.0
                    for i in range(len(candidate_words) - 1):
                        gap = candidate_words[i+1]['start'] - candidate_words[i]['end']
                        if gap >= max_gap:
                            max_gap = gap
                            best_split_idx = i
                
                new_segment_words = words_to_process[:best_split_idx + 1]
                if new_seg := _create_segment_from_words(new_segment_words): final_segments.append(new_seg)
                words_to_process = words_to_process[best_split_idx + 1:]
    return final_segments

# -----------------------------------------------------------------------------
#   DARABOLÁS + TRANSZKRIPCIÓ
# -----------------------------------------------------------------------------

def transcribe_long_audio_final(asr_model, audio_path, *, chunk_len_s: int, batch_size: int, device: torch.device, max_pause_s: float, padding_s: float, max_segment_s: float):
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)
        if audio.ndim > 1: audio = librosa.to_mono(audio)
    except Exception as e:
        warnings.warn(f"Betöltési hiba '{Path(audio_path).name}': {e}")
        return None

    print(f"  - Fájl: {Path(audio_path).name} – {len(audio)/SAMPLE_RATE:.1f}s, chunk={chunk_len_s}s")
    chunk_samples = chunk_len_s * SAMPLE_RATE
    with tempfile.TemporaryDirectory() as tmp:
        paths, offs = [], []
        for i, start in enumerate(range(0, len(audio), chunk_samples)):
            end = start + chunk_samples
            chunk = audio[start:end]
            if np.max(np.abs(chunk)) < 0.001: continue
            p = os.path.join(tmp, f"chunk_{i}.wav")
            sf.write(p, chunk, SAMPLE_RATE)
            paths.append(p)
            offs.append(start / SAMPLE_RATE)
        if not paths:
            warnings.warn("Nincs feldolgozható darab.")
            return None
        log_gpu_memory(device, "(before batch transcribe)")
        try:
            hyps = asr_model.transcribe(audio=paths, batch_size=batch_size, timestamps=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            warnings.warn("OOM – kisebb chunk szükséges.")
            return None
        except Exception as e:
            warnings.warn(f"Transzkripciós hiba: {e}")
            return None
        finally:
            log_gpu_memory(device, "(after batch transcribe)")
        word_segments = []
        for i, h in enumerate(hyps):
            off = offs[i]
            if hasattr(h, "timestamp") and isinstance(h.timestamp, dict) and h.timestamp.get("word"):
                for w in h.timestamp["word"]:
                    word_segments.append({"word": w.get("word", w.get("text", "")),"start": round(w["start"] + off, 3),"end": round(w["end"] + off, 3),"score": w.get("conf") or w.get("confidence")})
            elif hasattr(h, "words") and h.words and not isinstance(h.words[0], str):
                for w in h.words:
                    word_segments.append({"word": w.word,"start": round(w.start_time + off, 3),"end": round(w.end_time + off, 3),"score": round(getattr(w, "score", 0.0), 4)})
        if not word_segments: return None

        word_segments = adjust_word_timestamps(word_segments, padding_s=padding_s)
        initial_segments = sentence_segments_from_words(word_segments, max_pause_s=max_pause_s)
        final_segments = split_long_segments(initial_segments, max_duration_s=max_segment_s)
        
        print(f"  - {len(final_segments)} mondatszegmens generálva (max_pause={max_pause_s}s, padding={padding_s}s, max_dur={max_segment_s}s).")
        return {"segments": final_segments, "word_segments": word_segments, "language": "en"}

# -----------------------------------------------------------------------------
#   MAPPÁK FELDOLGOZÁSA
# -----------------------------------------------------------------------------

def process_project_directory(directory_path: str, auto_chunk: bool, fixed_chunk_s: int, max_pause_s: float, padding_s: float, max_segment_s: float):
    model_name = "nvidia/parakeet-tdt-0.6b-v2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nModell betöltése: {model_name} ({device})")
    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        asr_model.to(device)
        enable_word_timestamps(asr_model)
        log_gpu_memory(device, "(after model load)")
    except Exception as e:
        raise RuntimeError(f"Modell‑betöltési hiba: {e}")

    chunk_len_s = calibrate_chunk_size(asr_model, device) if auto_chunk and device.type == "cuda" else fixed_chunk_s
    if not auto_chunk: print(f"  - Automatikus kalibráció kikapcsolva → fix {chunk_len_s}s chunk")
    
    audio_files = [str(p) for p in Path(directory_path).iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not audio_files:
        print(f"Nem található támogatott hangfájl a '{directory_path}' mappában.")
        return

    print(f"\n{len(audio_files)} fájl feldolgozása indul {chunk_len_s}s darabokkal…")
    for audio_path_str in audio_files:
        audio_path = Path(audio_path_str)
        output_json_path = audio_path.with_suffix(".json")
        
        print("-" * 50)
        print(f"▶  Feldolgozás: {audio_path.name}")
        
        res = transcribe_long_audio_final(
            asr_model, audio_path_str, chunk_len_s=chunk_len_s, batch_size=8, device=device,
            max_pause_s=max_pause_s, padding_s=padding_s, max_segment_s=max_segment_s
        )
        if res and (res.get("word_segments") or res.get("segments")):
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=4, ensure_ascii=False)
            print(f"  ✔  Mentve: {output_json_path.name}")
        else:
            print(f"  ✖  Sikertelen vagy üres transzkripció: {audio_path.name}")
    print("\nKész.")

# -----------------------------------------------------------------------------
#   KONFIGURÁCIÓ ÉS CLI
# -----------------------------------------------------------------------------

def load_config_and_get_paths(project_name: str):
    """Betölti a config.json-t és visszaadja a projekt feldolgozandó mappájának útvonalát."""
    try:
        # A szkript a scripts/Nvidia_asr_eng/ mappában van, a config a gyökérben
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        config_path = project_root / "config.json"

        if not config_path.is_file():
            raise FileNotFoundError(f"A 'config.json' nem található a projekt gyökerében: {project_root}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        workdir = project_root / config["DIRECTORIES"]["workdir"]
        input_subdir = config["PROJECT_SUBDIRS"]["separated_audio_speech"]

        processing_path = workdir / project_name / input_subdir

        if not processing_path.is_dir():
             raise FileNotFoundError(f"A feldolgozandó mappa nem létezik: {processing_path}")

        print("Projekt beállítások betöltve:")
        print(f"  - Projekt név:     {project_name}")
        print(f"  - Feldolgozandó mappa (bemenet és kimenet): {processing_path}")

        return str(processing_path)
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Hiba a konfiguráció betöltése vagy az útvonalak meghatározása közben: {e}")
        print("Kérlek, ellenőrizd a 'config.json' fájlt és a projekt mappaszerkezetét.")
        exit(1)


def main():
    p = argparse.ArgumentParser(description="Transcribe audio files with NVIDIA NeMo (word‑level timestamps, v4.1-project-aware-inplace)")
    p.add_argument("-p", "--project-name", required=True, help="A projekt neve (a 'workdir' alatti mappa), amit fel kell dolgozni.")
    p.add_argument("--no-auto-chunk", action="store_true", help="Automatikus chunk-méret kalibráció kikapcsolása.")
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK_S, help=f"Fix chunk‑hossz másodpercben, ha az auto-chunk ki van kapcsolva (alapért.: {DEFAULT_CHUNK_S}s).")
    p.add_argument("--max-pause", type=float, default=DEFAULT_MAX_PAUSE_S, help=f"Mondatok közti maximális szünet másodpercben (alapért.: {DEFAULT_MAX_PAUSE_S}s).")
    p.add_argument("--timestamp-padding", type=float, default=DEFAULT_PADDING_S, help=f"Időbélyegek kiterjesztése a szünetek rovására (mp). 0 a kikapcsoláshoz (alapért.: {DEFAULT_PADDING_S}s).")
    p.add_argument("--max-segment-duration", type=float, default=DEFAULT_MAX_SEGMENT_S, help=f"Mondatszegmensek maximális hossza másodpercben (alapért.: {DEFAULT_MAX_SEGMENT_S}s). 0 a kikapcsoláshoz.")
    add_debug_argument(p)
    args = p.parse_args()
    configure_debug_mode(args.debug)

    processing_dir = load_config_and_get_paths(args.project_name)

    process_project_directory(
        directory_path=processing_dir,
        auto_chunk=not args.no_auto_chunk,
        fixed_chunk_s=args.chunk,
        max_pause_s=args.max_pause,
        padding_s=args.timestamp_padding,
        max_segment_s=args.max_segment_duration
    )

if __name__ == "__main__":
    main()

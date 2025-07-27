# ==============================================================================
#  NVIDIA PARAKEET + SORTFORMER TRANSCRIPTION SCRIPT (V5.5-FINAL-FIX)
# ==============================================================================
#  🔊  Hosszú hangfájl‑transzkripció és beszélőváltás-alapú mondatdarabolás
# ------------------------------------------------------------------------------
# Mit javít ez a verzió?
#   • ✨ **VÉGLEGES ASR BUGFIX**: A szó-kinyerési logika visszaállítva az
#     eredeti, robusztus, kétágú ellenőrzésre, amely garantáltan feldolgozza
#     a NeMo ASR kimenetét. Ez végleg megoldja a "Sikertelen transzkripció" hibát.
#   • ✨ Minden más funkció (Sortformer, kétlépcsős darabolás) változatlan.
# ==============================================================================

import torch
import json
import argparse
from pathlib import Path
import numpy as np
import tempfile
import os
import warnings
import datetime as _dt
import traceback
import soundfile as sf

try:
    import nemo.collections.asr as nemo_asr
    import librosa
    from omegaconf import open_dict, OmegaConf
    print("NVIDIA NeMo, librosa, soundfile és OmegaConf sikeresen betöltve.")
except ImportError as e:
    print(f"Hiba az importálás során: {e}")
    print("Kérlek, bizonyosodj meg róla, hogy a 'parakeet-fix' conda környezet aktív és a szükséges csomagok telepítve vannak.")
    exit(1)

# --- KONSTANSOK ÉS SEGÉDFÜGGVÉNYEK (VÁLTOZATLAN) ---
SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
SAMPLE_RATE = 16000
DEFAULT_CHUNK_S = 30
DEFAULT_MAX_PAUSE_S = 2.0
DEFAULT_PADDING_S = 0.2
DEFAULT_MAX_SEGMENT_S = 11.5
TARGET_GPU_UTIL = 0.9
MIN_CHUNK_S = 10
MAX_CHUNK_S = 180
CALIB_DUMMY_SEC = 120

def get_free_gpu_memory(device: torch.device):
    try:
        free, total = torch.cuda.mem_get_info(device.index)
        return free, total
    except Exception: return None, None

def log_gpu_memory(device: torch.device, ctx: str = ""):
    if device.type != "cuda": return
    free, total = get_free_gpu_memory(device)
    alloc, reserved, peak = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device), torch.cuda.max_memory_allocated(device)
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[GPU {ts}]{' ' + ctx if ctx else ''} / alloc: {alloc/1e9:.2f} GB, res: {reserved/1e9:.2f} GB, free: {free/1e9 if free else 0:.2f} GB, peak: {peak/1e9:.2f} GB")

def enable_word_timestamps(asr_model):
    try:
        with open_dict(asr_model.cfg.decoding) as decoding_cfg:
            decoding_cfg.preserve_alignments = True
            decoding_cfg.compute_timestamps = True
        asr_model.change_decoding_strategy(asr_model.cfg.decoding)
        print("  - Word‑timestamp konfiguráció engedélyezve (ASR).")
    except Exception as e:
        warnings.warn(f"Nem sikerült módosítani a dekóder konfigurációt: {e}")

def calibrate_chunk_size(asr_model, device: torch.device):
    free_mem, _ = get_free_gpu_memory(device)
    if free_mem is None: return DEFAULT_CHUNK_S
    log_gpu_memory(device, "(before calibration)")
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            tmp_wav = fp.name
            sf.write(fp.name, np.random.uniform(-0.2, 0.2, SAMPLE_RATE * CALIB_DUMMY_SEC).astype(np.float32), SAMPLE_RATE)
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        asr_model.transcribe([tmp_wav], batch_size=1, timestamps=True)
        after = torch.cuda.memory_allocated(device)
        bytes_per_sec = max(after - before, 1) / CALIB_DUMMY_SEC
        est_sec = max(MIN_CHUNK_S, min(int((free_mem * TARGET_GPU_UTIL) / bytes_per_sec), MAX_CHUNK_S))
        print(f"  - Kalibrált chunk‑méret ≈ {est_sec}s (≈ {bytes_per_sec/1e6:.0f} MB/s)")
        log_gpu_memory(device, "(after calibration)")
        os.remove(tmp_wav)
        return est_sec
    except Exception as e:
        warnings.warn(f"Kalibráció nem sikerült ({e}), marad a fix {DEFAULT_CHUNK_S}s.")
        if 'tmp_wav' in locals() and os.path.exists(tmp_wav): os.remove(tmp_wav)
        return DEFAULT_CHUNK_S

def adjust_word_timestamps(word_segments, padding_s: float):
    if not word_segments or padding_s <= 0: return word_segments
    adjusted = [w.copy() for w in word_segments]
    for i in range(len(adjusted) - 1):
        gap = adjusted[i+1]['start'] - adjusted[i]['end']
        if gap > (padding_s * 2):
            adj = min(padding_s, gap / 2.0)
            adjusted[i]['end'] = round(adjusted[i]['end'] + adj, 3)
            adjusted[i+1]['start'] = round(adjusted[i+1]['start'] - adj, 3)
    adjusted[0]['start'] = round(max(0.0, adjusted[0]['start'] - padding_s), 3)
    adjusted[-1]['end'] = round(adjusted[-1]['end'] + padding_s, 3)
    return adjusted

def _create_segment_from_words(words):
    if not words: return None
    text = " ".join(w['word'] for w in words).replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return {"start": words[0]['start'], "end": words[-1]['end'], "text": text, "words": words}

def sentence_segments_from_words(words, max_pause_s: float):
    segs, current = [], []
    for w in words:
        if current and (w["start"] - current[-1]["end"] > max_pause_s):
            if new_seg := _create_segment_from_words(current): segs.append(new_seg)
            current = []
        current.append(w)
        if w["word"].strip() and w["word"].strip()[-1] in {".", "!", "?"}:
            if new_seg := _create_segment_from_words(current): segs.append(new_seg)
            current = []
    if new_seg := _create_segment_from_words(current): segs.append(new_seg)
    return segs

def split_long_segments(segments, max_duration_s: float):
    if max_duration_s <= 0: return segments
    final_segments = []
    for segment in segments:
        words_to_process = segment['words']
        while words_to_process:
            if words_to_process[-1]['end'] - words_to_process[0]['start'] <= max_duration_s:
                if new_seg := _create_segment_from_words(words_to_process): final_segments.append(new_seg)
                break
            else:
                candidates = [w for w in words_to_process if w['end'] - words_to_process[0]['start'] <= max_duration_s]
                if not candidates: candidates = words_to_process[:1]
                best_split_idx = len(candidates) - 1
                if len(candidates) > 1:
                    max_gap = -1.0
                    for i in range(len(candidates) - 1):
                        gap = candidates[i+1]['start'] - candidates[i]['end']
                        if gap >= max_gap:
                            max_gap = gap
                            best_split_idx = i
                new_words = words_to_process[:best_split_idx + 1]
                if new_seg := _create_segment_from_words(new_words): final_segments.append(new_seg)
                words_to_process = words_to_process[best_split_idx + 1:]
    return final_segments

# -----------------------------------------------------------------------------
#   1. LÉPÉS: TELJES TRANSZKRIPCIÓ (ASR) - VÉGLEGES JAVÍTÁS
# -----------------------------------------------------------------------------
def transcribe_long_audio(asr_model, audio_path, **kwargs):
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        warnings.warn(f"Betöltési hiba '{Path(audio_path).name}': {e}")
        return None
    print(f"  - Fájl: {Path(audio_path).name} – {len(audio)/SAMPLE_RATE:.1f}s, chunk={kwargs['chunk_len_s']}s")
    
    word_segments = []
    with tempfile.TemporaryDirectory() as tmp:
        chunk_samples = kwargs['chunk_len_s'] * SAMPLE_RATE
        paths, offs = [], []
        for i, start in enumerate(range(0, len(audio), chunk_samples)):
            chunk = audio[start : start + chunk_samples]
            if np.max(np.abs(chunk)) < 0.001: continue
            p = os.path.join(tmp, f"c_{i}.wav")
            sf.write(p, chunk, SAMPLE_RATE)
            paths.append(p)
            offs.append(start / SAMPLE_RATE)
        
        if not paths: return None
        log_gpu_memory(kwargs['device'], "(before ASR batch transcribe)")
        try:
            # Visszaállítva a robusztus, kétágú szó-kinyerési logika
            hyps = asr_model.transcribe(audio=paths, batch_size=kwargs['batch_size'], timestamps=True)
            for i, h in enumerate(hyps):
                off = offs[i]
                # 1. ág: .timestamp['word'] ellenőrzése
                if hasattr(h, "timestamp") and isinstance(h.timestamp, dict) and h.timestamp.get("word"):
                    for w in h.timestamp["word"]:
                         word_segments.append({"word": w.get("word", ""),"start": round(w["start"] + off, 3),"end": round(w["end"] + off, 3)})
                # 2. ág: .words ellenőrzése fallback-ként
                elif hasattr(h, "words") and h.words and not isinstance(h.words[0], str):
                    for w in h.words:
                        word_segments.append({"word": w.word,"start": round(w.start_time + off, 3),"end": round(w.end_time + off, 3)})

        except Exception as e:
            warnings.warn(f"Transzkripciós hiba: {e}")
            traceback.print_exc()
            return None
        finally:
            log_gpu_memory(kwargs['device'], "(after ASR batch transcribe)")

    if not word_segments: return None
    padded_words = adjust_word_timestamps(word_segments, kwargs['padding_s'])
    initial_segments = sentence_segments_from_words(padded_words, kwargs['max_pause_s'])
    final_segments = split_long_segments(initial_segments, kwargs['max_segment_s'])
    print(f"  - ASR kész: {len(final_segments)} mondatszegmens generálva.")
    return {"segments": final_segments, "word_segments": padded_words, "language": "hu"}

# -----------------------------------------------------------------------------
#   2. LÉPÉS: BESZÉLŐVÁLTÁS-DETEKTÁLÁS ÉS DARABOLÁS (VÁLTOZATLAN)
# -----------------------------------------------------------------------------
def get_speaker_for_time(timestamp: float, speaker_turns: list):
    for turn in speaker_turns:
        if turn['start'] <= timestamp <= turn['end']:
            return turn['speaker']
    return None

def run_sortformer_on_segment(diar_model, audio_segment: np.ndarray, segment_offset: float):
    if len(audio_segment) / SAMPLE_RATE < 0.5:
        return [], 1
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_audio_path = os.path.join(tmpdir, "segment.wav")
            sf.write(tmp_audio_path, audio_segment, SAMPLE_RATE)
            diar_model.diarize(paths2audio_files=[tmp_audio_path], output_dir=tmpdir)
            rttm_files = list(Path(tmpdir).rglob("*.rttm"))
            if not rttm_files: return [], 0
            with open(rttm_files[0], 'r') as f:
                rttm_lines = f.readlines()
        
        speaker_turns, speakers = [], set()
        for line in rttm_lines:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER": continue
            local_start = float(parts[3])
            global_start = local_start + segment_offset
            global_end = global_start + float(parts[4])
            speaker = parts[7]
            speakers.add(speaker)
            speaker_turns.append({'start': round(global_start, 3), 'end': round(global_end, 3), 'speaker': speaker})
        return sorted(speaker_turns, key=lambda x: x['start']), len(speakers)
    except Exception as e:
        warnings.warn(f"Hiba a szegmens-diarizáció során: {e}")
        return [], 0

def resplit_segment_if_needed(segment: dict, speaker_turns: list):
    for word in segment['words']:
        word['speaker'] = get_speaker_for_time((word['start'] + word['end']) / 2, speaker_turns)
    new_segments, current_words, current_speaker = [], [], None
    for word in segment['words']:
        speaker = word.get('speaker')
        if current_speaker is None and speaker is not None:
            current_speaker = speaker
        if speaker is not None and speaker != current_speaker and current_words:
            if new_seg := _create_segment_from_words(current_words): new_segments.append(new_seg)
            current_words = []
        current_speaker = speaker if speaker is not None else current_speaker
        current_words.append(word)
    if new_seg := _create_segment_from_words(current_words): new_segments.append(new_seg)
    return new_segments

# -----------------------------------------------------------------------------
#   FŐ FELDOLGOZÓ FÜGGVÉNY (VÁLTOZATLAN)
# -----------------------------------------------------------------------------
def process_directory(directory_path: str, enable_diarization: bool, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ASR modell betöltése: nvidia/parakeet-tdt-0.6b-v2")
    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
        asr_model.to(device)
        enable_word_timestamps(asr_model)
        log_gpu_memory(device, "(after ASR model load)")
    except Exception as e:
        raise RuntimeError(f"ASR modell betöltési hiba: {e}")

    diar_model = None
    if enable_diarization:
        model_name = "nvidia/diar_sortformer_4spk-v1"
        print(f"Diarizációs modell betöltése: {model_name}")
        try:
            diar_model = nemo_asr.models.SortformerEncLabelModel.from_pretrained(model_name=model_name)
            diar_model.to(device)
            log_gpu_memory(device, "(after Diarization model load)")
        except Exception as e:
            print(f"\n{'='*60}\n!!! KRITIKUS HIBA: A DIARIZÁCIÓS MODELL BETÖLTÉSE SIKERTELEN !!!\n{e}\nA program folytatódik darabolás nélkül.\n{'='*60}\n")
            traceback.print_exc()
            diar_model = None
    
    kwargs['chunk_len_s'] = calibrate_chunk_size(asr_model, device) if kwargs['auto_chunk'] else kwargs['fixed_chunk_s']
    kwargs['device'] = device
    
    wavs = [str(p) for p in Path(directory_path).iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not wavs:
        print("Nem található támogatott hangfájl.")
        return

    print(f"\n{len(wavs)} fájl feldolgozása...")
    for ap in wavs:
        apath = Path(ap)
        out_json = apath.with_suffix(".json")
        print("-" * 50)
        print(f"▶  {apath.name}")

        res = transcribe_long_audio(asr_model, ap, **kwargs)
        if not (res and res.get("segments")):
            print("  ✖  Sikertelen vagy üres transzkripció.")
            continue

        if diar_model:
            print("  - Mondatszegmensek ellenőrzése és darabolása beszélőváltások mentén...")
            audio, _ = librosa.load(ap, sr=SAMPLE_RATE, mono=True)
            final_segments = []
            
            for i, segment in enumerate(res["segments"]):
                start_sample, end_sample = int(segment['start'] * SAMPLE_RATE), int(segment['end'] * SAMPLE_RATE)
                audio_segment = audio[start_sample:end_sample]
                speaker_turns, num_speakers = run_sortformer_on_segment(diar_model, audio_segment, segment['start'])
                
                if num_speakers > 1:
                    print(f"    - Beszélőváltás detektálva a(z) {i+1}. szegmensben. Darabolás...")
                    final_segments.extend(resplit_segment_if_needed(segment, speaker_turns))
                else:
                    final_segments.append(segment)
            
            print(f"  - Darabolás kész. Eredeti {len(res['segments'])} szegmensből {len(final_segments)} lett.")
            res["segments"] = final_segments
            all_words = []
            for seg in res["segments"]:
                for word in seg['words']:
                    word.pop('speaker', None)
                all_words.extend(seg['words'])
            res['word_segments'] = all_words

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=4, ensure_ascii=False)
        print(f"  ✔  Mentve: {out_json.name}")
            
    print("\nKész.")

# -----------------------------------------------------------------------------
#   PARANCSSORI INTERFÉSZ (CLI)
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Transcribe and Split-by-Speaker audio files with NVIDIA NeMo (v5.5-final-fix)")
    p.add_argument("-i", "--input-directory", required=True, help="Mappa, ahol a hangfájlok találhatók.")
    p.add_argument("--enable-diarization", action="store_true", help="Beszélőváltás-alapú mondatdarabolás engedélyezése.")
    p.add_argument("--no-auto-chunk", action="store_true", help="ASR chunk-kalibráció kikapcsolása.")
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK_S, help=f"Fix chunk‑hossz mp‑ben (alapért.: {DEFAULT_CHUNK_S}s).")
    p.add_argument("--max-pause", type=float, default=DEFAULT_MAX_PAUSE_S, help=f"Mondatok közti max. szünet (alapért.: {DEFAULT_MAX_PAUSE_S}s).")
    p.add_argument("--timestamp-padding", type=float, default=DEFAULT_PADDING_S, help=f"Időbélyeg kiterjesztés (alapért.: {DEFAULT_PADDING_S}s).")
    p.add_argument("--max-segment-duration", type=float, default=DEFAULT_MAX_SEGMENT_S, help=f"Mondatszegmensek max. hossza (alapért.: {DEFAULT_MAX_SEGMENT_S}s).")
    p.add_argument("--batch-size", type=int, default=8, help=f"ASR batch méret (alapért.: 8).")
    args = p.parse_args()

    process_directory(
        directory_path=args.input_directory,
        enable_diarization=args.enable_diarization,
        auto_chunk=not args.no_auto_chunk,
        fixed_chunk_s=args.chunk,
        max_pause_s=args.max_pause,
        padding_s=args.timestamp_padding,
        max_segment_s=args.max_segment_duration,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
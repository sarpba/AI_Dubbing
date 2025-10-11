# ==============================================================================
#           NVIDIA PARAKEET‑TDT TRANSCRIPTION SCRIPT  (FINAL, V3.9-PAUSE-SPLIT)
# ==============================================================================
#  🔊  Hosszú hangfájl‑transzkripció szó‑alapú időbélyegekkel (NVIDIA NeMo)
# ------------------------------------------------------------------------------
# Mit javít a v3.9-PAUSE-SPLIT?
#   • ✨ **Szünet-alapú mondattörés** – Ha két szó között a szünet nagyobb,
#     mint a megadott `max-pause` érték (alapértelmezetten 2s), a szkript
#     automatikusan új mondatszegmenst kezd.
#   • 🐞 `wavs` inicializálása – most mindig létrejön, függetlenül attól, hogy
#     auto‑chunk aktív‑e, megszüntetve az `UnboundLocalError`‑t.
#   • 📌 A `device` argumentum átadásra kerül a `transcribe_long_audio_final()`
#     hívásnál.
#   • 📝 Verzió‑ és help‑szövegek frissítve.
#   • Minden korábbi funkció (GPU‑monitor, auto‑chunk) változatlan.
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

try:
    import nemo.collections.asr as nemo_asr
    import librosa
    import soundfile as sf
    from omegaconf import open_dict
    print("NVIDIA NeMo, librosa, soundfile és OmegaConf sikeresen betöltve.")
except ImportError as e:
    print(f"Hiba az importálás során: {e}")
    print("Kérlek, bizonyosodj meg róla, hogy a 'parakeet-fix' conda környezet aktív és a szükséges csomagok telepítve vannak.")
    exit(1)

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
SAMPLE_RATE = 16000
DEFAULT_CHUNK_S = 30
DEFAULT_MAX_PAUSE_S = 2.0
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
#   MONDATSZEGMENTÁLÁS SEGÉD
# -----------------------------------------------------------------------------

def sentence_segments_from_words(words, max_pause_s: float):
    segs, current = [], []
    start_time = None
    for w in words:
        # Szünet alapú mondattörés ellenőrzése
        # Ha már van folyamatban lévő mondat, ÉS az új szó kezdete és az előző
        # szó vége között nagyobb a szünet, mint a megengedett, akkor lezárjuk
        # az előző mondatot.
        if current and (w["start"] - current[-1]["end"] > max_pause_s):
            segs.append({
                "start": start_time,
                "end": current[-1]["end"],
                "text": " ".join(t["word"] for t in current),
                "words": current.copy(),
            })
            # Új mondat indítása
            current, start_time = [], None

        # Eredeti logika: szó hozzáadása és írásjel-ellenőrzés
        if start_time is None:
            start_time = w["start"]
        current.append(w)
        token = w["word"].strip()
        if token in {".", "!", "?"} or token[-1] in {".", "!", "?"}:
            segs.append({
                "start": start_time,
                "end": w["end"],
                "text": " ".join(t["word"] for t in current).replace(" .", ".").replace(" !", "!").replace(" ?", "?"),
                "words": current.copy(),
            })
            current, start_time = [], None

    # Maradék mondat hozzáadása a végén
    if current:
        segs.append({
            "start": start_time,
            "end": current[-1]["end"],
            "text": " ".join(t["word"] for t in current),
            "words": current,
        })
    return segs

# -----------------------------------------------------------------------------
#   DARABOLÁS + TRANSZKRIPCIÓ
# -----------------------------------------------------------------------------

def transcribe_long_audio_final(asr_model, audio_path, *, chunk_len_s: int, batch_size: int, device: torch.device, max_pause_s: float):
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
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
            if np.max(np.abs(chunk)) < 0.001:
                continue
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
                    word_segments.append({
                        "word": w.get("word", w.get("text", "")),
                        "start": round(w["start"] + off, 3),
                        "end": round(w["end"] + off, 3),
                        "score": w.get("conf") or w.get("confidence"),
                    })
            elif hasattr(h, "words") and h.words and not isinstance(h.words[0], str):
                for w in h.words:
                    word_segments.append({
                        "word": w.word,
                        "start": round(w.start_time + off, 3),
                        "end": round(w.end_time + off, 3),
                        "score": round(getattr(w, "score", 0.0), 4),
                    })

        if not word_segments:
            return None

        sentence_segments = sentence_segments_from_words(word_segments, max_pause_s=max_pause_s)
        print(f"  - {len(sentence_segments)} mondatszegmens generálva (max_pause={max_pause_s}s).")

        return {"segments": sentence_segments, "word_segments": word_segments, "language": "en"}

# -----------------------------------------------------------------------------
#   MAPPÁK FELDOLGOZÁSA
# -----------------------------------------------------------------------------

def process_directory_final_v3(directory_path: str, auto_chunk: bool, fixed_chunk_s: int, max_pause_s: float):
    model_name = "nvidia/parakeet-tdt-0.6b-v2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Modell betöltése: {model_name}")
    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        asr_model.to(device)
        enable_word_timestamps(asr_model)
        log_gpu_memory(device, "(after model load)")
    except Exception as e:
        raise RuntimeError(f"Modell‑betöltési hiba: {e}")

    chunk_len_s = calibrate_chunk_size(asr_model, device) if auto_chunk and device.type == "cuda" else fixed_chunk_s
    if not auto_chunk:
        print(f"  - Automatikus kalibráció kikapcsolva → fix {chunk_len_s}s chunk")

    wavs = [str(p) for p in Path(directory_path).iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not wavs:
        print("Nem található támogatott hangfájl.")
        return

    print(f"\n{len(wavs)} fájl feldolgozása {chunk_len_s}s darabokkal…")
    for ap in wavs:
        apath = Path(ap)
        out_json = apath.with_suffix(".json")
        print("-" * 50)
        print(f"▶  {apath.name}")

        res = transcribe_long_audio_final(
            asr_model,
            ap,
            chunk_len_s=chunk_len_s,
            batch_size=8,
            device=device,
            max_pause_s=max_pause_s
        )

        if res and (res.get("word_segments") or res.get("segments")):
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=4, ensure_ascii=False)
            print(f"  ✔  Mentve: {out_json.name}")
        else:
            print("  ✖  Sikertelen vagy üres transzkripció.")
    print("\nKész.")

# -----------------------------------------------------------------------------
#   CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Transcribe audio files with NVIDIA NeMo (word‑level timestamps, v3.9-pause-split)")
    p.add_argument("-i", "--input-directory", required=True, help="Mappa, ahol a hangfájlok találhatók.")
    p.add_argument("--no-auto-chunk", action="store_true", help="Kalibráció kikapcsolása.")
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK_S, help=f"Fix chunk‑hossz mp‑ben auto‑chunk nélkül (alapért.: {DEFAULT_CHUNK_S}s).")
    p.add_argument("--max-pause", type=float, default=DEFAULT_MAX_PAUSE_S, help=f"Mondatok közti max. szünet mp-ben (alapért.: {DEFAULT_MAX_PAUSE_S}s).")
    args = p.parse_args()

    process_directory_final_v3(
        directory_path=args.input_directory,
        auto_chunk=not args.no_auto_chunk,
        fixed_chunk_s=args.chunk,
        max_pause_s=args.max_pause
    )

if __name__ == "__main__":
    main()
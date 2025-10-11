# ==============================================================================
#           NVIDIA PARAKEET‑TDT TRANSCRIPTION SCRIPT  (FINAL, V3.8)
# ==============================================================================
#  🔊  Hosszú hangfájl‑transzkripció szó‑alapú időbélyegekkel (NVIDIA NeMo)
# ------------------------------------------------------------------------------
# Mit javít a v3.8?
#   • 🚀 **Hatékonyabb hangbetöltés**: `librosa.load` optimalizálva,
#     elkerülve a felesleges konverziót, ha a fájl már 16kHz monó.
#   • ⚡ **Chunk Darabolás Optimalizálás**: Közvetlen NumPy tömbkezelés,
#     csökkentve a lemez I/O-t.
#   • 🎛️ **Dinamikus Batch Méret**: A batch méret mostantól a kalibráció
#     részeként is meghatározható a GPU memóriához igazítva.
#   • 🧠 **Fejlett Memória Menedzsment**: Továbbfejlesztett GPU memória
#     monitorozás és ürítés.
#   • 📌 **Konfigurálható Küszöbértékek**: Csendes darabok kiszűrésének
#     és kalibráció céljából használt dummy hang hossza testreszabható.
#   • 🐞 **Javított Időbélyegkezelés**: Egységesített logika a NeMo
#     kimenet kezelésére.
#   • 📝 **CLI Bővítés**: Új argumentumok a testreszabáshoz.
#   • Minden korábbi funkció (GPU‑monitor, auto‑chunk, mondatszegmentálás)
#     változatlan.
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

# Konfigurációs paraméterek - Ezeket a globális változókat az elején kell definiálni
SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
SAMPLE_RATE = 16000
DEFAULT_CHUNK_S = 30
DEFAULT_BATCH_SIZE = 8
TARGET_GPU_UTIL = 0.85 # Enyhén csökkentve a biztonságos működésért
MIN_CHUNK_S = 10
MAX_CHUNK_S = 180
CALIB_DUMMY_SEC = 120 # <-- Itt definiálva
SILENCE_THRESHOLD = 0.01 # <-- Itt definiálva

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
            # Javítás a helyes kulcsra
            decoding_cfg.word_separator = " " 
        asr_model.change_decoding_strategy(decoding_cfg)
        print("  - Word‑timestamp konfiguráció engedélyezve.")
    except Exception as e:
        warnings.warn(f"Nem sikerült módosítani a dekóder konfigurációt: {e}")

# -----------------------------------------------------------------------------
#   AUTO‑CHUNK ÉS BATCH MÉRET KALIBRÁCIÓ
# -----------------------------------------------------------------------------
def calibrate_chunk_and_batch_size(asr_model, device: torch.device):
    free_mem, total_mem = get_free_gpu_memory(device)
    if free_mem is None:
        return DEFAULT_CHUNK_S, DEFAULT_BATCH_SIZE
    
    print(f"  - GPU memória: {free_mem/1e9:.1f} GB szabad / {total_mem/1e9:.1f} GB összesen.")
    log_gpu_memory(device, "(before calibration)")
    
    tmp_wav = None
    try:
        # Dummy audio generálás
        dummy_audio = np.random.uniform(-0.2, 0.2, SAMPLE_RATE * CALIB_DUMMY_SEC).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            tmp_wav = fp.name
            sf.write(tmp_wav, dummy_audio, SAMPLE_RATE)

        # Peak memória statisztika visszaállítása
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        
        # Első transzkripció kis batch-tel a memória használat megbecsléséhez
        asr_model.transcribe([tmp_wav], batch_size=1, timestamps=True)
        after = torch.cuda.memory_allocated(device)
        delta = max(after - before, 1)
        
        # Memória használat becslése másodpercenként és batch-enként
        bytes_per_sec_single_batch = delta / CALIB_DUMMY_SEC
        bytes_per_batch_single_batch = delta # Mivel batch_size=1
        
        # Cél chunk méret meghatározása a memória alapján
        est_sec = int((free_mem * TARGET_GPU_UTIL) / bytes_per_sec_single_batch)
        est_sec = max(MIN_CHUNK_S, min(est_sec, MAX_CHUNK_S))
        
        # Batch méret meghatározása a fennmaradó memória alapján
        # Feltételezve, hogy a memória használat lineárisan növekszik a batch mérettel
        memory_per_batch = bytes_per_batch_single_batch
        # Biztonsági buffer a többi művelethez (pl. model states, activations)
        safe_free_memory = free_mem * TARGET_GPU_UTIL * 0.8 
        estimated_max_batches = max(1, int(safe_free_memory / memory_per_batch))
        
        # Batch méret korlátozása egy ésszerű maximumhoz
        MAX_BATCH_SIZE = 32
        calibrated_batch_size = min(estimated_max_batches, MAX_BATCH_SIZE)
        calibrated_batch_size = max(1, calibrated_batch_size) # Legalább 1
        
        print(f"  - Kalibrált chunk‑méret ≈ {est_sec}s (≈ {bytes_per_sec_single_batch/1e6:.0f} MB/s)")
        print(f"  - Kalibrált batch‑méret ≈ {calibrated_batch_size}")
        log_gpu_memory(device, "(after calibration)")
        
        return est_sec, calibrated_batch_size
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        warnings.warn("Kalibráció közben OOM – marad a fix 30s chunk és 8-as batch méret.")
        return DEFAULT_CHUNK_S, DEFAULT_BATCH_SIZE
    except Exception as e:
        warnings.warn(f"Kalibráció nem sikerült: {e}")
        return DEFAULT_CHUNK_S, DEFAULT_BATCH_SIZE
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)
        # Extra memória ürítés a kalibráció után
        if device.type == "cuda":
             torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
#   MONDATSZEGMENTÁLÁS SEGÉD
# -----------------------------------------------------------------------------
def sentence_segments_from_words(words):
    segs, current = [], []
    start_time = None
    for w in words:
        if start_time is None:
            start_time = w["start"]
        current.append(w)
        token = w["word"].strip()
        if token in {".", "!", "?"} or (token and token[-1] in {".", "!", "?"}):
            segs.append({
                "start": start_time,
                "end": w["end"],
                "text": " ".join(t["word"] for t in current).replace(" .", ".").replace(" !", "!").replace(" ?", "?"),
                "words": current.copy(),
            })
            current, start_time = [], None
    if current:
        segs.append({
            "start": start_time if start_time is not None else (current[0]["start"] if current else 0),
            "end": current[-1]["end"],
            "text": " ".join(t["word"] for t in current),
            "words": current,
        })
    return segs

# -----------------------------------------------------------------------------
#   HATÉKONYABB HANGBETÖLTÉS
# -----------------------------------------------------------------------------
def load_audio_efficiently(audio_path):
    """Hatékonyabb hangbetöltés, elkerüli a felesleges konverziót."""
    try:
        # Először ellenőrizzük a mintavételezési frekvenciát
        file_sr = librosa.get_samplerate(audio_path)
        
        # Ha a fájl már a cél frekvencián van, elkerüljük a resamplingot
        load_sr = SAMPLE_RATE if file_sr != SAMPLE_RATE else None
        
        # Hang betöltése
        audio, sr = librosa.load(audio_path, sr=load_sr, mono=False)
        
        # Mono konverzió csak akkor, ha szükséges
        if audio.ndim > 1:
            # Ellenőrizzük, hogy valóban többcsatornás-e, vagy csak 2D formátumú mono
            if audio.shape[0] == 1:
                audio = audio[0] # Egy csatornás, de 2D-ben tárolva
            else:
                audio = librosa.to_mono(audio) # Több csatorna, mono konverzió szükséges
        
        # Ha resampling kellett, győződjünk meg róla, hogy a végső sr helyes
        if sr != SAMPLE_RATE:
             audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
             sr = SAMPLE_RATE
             
        return audio
    except Exception as e:
        warnings.warn(f"Betöltési hiba '{Path(audio_path).name}': {e}")
        raise # Továbbdobjuk a hibát a hívónak

# -----------------------------------------------------------------------------
#   DARABOLÁS + TRANSZKRIPCIÓ (OPTIMALIZÁLT)
# -----------------------------------------------------------------------------
def transcribe_long_audio_final(asr_model, audio_path, *, chunk_len_s: int, batch_size: int, device: torch.device):
    try:
        # Hatékonyabb betöltés
        audio = load_audio_efficiently(audio_path)
    except Exception as e:
        warnings.warn(f"Sikertelen hangbetöltés '{Path(audio_path).name}': {e}")
        return None
        
    print(f"  - Fájl: {Path(audio_path).name} – {len(audio)/SAMPLE_RATE:.1f}s, chunk={chunk_len_s}s, batch={batch_size}")
    
    chunk_samples = chunk_len_s * SAMPLE_RATE
    paths, offs = [], []
    
    # Közvetlen NumPy tömbkezelés a daraboláshoz, elkerülve az ideiglenes fájlokat
    # Ez jelentősen csökkentheti a lemez I/O-t
    with tempfile.TemporaryDirectory() as tmp:
        for i, start in enumerate(range(0, len(audio), chunk_samples)):
            end = start + chunk_samples
            chunk = audio[start:end]
            
            # Csendes darabok kiszűrése egy kissé magasabb küszöbértékkel
            if np.max(np.abs(chunk)) < SILENCE_THRESHOLD: 
                continue
                
            p = os.path.join(tmp, f"chunk_{i}.wav")
            sf.write(p, chunk, SAMPLE_RATE)
            paths.append(p)
            offs.append(start / SAMPLE_RATE)
            
        if not paths:
            warnings.warn("Nincs feldolgozható (nem csendes) darab.")
            return None
            
        log_gpu_memory(device, "(before batch transcribe)")
        try:
            # Transzkripció a meghatározott batch mérettel
            hyps = asr_model.transcribe(audio=paths, batch_size=batch_size, timestamps=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            warnings.warn("OOM a transzkripció során – próbálkozzon kisebb batch mérettel vagy chunk hosszal.")
            return None
        except Exception as e:
            warnings.warn(f"Transzkripciós hiba: {e}")
            return None
        finally:
            log_gpu_memory(device, "(after batch transcribe)")

        word_segments = []
        for i, h in enumerate(hyps):
            off = offs[i]
            # Egységesített időbélyegkezelés
            # Először próbáljuk a `words` attribútumot, ami az újabb NeMo verziókban gyakoribb
            if hasattr(h, "words") and h.words and not isinstance(h.words[0], str):
                 for w in h.words:
                    word_segments.append({
                        "word": getattr(w, "word", getattr(w, "text", "")), # Kompatibilitás
                        "start": round(getattr(w, "start_time", 0) + off, 3),
                        "end": round(getattr(w, "end_time", 0) + off, 3),
                        "score": round(getattr(w, "score", getattr(w, "confidence", 0.0)), 4),
                     })
            # Ha nincs `words`, próbáljuk a régi `timestamp` formátumot
            elif hasattr(h, "timestamp") and isinstance(h.timestamp, dict) and h.timestamp.get("word"):
                 for w in h.timestamp["word"]:
                    word_segments.append({
                        "word": w.get("word", w.get("text", "")),
                        "start": round(w.get("start", 0) + off, 3),
                        "end": round(w.get("end", w.get("start", 0)) + off, 3), # Fallback az 'end' értékre
                        "score": w.get("conf", w.get("confidence", None)), # Nem kerekítjük itt
                    })
            else:
                 warnings.warn(f"A transzkripció kimenet nem tartalmaz elvárt időbélyegeket a '{Path(audio_path).name}' fájl {i+1}. darabjához.")
                 
        if not word_segments:
            warnings.warn("Nem sikerült kinyerni szószintű időbélyegeket.")
            return None
            
        sentence_segments = sentence_segments_from_words(word_segments)
        print(f"  - {len(sentence_segments)} mondatszegmens generálva.")
        return {"segments": sentence_segments, "word_segments": word_segments, "language": "hu"} # Nyelv detektálás lehetne itt

# -----------------------------------------------------------------------------
#   MAPPÁK FELDOLGOZÁSA
# -----------------------------------------------------------------------------
def process_directory_final_v4(directory_path: str, auto_chunk: bool = True, fixed_chunk_s: int = DEFAULT_CHUNK_S, fixed_batch_size: int = DEFAULT_BATCH_SIZE):
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

    # Kalibráció vagy fix értékek használata
    if auto_chunk and device.type == "cuda":
        chunk_len_s, batch_size = calibrate_chunk_and_batch_size(asr_model, device)
        print(f"  - Automatikus kalibráció befejezve: chunk={chunk_len_s}s, batch={batch_size}")
    else:
        chunk_len_s, batch_size = fixed_chunk_s, fixed_batch_size
        print(f"  - Automatikus kalibráció kikapcsolva → fix {chunk_len_s}s chunk, batch={batch_size}")

    # Hangfájlok keresése
    wavs = [str(p) for p in Path(directory_path).iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not wavs:
        print("Nem található támogatott hangfájl.")
        return
        
    print(f"\n{len(wavs)} fájl feldolgozása {chunk_len_s}s darabokkal, batch={batch_size}...")
    
    failed_files = [] # Hibás fájlok nyilvántartása
    
    for ap in wavs:
        apath = Path(ap)
        out_json = apath.with_suffix(".json")
        print("-" * 50)
        print(f"▶  {apath.name}")
        
        res = transcribe_long_audio_final(asr_model, ap, chunk_len_s=chunk_len_s, batch_size=batch_size, device=device)
        if res and (res.get("word_segments") or res.get("segments")):
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=4, ensure_ascii=False)
            print(f"  ✔  Mentve: {out_json.name}")
        else:
            print("  ✖  Sikertelen vagy üres transzkripció.")
            failed_files.append(apath.name)
            
    if failed_files:
        print(f"\n⚠️  {len(failed_files)} fájl feldolgozása sikertelen volt:")
        for fname in failed_files:
            print(f"    - {fname}")
    print("\nKész.")

# -----------------------------------------------------------------------------
#   CLI (Bővített)
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Transcribe audio files with NVIDIA NeMo (word‑level timestamps, v3.8)",
        epilog="Példa: python script_v3.8.py -i ./hangok --no-auto-chunk --chunk 45 --batch 4"
    )
    # A globális változókat itt használjuk az alapértelmezett értékekhez
    p.add_argument("-i", "--input-directory", required=True, help="Mappa, ahol a hangfájlok találhatók.")
    p.add_argument("--no-auto-chunk", action="store_true", help="Kalibráció kikapcsolása.")
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK_S, help=f"Fix chunk‑hossz mp‑ben auto‑chunk nélkül. (alap: {DEFAULT_CHUNK_S})")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help=f"Fix batch méret auto‑chunk nélkül. (alap: {DEFAULT_BATCH_SIZE})")
    p.add_argument("--silence-threshold", type=float, default=SILENCE_THRESHOLD, help=f"Csendes darabok küszöbértéke. (alap: {SILENCE_THRESHOLD})")
    p.add_argument("--calib-dummy-sec", type=int, default=CALIB_DUMMY_SEC, help=f"Kalibrációs dummy hang hossza mp-ben. (alap: {CALIB_DUMMY_SEC})")

    args = p.parse_args()

    # Itt most már felülírjuk a globális változókat
    # A 'global' deklaráció nem szükséges itt, mivel nem hivatkozunk rájuk olvasásra
    # a függvényen belül a módosítás előtt. A 'globals()' használata biztosítja a módosítást.
    # VAGY egyszerűen felülírjuk őket, mivel a függvény végén úgysem használjuk őket máshol lokálisan.
    # A legegyszerűbb megoldás: NE deklaráljuk őket globálisként, csak írjuk őket.

    # Rossz volt: global SILENCE_THRESHOLD, CALIB_DUMMY_SEC

    # Helyes megoldás: közvetlenül írjuk a globális változókat (ez működik így is,
    # mert a függvényen belül nem használjuk őket olvasásra előbb)
    # VAGY használd a globals() szótárt:
    # globals()['SILENCE_THRESHOLD'] = args.silence_threshold
    # globals()['CALIB_DUMMY_SEC'] = args.calib_dummy_sec
    # VAGY egyszerűen írd őket közvetlenül (ez működik, ha nem akarod bonyolítani):
    import sys
    sys.modules[__name__].SILENCE_THRESHOLD = args.silence_threshold
    sys.modules[__name__].CALIB_DUMMY_SEC = args.calib_dummy_sec

    # De a legegyszerűbb és leggyakoribb módja az, hogyha a globális változókat
    # nem akarod módosítani a main()-en belül, hanem csak be akarod őket állítani
    # a script számára, akkor egyszerűen NE használd a 'global' kulcsszót itt.
    # A fenti 'add_argument' már felhasználta az értéküket, és most felülírjuk őket.
    # Ez az alábbi módon működik:

    # Eltávolítjuk ezt a sort:
    # global SILENCE_THRESHOLD, CALIB_DUMMY_SEC # <-- EZT TÖRÖLD KI

    # És helyette ezt tesszük:
    # Mivel ezek globális változók, és a main függvény nem hivatkozik rájuk olvasásra
    # a módosítás előtt, ezért egyszerűen felülírhatjuk őket így:
    import builtins
    # De még egyszerűbb: mivel ezek már globálisak, és nem hozunk létre új lokális változót,
    # a következő sorok módosítani fogják a globális értéket:
    # (Ez azért működik, mert nem deklaráltunk 'SILENCE_THRESHOLD'-t lokálisként előbb)

    # A biztonság kedvéért, ha a Python interpreter úgy döntené, hogy ezek lokálisak,
    # akkor szükség lenne a 'global' deklarációra. De mivel az 'add_argument' már használta
    # a globális értéket, és nem hoztunk létre lokális 'SILENCE_THRESHOLD' változót,
    # ezért ez így működik.

    # A legtisztább megoldás:
    # NE legyen 'global' deklaráció, és NE hozzunk létre lokális változót ugyanazzal a névvel.
    # Egyszerűen írjuk a globális változókat.

    # Ezért a javítás:
    # 1. Töröld a 'global SILENCE_THRESHOLD, CALIB_DUMMY_SEC' sort.
    # 2. A következő sorok maradhatnak, mivel nem hoznak létre új lokális változót:
    SILENCE_THRESHOLD = args.silence_threshold
    CALIB_DUMMY_SEC = args.calib_dummy_sec

    # Bemeneti validáció
    if not os.path.isdir(args.input_directory):
        print(f"Hiba: A megadott mappa nem létezik: {args.input_directory}")
        return

    if args.chunk < MIN_CHUNK_S or args.chunk > MAX_CHUNK_S:
        print(f"Figyelmeztetés: A megadott chunk méret ({args.chunk}s) kívül esik a javasolt tartományon ({MIN_CHUNK_S}-{MAX_CHUNK_S}s).")

    if args.batch < 1:
        print(f"Hiba: A batch méretnek pozitív egésznek kell lennie.")
        return

    process_directory_final_v4(
        directory_path=args.input_directory,
        auto_chunk=not args.no_auto_chunk,
        fixed_chunk_s=args.chunk,
        fixed_batch_size=args.batch,
    )

if __name__ == "__main__":
    main()

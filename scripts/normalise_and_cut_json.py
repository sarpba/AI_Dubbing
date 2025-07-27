# --- START OF FILE 4, normalise_and_cut.py ---

import os
import argparse
import wave
import contextlib
import webrtcvad
import subprocess
import tempfile
import multiprocessing
import time
import sys
from pathlib import Path
from pydub import AudioSegment
import datetime # ### VÁLTOZTATÁS ###

try:
    from tqdm import tqdm
except ImportError:
    print("A 'tqdm' könyvtár nem található. A folyamatjelző nem lesz elérhető.")
    print("Telepítés: pip install tqdm")
    def tqdm(iterable, **kwargs):
        return iterable

# --- VAD (Voice Activity Detection) logika... (Változatlan) ---
ALLOWED_SAMPLE_RATES = (8000, 16000, 32000, 48000)
TARGET_SAMPLE_RATE = 16000

def convert_to_allowed_format(path):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()
    try:
        command = ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-acodec", "pcm_s16le", temp_file.name]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        os.remove(temp_file.name)
        raise RuntimeError("ffmpeg nem található. Kérlek, telepítsd és add hozzá a PATH-hoz.")
    except subprocess.CalledProcessError as e:
        os.remove(temp_file.name)
        error_message = e.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f"FFmpeg konverzió sikertelen a(z) '{path}' fájlnál: {error_message}")
    return temp_file.name

def read_wave_for_vad(path):
    converted_path = None
    try:
        converted_path = convert_to_allowed_format(path)
        with contextlib.closing(wave.open(converted_path, 'rb')) as wf:
            return wf.readframes(wf.getnframes()), wf.getframerate(), wf.getsampwidth()
    finally:
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)

def frame_generator(frame_duration_ms, audio, sample_rate, sample_width):
    bytes_per_frame = int(sample_rate * (frame_duration_ms / 1000.0) * sample_width)
    offset, timestamp, frame_duration = 0, 0.0, frame_duration_ms / 1000.0
    while offset + bytes_per_frame <= len(audio):
        yield audio[offset:offset+bytes_per_frame], timestamp
        timestamp += frame_duration
        offset += bytes_per_frame

def get_speech_start_time(audio_path):
    try:
        audio, sample_rate, sample_width = read_wave_for_vad(audio_path)
        vad = webrtcvad.Vad(3)
        for frame, timestamp in frame_generator(30, audio, sample_rate, sample_width):
            if vad.is_speech(frame, sample_rate):
                return timestamp
    except Exception:
        return None
    return None

# --- Fájlfeldolgozó függvények (Változatlan) ---
def process_single_file_timing(input_audio_path, reference_audio_path, delete_empty):
    result = {"status": "skipped_timing", "time_diff": 0.0, "action": "none"}
    input_start_time = get_speech_start_time(input_audio_path)
    reference_start_time = get_speech_start_time(reference_audio_path)
    if input_start_time is None or reference_start_time is None:
        result["status"] = "error_vad"
        if delete_empty and input_start_time is None and os.path.exists(input_audio_path):
            try:
                os.remove(input_audio_path)
                result["status"] = "deleted_no_speech"
            except OSError:
                result["status"] = "error_deleting"
        return result
    time_difference = reference_start_time - input_start_time
    result["time_diff"] = time_difference
    if abs(time_difference) < 0.001:
        result["status"] = "success"
        return result
    try:
        audio = AudioSegment.from_file(input_audio_path)
    except Exception:
        result["status"] = "error_loading_audio"
        return result
    try:
        if time_difference > 0:
            result["action"] = "added_silence"
            adjusted_audio = AudioSegment.silent(duration=time_difference * 1000) + audio
        else:
            trim_duration = abs(time_difference) * 1000
            if trim_duration >= len(audio):
                result["status"] = "error_trim_too_long"
                if delete_empty: os.remove(input_audio_path)
                return result
            result["action"] = "trimmed_silence"
            adjusted_audio = audio[trim_duration:]
        file_format = os.path.splitext(input_audio_path)[1].lstrip('.').lower()
        adjusted_audio.export(input_audio_path, format=file_format if file_format in ['mp3', 'wav'] else 'wav')
        result["status"] = "success"
    except Exception:
        result["status"] = "error_exporting"
    return result

def synchronize_single_file_loudness(sync_path, reference_path, min_db):
    result = {"status": "skipped_loudness", "gain_applied": 0.0}
    try:
        reference_audio = AudioSegment.from_file(reference_path)
        reference_peak = reference_audio.max_dBFS
        if reference_peak < min_db:
            result["status"] = "skipped_low_ref_db"
            return result
        sync_audio = AudioSegment.from_file(sync_path)
        gain = reference_peak - sync_audio.max_dBFS
        result["gain_applied"] = gain
        if abs(gain) < 0.1:
            result["status"] = "success"
            return result
        adjusted_sync_audio = sync_audio.apply_gain(gain)
        file_format = os.path.splitext(sync_path)[1].lstrip('.').lower()
        adjusted_sync_audio.export(sync_path, format=file_format if file_format in ['mp3', 'wav'] else 'wav')
        result["status"] = "success"
    except Exception:
        result["status"] = "error_syncing_loudness"
    return result

# --- ### VÁLTOZTATÁS KEZDETE: Új segédfüggvény és átírt munkás függvény ### ---

def parse_filename_to_seconds(filename: str):
    """
    Átalakítja a 'HH-MM-SS-ms_HH-MM-SS-ms.wav' formátumú fájlnevet
    (start_másodperc, end_másodperc) tuple-re.
    """
    try:
        base_name = os.path.splitext(filename)[0]
        start_str, end_str = base_name.split('_')
        
        def parse_time(time_str):
            parts = time_str.split('-')
            h, m, s, ms = map(int, parts)
            return h * 3600 + m * 60 + s + ms / 1000.0

        return parse_time(start_str), parse_time(end_str)
    except Exception:
        return None, None

def process_file_pair(task_args):
    """
    Egyetlen fájl teljes feldolgozását végzi. Dinamikusan létrehozza
    a referencia audiót a forrásfájlból a fájlnév alapján.
    """
    input_path, source_audio, args = task_args
    filename = os.path.basename(input_path)
    
    final_result = {"file": filename, "timing": None, "loudness": None}

    start_sec, end_sec = parse_filename_to_seconds(filename)
    if start_sec is None or end_sec is None:
        final_result["timing"] = {"status": "error_parsing_filename"}
        return final_result

    try:
        # A referencia audió szelet kivágása
        start_ms = start_sec * 1000
        end_ms = end_sec * 1000
        reference_chunk = source_audio[start_ms:end_ms]

        # Ideiglenes fájl létrehozása a referenciának
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_ref_file:
            reference_chunk.export(tmp_ref_file.name, format="wav")
            
            # 1. Lépés: Időzítés normalizálása
            final_result["timing"] = process_single_file_timing(input_path, tmp_ref_file.name, args.delete_empty)
            
            # 2. Lépés: Hangerő szinkronizálása
            timing_success = final_result["timing"]["status"] == "success"
            if timing_success and args.sync_loudness and os.path.exists(input_path):
                final_result["loudness"] = synchronize_single_file_loudness(input_path, tmp_ref_file.name, args.min_db)

    except Exception as e:
        final_result["timing"] = {"status": f"error_creating_ref: {e}"}

    return final_result

# --- ### VÁLTOZTATÁS VÉGE --- ###

# --- STATISZTIKA (Változatlan) ---
def print_statistics(results, duration):
    total_files = len(results)
    timing_success = sum(1 for r in results if r['timing'] and r['timing']['status'] == 'success')
    silence_added = [r['timing']['time_diff'] for r in results if r['timing'] and r['timing']['action'] == 'added_silence']
    silence_trimmed = [abs(r['timing']['time_diff']) for r in results if r['timing'] and r['timing']['action'] == 'trimmed_silence']
    timing_errors = total_files - timing_success
    loudness_results = [r['loudness'] for r in results if r.get('loudness')]
    loudness_success = sum(1 for r in loudness_results if r['status'] == 'success')
    gains_applied = [r['gain_applied'] for r in loudness_results if r['status'] == 'success' and abs(r['gain_applied']) >= 0.1]
    loudness_errors = len(loudness_results) - loudness_success
    deleted = sum(1 for r in results if r['timing'] and r['timing']['status'] in ['deleted_no_speech', 'error_trim_too_long'])

    print("\n" + "="*50)
    print("FELDOLGOZÁSI STATISZTIKA")
    print("="*50)
    print(f"Teljes feldolgozási idő: {duration:.2f} másodperc")
    print(f"Feldolgozott fájlok száma: {total_files}")
    print("\n--- Időzítés normalizálása ---")
    print(f"  Sikeres: {timing_success}")
    print(f"  Hibás/Kihagyott: {timing_errors}")
    print(f"  Törölt fájlok: {deleted}")
    if silence_added:
        print(f"  Hozzáadott csend (átlag): {sum(silence_added)/len(silence_added):.3f} s")
    if silence_trimmed:
        print(f"  Levágott csend (átlag): {sum(silence_trimmed)/len(silence_trimmed):.3f} s")
    print("\n--- Hangerő szinkronizálása ---")
    print(f"  Feldolgozásra jelölt: {len(loudness_results)}")
    print(f"  Sikeres: {loudness_success}")
    print(f"  Hibás/Kihagyott: {loudness_errors}")
    if gains_applied:
        print(f"  Alkalmazott erősítés (átlag): {sum(gains_applied)/len(gains_applied):.2f} dB")
    print("="*50)


# --- FŐ VEZÉRLŐ LOGIKA ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dinamikusan normalizálja az audió fájlok elején lévő csendet és a hangerőt. "
                    "Minden fájlhoz a forrás audióból vágja ki a referenciát a fájlnévben kódolt időbélyegek alapján.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- ### VÁLTOZTATÁS KEZDETE: Új argumentumok ---
    parser.add_argument("-i", "--input_dir", required=True, help="A módosítandó, szegmentált audió fájlokat tartalmazó könyvtár.")
    parser.add_argument("--source_audio_dir", required=True, help="A nagy, egybefüggő forrás audiófájlt tartalmazó könyvtár.")
    parser.add_argument("--source_json_dir", required=True, help="A forrás audióhoz tartozó szegmentációs JSON-t tartalmazó könyvtár (jelenleg csak konzisztencia miatt).")
    # --- ### VÁLTOZTATÁS VÉGE ---
    parser.add_argument("--delete_empty", action="store_true", help="Törli az audió fájlt, ha a VAD nem talál benne beszédet.")
    parser.add_argument('--no-sync-loudness', dest='sync_loudness', action='store_false', help='Kikapcsolja a hangerő szinkronizálását.')
    parser.set_defaults(sync_loudness=True)
    parser.add_argument("-db", "--min_db", type=float, default=-40.0, help="Minimális referencia hangerőszint (dB) a szinkronizáláshoz.")
    
    args = parser.parse_args()
    start_time = time.time()
    
    # --- ### VÁLTOZTATÁS KEZDETE: Forrásfájlok betöltése ### ---
    print("Forrás audió és JSON keresése...")
    source_audio_dir = Path(args.source_audio_dir)
    source_audio_files = list(source_audio_dir.glob('*.wav')) + list(source_audio_dir.glob('*.mp3'))
    if not source_audio_files:
        print(f"[HIBA] Nem található forrás audió fájl a könyvtárban: {source_audio_dir}")
        sys.exit(1)
    if len(source_audio_files) > 1:
        print(f"[FIGYELEM] Több forrás audiófájl található, a következőt használom: {source_audio_files[0].name}")
    source_audio_path = source_audio_files[0]

    # A nagy forrásfájl betöltése a memóriába egyszer, a pydub segítségével
    try:
        print(f"A forrás audió betöltése: {source_audio_path}...")
        global_source_audio = AudioSegment.from_file(source_audio_path)
        print("A forrás audió sikeresen betöltve a memóriába.")
    except Exception as e:
        print(f"[HIBA] Nem sikerült betölteni a forrás audiófájlt: {e}")
        sys.exit(1)
    # --- ### VÁLTOZTATÁS VÉGE ---

    # 1. LÉPÉS: A feladatok összegyűjtése
    tasks_to_process = []
    print("Feldolgozandó fájlok keresése...")
    for dirpath, _, filenames in os.walk(args.input_dir):
        for filename in filenames:
            if filename.lower().endswith((".mp3", ".wav")):
                input_path = os.path.join(dirpath, filename)
                # A feladathoz hozzáadjuk a betöltött audió objektumot is
                tasks_to_process.append((input_path, global_source_audio, args))

    if not tasks_to_process:
        print("Nincs feldolgozandó fájl. A program leáll.")
    else:
        # 2. LÉPÉS: Párhuzamos végrehajtás
        cpu_count = os.cpu_count() or 1
        print(f"\nÖsszesen {len(tasks_to_process)} feladat feldolgozása indul {cpu_count} CPU magon...")
        results = []
        with multiprocessing.Pool(processes=cpu_count) as pool:
            for result in tqdm(pool.imap_unordered(process_file_pair, tasks_to_process), total=len(tasks_to_process), desc="Fájlok feldolgozása", unit="fájl"):
                results.append(result)
        
        duration = time.time() - start_time
        print_statistics(results, duration)
# --- START OF FILE normalise_and_cut_json_easy.py ---

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
import json

try:
    from tqdm import tqdm
except ImportError:
    print("A 'tqdm' könyvtár nem található. A folyamatjelző nem lesz elérhető.")
    print("Telepítés: pip install tqdm")
    def tqdm(iterable, **kwargs):
        return iterable

# --- VAD (Voice Activity Detection) logika ---
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

# --- Fájlfeldolgozó függvények ---
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
                if delete_empty and os.path.exists(input_audio_path):
                    os.remove(input_audio_path)
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

# --- Munkás függvények ---

worker_source_audio = None

def init_worker(audio_path_str: str):
    global worker_source_audio
    try:
        print(f"Munkás processz (PID: {os.getpid()}) betölti a forrás audiót: {audio_path_str}")
        worker_source_audio = AudioSegment.from_file(audio_path_str)
        print(f"Munkás processz (PID: {os.getpid()}) sikeresen betöltötte a forrás audiót.")
    except Exception as e:
        print(f"[HIBA] Munkás processz (PID: {os.getpid()}) nem tudta betölteni az audiót: {e}")
        worker_source_audio = None

def parse_filename_to_seconds(filename: str):
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
    input_path, args = task_args
    filename = os.path.basename(input_path)
    
    global worker_source_audio
    
    final_result = {"file": filename, "timing": None, "loudness": None}

    if worker_source_audio is None:
        final_result["timing"] = {"status": "error_worker_initialization_failed"}
        return final_result

    start_sec, end_sec = parse_filename_to_seconds(filename)
    if start_sec is None or end_sec is None:
        final_result["timing"] = {"status": "error_parsing_filename"}
        return final_result

    try:
        start_ms = start_sec * 1000
        end_ms = end_sec * 1000
        reference_chunk = worker_source_audio[start_ms:end_ms]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_ref_file:
            reference_chunk.export(tmp_ref_file.name, format="wav")
            
            final_result["timing"] = process_single_file_timing(input_path, tmp_ref_file.name, args.delete_empty)
            
            timing_success = final_result.get("timing", {}).get("status") in ["success", "deleted_no_speech", "error_trim_too_long"]
            if timing_success and args.sync_loudness and os.path.exists(input_path):
                final_result["loudness"] = synchronize_single_file_loudness(input_path, tmp_ref_file.name, args.min_db)

    except Exception as e:
        final_result["timing"] = {"status": f"error_creating_ref: {e}"}

    return final_result


# --- STATISZTIKA (JAVÍTOTT) ---
def print_statistics(results, duration):
    total_files = len(results)
    timing_success = sum(1 for r in results if r.get('timing') and r['timing'].get('status') == 'success')
    
    # A .get('action') használatával elkerüljük a KeyError-t, ha a kulcs hiányzik.
    silence_added = [r['timing']['time_diff'] for r in results if r.get('timing') and r['timing'].get('action') == 'added_silence']
    silence_trimmed = [abs(r['timing']['time_diff']) for r in results if r.get('timing') and r['timing'].get('action') == 'trimmed_silence']

    timing_errors = total_files - timing_success
    loudness_results = [r['loudness'] for r in results if r.get('loudness')]
    loudness_success = sum(1 for r in loudness_results if r.get('status') == 'success')
    gains_applied = [r['gain_applied'] for r in loudness_results if r.get('status') == 'success' and abs(r.get('gain_applied', 0)) >= 0.1]
    loudness_errors = len(loudness_results) - loudness_success
    deleted = sum(1 for r in results if r.get('timing') and r['timing'].get('status') in ['deleted_no_speech', 'error_trim_too_long'])

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
        description="Dinamikusan normalizálja az audió fájlok elején lévő csendet és a hangerőt "
                    "egy projekt mappán belül a config.json alapján.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("project_name", help="A projektmappa neve a 'workdir'-en belül.")
    parser.add_argument("--delete_empty", action="store_true", help="Törli az audió fájlt, ha a VAD nem talál benne beszédet.")
    parser.add_argument('--no-sync-loudness', dest='sync_loudness', action='store_false', help='Kikapcsolja a hangerő szinkronizálását.')
    parser.set_defaults(sync_loudness=True)
    parser.add_argument("-db", "--min_db", type=float, default=-40.0, help="Minimális referencia hangerőszint (dB) a szinkronizáláshoz.")
    
    args = parser.parse_args()
    start_time = time.time()
    
    try:
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir.parent / 'config.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"[HIBA] A konfigurációs fájl nem található: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[HIBA] A konfigurációs fájl hibás formátumú: {config_path}")
        sys.exit(1)

    workdir_base = Path(config['DIRECTORIES']['workdir'])
    project_path = workdir_base / args.project_name
    subdirs = config['PROJECT_SUBDIRS']
    input_dir = project_path / subdirs['translated_splits']
    source_audio_dir = project_path / subdirs['separated_audio_speech']
    translated_json_dir = project_path / subdirs['translated']

    print(f"Projekt: {args.project_name}")
    print(f"Konfiguráció betöltve: {config_path.name}")
    print("-" * 30)
    print(f"Forrás JSON könyvtár (ellenőrzés): {translated_json_dir}")
    print(f"Forrás audió könyvtár:             {source_audio_dir}")
    print(f"Bemeneti könyvtár (splits):         {input_dir}")
    print("-" * 30)

    if not project_path.is_dir():
        print(f"[HIBA] A projekt könyvtár nem található: {project_path}")
        sys.exit(1)
    if not translated_json_dir.is_dir() or not any(translated_json_dir.glob('*.json')):
        print(f"[HIBA] A fordítási JSON könyvtár ('{subdirs['translated']}') vagy a benne lévő JSON fájl nem található: {translated_json_dir}")
        sys.exit(1)
    if not source_audio_dir.is_dir():
        print(f"[HIBA] A forrás audiót tartalmazó könyvtár ('{subdirs['separated_audio_speech']}') nem található: {source_audio_dir}")
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"[HIBA] A bemeneti könyvtár ('{subdirs['translated_splits']}') nem található: {input_dir}")
        sys.exit(1)
    
    print("Forrás audió keresése...")
    source_audio_files = list(source_audio_dir.glob('*.wav')) + list(source_audio_dir.glob('*.mp3'))
    if not source_audio_files:
        print(f"[HIBA] Nem található forrás audió fájl a könyvtárban: {source_audio_dir}")
        sys.exit(1)
    if len(source_audio_files) > 1:
        print(f"[FIGYELEM] Több forrás audiófájl található, a következőt használom: {source_audio_files[0].name}")
    source_audio_path = source_audio_files[0]
    
    tasks_to_process = []
    print("Feldolgozandó fájlok keresése...")
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.lower().endswith((".mp3", ".wav")):
                input_path = os.path.join(dirpath, filename)
                tasks_to_process.append((input_path, args))

    if not tasks_to_process:
        print("Nincs feldolgozandó fájl. A program leáll.")
    else:
        cpu_count = os.cpu_count() or 1
        print(f"\nÖsszesen {len(tasks_to_process)} feladat feldolgozása indul {cpu_count} CPU magon...")
        results = []
        
        with multiprocessing.Pool(
            processes=cpu_count, 
            initializer=init_worker, 
            initargs=(str(source_audio_path),)
        ) as pool:
            for result in tqdm(pool.imap_unordered(process_file_pair, tasks_to_process), total=len(tasks_to_process), desc="Fájlok feldolgozása", unit="fájl"):
                results.append(result)
        
        duration = time.time() - start_time
        print_statistics(results, duration)
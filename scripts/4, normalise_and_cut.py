import os
import argparse
import wave
import contextlib
import webrtcvad
import subprocess
import tempfile
import multiprocessing
import time
from pydub import AudioSegment
try:
    from tqdm import tqdm
except ImportError:
    print("A 'tqdm' könyvtár nem található. A folyamatjelző nem lesz elérhető.")
    print("Telepítés: pip install tqdm")
    # Helyettesítő függvény, ha a tqdm nem elérhető
    def tqdm(iterable, **kwargs):
        return iterable

# --- VAD (Voice Activity Detection) logika a dinamikus időbélyeg-generáláshoz ---
# (Ez a rész változatlan, a rövidség kedvéért most nem másolom be újra,
# de a teljes kódban természetesen szerepelnie kell.)
ALLOWED_SAMPLE_RATES = (8000, 16000, 32000, 48000)
TARGET_SAMPLE_RATE = 16000

def convert_to_allowed_format(path):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()
    try:
        command = [
            "ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
            "-acodec", "pcm_s16le", temp_file.name
        ]
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
        # Párhuzamos környezetben a hibaüzeneteket a munkás függvényben kezeljük
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

# --- PÁRHUZAMOSÍTÁSHOZ SZÜKSÉGES MUNKÁS FÜGGVÉNY ---

def process_file_pair(task_args):
    """
    Egyetlen fájlpár teljes feldolgozását végző "munkás" függvény.
    Egy szótárral tér vissza, ami a feldolgozás eredményét összegzi.
    """
    input_audio_path, reference_audio_path, args = task_args
    
    final_result = {"file": os.path.basename(input_audio_path), "timing": None, "loudness": None}

    # 1. Lépés: Időzítés
    final_result["timing"] = process_single_file_timing(input_audio_path, reference_audio_path, args.delete_empty)
    
    # 2. Lépés: Hangerő
    timing_success = final_result["timing"]["status"] == "success"
    if timing_success and args.sync_loudness and os.path.exists(input_audio_path):
        final_result["loudness"] = synchronize_single_file_loudness(input_audio_path, reference_audio_path, args.min_db)
    
    return final_result

# --- STATISZTIKA ---

def print_statistics(results, duration):
    """Kiírja az összesített statisztikát a feldolgozás végén."""
    total_files = len(results)
    
    # Időzítés statisztika
    timing_success = sum(1 for r in results if r['timing']['status'] == 'success')
    silence_added = [r['timing']['time_diff'] for r in results if r['timing']['action'] == 'added_silence']
    silence_trimmed = [abs(r['timing']['time_diff']) for r in results if r['timing']['action'] == 'trimmed_silence']
    timing_errors = total_files - timing_success
    
    # Hangerő statisztika
    loudness_results = [r['loudness'] for r in results if r.get('loudness')]
    loudness_success = sum(1 for r in loudness_results if r['status'] == 'success')
    gains_applied = [r['gain_applied'] for r in loudness_results if r['status'] == 'success' and abs(r['gain_applied']) >= 0.1]
    loudness_errors = len(loudness_results) - loudness_success
    
    deleted = sum(1 for r in results if r['timing']['status'] in ['deleted_no_speech', 'error_trim_too_long'])

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
        description="Párhuzamosan, rekurzívan normalizálja az audió fájlok elején lévő csendet és a hangerőt "
                    "egy referencia könyvtár alapján. Függőségek: pydub, webrtcvad, tqdm. Rendszerkövetelmény: ffmpeg.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Argumentumok... (változatlan)
    parser.add_argument("-i", "--input_dir", required=True, help="A módosítandó audió fájlokat tartalmazó gyökérkönyvtár.")
    parser.add_argument("-r", "--reference_dir", required=True, help="A referencia audió fájlokat tartalmazó gyökérkönyvtár.")
    parser.add_argument("--delete_empty", action="store_true", help="Törli az audió fájlt, ha a VAD nem talál benne beszédet.")
    parser.add_argument('--no-sync-loudness', dest='sync_loudness', action='store_false', help='Kikapcsolja a hangerő szinkronizálását.')
    parser.set_defaults(sync_loudness=True)
    parser.add_argument("-db", "--min_db", type=float, default=-40.0, help="Minimális referencia hangerőszint (dB) a szinkronizáláshoz.")
    
    args = parser.parse_args()

    start_time = time.time()
    
    # 1. LÉPÉS: A feladatok összegyűjtése
    tasks_to_process = []
    print("Fájlok keresése és a feladatlista összeállítása...")
    for dirpath, _, filenames in os.walk(args.input_dir):
        for filename in filenames:
            if not filename.lower().endswith((".mp3", ".wav")):
                continue

            input_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(dirpath, args.input_dir)
            base_name = os.path.splitext(filename)[0]
            
            ref_path_wav = os.path.join(args.reference_dir, relative_path, base_name + '.wav')
            ref_path_mp3 = os.path.join(args.reference_dir, relative_path, base_name + '.mp3')

            reference_path = None
            if os.path.exists(ref_path_wav): reference_path = ref_path_wav
            elif os.path.exists(ref_path_mp3): reference_path = ref_path_mp3
            
            if reference_path:
                tasks_to_process.append((input_path, reference_path, args))
            else:
                print(f"[INFO] A(z) '{input_path}' fájlhoz nem található referencia pár. Kihagyás.")

    if not tasks_to_process:
        print("Nincs feldolgozandó fájl. A program leáll.")
    else:
        # 2. LÉPÉS: A feladatok párhuzamos végrehajtása folyamatjelzővel
        cpu_count = os.cpu_count() or 1
        print(f"\nÖsszesen {len(tasks_to_process)} feladat feldolgozása indul {cpu_count} CPU magon...")
        
        results = []
        with multiprocessing.Pool(processes=cpu_count) as pool:
            # tqdm-et használunk az imap_unordered iterátoron, ami a folyamatjelzést biztosítja
            # A 'desc' a leírás, a 'unit' pedig a számláló egysége
            for result in tqdm(pool.imap_unordered(process_file_pair, tasks_to_process), 
                               total=len(tasks_to_process), 
                               desc="Fájlok feldolgozása", 
                               unit="fájl"):
                results.append(result)
        
        duration = time.time() - start_time
        
        # 3. LÉPÉS: Statisztika kiírása
        print_statistics(results, duration)
import argparse
import subprocess
import json
import os
import sys
import glob
from pathlib import Path

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

def find_first_video_file(directory):
    """Megkeresi az első videófájlt a megadott könyvtárban."""
    supported_formats = ('.mkv', '.mp4', '.avi', '.mov', '.webm')
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(supported_formats):
            return os.path.join(directory, f)
    return None

def find_latest_audio_file(directory):
    """Megkeresi a legfrissebb .wav fájlt a megadott könyvtárban."""
    wav_files = glob.glob(os.path.join(directory, '*.wav'))
    if not wav_files: return None
    return max(wav_files, key=os.path.getmtime)

def get_target_sample_rate(video_file):
    """Lekéri a video első audió streamjének mintavételi rátáját."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=sample_rate', '-of', 'json', video_file]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        info = json.loads(result.stdout)
        return info['streams'][0].get('sample_rate', '48000')
    except Exception:
        return '48000'

def process_audio_to_raw_aac(audio_wav_file, target_sample_rate):
    """A bemeneti WAV fájlt átkódolja nyers .aac (ADTS) formátumra."""
    processed_audio_path = "processed_audio_temp.aac"
    try:
        cmd = ['ffmpeg', '-y', '-i', audio_wav_file, '-c:a', 'aac', '-b:a', '128k', '-ar', str(target_sample_rate), '-ac', '2', processed_audio_path]
        print("\n--- FFMPEG LOG START (Creating raw .aac) ---")
        subprocess.run(cmd, check=True)
        print("--- FFMPEG LOG END (Creating raw .aac) ---\n")
        return processed_audio_path
    except subprocess.CalledProcessError:
        print(f"Hiba az audió .aac-re való feldolgozása során. Lásd a fenti ffmpeg logot."); sys.exit(1)

def add_audio_to_video_final(video_file, new_audio_aac_file, language, target_sample_rate, output_dir):
    """Hozzáadja az új hangsávot a videóhoz a bizonyítottan működő, letisztult mappinggal."""
    video_basename = os.path.basename(video_file)
    video_name, video_ext = os.path.splitext(video_basename)
    output_file = os.path.join(output_dir, f"{video_name}_with_{language}_dub{video_ext}")

    try:
        # A parancs, ami a működő logikát másolja: explicit, nem mohó mapping
        cmd = [
            'ffmpeg', '-y',
            '-i', video_file,
            '-i', new_audio_aac_file,
            '-map', '0:v:0',    # Csak az első videósáv az eredetiből
            '-map', '0:a:0',    # Csak az első audiosáv az eredetiből
            '-map', '1:a',      # Az új audiosáv
            '-c:v', 'copy',     # Videó másolása
        ]
        
        # A kimeneti sávok indexelése most már egyszerű:
        # 0: video (másolva)
        # 1: eredeti audio (másolva, ez lesz a:0)
        # 2: új audio (kódolva, ez lesz a:1)
        
        cmd.extend([
            '-c:a:0', 'copy',   # Eredeti hang másolása
            '-c:a:1', 'aac',    # Új hang kódolása
            '-ar:a:1', str(target_sample_rate),
            '-b:a:1', '128k',
            '-metadata:s:a:1', f'language={language}', # A kimeneti második hangsáv (a:1) metaadata
        ])
        
        cmd.append(output_file)

        print(f"FFmpeg parancs futtatása:\n{' '.join(cmd)}")
        print("\n--- FFMPEG LOG START (Final Muxing) ---")
        subprocess.run(cmd, check=True)
        print("--- FFMPEG LOG END (Final Muxing) ---\n")
        return output_file
    except subprocess.CalledProcessError:
        print(f"Hiba az audiosáv hozzáadása során. Lásd a fenti ffmpeg logot."); sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Videóhoz új, szinkronizált audiosávot fűz a projektstruktúra alapján.")
    parser.add_argument('project_name', help='A feldolgozandó projekt könyvtár neve a "workdir"-en belül.')
    parser.add_argument('-lang', '--language', required=True, help='A hozzáadandó audiosáv nyelvi címkéje (pl. hun, eng).')
    add_debug_argument(parser)
    args = parser.parse_args()
    configure_debug_mode(args.debug)

    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, '..', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f: config = json.load(f)
    except Exception as e: print(f"Hiba a config.json betöltése közben: {e}"); sys.exit(1)

    try:
        workdir_base = config['DIRECTORIES']['workdir']
        subdirs = config['PROJECT_SUBDIRS']
        project_path = os.path.join(workdir_base, args.project_name)
        upload_dir = os.path.join(project_path, subdirs['upload'])
        audio_dir = os.path.join(project_path, subdirs['film_dubbing'])
        output_dir = os.path.join(project_path, subdirs['download'])
    except KeyError as e: print(f"Hiba: Hiányzó kulcs a config.json fájlban: {e}"); sys.exit(1)

    input_video = find_first_video_file(upload_dir)
    if not input_video: print(f"Hiba: Nem található videófájl a(z) '{upload_dir}' könyvtárban."); sys.exit(1)
    print(f"Bemeneti videó: {input_video}")

    input_wav_audio = find_latest_audio_file(audio_dir)
    if not input_wav_audio: print(f"Hiba: Nem található .wav audiófájl a(z) '{audio_dir}' könyvtárban."); sys.exit(1)
    print(f"Bemeneti audió (WAV): {input_wav_audio}")

    os.makedirs(output_dir, exist_ok=True)
    
    target_sample_rate = get_target_sample_rate(input_video)
    print(f"Cél mintavételezési ráta: {target_sample_rate}Hz")

    processed_aac_path = process_audio_to_raw_aac(input_wav_audio, target_sample_rate)
    print(f"Ideiglenes nyers .aac fájl létrehozva: {processed_aac_path}")

    final_video = add_audio_to_video_final(input_video, processed_aac_path, args.language, target_sample_rate, output_dir)
    
    if os.path.exists(processed_aac_path):
        os.remove(processed_aac_path)
        print(f"Ideiglenes audiófájl törölve: {processed_aac_path}")

    print(f"\nSikeres végrehajtás! A kimeneti fájl itt található: {final_video}")

if __name__ == "__main__":
    main()

import argparse
import subprocess
import json
import os
import sys
import glob
import tempfile
import shutil
from pathlib import Path

# Hozzáadja a 'tools' könyvtárat a Python útvonalhoz
for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

# --- FÁJLKERESŐ ÉS ELŐKÉSZÍTŐ FÜGGVÉNYEK (Változatlanok) ---

def find_first_video_file(directory):
    supported_formats = ('.mkv', '.mp4', '.avi', '.mov', '.webm')
    for file_name in sorted(os.listdir(directory)):
        if file_name.lower().endswith(supported_formats):
            return os.path.join(directory, file_name)
    return None

def find_latest_audio_file(directory):
    wav_files = glob.glob(os.path.join(directory, '*.wav'))
    if not wav_files: return None
    return max(wav_files, key=os.path.getmtime)

def get_target_sample_rate(video_file):
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=sample_rate', '-of', 'json', video_file]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        info = json.loads(result.stdout)
        return info['streams'][0].get('sample_rate', '48000')
    except Exception:
        return '48000'

def process_audio_to_raw_aac(audio_wav_file, target_sample_rate):
    processed_audio_path = "processed_audio_temp.aac"
    try:
        cmd = ['ffmpeg', '-y', '-i', audio_wav_file, '-c:a', 'aac', '-b:a', '128k', '-ar', str(target_sample_rate), '-ac', '2', processed_audio_path]
        print("\n--- FFMPEG LOG START (Creating raw .aac) ---")
        subprocess.run(cmd, check=True)
        print("--- FFMPEG LOG END (Creating raw .aac) ---\n")
        return processed_audio_path
    except subprocess.CalledProcessError:
        print("Hiba az audió AAC formátumra alakítása során.")
        sys.exit(1)


def get_project_root() -> Path:
    """
    Felkeresi a projekt gyökerét a config.json alapján.
    """
    for candidate in Path(__file__).resolve().parents:
        config_candidate = candidate / "config.json"
        if config_candidate.is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")

# --- DEMUX-REMUX LOGIKA (Felirat metaadatokkal bővítve) ---

def get_stream_info(video_file):
    """Lekéri a videó összes sávjának részletes adatait JSON formátumban."""
    cmd = ['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', video_file]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return json.loads(result.stdout)['streams']

def demux_streams(video_file, streams, temp_dir):
    """Kimenti az összes sávot külön fájlokba."""
    demuxed_files = []
    print("\n--- STARTING DEMUXING (Sávok szétválasztása) ---")
    for stream in streams:
        stream_index = stream['index']
        codec_type = stream['codec_type']
        
        ext = stream.get('codec_name', 'bin')
        if codec_type == 'video': ext = 'mkv'
        if codec_type == 'subtitle' and ext == 'subrip': ext = 'srt'
        if codec_type == 'subtitle' and ext == 'ass': ext = 'ass'

        output_path = os.path.join(temp_dir, f"stream_{stream_index}.{ext}")
        
        cmd = ['ffmpeg', '-y', '-i', video_file, '-map', f'0:{stream_index}', '-c', 'copy', output_path]
        
        print(f"Extracting {codec_type} stream #{stream_index} -> {output_path}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        demuxed_files.append({'path': output_path, 'stream_info': stream})
        
    print("--- DEMUXING FINISHED ---\n")
    return demuxed_files

def remux_video(demuxed_files, new_audio_file, language, output_file):
    """Újraépíti a videót a különválasztott sávokból és az új hangból."""
    print("\n--- STARTING REMUXING (Sávok újraegyesítése) ---")
    
    cmd = ['ffmpeg', '-y']
    
    # Bemeneti fájlok hozzáadása
    input_files = [f['path'] for f in demuxed_files] + [new_audio_file]
    for file_path in input_files:
        cmd.extend(['-i', file_path])
        
    # Sávok feltérképezése (mapping)
    num_original_streams = len(demuxed_files)
    for i in range(num_original_streams):
        cmd.extend(['-map', f'{i}:0'])
    cmd.extend(['-map', f'{num_original_streams}:0'])
    
    # Kodekek beállítása másolásra
    cmd.extend(['-c', 'copy'])
    
    # Metaadatok visszaállítása és beállítása
    original_audio_count = 0
    original_subtitle_count = 0
    for i, demuxed in enumerate(demuxed_files):
        stream = demuxed['stream_info']
        stream_type = stream['codec_type']
        
        if stream_type == 'audio':
            lang_tag = stream.get('tags', {}).get('language', 'und')
            cmd.extend([f'-metadata:s:a:{original_audio_count}', f'language={lang_tag}'])
            original_audio_count += 1
            
        elif stream_type == 'subtitle':
            # --- JAVÍTÁS: Felirat nyelvi kódjának beállítása ---
            lang_tag = stream.get('tags', {}).get('language', 'und')
            cmd.extend([f'-metadata:s:s:{original_subtitle_count}', f'language={lang_tag}'])
            original_subtitle_count += 1
            
    # Új audiosáv metaadatainak és viselkedésének beállítása
    cmd.extend([f'-metadata:s:a:{original_audio_count}', f'language={language}'])
    cmd.extend([f'-metadata:s:a:{original_audio_count}', f'title={language.upper()} dub'])
    cmd.extend([f'-disposition:a:{original_audio_count}', 'default+dub'])
    
    cmd.append(output_file)

    print(f"FFmpeg parancs futtatása:\n{' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("Hiba a végső videó összefűzése során.")
        sys.exit(1)
    finally:
        print("--- REMUXING FINISHED ---\n")

def main():
    parser = argparse.ArgumentParser(description="Videót atomjaira szed, majd új hangsávval rakja össze a maximális stabilitás érdekében.")
    parser.add_argument('project_name', help='A feldlogozandó projekt könyvtár neve a \"workdir\"-en belül.')
    parser.add_argument('-lang', '--language', required=True, help='A hozzáadandó audiosáv nyelvi címkéje (pl. hun, eng).')
    add_debug_argument(parser)
    args = parser.parse_args()
    configure_debug_mode(args.debug)

    # --- Konfiguráció és útvonalak beállítása (Változatlan) ---
    try:
        project_root = get_project_root()
    except FileNotFoundError as exc:
        print(f"Hiba: {exc}")
        sys.exit(1)

    config_path = project_root / "config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Hiba: A config.json fájl nem található itt: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Hiba: A config.json fájl hibás formátumú ({config_path}): {exc}")
        sys.exit(1)

    try:
        workdir_base = project_root / config['DIRECTORIES']['workdir']
        subdirs = config['PROJECT_SUBDIRS']
        project_path = workdir_base / args.project_name
        upload_dir = project_path / subdirs['upload']
        audio_dir = project_path / subdirs['film_dubbing']
        output_dir = project_path / subdirs['download']
    except KeyError as e:
        print(f"Hiba: Hiányzó kulcs a config.json fájlban: {e}")
        sys.exit(1)

    input_video = find_first_video_file(str(upload_dir))
    if not input_video:
        print(f"Hiba: Nem található videófájl a(z) '{upload_dir}' könyvtárban.")
        sys.exit(1)
    
    input_wav_audio = find_latest_audio_file(str(audio_dir))
    if not input_wav_audio:
        print(f"Hiba: Nem található WAV audiófájl a(z) '{audio_dir}' könyvtárban.")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = tempfile.mkdtemp(prefix="demux_")
    
    try:
        streams = get_stream_info(input_video)
        demuxed_files = demux_streams(input_video, streams, temp_dir)
        
        target_sample_rate = get_target_sample_rate(input_video)
        processed_aac_path = process_audio_to_raw_aac(input_wav_audio, target_sample_rate)
        
        video_basename = os.path.basename(input_video)
        video_name, video_ext = os.path.splitext(video_basename)
        output_file = str(output_dir / f"{video_name}_with_{args.language}_dub{video_ext}")
        
        remux_video(demuxed_files, processed_aac_path, args.language, output_file)

        if os.path.exists(processed_aac_path):
            os.remove(processed_aac_path)
            print(f"Ideiglenes audiófájl törölve: {processed_aac_path}")
            
        print(f"\nSikeres végrehajtás! A kimeneti fájl itt található: {output_file}")

    finally:
        print(f"Takarítás: ideiglenes könyvtár törlése ({temp_dir})")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()

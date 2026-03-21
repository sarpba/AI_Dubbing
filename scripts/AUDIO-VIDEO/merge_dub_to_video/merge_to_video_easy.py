import argparse
import subprocess
import json
import os
import sys
import glob
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

def get_project_root() -> Path:
    """
    Felkeresi a projekt gyökerét a config.json alapján.
    """
    for candidate in Path(__file__).resolve().parents:
        config_candidate = candidate / "config.json"
        if config_candidate.is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")

# --- MUX LOGIKA (Az eredeti sávok megtartásával) ---

def get_stream_info(video_file):
    """Lekéri a videó összes sávjának részletes adatait JSON formátumban."""
    cmd = ['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', video_file]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return json.loads(result.stdout)['streams']


def get_container_specific_mux_options(output_file):
    """Container-specifikus mux opciók a jobb lejátszó-kompatibilitásért."""
    output_ext = Path(output_file).suffix.lower()
    if output_ext in {'.mp4', '.m4v', '.mov'}:
        return ['-movflags', '+faststart']
    return []

def merge_video_with_new_audio(video_file, streams, new_audio_file, language, output_file, target_sample_rate, include_original_audio=True):
    """Az eredeti videó minden sávját megtartja, és új, kompatibilis AAC hangsávot fűz hozzá."""
    print("\n--- STARTING MUXING (Új audiosáv hozzáadása) ---")

    cmd = [
        'ffmpeg', '-y',
        '-fflags', '+genpts',
        '-i', video_file,
        '-i', new_audio_file,
    ]
    cmd.extend(['-map_metadata', '0', '-map_chapters', '0'])

    if include_original_audio:
        cmd.extend(['-map', '0'])
    else:
        print("Az eredeti hangsávok kihagyásra kerülnek; csak az új dub kerül a kimenetbe.")
        cmd.extend(['-map', '0', '-map', '-0:a'])

    cmd.extend(['-map', '1:a:0', '-c', 'copy'])

    original_audio_count = sum(1 for stream in streams if stream['codec_type'] == 'audio') if include_original_audio else 0

    if include_original_audio:
        for audio_index in range(original_audio_count):
            cmd.extend([f'-disposition:a:{audio_index}', '0'])

    # A hozzáadott dub sávot újrakódoljuk stabil AAC-LC formátumba, tiszta kezdő PTS-sel.
    cmd.extend([
        f'-c:a:{original_audio_count}', 'aac',
        f'-profile:a:{original_audio_count}', 'aac_low',
        f'-b:a:{original_audio_count}', '192k',
        f'-ar:a:{original_audio_count}', str(target_sample_rate),
        f'-ac:a:{original_audio_count}', '2',
        f'-filter:a:{original_audio_count}', 'aresample=async=1:first_pts=0',
    ])

    cmd.extend([f'-metadata:s:a:{original_audio_count}', f'language={language}'])
    cmd.extend([f'-metadata:s:a:{original_audio_count}', f'title={language.upper()} dub'])
    cmd.extend([f'-disposition:a:{original_audio_count}', 'default+dub'])
    cmd.extend(['-max_interleave_delta', '0', '-avoid_negative_ts', 'make_zero'])
    cmd.extend(get_container_specific_mux_options(output_file))
    cmd.append(output_file)

    print(f"FFmpeg parancs futtatása:\n{' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("Hiba a végső videó összefűzése során.")
        sys.exit(1)
    finally:
        print("--- MUXING FINISHED ---\n")

def main():
    parser = argparse.ArgumentParser(description="Az eredeti videó sávjait megtartja, és új hangsávot fűz a kimeneti fájlhoz.")
    parser.add_argument('project_name', help='A feldlogozandó projekt könyvtár neve a \"workdir\"-en belül.')
    parser.add_argument('-lang', '--language', required=True, help='A hozzáadandó audiosáv nyelvi címkéje (pl. hun, eng).')
    parser.add_argument('--only-new-audio', action='store_true', help='Csak az új hangsáv kerüljön a kimenő videóba, az eredeti audiók kihagyásával.')
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
    
    streams = get_stream_info(input_video)
    target_sample_rate = get_target_sample_rate(input_video)

    video_basename = os.path.basename(input_video)
    video_name, video_ext = os.path.splitext(video_basename)
    output_file = str(output_dir / f"{video_name}_with_{args.language}_dub{video_ext}")

    include_original_audio = not args.only_new_audio
    merge_video_with_new_audio(
        input_video,
        streams,
        input_wav_audio,
        args.language,
        output_file,
        target_sample_rate,
        include_original_audio=include_original_audio,
    )

    print(f"\nSikeres végrehajtás! A kimeneti fájl itt található: {output_file}")

if __name__ == "__main__":
    main()

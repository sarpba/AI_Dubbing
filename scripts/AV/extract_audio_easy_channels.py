import os
import sys
import subprocess
import json
import argparse
from pathlib import Path

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

def get_audio_stream_info(video_path):
    """
    Lekérdezi a legelső audió sáv (a:0) minden jellemzőjét JSON formátumban.
    Visszatér a teljes 'stream' objektummal, vagy None-nal hiba esetén.
    """
    try:
        command = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream',
            '-of', 'json', video_path
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        stream_info = json.loads(result.stdout)['streams'][0]
        return stream_info
    except FileNotFoundError:
        print("Hiba: Az 'ffprobe' parancs nem található. Kérlek, telepítsd az FFmpeg-et.")
        return None
    except (subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Hiba az audió sáv információinak kiolvasása közben: {e}")
        return None

def save_audio_info_to_json(stream_info, output_path):
    """
    Elmenti a kapott, teljes információcsomagot egy JSON fájlba.
    Ez a fájl felhasználható egy másik szkript által a helyreállításhoz.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stream_info, f, ensure_ascii=False, indent=4)
        print(f"A teljes audió adatcsomag elmentve ide: {output_path}")
        return True
    except (IOError, TypeError) as e:
        print(f"Hiba az audió információk fájlba írása közben: {e}")
        return False

def extract_individual_channels(video_path, output_dir, channels, layout_string):
    """
    Kinyeri az audió sávokat külön fájlokba. Ha a layout_string ismert,
    akkor a szabványos neveket használja. Ellenkező esetben általános neveket (C0, C1..).
    Visszatér (True, file_map) siker esetén, (False, {}) hiba esetén.
    """
    LAYOUT_TO_NAMES_MAP = {
        '5.1(side)': ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR'],
        '5.1':       ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR'],
        'stereo':    ['FL', 'FR'],
        '7.1':       ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR', 'SL', 'SR'],
    }
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    command = ['ffmpeg', '-i', video_path]
    output_file_map = {}
    
    channel_names = LAYOUT_TO_NAMES_MAP.get(layout_string)

    if channel_names:
        print(f"'{layout_string}' elrendezés alapján {len(channel_names)} csatorna kinyerése folyamatban...")
        if len(channel_names) != channels:
            print(f"Figyelem: Eltérés van a csatornaszám ({channels}) és a nevek száma ({len(channel_names)}) között.")

        filter_pads = ''.join([f'[{name}]' for name in channel_names])
        filter_complex_str = f"[0:a:0]channelsplit=channel_layout={layout_string}{filter_pads}"
        command.extend(['-filter_complex', filter_complex_str])
    else:
        print(f"Figyelem: Ismeretlen ('{layout_string}') vagy hiányzó csatornaelrendezés.")
        print(f"A kinyerés általános módszerrel történik {channels} csatornára (C0, C1...).")
        
        if channels is None or channels == 0:
            print("Hiba: A csatornák száma 0 vagy nem meghatározható, a szétválasztás nem lehetséges.")
            return False, {}
            
        channel_names = [f'C{i}' for i in range(channels)]
        pan_filters = [f"[0:a]pan=1c|c0=c{i}[{name}]" for i, name in enumerate(channel_names)]
        filter_complex_str = ";".join(pan_filters)
        command.extend(['-filter_complex', filter_complex_str])

    for name in channel_names:
        output_filename = f"{base_name}_{name}.wav"
        output_path = os.path.join(output_dir, output_filename)
        output_file_map[name] = output_filename

        print(f"-> {name} mentése ide: {output_path}")
        command.extend(['-map', f'[{name}]'])
        command.extend(['-acodec', 'pcm_s16le', '-ar', '44100'])
        command.append(output_path)
    
    command.append('-y')

    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("\nMinden csatorna sikeresen kinyerve.")
        return True, output_file_map
    except subprocess.CalledProcessError as e:
        print("\nHIBA a csatornák kinyerése közben.")
        print(f"A futtatott parancs: {' '.join(command)}")
        print(f"FFmpeg hibaüzenet:\n{e.stderr}")
        return False, {}

def extract_stereo_audio(video_path, audio_path):
    """Kinyeri az audió sávot sztereó WAV fájlként."""
    try:
        output_dir = os.path.dirname(audio_path)
        os.makedirs(output_dir, exist_ok=True)
        command = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', '-y', audio_path]
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"A sztereó audió sikeresen kinyerve ide: {audio_path}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Hiba a sztereó audió kinyerése közben: {e}")
        if hasattr(e, 'stderr'): print(f"FFmpeg hiba: {e.stderr}")
        return False

# ==============================================================================
# ===                         PROGRAMFUTTATÁS INDÍTÁSA                         ===
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audiót nyeri ki videófájlból és elmenti a metaadatokat.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("project_dir_name", help="A projekt mappa neve a 'workdir'-en belül.")
    parser.add_argument(
        "--keep_channels",
        action="store_true",
        help="Többcsatornás audió esetén minden csatornát külön .wav fájlba ment.\n"
             "Ha a forrás hangsáv sztereó (2 csatornás), ez a kapcsoló figyelmen kívül lesz hagyva,\n"
             "és a kinyerés egyetlen sztereó fájlba történik."
    )
    add_debug_argument(parser)
    args = parser.parse_args()
    configure_debug_mode(args.debug)
    
    try:
        with open('config.json', 'r', encoding='utf-8') as f: config = json.load(f)
        workdir = config['DIRECTORIES']['workdir']
        upload_subdir = config['PROJECT_SUBDIRS']['upload']
        extracted_audio_subdir = config['PROJECT_SUBDIRS']['extracted_audio']
    except (FileNotFoundError, KeyError) as e:
        print(f"Hiba a 'config.json' beolvasásakor: {e}")
        sys.exit(1)

    project_path = os.path.join(workdir, args.project_dir_name)
    upload_dir = os.path.join(project_path, upload_subdir)
    extracted_audio_dir = os.path.join(project_path, extracted_audio_subdir)
    
    VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.mts')
    video_files = [f for f in os.listdir(upload_dir) if f.lower().endswith(VIDEO_EXTENSIONS)]
    if not video_files:
        print(f"Hiba: Nem található videófájl: {upload_dir}")
        sys.exit(1)
    
    video_filename = video_files[0]
    video_path = os.path.join(upload_dir, video_filename)
    if len(video_files) > 1:
        print(f"Figyelem: Több videó található, az elsőt használom: {video_filename}")

    print(f"\nProjekt: {args.project_dir_name}\nBemeneti videó: {video_path}")

    # 1. Eredeti audió infók beolvasása memóriába
    original_audio_info = get_audio_stream_info(video_path)
    if not original_audio_info:
        print("Kritikus hiba: Nem sikerült lekérdezni az eredeti audió sáv adatait. A folyamat leáll.")
        sys.exit(1)
    
    codec_name = original_audio_info.get('codec_name', 'ismeretlen')
    channels = original_audio_info.get('channels')
    print(f"Eredeti audió sáv adatai sikeresen beolvasva (kódek: {codec_name}, csatornák: {channels}).")
    
    # 2. Audó kinyerése a választott módon
    success = False

    # === MÓDOSÍTOTT LOGIKA ===
    # Ha a --keep_channels aktív ÉS a csatornák száma NEM 2, akkor választjuk szét.
    # Minden más esetben (nincs --keep_channels VAGY 2 csatorna van) sztereó kinyerés történik.
    if args.keep_channels and channels != 2:
        print("\nMód: Csatornák szétválasztása egyedi fájlokba.")
        layout_string = original_audio_info.get('channel_layout')
        
        success, extracted_files_map = extract_individual_channels(video_path, extracted_audio_dir, channels, layout_string)
        
        if success:
            original_audio_info['extraction_mode'] = 'channels'
            original_audio_info['extracted_channel_files'] = extracted_files_map
    else:
        # Ez az ág fut, ha nincs --keep_channels, vagy ha a sáv sztereó.
        if args.keep_channels and channels == 2:
            print("\nFigyelem: A forrásfájl sztereó (2 csatornás).")
            print("A '--keep_channels' kapcsoló ilyenkor figyelmen kívül lesz hagyva, a sáv egyetlen sztereó fájlként lesz kinyerve.")

        print("\nMód: Sztereó audió kinyerése.")
        base_name = os.path.splitext(video_filename)[0]
        audio_filename = f"{base_name}_stereo.wav"
        audio_path = os.path.join(extracted_audio_dir, audio_filename)
        
        success = extract_stereo_audio(video_path, audio_path)
        if success:
            original_audio_info['extraction_mode'] = 'stereo'
            original_audio_info['extracted_file'] = audio_filename

    # 3. A teljes információs csomag mentése, CSAK ha a kinyerés sikeres volt
    if success:
        info_file_path = os.path.join(extracted_audio_dir, 'original_audio_info.json')
        if not save_audio_info_to_json(original_audio_info, info_file_path):
            print("Figyelem: Az audió kinyerése sikeres volt, de az információs fájl mentése nem.")
            sys.exit(1)

    sys.exit(0 if success else 1)

# Mentsd el extract_audio_channels.py néven
import os
import sys
import subprocess
import json
import argparse

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

# ==============================================================================
# ===                            JAVÍTOTT FUNKCIÓ                            ===
# ==============================================================================
def extract_individual_channels(video_path, output_dir, channels, layout_string):
    """
    Kinyeri az audió sávokat külön fájlokba.
    Visszatér (True, file_map) siker esetén, ahol a file_map egy szótár,
    ami a csatornakódot a fájlnévhez rendeli. Hiba esetén (False, {})-t ad vissza.
    """
    LAYOUT_TO_NAMES_MAP = {
        '5.1(side)': ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR'],
        '5.1':       ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR'],
        'stereo':    ['FL', 'FR'],
        '7.1':       ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR', 'SL', 'SR'],
    }
    channel_names = LAYOUT_TO_NAMES_MAP.get(layout_string)
    if not channel_names:
        print(f"Hiba: A szkript nem ismeri a '{layout_string}' elrendezést.")
        return False, {}
    
    if len(channel_names) != channels:
        print(f"Figyelem: Eltérés van a csatornaszám ({channels}) és a nevek száma ({len(channel_names)}) között.")

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    command = ['ffmpeg', '-i', video_path]
    filter_pads = ''.join([f'[{name}]' for name in channel_names])
    filter_complex_str = f"[0:a:0]channelsplit=channel_layout={layout_string}{filter_pads}"
    command.extend(['-filter_complex', filter_complex_str])

    print(f"{len(channel_names)} csatorna kinyerése folyamatban...")
    output_file_map = {}
    for name in channel_names:
        # Fájlnevet generálunk, de a teljes elérési utat használjuk a parancshoz
        output_filename = f"{base_name}_{name}.wav"
        output_path = os.path.join(output_dir, output_filename)
        output_file_map[name] = output_filename # A szótárba csak a tiszta fájlnevet tesszük

        print(f"-> {name} mentése ide: {output_path}")
        command.extend(['-map', f'[{name}]'])
        command.extend(['-acodec', 'pcm_s16le', '-ar', '44100'])
        command.append(output_path)
    
    command.append('-y')

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audiót nyeri ki videófájlból és elmenti a metaadatokat.",
    )
    parser.add_argument("project_dir_name", help="A projekt mappa neve a 'workdir'-en belül.")
    parser.add_argument(
        "--keep_channels",
        action="store_true",
        help="Minden audió csatornát külön .wav fájlba ment."
    )
    args = parser.parse_args()
    
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
    print("Eredeti audió sáv adatai sikeresen beolvasva.")
    
    # 2. Audó kinyerése a választott módon
    success = False
    if args.keep_channels:
        print("\nMód: Csatornák szétválasztása egyedi fájlokba.")
        channels = original_audio_info.get('channels')
        layout_string = original_audio_info.get('channel_layout')
        
        if not channels or not layout_string:
             print("Hiba: A csatornák száma vagy elrendezése nem határozható meg.")
             sys.exit(1)
        
        # A kinyerés itt történik, és visszakapjuk a fájlok térképét
        success, extracted_files_map = extract_individual_channels(video_path, extracted_audio_dir, channels, layout_string)
        
        if success:
            # Sikeres kinyerés után bővítjük az információs csomagot
            original_audio_info['extraction_mode'] = 'channels'
            original_audio_info['extracted_channel_files'] = extracted_files_map
    else:
        print("\nMód: Sztereó audió kinyerése.")
        base_name = os.path.splitext(video_filename)[0]
        audio_filename = f"{base_name}.wav"
        audio_path = os.path.join(extracted_audio_dir, audio_filename)
        
        success = extract_stereo_audio(video_path, audio_path)
        if success:
            # Itt is jelezzük a fájl nevét és a módot
            original_audio_info['extraction_mode'] = 'stereo'
            original_audio_info['extracted_file'] = audio_filename

    # 3. A teljes információs csomag mentése, CSAK ha a kinyerés sikeres volt
    if success:
        info_file_path = os.path.join(extracted_audio_dir, 'original_audio_info.json')
        if not save_audio_info_to_json(original_audio_info, info_file_path):
            print("Figyelem: Az audió kinyerése sikeres volt, de az információs fájl mentése nem.")
            sys.exit(1) # Dönthetünk úgy, hogy ez is kritikus hiba

    sys.exit(0 if success else 1)
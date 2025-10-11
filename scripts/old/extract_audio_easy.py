import os
import sys
import subprocess
import json

def extract_audio(video_path, audio_path):
    """
    Kinyeri az audió sávot egy videófájlból az FFmpeg segítségével.
    A kimenet egy 44.1kHz-es, 16-bites, sztereó PCM WAV fájl lesz.
    """
    try:
        # Győződjünk meg róla, hogy a kimeneti mappa létezik
        output_dir = os.path.dirname(audio_path)
        os.makedirs(output_dir, exist_ok=True)

        print("Audió kinyerése a következő paranccsal...")
        # FFmpeg paranccsal kinyerjük az audiót
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',              # Csak audió (video sáv eldobása)
            '-acodec', 'pcm_s16le',  # 16-bit PCM formátum (WAV)
            '-ar', '44100',     # 44.1kHz mintavételi frekvencia
            '-ac', '2',        # Sztereó csatornák
            '-y',              # Felülírás engedélyezése, ha a fájl már létezik
            audio_path
        ]
        print(f"Parancs: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Az audió sikeresen kinyerve ide: {audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Hiba az audió kinyerése közben: {e}")
        print(f"FFmpeg hibaüzenet:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print("Hiba: Az 'ffmpeg' parancs nem található. Kérlek, telepítsd az FFmpeg-et és győződj meg róla, hogy elérhető a rendszer PATH-jában.")
        return False
    except Exception as e:
        print(f"Váratlan hiba: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Használat: python extract_audio.py <projekt_mappa_neve>")
        sys.exit(1)
        
    project_dir_name = sys.argv[1]
    
    # Konfigurációs fájl betöltése
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        workdir = config['DIRECTORIES']['workdir']
        upload_subdir = config['PROJECT_SUBDIRS']['upload']
        extracted_audio_subdir = config['PROJECT_SUBDIRS']['extracted_audio']
    except FileNotFoundError:
        print("Hiba: A 'config.json' fájl nem található az aktuális könyvtárban.")
        sys.exit(1)
    except KeyError as e:
        print(f"Hiba: Hiányzó kulcs a config.json fájlban: {e}")
        sys.exit(1)

    # Elérési utak összeállítása
    project_path = os.path.join(workdir, project_dir_name)
    upload_dir = os.path.join(project_path, upload_subdir)
    extracted_audio_dir = os.path.join(project_path, extracted_audio_subdir)
    
    # Bemeneti videófájl megkeresése
    if not os.path.isdir(upload_dir):
        print(f"A projekt feltöltési mappája nem található: {upload_dir}")
        sys.exit(1)
        
    # *** JAVÍTÁS: Csak videó kiterjesztésű fájlokat keresünk ***
    VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv')
    
    video_files = [
        f for f in os.listdir(upload_dir) 
        if os.path.isfile(os.path.join(upload_dir, f)) and f.lower().endswith(VIDEO_EXTENSIONS)
    ]
    
    if not video_files:
        print(f"Nem található videófájl a következő mappában: {upload_dir}")
        print(f"Támogatott kiterjesztések: {', '.join(VIDEO_EXTENSIONS)}")
        sys.exit(1)
    
    video_filename = video_files[0]
    video_path = os.path.join(upload_dir, video_filename)
    
    if len(video_files) > 1:
        print(f"Figyelem: Több videófájl található a feltöltési mappában. Az elsőt használom: {video_filename}")

    # Kimeneti audiofájl nevének létrehozása
    base_name = os.path.splitext(video_filename)[0]
    audio_filename = f"{base_name}.wav"
    audio_path = os.path.join(extracted_audio_dir, audio_filename)
    
    print(f"Projekt: {project_dir_name}")
    print(f"Bemeneti videó: {video_path}")
    print(f"Kimeneti audió: {audio_path}")
    
    success = extract_audio(video_path, audio_path)
    
    sys.exit(0 if success else 1)
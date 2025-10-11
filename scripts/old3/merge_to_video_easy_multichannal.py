import argparse
import subprocess
import json
import os
import sys
import glob

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

def process_audio(audio_wav_file, target_sample_rate, original_audio_info):
    """
    A bemeneti WAV fájlt átkódolja a videó eredeti hangsávjának megfelelő formátumra.
    Ha az eredeti többcsatornás, a kimenet is az lesz. Egyébként sztereó AAC.
    """
    # Alapértelmezett értékek (sztereó AAC), ha nincs extra infó
    target_codec = 'aac'
    target_channels = 2
    target_bitrate = '192k' # Emelt minőségű sztereó bitrate
    output_extension = 'aac'

    if original_audio_info:
        original_channels = original_audio_info.get('channels')
        original_codec = original_audio_info.get('codec_name')
        original_bitrate = original_audio_info.get('bit_rate')

        if original_channels and original_channels > 2:
            print(f"Az eredeti hangsáv többcsatornás ({original_channels} csatorna, {original_codec}). A kimenet ehhez lesz igazítva.")
            target_channels = original_channels
            
            # Kodek és bitráta választása az eredeti alapján
            if original_codec == 'eac3':
                target_codec = 'eac3'
                output_extension = 'eac3'
                # Bitráta beállítása: használjuk az eredetit, ha van, különben egy ésszerű alapértéket.
                target_bitrate = original_bitrate if original_bitrate else '640k'
            else: # Más többcsatornás esetben (pl. aac, dts) AAC-t használunk
                target_codec = 'aac'
                output_extension = 'aac'
                # Bitráta becslése csatornánként (pl. 64kbps/csatorna)
                target_bitrate = f"{target_channels * 64}k"
    else:
        print("Nincs információ az eredeti hangsávról, vagy az sztereó. A kimenet sztereó AAC lesz.")

    processed_audio_path = f"processed_audio_temp.{output_extension}"
    
    try:
        cmd = [
            'ffmpeg', '-y', '-i', audio_wav_file,
            '-c:a', target_codec,
            '-b:a', target_bitrate,
            '-ar', str(target_sample_rate),
            '-ac', str(target_channels),
            processed_audio_path
        ]
        print("\n--- FFMPEG LOG START (Audio Processing) ---")
        print(f"Futtatandó audió konverziós parancs: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stderr=subprocess.STDOUT)
        print("--- FFMPEG LOG END (Audio Processing) ---\n")
        return processed_audio_path
    except subprocess.CalledProcessError as e:
        print(f"Hiba az audió feldolgozása során. Lásd a fenti ffmpeg logot.\nHiba: {e}"); 
        sys.exit(1)


def add_audio_to_video_final(video_file, new_audio_file, language, output_dir):
    """Hozzáadja az új, már feldolgozott hangsávot a videóhoz stream másolással."""
    video_basename = os.path.basename(video_file)
    video_name, video_ext = os.path.splitext(video_basename)
    output_file = os.path.join(output_dir, f"{video_name}_with_{language}_dub{video_ext}")

    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', video_file,        # Bemenet 0
            '-i', new_audio_file,    # Bemenet 1 (az általunk kódolt hang)
            '-map', '0:v:0',         # Videósáv a 0. bemenetről
            '-map', '0:a:0',         # Eredeti hangsáv a 0. bemenetről
            '-map', '1:a:0',         # Új hangsáv az 1. bemenetről
            '-c:v', 'copy',          # Videó másolása (nincs újrakódolás)
            '-c:a:0', 'copy',        # Eredeti hang másolása
            '-c:a:1', 'copy',        # Új, már kódolt hangunk másolása
            '-metadata:s:a:1', f'language={language}', # Nyelvi címke beállítása az új hangsávra (a:1)
        ]
        
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
    args = parser.parse_args()

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
        extracted_audio_dir = os.path.join(project_path, subdirs['extracted_audio'])
    except KeyError as e: print(f"Hiba: Hiányzó kulcs a config.json fájlban: {e}"); sys.exit(1)

    # Eredeti audio információk beolvasása
    original_audio_info = None
    original_audio_info_path = os.path.join(extracted_audio_dir, 'original_audio_info.json')
    if os.path.exists(original_audio_info_path):
        try:
            with open(original_audio_info_path, 'r', encoding='utf-8') as f:
                original_audio_info = json.load(f)
            print(f"Eredeti audió információk sikeresen betöltve: {original_audio_info_path}")
        except Exception as e:
            print(f"Figyelmeztetés: Hiba történt az 'original_audio_info.json' betöltésekor: {e}. Sztereó fallback lesz használva.")
    else:
        print("Figyelmeztetés: Nem található 'original_audio_info.json'. Sztereó fallback lesz használva.")

    input_video = find_first_video_file(upload_dir)
    if not input_video: print(f"Hiba: Nem található videófájl a(z) '{upload_dir}' könyvtárban."); sys.exit(1)
    print(f"Bemeneti videó: {input_video}")

    input_wav_audio = find_latest_audio_file(audio_dir)
    if not input_wav_audio: print(f"Hiba: Nem található .wav audiófájl a(z) '{audio_dir}' könyvtárban."); sys.exit(1)
    print(f"Bemeneti audió (WAV): {input_wav_audio}")

    os.makedirs(output_dir, exist_ok=True)
    
    target_sample_rate = get_target_sample_rate(input_video)
    print(f"Cél mintavételezési ráta: {target_sample_rate}Hz")

    # Hang feldolgozása az eredeti formátum alapján
    processed_audio_path = process_audio(input_wav_audio, target_sample_rate, original_audio_info)
    print(f"Ideiglenes feldolgozott audiófájl létrehozva: {processed_audio_path}")
    
    # Feldolgozott hang hozzáadása a videóhoz
    final_video = add_audio_to_video_final(input_video, processed_audio_path, args.language, output_dir)
    
    if os.path.exists(processed_audio_path):
        os.remove(processed_audio_path)
        print(f"Ideiglenes audiófájl törölve: {processed_audio_path}")

    print(f"\nSikeres végrehajtás! A kimeneti fájl itt található: {final_video}")

if __name__ == "__main__":
    main()
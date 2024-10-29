import argparse
import subprocess
import json
import os
import sys
import glob
import datetime

def get_audio_codec(video_file):
    """
    Retrieves the audio codec of the first audio stream in the video file using ffprobe.
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'json',
            video_file
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        info = json.loads(result.stdout)
        codec = info['streams'][0]['codec_name']
        return codec
    except (subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Hiba az audio codec lekérése során: {e}")
        sys.exit(1)

def process_audio(audio_file, codec):
    """
    Re-encodes the audio file to match the codec and ensures it has at most stereo channels.
    Returns the path to the processed audio file.
    """
    processed_audio = "processed_audio." + codec
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite without asking
            '-i', audio_file,
            '-c:a', codec,
            '-ac', '2',  # Max stereo
            processed_audio
        ]
        subprocess.run(cmd, check=True)
        return processed_audio
    except subprocess.CalledProcessError as e:
        print(f"Hiba az audió feldolgozása során: {e}")
        sys.exit(1)

def add_audio_to_video(video_file, new_audio_file, language, codec, output_dir):
    """
    Adds the new audio track to the video file with the specified language tag.
    The new audio is encoded with the given codec.
    """
    # Derive the output file name based on the input video file name
    video_basename = os.path.basename(video_file)
    video_name, video_ext = os.path.splitext(video_basename)
    output_file = os.path.join(output_dir, f"{video_name}_with_new_audio{video_ext}")

    try:
        # Build the ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', video_file,
            '-i', new_audio_file,
            '-c', 'copy',  # Copy all streams by default
            '-c:a:1', codec,  # Encode the new audio stream
            '-metadata:s:a:1', f'language={language}',
            '-map', '0:v',      # Map video from first input
            '-map', '0:a?',     # Map existing audio if present
            '-map', '1:a',      # Map new audio
            output_file
        ]
        subprocess.run(cmd, check=True)
        print(f"Új audiosáv hozzáadva a videóhoz: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Hiba az audiosáv hozzáadása során: {e}")
        sys.exit(1)

def rename_existing_files(output_dir):
    """
    Ellenőrzi, hogy van-e *_with_new_audio.mkv nevű fájl a megadott könyvtárban.
    Ha igen, átnevezi a fájlt a létrehozás dátumával a formátumban "yyyy.mm.dd-hh_mm".
    """
    pattern = os.path.join(output_dir, '*_with_new_audio.mkv')
    matches = glob.glob(pattern)
    
    for file_path in matches:
        try:
            # Lekérjük a fájl létrehozási idejét
            # Megjegyzés: Egyes rendszereken a létrehozási idő helyett a módosítási időt kapjuk
            ctime = os.path.getctime(file_path)
            dt = datetime.datetime.fromtimestamp(ctime)
            formatted_date = dt.strftime("%Y.%m.%d-%H_%M")
            
            # Felbontjuk a fájlnevet és az extension-t
            dirname, basename = os.path.split(file_path)
            name, ext = os.path.splitext(basename)
            
            # Új név összeállítása
            new_name = f"{name}_{formatted_date}{ext}"
            new_path = os.path.join(dirname, new_name)
            
            # Átnevezés
            os.rename(file_path, new_path)
            print(f"Átnevezve: {file_path} -> {new_path}")
        except Exception as e:
            print(f"Hiba az átnevezés során a fájlhoz {file_path}: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Videó és audió fájl összefűzése új audiosáv hozzáadásával.")
    parser.add_argument('-i', '--input_video', required=True, help='A bemeneti videófájl elérési útja.')
    parser.add_argument('-ia', '--input_audio', required=True, help='A bemeneti audiófájl elérési útja.')
    parser.add_argument('-lang', '--language', required=True, help='A nyelvi címke (pl. eng, hun).')
    parser.add_argument('-o', '--output_dir', default='.', help='A kimeneti könyvtár elérési útja. Alapértelmezett: jelenlegi könyvtár.')

    args = parser.parse_args()

    # Ellenőrizzük, hogy a bemeneti fájlok léteznek
    if not os.path.isfile(args.input_video):
        print(f"A megadott videófájl nem található: {args.input_video}")
        sys.exit(1)
    if not os.path.isfile(args.input_audio):
        print(f"A megadott audiófájl nem található: {args.input_audio}")
        sys.exit(1)

    # Ellenőrizzük, hogy a kimeneti könyvtár létezik, ha nem, létrehozzuk
    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"Létrehozva a kimeneti könyvtár: {args.output_dir}")
        except OSError as e:
            print(f"Hiba a kimeneti könyvtár létrehozása során: {e}")
            sys.exit(1)

    # Ellenőrizzük és átnevezzük a meglévő *_with_new_audio.mkv fájlokat
    rename_existing_files(args.output_dir)

    # Lekérjük a videó audio codecét
    audio_codec = get_audio_codec(args.input_video)
    print(f"Eredeti audio codec: {audio_codec}")

    # Feldolgozzuk az új audiót
    processed_audio = process_audio(args.input_audio, audio_codec)
    print(f"Feldolgozott audió fájl: {processed_audio}")

    # Hozzáadjuk az új audiosávot a videóhoz
    add_audio_to_video(args.input_video, processed_audio, args.language, audio_codec, args.output_dir)

    # Tisztítjuk a feldolgozott audió fájlt
    if os.path.exists(processed_audio):
        os.remove(processed_audio)
        print(f"Törölve a feldolgozott audió fájl: {processed_audio}")

if __name__ == "__main__":
    main()


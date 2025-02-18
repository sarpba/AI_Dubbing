import argparse
import subprocess
import json
import os
import sys
import glob
import datetime

def get_audio_codec_and_sample_rate(video_file):
    """
    Lekéri a video első audió streamjének codec-jét és mintavételi arányát ffprobe segítségével.
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_name,sample_rate',
            '-of', 'json',
            video_file
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        info = json.loads(result.stdout)
        codec = info['streams'][0]['codec_name']
        sample_rate = info['streams'][0]['sample_rate']
        return codec, sample_rate
    except (subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Hiba az audio codec és sample rate lekérése során: {e}")
        sys.exit(1)

def process_audio(audio_file, codec, sample_rate):
    """
    Újrakódolja az audio fájlt, hogy a codec és a mintavételi arány megegyezzen az eredetivel,
    és legfeljebb sztereó csatornákat tartalmazzon.
    Ha az eredeti codec DTS, akkor AAC-t használ.
    Visszaadja a feldolgozott audio fájl útvonalát és a használt codec-et.
    """
    # Ha az eredeti codec 'opus', akkor a 'libopus'-t használjuk
    if codec.lower() == 'opus':
        codec_used = 'libopus'
        output_extension = 'opus'
    # Ha az eredeti codec DTS, akkor AAC-t használunk (így nem használ DTS-t)
    elif codec.lower() == 'dts':
        codec_used = 'aac'
        output_extension = 'm4a'
    else:
        codec_used = codec
        output_extension = codec.split('.')[-1]

    processed_audio = f"processed_audio.{output_extension}"
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Felülírja a már létező fájlt
            '-i', audio_file,
            '-c:a', codec_used,
            '-b:a', '128k',  # Bitrate beállítása
            '-ar', sample_rate,  # Mintavételi arány beállítása
            '-ac', '2',  # Legfeljebb sztereó
            processed_audio
        ]
        subprocess.run(cmd, check=True)
        return processed_audio, codec_used
    except subprocess.CalledProcessError as e:
        print(f"Hiba az audió feldolgozása során: {e}")
        sys.exit(1)

def add_audio_to_video(video_file, new_audio_file, language, codec, sample_rate, output_dir):
    """
    Hozzáadja az új audiosávot a videóhoz a megadott nyelvi címkével.
    Az új audió a megadott codec-et és mintavételi arányt használja.
    Az eredeti videóból csak a fő videó stream kerül átmásolásra.
    """
    # A kimeneti fájl neve az input videófájl alapján
    video_basename = os.path.basename(video_file)
    video_name, video_ext = os.path.splitext(video_basename)
    output_file = os.path.join(output_dir, f"{video_name}_with_new_audio{video_ext}")

    try:
        cmd = [
            'ffmpeg',
            '-i', video_file,
            '-i', new_audio_file,
            '-map', '0:v:0',      # Csak az első videó stream
            '-map', '0:a?',       # Esetlegesen meglévő audió stream-ek
            '-map', '1:a',        # Az új audió stream
            '-c:v', 'copy',       # Videó stream másolása kódolás nélkül
            '-c:a', 'copy',       # Az eredeti audió stream-ek másolása
            '-c:a:1', codec,      # Az új audió stream kódolása a megadott codec-szel
            '-ar:a:1', sample_rate,  # Az új audió mintavételi aránya
            '-metadata:s:a:1', f'language={language}',
            output_file
        ]
        subprocess.run(cmd, check=True)
        print(f"Új audiosáv hozzáadva a videóhoz: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Hiba az audiosáv hozzáadása során: {e}")
        sys.exit(1)

def rename_existing_files(output_dir):
    """
    Ellenőrzi az output könyvtárban az *_with_new_audio.mkv fájlokat.
    Ha talál ilyeneket, a létrehozás dátumát hozzáfűzi a fájl nevéhez.
    """
    pattern = os.path.join(output_dir, '*_with_new_audio.mkv')
    matches = glob.glob(pattern)
    
    for file_path in matches:
        try:
            ctime = os.path.getctime(file_path)
            dt = datetime.datetime.fromtimestamp(ctime)
            formatted_date = dt.strftime("%Y.%m.%d-%H_%M")
            
            dirname, basename = os.path.split(file_path)
            name, ext = os.path.splitext(basename)
            
            new_name = f"{name}_{formatted_date}{ext}"
            new_path = os.path.join(dirname, new_name)
            
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

    # Ellenőrzi, hogy a megadott fájlok léteznek-e
    if not os.path.isfile(args.input_video):
        print(f"A megadott videófájl nem található: {args.input_video}")
        sys.exit(1)
    if not os.path.isfile(args.input_audio):
        print(f"A megadott audiófájl nem található: {args.input_audio}")
        sys.exit(1)

    # Ellenőrzi, hogy a kimeneti könyvtár létezik-e, ha nem, létrehozza
    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"Létrehozva a kimeneti könyvtár: {args.output_dir}")
        except OSError as e:
            print(f"Hiba a kimeneti könyvtár létrehozása során: {e}")
            sys.exit(1)

    # Átnevezi a már létező *_with_new_audio.mkv fájlokat
    rename_existing_files(args.output_dir)

    # Lekéri az eredeti video audió codec-jét és mintavételi arányát
    audio_codec, sample_rate = get_audio_codec_and_sample_rate(args.input_video)
    print(f"Eredeti audio codec: {audio_codec}")
    print(f"Eredeti sample rate: {sample_rate} Hz")

    # Feldolgozza az új audiót, hogy megfeleljen a paramétereknek
    processed_audio, codec_used = process_audio(args.input_audio, audio_codec, sample_rate)
    print(f"Feldolgozott audió fájl: {processed_audio}")
    print(f"Használt codec: {codec_used}")

    # Hozzáadja az új audiosávot a videóhoz
    add_audio_to_video(args.input_video, processed_audio, args.language, codec_used, sample_rate, args.output_dir)

    # A létrehozott kimeneti fájl elérési útja
    video_basename = os.path.basename(args.input_video)
    video_name, video_ext = os.path.splitext(video_basename)
    output_file = os.path.join(args.output_dir, f"{video_name}_with_new_audio{video_ext}")

    # Törli a feldolgozott audió fájlt
    if os.path.exists(processed_audio):
        os.remove(processed_audio)
        print(f"Törölve a feldolgozott audió fájl: {processed_audio}")

    print("Script végrehajtása sikeresen befejeződött.")

if __name__ == "__main__":
    main()

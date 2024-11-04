import argparse
import subprocess
import json
import os
import sys
import glob
import datetime

def get_audio_codec_and_sample_rate(video_file):
    """
    Retrieves the audio codec and sample rate of the first audio stream in the video file using ffprobe.
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
    Re-encodes the audio file to match the codec and sample rate, and ensures it has at most stereo channels.
    Returns the path to the processed audio file.
    """
    processed_audio = f"processed_audio.{codec}"
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite without asking
            '-i', audio_file,
            '-c:a', codec,
            '-b:a', '128k',  # Bitrate beállítása
            '-ar', sample_rate,  # Mintaarány beállítása az eredetihoz
            '-ac', '2',  # Max stereo
            processed_audio
        ]
        subprocess.run(cmd, check=True)
        return processed_audio
    except subprocess.CalledProcessError as e:
        print(f"Hiba az audió feldolgozása során: {e}")
        sys.exit(1)

def add_audio_to_video(video_file, new_audio_file, language, codec, sample_rate, output_dir):
    """
    Adds the new audio track to the video file with the specified language tag.
    The new audio is encoded with the given codec and sample rate.
    Excludes any existing cover images by not mapping additional video streams.
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
            '-map', '0:v:0',      # Map only the primary video stream
            '-map', '0:a?',       # Map existing audio streams if present
            '-map', '1:a',        # Map new audio
            '-c:v', 'copy',       # Copy video stream without re-encoding
            '-c:a', 'copy',       # Copy existing audio streams without re-encoding
            '-c:a:1', codec,      # Encode the new audio stream
            '-ar:a:1', sample_rate,  # Ensure new audio has the same sample rate
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
    Checks for files matching *_with_new_audio.mkv in the specified directory.
    If found, renames them by appending the creation date in the format "yyyy.mm.dd-hh_mm".
    """
    pattern = os.path.join(output_dir, '*_with_new_audio.mkv')
    matches = glob.glob(pattern)
    
    for file_path in matches:
        try:
            # Retrieve the file creation time
            ctime = os.path.getctime(file_path)
            dt = datetime.datetime.fromtimestamp(ctime)
            formatted_date = dt.strftime("%Y.%m.%d-%H_%M")
            
            # Split the filename and extension
            dirname, basename = os.path.split(file_path)
            name, ext = os.path.splitext(basename)
            
            # Construct the new name
            new_name = f"{name}_{formatted_date}{ext}"
            new_path = os.path.join(dirname, new_name)
            
            # Rename the file
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

    # Check if input files exist
    if not os.path.isfile(args.input_video):
        print(f"A megadott videófájl nem található: {args.input_video}")
        sys.exit(1)
    if not os.path.isfile(args.input_audio):
        print(f"A megadott audiófájl nem található: {args.input_audio}")
        sys.exit(1)

    # Check if output directory exists, if not, create it
    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"Létrehozva a kimeneti könyvtár: {args.output_dir}")
        except OSError as e:
            print(f"Hiba a kimeneti könyvtár létrehozása során: {e}")
            sys.exit(1)

    # Rename existing *_with_new_audio.mkv files
    rename_existing_files(args.output_dir)

    # Get audio codec and sample rate from the original video
    audio_codec, sample_rate = get_audio_codec_and_sample_rate(args.input_video)
    print(f"Eredeti audio codec: {audio_codec}")
    print(f"Eredeti sample rate: {sample_rate} Hz")

    # Process the new audio to match codec and sample rate
    processed_audio = process_audio(args.input_audio, audio_codec, sample_rate)
    print(f"Feldolgozott audió fájl: {processed_audio}")

    # Add the new audio track to the video
    add_audio_to_video(args.input_video, processed_audio, args.language, audio_codec, sample_rate, args.output_dir)

    # Define the path to the newly created file
    video_basename = os.path.basename(args.input_video)
    video_name, video_ext = os.path.splitext(video_basename)
    output_file = os.path.join(args.output_dir, f"{video_name}_with_new_audio{video_ext}")

    # Clean up the processed audio file
    if os.path.exists(processed_audio):
        os.remove(processed_audio)
        print(f"Törölve a feldolgozott audió fájl: {processed_audio}")

    print("Script végrehajtása sikeresen befejeződött.")

if __name__ == "__main__":
    main()

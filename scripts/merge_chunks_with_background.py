import os
import re
import argparse
from datetime import datetime
from pydub import AudioSegment

def parse_time_string(time_str):
    """
    Parse a time string in HH-MM-SS-mmm format to milliseconds.
    """
    pattern = r"^(\d{2})-(\d{2})-(\d{2})-(\d{3})$"
    match = re.match(pattern, time_str)
    if not match:
        raise ValueError(f"Time string '{time_str}' does not match the expected format HH-MM-SS-mmm.")
    
    h, m, s, ms = match.groups()
    
    # Convert to milliseconds
    total_ms = (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)
    return total_ms

def parse_filename(filename):
    """
    Parse the filename to extract start and end times in milliseconds.
    Expected format: HH-MM-SS-mmm_HH-MM-SS-mmm[_EXTRA].wav
    Example: 00-00-10-287_00-00-13-811_SPEAKER_20.wav
    """
    parts = filename.split('_')
    if len(parts) < 2:
        raise ValueError(f"Filename '{filename}' does not contain enough parts separated by underscores.")
        
    start_time_str = parts[0]
    end_time_str = parts[1]
    
    try:
        start_time = parse_time_string(start_time_str)
        end_time = parse_time_string(end_time_str)
    except ValueError as e:
        # Re-raise with the original filename for context
        raise ValueError(f"Error parsing filename '{filename}': {e}")
        
    return start_time, end_time

def merge_wav_files(input_folder, output_file, background_file=None):
    """
    Merge WAV files from the input_folder into a single WAV file.
    Each file is overlaid at the start time specified in its filename.
    Gaps are filled with silence.
    Overlapping files are mixed together, preserving original volumes.
    Optionally overlays the merged audio onto a background music file.
    """
    timeline = []
    max_end_time = 0
    
    # List all WAV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            try:
                start_time, end_time = parse_filename(filename)
                filepath = os.path.join(input_folder, filename)
                audio = AudioSegment.from_wav(filepath)
                # Optional: Verify that audio duration matches the filename
                expected_duration = end_time - start_time
                actual_duration = len(audio)
                if abs(expected_duration - actual_duration) > 10:  # Allow 10 ms discrepancy
                    print(f"Warning: Duration mismatch in '{filename}'. Expected {expected_duration} ms, got {actual_duration} ms.")
                timeline.append((start_time, audio))
                if end_time > max_end_time:
                    max_end_time = end_time
            except ValueError as ve:
                print(f"Skipping file: {ve}")
    
    if not timeline:
        print("No valid WAV files found to merge.")
        return
    
    # Sort the timeline by start_time
    timeline.sort(key=lambda x: x[0])
    
    # Initialize the final audio with silence of total duration
    final_audio = AudioSegment.silent(duration=max_end_time)
    
    for start_time, audio in timeline:
        # Overlay the current audio at the specified start_time without altering its volume
        final_audio = final_audio.overlay(audio, position=start_time)
    
    # If background music is provided, overlay the final_audio onto the background
    if background_file:
        if not os.path.isfile(background_file):
            print(f"Hiba: A háttérzene fájlja '{background_file}' nem található.")
            return
        
        try:
            background = AudioSegment.from_file(background_file)
        except Exception as e:
            print(f"Hiba a háttérzene fájl betöltésekor: {e}")
            return
        
        # Determine the duration of the final audio
        final_duration = len(final_audio)
        
        # Loop the background music if it's shorter than the final audio
        if len(background) < final_duration:
            repeats = final_duration // len(background) + 1
            background = background * repeats
        
        # Trim the background music to match the final audio duration
        background = background[:final_duration]
        
        # Overlay the final audio onto the background
        final_with_bg = background.overlay(final_audio)
        
        # Export the final audio with background
        final_with_bg.export(output_file, format="wav")
        print(f"Merged audio with background saved to '{output_file}'")
    else:
        # Export the final audio without background
        final_audio.export(output_file, format="wav")
        print(f"Merged audio saved to '{output_file}'")

def main():
    parser = argparse.ArgumentParser(description="Összeilleszt WAV fájlokat időalapú neveik szerint, csenddel kitöltve a köztes részeket és összemixelve az átfedéseket. Opcionálisan háttérzenével is kiegészíthető.")
    parser.add_argument('-i', '--input', required=True, help='Az input könyvtár, amely tartalmazza a WAV fájlokat.')
    parser.add_argument('-o', '--output', required=True, help='A kimeneti könyvtár, ahol a generált WAV fájl lesz elmentve. A fájl neve a futtatás ideje alapján automatikusan generálódik (YYYY.MM.DD_HH.MM.SS.wav).')
    parser.add_argument('-bg', '--background', required=False, help='A háttérzene WAV fájlja.')
    
    args = parser.parse_args()
    
    input_folder = args.input
    output_dir = args.output
    background_file = args.background

    # Ellenőrizzük, hogy az input könyvtár létezik-e
    if not os.path.isdir(input_folder):
        print(f"Hiba: Az input könyvtár '{input_folder}' nem létezik vagy nem elérhető.")
        return

    # Ellenőrizzük, hogy az output könyvtár létezik-e, ha nem, létrehozzuk
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Létrehozva az output könyvtár: {output_dir}")
        except Exception as e:
            print(f"Hiba az output könyvtár létrehozásakor: {e}")
            return

    # Generáljuk a fájlnevet a futtatás ideje alapján (YYYY.MM.DD_HH.MM.SS.wav)
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    output_file = os.path.join(output_dir, f"{timestamp}.wav")
    print(f"A kimeneti fájl neve: {output_file}")
    
    merge_wav_files(input_folder, output_file, background_file)

if __name__ == "__main__":
    main()

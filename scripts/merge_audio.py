import os
import re
import argparse
from pydub import AudioSegment

def parse_filename(filename):
    """
    Parse the filename to extract start and end times in milliseconds.
    Expected format: HH-MM-SS.mmm-HH-MM-SS.mmm[_EXTRA].wav
    Example: 00-00-10.287-00-00-13.811_SPEAKER_20.wav
    """
    # Extract the first part before any underscore
    time_part = filename.split('_')[0]
    
    # Define the regex pattern to match the time part
    pattern = r"^(\d{2})-(\d{2})-(\d{2}\.\d{3})-(\d{2})-(\d{2})-(\d{2}\.\d{3})$"
    match = re.match(pattern, time_part)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match the expected format.")
    
    groups = match.groups()
    start_h, start_m, start_s_ms = groups[0], groups[1], groups[2]
    end_h, end_m, end_s_ms = groups[3], groups[4], groups[5]
    
    # Convert start and end times to milliseconds
    start_time = (int(start_h) * 3600 + int(start_m) * 60 + float(start_s_ms)) * 1000  # in ms
    end_time = (int(end_h) * 3600 + int(end_m) * 60 + float(end_s_ms)) * 1000  # in ms
    
    return int(start_time), int(end_time)

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
        
        # **Hangerejének Megőrzése:** Eltávolítjuk a hangerő csökkentését
        # background = background - 20  # Eltávolítva, hogy a háttérzene eredeti hangerőn maradjon
        
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
    parser.add_argument('-o', '--output', required=True, help='A kimeneti WAV fájl neve vagy mappája.')
    parser.add_argument('-bg', '--background', required=False, help='A háttérzene WAV fájlja.')
    
    args = parser.parse_args()
    
    input_folder = args.input
    output_file = args.output
    background_file = args.background
    
    if not os.path.isdir(input_folder):
        print(f"Hiba: Az input könyvtár '{input_folder}' nem létezik vagy nem elérhető.")
        return
    
    # Ellenőrizzük, hogy az output path egy mappa-e
    if os.path.isdir(output_file):
        # Ha az output path egy mappa, hozzunk létre egy alapértelmezett fájlnevet
        output_file = os.path.join(output_file, "merged_output.wav")
        print(f"Az output mappa meg van adva. A kimeneti fájl: {output_file}")
    else:
        # Ha az output path nem létezik, ellenőrizzük, hogy a szülő könyvtár létezik-e
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"Létrehozva az output könyvtár: {output_dir}")
            except Exception as e:
                print(f"Hiba az output könyvtár létrehozásakor: {e}")
                return
    
    merge_wav_files(input_folder, output_file, background_file)

if __name__ == "__main__":
    main()


import os
import re
import argparse
import json
from datetime import datetime
from pathlib import Path
from pydub import AudioSegment

import sys

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

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
    # JAVÍTÁS: Először távolítsuk el a fájlkiterjesztést, hogy a split biztosan jól működjön.
    name_without_ext, _ = os.path.splitext(filename)
    parts = name_without_ext.split('_')
    
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
    parser = argparse.ArgumentParser(
        description="Egy projekt könyvtáron belül automatikusan összeilleszti a lefordított hang darabokat ('translated_splits') "
                    "a megfelelő háttérzenére ('separated_audio_background'), és az eredményt a 'film_dubbing' könyvtárba menti. "
                    "A könyvtárstruktúrát a 'config.json' fájl alapján határozza meg."
    )
    parser.add_argument('project_name', help='A feldolgozandó projekt könyvtár neve a "workdir"-en belül.')
    add_debug_argument(parser)
    args = parser.parse_args()
    configure_debug_mode(args.debug)
    project_name = args.project_name

    try:
        # A szkript a 'scripts' könyvtárban van, a config.json egy szinttel feljebb
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, '..', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Hiba: A config.json fájl nem található itt: {config_path}")
        return
    except json.JSONDecodeError:
        print(f"Hiba: A config.json fájl hibás formátumú.")
        return
    
    try:
        workdir_base = config['DIRECTORIES']['workdir']
        subdirs = config['PROJECT_SUBDIRS']
        input_subdir = subdirs['translated_splits']
        output_subdir = subdirs['film_dubbing']
        background_subdir = subdirs['separated_audio_background']
    except KeyError as e:
        print(f"Hiba: Hiányzó kulcs a config.json fájlban: {e}")
        return

    # Teljes útvonalak összeállítása
    project_path = os.path.join(workdir_base, project_name)
    input_folder = os.path.join(project_path, input_subdir)
    output_dir = os.path.join(project_path, output_subdir)
    background_dir = os.path.join(project_path, background_subdir)

    # Háttérzene fájl automatikus megkeresése
    background_file = None
    if os.path.isdir(background_dir):
        supported_formats = ('.wav', '.mp3', '.aac', '.flac', 'm4a', 'ogg')
        # Fájlok rendezése a determinisztikus viselkedésért
        for f in sorted(os.listdir(background_dir)):
            if f.lower().endswith(supported_formats):
                background_file = os.path.join(background_dir, f)
                print(f"Megtalált háttérzene fájl: {background_file}")
                break  # Az első találtat használjuk
    
    if not background_file:
        print("Nem található háttérzene fájl. A folyamat háttérzene nélkül folytatódik.")

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

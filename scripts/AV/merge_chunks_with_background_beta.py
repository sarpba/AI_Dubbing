import os
import re
import argparse
import json
from datetime import datetime
from pathlib import Path
import math

import sys

from pydub import AudioSegment

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode


def get_project_root() -> Path:
    """
    Felkeresi a projekt gyökerét a config.json alapján.
    """
    for candidate in Path(__file__).resolve().parents:
        config_candidate = candidate / "config.json"
        if config_candidate.is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")

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

def apply_volume_percent(audio_segment, percent):
    """
    Adjust the volume of an AudioSegment based on a percentage value.
    100% leaves the audio unchanged.
    """
    if percent == 100:
        return audio_segment
    gain_db = 20 * math.log10(percent / 100.0)
    return audio_segment.apply_gain(gain_db)


def apply_speedup_percent(audio_segment, percent):
    """
    Speed up an AudioSegment by the given percentage while preserving the original frame rate.
    A 10% value shortens the audio duration by roughly 10%.
    """
    if percent <= 0:
        return audio_segment

    factor = 1 + (percent / 100.0)
    base_frame_rate = audio_segment.frame_rate
    sped_up = audio_segment._spawn(audio_segment.raw_data, overrides={"frame_rate": int(base_frame_rate * factor)})
    return sped_up.set_frame_rate(base_frame_rate)


def merge_wav_files(
    input_folder,
    output_file,
    background_file=None,
    background_volume_percent=100,
    max_speedup_percent=10,
):
    """
    Merge WAV files from the input_folder into a single WAV file.
    Segments are sorted by their intended start times, but any overlaps are resolved
    by shifting later segments so that playback always advances forward in time.
    Gaps are preserved as silence.
    Optionally overlays the merged audio onto a background music file.
    Segments can be sped up by up to max_speedup_percent to reduce the accumulated delay.
    """
    timeline = []

    # List all WAV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            try:
                start_time, end_time = parse_filename(filename)
                filepath = os.path.join(input_folder, filename)
                audio = AudioSegment.from_wav(filepath)
                expected_duration = end_time - start_time
                actual_duration = len(audio)
                if actual_duration - expected_duration > 10:  # Allow 10 ms discrepancy
                    print(f"Warning: Duration mismatch in '{filename}'. Expected {expected_duration} ms, got {actual_duration} ms.")
                adjusted_audio = apply_speedup_percent(audio, max_speedup_percent)
                timeline.append((start_time, adjusted_audio, filename))
            except ValueError as ve:
                print(f"Skipping file: {ve}")

    if not timeline:
        print("No valid WAV files found to merge.")
        return

    # Sort the timeline by intended start_time
    timeline.sort(key=lambda x: x[0])

    final_audio = AudioSegment.silent(duration=0)
    current_position = 0

    for start_time, audio, filename in timeline:
        if start_time > current_position:
            gap = start_time - current_position
            final_audio += AudioSegment.silent(duration=gap)
            current_position += gap
        elif start_time < current_position:
            overlap = current_position - start_time
            print(f"Segment '{filename}' overlaps the previous one by {overlap} ms; shifting to {current_position} ms.")

        final_audio += audio
        current_position += len(audio)

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

        background = apply_volume_percent(background, background_volume_percent)
        
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
    parser.add_argument('-narrator', '--narrator', action='store_true',
                        help='Narrátor mód aktiválása: háttér a "extracted_audio" mappából.')
    parser.add_argument('--background-volume', type=int, choices=range(1, 101), metavar='PERCENT',
                        help='Narrátor módban a háttér hangereje százalékban (1-100).')
    parser.add_argument(
        '--max-speedup',
        type=int,
        choices=range(0, 101),
        metavar='PERCENT',
        default=10,
        help='A szegmensek maximális gyorsítása százalékban (0-100). 10 azt jelenti, hogy a darabok 10%-kal rövidebbek lesznek.',
    )
    add_debug_argument(parser)
    args = parser.parse_args()

    if args.background_volume is not None and not args.narrator:
        parser.error("A '--background-volume' opció csak narrátor módban használható.")

    background_volume_percent = args.background_volume if args.background_volume is not None else 100
    configure_debug_mode(args.debug)
    project_name = args.project_name

    try:
        project_root = get_project_root()
    except FileNotFoundError as exc:
        print(f"Hiba: {exc}")
        return

    config_path = project_root / "config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Hiba: A config.json fájl nem található itt: {config_path}")
        return
    except json.JSONDecodeError as exc:
        print(f"Hiba: A config.json fájl hibás formátumú ({config_path}): {exc}")
        return
    
    try:
        workdir_base = project_root / config['DIRECTORIES']['workdir']
        subdirs = config['PROJECT_SUBDIRS']
        input_subdir = subdirs['translated_splits']
        output_subdir = subdirs['film_dubbing']
        background_key = 'extracted_audio' if args.narrator else 'separated_audio_background'
        if background_key not in subdirs:
            print(f"Hiba: A config.json nem tartalmazza a(z) '{background_key}' bejegyzést a PROJECT_SUBDIRS alatt.")
            return
        background_subdir = subdirs[background_key]
    except KeyError as e:
        print(f"Hiba: Hiányzó kulcs a config.json fájlban: {e}")
        return

    # Teljes útvonalak összeállítása
    project_path = workdir_base / project_name
    input_folder = project_path / input_subdir
    output_dir = project_path / output_subdir
    background_dir = project_path / background_subdir

    # Háttérzene fájl automatikus megkeresése
    background_file = None
    if background_dir.is_dir():
        supported_formats = ('.wav', '.mp3', '.aac', '.flac', 'm4a', 'ogg')
        # Fájlok rendezése a determinisztikus viselkedésért
        for audio_file in sorted(background_dir.iterdir()):
            if audio_file.suffix.lower() in supported_formats:
                background_file = str(audio_file)
                print(f"Megtalált háttérzene fájl: {background_file}")
                break  # Az első találtat használjuk
    
    if not background_file:
        print("Nem található háttérzene fájl. A folyamat háttérzene nélkül folytatódik.")

    # Ellenőrizzük, hogy az input könyvtár létezik-e
    if not input_folder.is_dir():
        print(f"Hiba: Az input könyvtár '{input_folder}' nem létezik vagy nem elérhető.")
        return

    # Ellenőrizzük, hogy az output könyvtár létezik-e, ha nem, létrehozzuk
    if not output_dir.is_dir():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Létrehozva az output könyvtár: {output_dir}")
        except Exception as e:
            print(f"Hiba az output könyvtár létrehozásakor: {e}")
            return

    # Generáljuk a fájlnevet a futtatás ideje alapján (YYYY.MM.DD_HH.MM.SS.wav)
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    output_file = str(output_dir / f"{timestamp}.wav")
    print(f"A kimeneti fájl neve: {output_file}")
    
    merge_wav_files(
        str(input_folder),
        output_file,
        background_file,
        background_volume_percent,
        args.max_speedup,
    )

if __name__ == "__main__":
    main()

import os
import re
import argparse
import json
from datetime import datetime
from pathlib import Path
from pydub import AudioSegment
import math
import sys

# A time-stretching funkcióhoz szükséges importok
import numpy as np
import pyrubberband as pyrb


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

def speed_up_audio(audio_segment, target_duration_ms):
    """
    Felgyorsít egy pydub AudioSegment-et egy célértékre a pyrubberband segítségével
    a magasabb minőség és a torzítás elkerülése érdekében.
    """
    original_duration_ms = len(audio_segment)
    if original_duration_ms <= target_duration_ms or target_duration_ms <= 0:
        return audio_segment

    rate = original_duration_ms / target_duration_ms
    
    # 1. pydub AudioSegment konvertálása numpy tömbbé
    samples = np.array(audio_segment.get_array_of_samples())
    # Sztereó hang kezelése
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels))

    # 2. Time-stretching alkalmazása a pyrubberband segítségével
    # A pyrubberband a mintavételezési frekvenciát is paraméterként várja
    stretched_samples = pyrb.time_stretch(samples, audio_segment.frame_rate, rate)

    # 3. A numpy tömb visszaalakítása pydub AudioSegment-té
    # A pyrubberband float tömböt ad vissza, ezt vissza kell alakítani az eredeti integer típusra
    dtype = f"int{audio_segment.sample_width * 8}"
    stretched_samples_int = stretched_samples.astype(getattr(np, dtype))
    
    # Az új AudioSegment létrehozása a feldolgozott bájtokból
    new_audio = AudioSegment(
        stretched_samples_int.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=audio_segment.sample_width,
        channels=audio_segment.channels
    )

    return new_audio

def merge_wav_files(input_folder, output_file, background_file=None, background_volume_percent=100, time_stretching=False):
    """
    Összefűzi a WAV fájlokat egyetlen WAV fájllá.
    Minden fájlt a fájlnevében megadott időponttól kezdve illeszt be.
    A réseket csenddel tölti ki.
    Az egymást átfedő fájlokat összekeveri.
    Opcionálisan a kész hangot háttérzenére illeszti.
    Ha a time_stretching True, felgyorsítja azokat a klipeket, amelyek átfedésben lennének a következővel.
    """
    timeline = []
    max_end_time = 0
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            try:
                start_time, end_time = parse_filename(filename)
                filepath = os.path.join(input_folder, filename)
                audio = AudioSegment.from_wav(filepath)
                expected_duration = end_time - start_time
                actual_duration = len(audio)
                if actual_duration - expected_duration > 10:
                    print(f"Warning: Duration mismatch in '{filename}'. Expected {expected_duration} ms, got {actual_duration} ms.")
                
                timeline.append({'start': start_time, 'end': end_time, 'audio': audio, 'filename': filename})
                
                if end_time > max_end_time:
                    max_end_time = end_time
            except ValueError as ve:
                print(f"Skipping file: {ve}")
    
    if not timeline:
        print("No valid WAV files found to merge.")
        return
    
    timeline.sort(key=lambda x: x['start'])
    
    # --- TIME-STRETCHING LOGIKA ---
    processed_timeline = []
    if time_stretching:
        print("Time-stretching engedélyezve. Átfedések ellenőrzése...")
        for i, current_item in enumerate(timeline):
            # Az utolsó elemen nincs mit ellenőrizni
            if i == len(timeline) - 1:
                processed_timeline.append(current_item)
                continue

            next_item = timeline[i+1]
            current_audio = current_item['audio']
            current_start_time = current_item['start']
            
            current_actual_end_time = current_start_time + len(current_audio)
            next_start_time = next_item['start']

            if current_actual_end_time > next_start_time:
                target_duration = next_start_time - current_start_time
                # Csak akkor gyorsítunk, ha van értelme (a célhossz pozitív)
                if target_duration > 0:
                    print(f"  -> Time-stretching alkalmazása: '{current_item['filename']}'.")
                    print(f"     Eredeti hossz: {len(current_audio)} ms. Átfedés: {current_actual_end_time - next_start_time} ms.")
                    print(f"     Cél hossz: {target_duration} ms.")
                    
                    stretched_audio = speed_up_audio(current_audio, target_duration)
                    current_item['audio'] = stretched_audio
                else:
                    print(f"  -> Figyelmeztetés: Nem lehet time-stretching-et alkalmazni a(z) '{current_item['filename']}' fájlra, mert a következő klip túl korán kezdődik.")
            
            processed_timeline.append(current_item)
    else:
        processed_timeline = timeline

    final_audio = AudioSegment.silent(duration=max_end_time)
    
    for item in processed_timeline:
        final_audio = final_audio.overlay(item['audio'], position=item['start'])
    
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
        
        final_duration = len(final_audio)
        
        if len(background) < final_duration:
            repeats = -(-final_duration // len(background)) # Ceiling division
            background = background * repeats
        
        background = background[:final_duration]
        final_with_bg = background.overlay(final_audio)
        final_with_bg.export(output_file, format="wav")
        print(f"A hangfájlok háttérzenével összefűzve és elmentve ide: '{output_file}'")
    else:
        final_audio.export(output_file, format="wav")
        print(f"A hangfájlok összefűzve és elmentve ide: '{output_file}'")

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
    parser.add_argument('--time-stretching', action='store_true',
                        help='Aktiválja az automatikus time-stretching funkciót, hogy elkerülje az audiók átfedését.')
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

    project_path = workdir_base / project_name
    input_folder = project_path / input_subdir
    output_dir = project_path / output_subdir
    background_dir = project_path / background_subdir

    background_file = None
    if background_dir.is_dir():
        supported_formats = ('.wav', '.mp3', '.aac', '.flac', 'm4a', 'ogg')
        for audio_file in sorted(background_dir.iterdir()):
            if audio_file.suffix.lower() in supported_formats:
                background_file = str(audio_file)
                print(f"Megtalált háttérzene fájl: {background_file}")
                break
    
    if not background_file:
        print("Nem található háttérzene fájl. A folyamat háttérzene nélkül folytatódik.")

    if not input_folder.is_dir():
        print(f"Hiba: Az input könyvtár '{input_folder}' nem létezik vagy nem elérhető.")
        return

    if not output_dir.is_dir():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Létrehozva az output könyvtár: {output_dir}")
        except Exception as e:
            print(f"Hiba az output könyvtár létrehozásakor: {e}")
            return

    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    output_file = str(output_dir / f"{timestamp}.wav")
    print(f"A kimeneti fájl neve: {output_file}")
    
    merge_wav_files(str(input_folder), output_file, background_file, background_volume_percent, time_stretching=args.time_stretching)

if __name__ == "__main__":
    main()
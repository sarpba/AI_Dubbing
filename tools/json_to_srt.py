import json
import argparse
import os
from pathlib import Path

try:
    from tools.debug_utils import add_debug_argument, configure_debug_mode
except ImportError:  # pragma: no cover - allows running standalone
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from tools.debug_utils import add_debug_argument, configure_debug_mode

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def json_to_srt(input_file):
    # Betöltjük a JSON fájlt
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON dekódolási hiba: {e}")
        return

    # Ellenőrizzük, hogy a 'word_segments' létezik-e
    if "word_segments" not in data:
        print("Hiba: A JSON fájl nem tartalmaz 'word_segments' kulcsot.")
        print("Elérhető kulcsok:", list(data.keys()))
        return

    word_segments = data["word_segments"]

    # Felirat file név generálása
    output_file = os.path.splitext(input_file)[0] + "_words.srt"
    
    with open(output_file, 'w', encoding='utf-8') as srt_file:
        index = 1
        for segment in word_segments:
            # Ellenőrizzük, hogy a szükséges kulcsok jelen vannak-e
            if all(k in segment for k in ("start", "end", "word")):
                start_time = segment["start"]
                end_time = segment["end"]
                word = segment["word"]

                # SRT formátumú idő átalakítása
                start_srt_time = format_timestamp(start_time)
                end_srt_time = format_timestamp(end_time)

                # SRT formátum írása
                srt_file.write(f"{index}\n")
                srt_file.write(f"{start_srt_time} --> {end_srt_time}\n")
                srt_file.write(f"{word}\n\n")
                index += 1
            else:
                # Ha nem tartalmazza a szükséges kulcsokat, egyszerűen átugorjuk
                continue

    print(f"SRT fájl elkészült: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate word-level SRT file from JSON file.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input JSON file")
    add_debug_argument(parser)
    args = parser.parse_args()
    configure_debug_mode(args.debug)
    
    json_to_srt(args.input)

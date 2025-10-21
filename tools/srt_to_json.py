import argparse
import json
import re
import random
from pathlib import Path

try:
    from tools.debug_utils import add_debug_argument, configure_debug_mode
except ImportError:  # pragma: no cover - allows running as a standalone script
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from tools.debug_utils import add_debug_argument, configure_debug_mode

def srt_time_to_seconds(time_str):
    """
    Átalakítja a srt időformátumot (hh:mm:ss,ms) lebegőpontos másodpercekké.
    Például: "00:00:26,109" -> 26.109
    """
    hours, minutes, seconds_ms = time_str.split(':')
    seconds, ms = seconds_ms.split(',')
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms) / 1000
    return total_seconds

def remove_formatting(text):
    """
    Eltávolítja a HTML-szerű tageket, például <i> és </i>
    """
    return re.sub(r'<[^>]+>', '', text)

def parse_srt(srt_content):
    """
    Feldolgozza az srt fájl tartalmát, és listát ad vissza a blokkokból,
    ahol minden blokk egy dict a következő kulcsokkal:
      - start: lebegőpontos kezdőidő (másodpercben)
      - end: lebegőpontos végidő (másodpercben)
      - text: a formázás nélküli szöveg
    """
    segments = []
    # A srt blokk általában üres sorral van elválasztva
    blocks = srt_content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 3:
            # Az első sor az index, a második sor az időintervallum,
            # a többi sor pedig a felirat szöveg
            index = lines[0].strip()
            time_line = lines[1].strip()
            # A formátum: "00:00:26,109 --> 00:00:27,152"
            try:
                start_str, end_str = [t.strip() for t in time_line.split('-->')]
            except ValueError:
                # Ha nem megfelelő a formátum, ugorjunk rá
                continue
            start = srt_time_to_seconds(start_str)
            end = srt_time_to_seconds(end_str)
            # A szövegek összefűzése, és a formázások eltávolítása
            raw_text = ' '.join(lines[2:])
            text = remove_formatting(raw_text).strip()
            segments.append({
                'start': start,
                'end': end,
                'text': " " + text  # a mintában a text előtt van egy szóköz
            })
    return segments

def create_json_structure(segments):
    """
    A kapott segment listából előállítja a kívánt JSON struktúrát.
    Minden segmenthez hozzárendel egy "words" listát (itt csak egy elemet tartalmaz),
    valamint egy "speaker" kulcsot.
    
    Mivel az srt fájl nem tartalmaz speaker információt, itt alapértelmezett értéket adunk,
    például "SPEAKER_1". A score értékét véletlenszerűen generáljuk (példa kedvéért).
    """
    json_segments = []
    # Alapértelmezett speaker; ha szeretnénk változatosabbá tenni, később ezt módosíthatjuk.
    default_speaker = "SPEAKER_1"
    for seg in segments:
        score = round(random.uniform(0.1, 1.0), 3)  # véletlenszerű score 0.1 és 1.0 között, 3 tizedesjegy pontossággal
        word_obj = {
            "word": seg['text'].strip(),
            "start": seg['start'],
            "end": seg['end'],
            "score": score,
            "speaker": default_speaker
        }
        seg_obj = {
            "start": seg['start'],
            "end": seg['end'],
            "text": seg['text'],
            "words": [word_obj],
            "speaker": default_speaker
        }
        json_segments.append(seg_obj)
    return {"segments": json_segments}

def main():
    parser = argparse.ArgumentParser(
        description="SRT fájl konvertálása a megadott JSON formátumra.")
    parser.add_argument('-i', '--input', required=True,
                        help='Bemeneti SRT fájl neve')
    parser.add_argument('-o', '--output', required=True,
                        help='Kimeneti JSON fájl neve')
    add_debug_argument(parser)
    args = parser.parse_args()
    configure_debug_mode(args.debug)

    # SRT fájl beolvasása
    try:
        with open(args.input, 'r', encoding='utf-8') as infile:
            srt_content = infile.read()
    except Exception as e:
        print(f"Hiba a bemeneti fájl olvasása közben: {e}")
        return

    segments = parse_srt(srt_content)
    json_structure = create_json_structure(segments)

    # JSON fájlba írása
    try:
        with open(args.output, 'w', encoding='utf-8') as outfile:
            json.dump(json_structure, outfile, indent=4, ensure_ascii=False)
        print(f"A konvertálás kész. Az új fájl neve: {args.output}")
    except Exception as e:
        print(f"Hiba a kimeneti fájl írása közben: {e}")

if __name__ == '__main__':
    main()

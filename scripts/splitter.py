import os
import argparse
import json
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import timedelta

def sanitize_filename(filename):
    """
    Csere a tiltott karaktereket a fájlnévben.
    Itt a kettőspontot kötőjelre cseréljük.
    """
    return filename.replace(":", "-").replace(",", ".")

def format_timedelta_seconds(seconds):
    """
    Formázza a másodperceket SRT-szerű időbélyeg formátumra: HH-MM-SS.mmm
    """
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000) if hasattr(td, 'microseconds') else 0
    return f"{hours:02}-{minutes:02}-{secs:02}.{int((seconds - int(seconds)) * 1000):03}"

def process_json_file(args):
    json_path, audio_dir, output_dir, relative_path = args
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.opus']
    audio_file = None
    original_extension = None
    for ext in audio_extensions:
        potential_audio_path = os.path.join(audio_dir, base_name + ext)
        if os.path.exists(potential_audio_path):
            audio_file = potential_audio_path
            original_extension = ext.lstrip('.').lower()
            break

    if not audio_file:
        return f"Audio fájl nem található: '{base_name}' számára. (JSON: '{json_path}')"

    try:
        # Speciális fájlformátumok kezelése
        if original_extension == 'opus':
            # Az .opus fájlokat 'ogg' formátumként kezeljük
            audio = AudioSegment.from_file(audio_file, format='ogg')
            export_format = 'ogg'
            export_extension = 'ogg'
        elif original_extension == 'm4a':
            # Az .m4a fájlokat 'mp4' formátumként kezeljük
            audio = AudioSegment.from_file(audio_file, format='mp4')
            export_format = 'mp4'
            export_extension = 'mp4'
        else:
            audio = AudioSegment.from_file(audio_file, format=original_extension)
            export_format = original_extension
            export_extension = original_extension
    except Exception as e:
        return f"Hiba történt az audio fájl betöltése közben '{audio_file}': {e}"

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
            segments = json_content.get("segments", [])
    except Exception as e:
        return f"Hiba történt a JSON fájl betöltése közben '{json_path}': {e}"

    sentence_counter = 0
    errors = []

    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        text = segment.get("text", "").strip()
        speaker = segment.get("speaker", "UNKNOWN")

        if start is None or end is None:
            errors.append(f"Hiányzó 'start' vagy 'end' érték a '{base_name}' fájl '{sentence_counter + 1}' szegmensénél.")
            continue

        if not text:
            continue  # Üres szöveg, kihagyjuk

        if start >= end:
            errors.append(f"Nem érvényes vágási pontok a '{base_name}' fájl '{sentence_counter + 1}' szegmensénél.")
            continue

        # Átalakítjuk a timestampokat miliszekundumokra
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)

        # Mappastruktúra létrehozása a relatív útvonal alapján
        output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
        os.makedirs(output_subdir, exist_ok=True)

        try:
            audio_segment = audio[start_ms:end_ms]
        except Exception as e:
            errors.append(f"Hiba történt az audio szakasz kivágása közben '{base_name}' szegmens {sentence_counter + 1}-nél: {e}")
            continue

        # Timestamp formázása a fájlnévhez
        start_timestamp = format_timedelta_seconds(start)  # Példa: 00-00-09.287
        end_timestamp = format_timedelta_seconds(end)      # Példa: 00-00-17.312

        # Fájlnév összeállítása és sanitizálása
        filename_base = f"{sanitize_filename(start_timestamp)}-{sanitize_filename(end_timestamp)}"

        # Ha a beszélő információ is szükséges a fájlnévben, hozzáadhatjuk
        filename_base = f"{filename_base}_{sanitize_filename(speaker)}"

        output_audio_path = os.path.join(output_subdir, f"{filename_base}.{export_extension}")
        try:
            # Exportáljuk az audio szakaszt a megfelelő formátumban
            audio_segment.export(output_audio_path, format=export_format)
        except Exception as e:
            error_msg = f"Hiba történt az audio szakasz exportálása közben '{output_audio_path}': {e}"
            errors.append(error_msg)
            continue

        output_text_path = os.path.join(output_subdir, f"{filename_base}.txt")
        try:
            with open(output_text_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)
        except Exception as e:
            error_msg = f"Hiba történt a szövegfájl írása közben '{output_text_path}': {e}"
            errors.append(error_msg)
            continue

        sentence_counter += 1

    stats = f"Fájl: '{base_name}'\n" \
            f"Feldolgozott szegmensek: {sentence_counter}"

    if errors:
        return f"{stats}\nHibák:\n" + "\n".join(errors)
    else:
        return f"{stats}\nFeldolgozás sikeresen befejezve."

def process_directory(input_dir, output_dir):
    json_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                audio_dir = root
                relative_path = os.path.relpath(json_path, input_dir)
                json_files.append((json_path, audio_dir, output_dir, relative_path))

    total_files = len(json_files)
    if total_files == 0:
        print("Nincs feldolgozandó JSON fájl a megadott bemeneti könyvtárban.")
        return

    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_json_file, args): args[0] for args in json_files}
        for future in tqdm(as_completed(future_to_file), total=total_files, desc="Feldolgozás"):
            json_path = future_to_file[future]
            try:
                result = future.result()
                if result:
                    print(result)
            except Exception as exc:
                print(f"Hiba történt a '{json_path}' fájl feldolgozása közben: {exc}")

def main():
    parser = argparse.ArgumentParser(
        description="JSON és audio fájlok feldolgozása szegmensekre bontáshoz és audio chunk-ok kivágásához.",
        epilog="Példa használat:\n  python splitter_v4_json.py --input_dir ./audio/ --output_dir ./angol_darabok/",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        required=True,
        help='A bemeneti könyvtár útvonala, ahol a JSON és audio fájlok találhatók.'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        required=True,
        help='A kimeneti könyvtár útvonala, ahol a feldolgozott audio és szövegfájlok mentésre kerülnek.'
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()


import os
import argparse
import json
import re
import math
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import timedelta

def sanitize_filename(filename):
    """
    Csere a tiltott karaktereket a fájlnévben.
    Itt a kettőspontot kötőjelre, a vesszőt pontokra cseréljük.
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

    # Feldolgozzuk az egyes JSON szegmenseket
    for segment in segments:
        seg_start = segment.get("start")
        seg_end = segment.get("end")
        text = segment.get("text", "").strip()
        speaker = segment.get("speaker", "UNKNOWN")

        if seg_start is None or seg_end is None:
            errors.append(f"Hiányzó 'start' vagy 'end' érték a '{base_name}' fájl egy szegmensénél.")
            continue

        if not text:
            continue  # Üres szöveg esetén kihagyjuk a szegmenst

        if seg_start >= seg_end:
            errors.append(f"Nem érvényes vágási pontok a '{base_name}' fájl egy szegmensénél.")
            continue

        total_duration = seg_end - seg_start  # A teljes szakasz hossza (másodpercben)
        # A kimeneti almappát a JSON fájl relatív útvonala alapján hozzuk létre
        output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
        os.makedirs(output_subdir, exist_ok=True)

        # Mondatokra bontás: próbáljuk meg a teljes szöveget mondatokra bontani.
        # A regex a mondatvégi írásjelek (., !, ?) utáni szóközök mentén választja el a mondatokat.
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            # Ha nem sikerült bontani (például rossz formázású szöveg esetén), az egész szöveget egy darabnak vesszük
            sentences = [text]
        
        # A mondatokhoz rendelt idő meghatározásához számoljuk az összes karakter hosszát
        total_chars = sum(len(s) for s in sentences)
        current_time = seg_start

        for sentence in sentences:
            sentence_length = len(sentence)
            # Arányos időkeret, feltételezve, hogy a beszéd sebessége állandó:
            sentence_duration = total_duration * (sentence_length / total_chars) if total_chars > 0 else total_duration

            # Ha a mondat hossza legfeljebb 13 másodperc, akkor egyetlen darabként dolgozzuk fel
            if sentence_duration <= 13:
                chunk_start = current_time
                chunk_end = current_time + sentence_duration
                chunk_start_ms = int(chunk_start * 1000)
                chunk_end_ms = int(chunk_end * 1000)
                try:
                    audio_chunk = audio[chunk_start_ms:chunk_end_ms]
                except Exception as e:
                    errors.append(f"Hiba történt az audio szakasz kivágása közben '{base_name}' szegmensnél: {e}")
                    current_time += sentence_duration
                    continue

                start_timestamp = format_timedelta_seconds(chunk_start)
                end_timestamp = format_timedelta_seconds(chunk_end)
                filename_base = f"{sanitize_filename(start_timestamp)}-{sanitize_filename(end_timestamp)}_{sanitize_filename(speaker)}"
                output_audio_path = os.path.join(output_subdir, f"{filename_base}.{export_extension}")
                try:
                    audio_chunk.export(output_audio_path, format=export_format)
                except Exception as e:
                    errors.append(f"Hiba történt az audio szakasz exportálása közben '{output_audio_path}': {e}")
                    current_time += sentence_duration
                    continue

                output_text_path = os.path.join(output_subdir, f"{filename_base}.txt")
                try:
                    with open(output_text_path, 'w', encoding='utf-8') as txt_file:
                        txt_file.write(sentence)
                except Exception as e:
                    errors.append(f"Hiba történt a szövegfájl írása közben '{output_text_path}': {e}")
                    current_time += sentence_duration
                    continue

                sentence_counter += 1
                current_time += sentence_duration
            else:
                # Ha a mondat hossza meghaladja a 13 másodpercet, további kisebb darabokra bontjuk.
                num_chunks = math.ceil(sentence_duration / 13)
                # Szavakra bontjuk, hogy ne szakítsuk el a szavakat véletlenül.
                words = sentence.split()
                if len(words) == 0:
                    words = [sentence]
                chunk_word_count = math.ceil(len(words) / num_chunks)
                for i in range(num_chunks):
                    chunk_text = " ".join(words[i*chunk_word_count : (i+1)*chunk_word_count])
                    # A darabhoz tartozó időkeret:
                    chunk_start = current_time + i * (sentence_duration / num_chunks)
                    chunk_end = current_time + (i+1) * (sentence_duration / num_chunks)
                    chunk_start_ms = int(chunk_start * 1000)
                    chunk_end_ms = int(chunk_end * 1000)
                    try:
                        audio_chunk = audio[chunk_start_ms:chunk_end_ms]
                    except Exception as e:
                        errors.append(f"Hiba történt az audio szakasz kivágása közben '{base_name}' szegmensnél: {e}")
                        continue

                    start_timestamp = format_timedelta_seconds(chunk_start)
                    end_timestamp = format_timedelta_seconds(chunk_end)
                    filename_base = f"{sanitize_filename(start_timestamp)}-{sanitize_filename(end_timestamp)}_{sanitize_filename(speaker)}"
                    output_audio_path = os.path.join(output_subdir, f"{filename_base}.{export_extension}")
                    try:
                        audio_chunk.export(output_audio_path, format=export_format)
                    except Exception as e:
                        errors.append(f"Hiba történt az audio szakasz exportálása közben '{output_audio_path}': {e}")
                        continue

                    output_text_path = os.path.join(output_subdir, f"{filename_base}.txt")
                    try:
                        with open(output_text_path, 'w', encoding='utf-8') as txt_file:
                            txt_file.write(chunk_text)
                    except Exception as e:
                        errors.append(f"Hiba történt a szövegfájl írása közben '{output_text_path}': {e}")
                        continue

                    sentence_counter += 1
                current_time += sentence_duration

    stats = f"Fájl: '{base_name}'\nFeldolgozott szegmensek: {sentence_counter}"
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
        description="JSON és audio fájlok feldolgozása szegmensekre bontáshoz, egész mondatokra bontva, "
                    "és audio chunk-ok kivágásához (max. 13 másodperc per darab).",
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

import os
import argparse
import deepl
import json
import re

def extract_timestamp(filename):
    """
    Kinyeri az első timestampet a fájlnévből.
    Példa: '00-52-39.253-00-52-39.641_SPEAKER_24.txt' -> '00-52-39.253'
    """
    match = re.match(r'(\d{2}-\d{2}-\d{2}\.\d{3})-', filename)
    if match:
        return match.group(1)
    else:
        return ''

def main(input_dir, output_dir, input_lang, output_lang, auth_key):
    # DeepL fordító inicializálása
    translator = deepl.Translator(auth_key)

    # Ellenőrizzük, hogy a kimeneti könyvtár létezik-e, ha nem, létrehozzuk
    os.makedirs(output_dir, exist_ok=True)

    # Az input_dir összes txt fájljának listázása
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    # Fájlok rendezése az első timestamp alapján
    sorted_files = sorted(txt_files, key=lambda x: extract_timestamp(x))

    # Összes szöveg sorainak összegyűjtése
    lines = []
    for filename in sorted_files:
        input_path = os.path.join(input_dir, filename)
        with open(input_path, 'r', encoding='utf-8') as file:
            # Assuming each file contains logically distinct text segments,
            # read the whole file content as one item.
            # Use strip() to remove leading/trailing whitespace but keep internal newlines.
            file_content = file.read().strip()
            # Store filename and content; use a placeholder if content is empty
            lines.append((filename, file_content if file_content else " "))


    # Combine all text content into a single string, separated by newlines.
    # Keep track of the original filenames associated with each segment.
    original_filenames = [item[0] for item in lines]
    text_to_translate = '\n'.join([item[1] for item in lines])

    print(f"Összes fájl (szegmens) száma: {len(original_filenames)}")
    print("Teljes szöveg fordítása...")

    # Translate the entire text block
    try:
        translation_result = translator.translate_text(
            text_to_translate,
            source_lang=input_lang,
            target_lang=output_lang
        )
        # Split the translated text back into segments based on newlines
        translated_lines = translation_result.text.split('\n')
    except deepl.DeepLException as e:
        print(f"Hiba történt a DeepL API hívásakor: {e}")
        # Handle potential API errors, e.g., text too large if limit is exceeded
        if "request entity too large" in str(e).lower():
             print("A fordítandó szöveg mérete meghaladja a DeepL API limitjét.")
             print("Fontolja meg a szöveg kisebb részekre bontását vagy a batch-feldolgozás használatát.")
        return # Exit if translation failed

    # Check if the number of translated segments matches the number of original files
    if len(translated_lines) != len(original_filenames):
        print("Figyelmeztetés: A fordított szegmensek száma nem egyezik az eredeti fájlok számával.")
        print(f"Eredeti fájlok: {len(original_filenames)}, Fordított szegmensek: {len(translated_lines)}")
        # Decide how to handle mismatch: stop, pad, or truncate?
        # For now, we'll proceed but only write out matching segments.

    # Save translated segments to corresponding output files
    for idx, filename in enumerate(original_filenames):
        output_path = os.path.join(output_dir, filename)
        # Ensure there is a corresponding translated segment
        if idx < len(translated_lines):
            translated_segment = translated_lines[idx].strip()
        else:
            # Handle missing translation for a file (e.g., write empty string or log error)
            translated_segment = ""
            print(f"Figyelmeztetés: Nincs fordított szegmens a(z) {filename} fájlhoz.")
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(translated_segment)
        # Optional: print confirmation for each file saved
        # print(f"Fordítás elmentve: {output_path}")

    print(f"\nFordítás befejezve. A fájlok a(z) '{output_dir}' könyvtárba mentve.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fordítás DeepL segítségével - A teljes szöveget egyben fordítja le.')
    parser.add_argument('-input_dir', required=True, help='A könyvtár, amely tartalmazza a fordítandó .txt fájlokat (fájlonként egy szegmens).')
    parser.add_argument('-output_dir', required=True, help='A könyvtár, ahová a lefordított fájlok kerülnek (megtartva az eredeti fájlneveket).')
    parser.add_argument('-input_language', required=True, help='A bemeneti nyelv kódja (pl. EN, HU)')
    parser.add_argument('-output_language', required=True, help='A kimeneti nyelv kódja (pl. EN, HU)')
    parser.add_argument('-auth_key', required=True, help='A DeepL API hitelesítési kulcs')

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.input_language, args.output_language, args.auth_key)

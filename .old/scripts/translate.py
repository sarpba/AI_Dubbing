import os
import argparse
import deepl
import json
import re

MAX_REQUEST_SIZE = 128 * 1024  # 128 KiB

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

def split_into_batches(lines, max_size):
    """
    Felosztja a sorokat olyan batch-ekre, hogy minden batch összmérete ne lépje túl a max_size-t.
    """
    batches = []
    current_batch = []
    current_size = 0

    for line in lines:
        # +1 a newline karakter miatt
        line_size = len(line.encode('utf-8')) + 1
        if current_size + line_size > max_size:
            if current_batch:
                batches.append(current_batch)
            current_batch = [line]
            current_size = line_size
        else:
            current_batch.append(line)
            current_size += line_size

    if current_batch:
        batches.append(current_batch)

    return batches

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
            file_lines = file.read().strip().split('\n')
            lines.extend(file_lines)

    # Batch-ek létrehozása a maximális kérésméret alapján
    batches = split_into_batches(lines, MAX_REQUEST_SIZE)

    translated_lines = []
    print(f"Összes sor száma: {len(lines)}")
    print(f"Batch-ek száma: {len(batches)}")

    # Minden batch fordítása
    for i, batch in enumerate(batches, 1):
        batch_text = '\n'.join(batch)
        print(f"Batch {i}/{len(batches)} fordítása...")
        translated_batch = translator.translate_text(
            batch_text,
            source_lang=input_lang,
            target_lang=output_lang
        )
        translated_batch_lines = translated_batch.text.split('\n')
        translated_lines.extend(translated_batch_lines)

    # Ellenőrizzük, hogy a fordított sorok száma megegyezik-e az eredeti sorok számával
    if len(translated_lines) != len(lines):
        print("Figyelmeztetés: A fordított sorok száma nem egyezik az eredeti sorok számával.")
        print(f"Eredeti sorok: {len(lines)}, Fordított sorok: {len(translated_lines)}")

    # Fordított sorok mentése az egyes kimeneti fájlokba
    for idx, filename in enumerate(sorted_files):
        output_path = os.path.join(output_dir, filename)
        # Biztosítjuk, hogy van megfelelő fordított sor
        if idx < len(translated_lines):
            translated_line = translated_lines[idx].strip()
        else:
            translated_line = ""
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(translated_line)
        print(f"Fordítás elmentve: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fordítás DeepL segítségével - Összesítés és szétbontás (128 KiB limit)')
    parser.add_argument('-input_dir', required=True, help='A könyvtár, amely tartalmazza a fordítandó .txt fájlokat')
    parser.add_argument('-output_dir', required=True, help='A könyvtár, ahová a lefordított fájlok kerülnek')
    parser.add_argument('-input_language', required=True, help='A bemeneti nyelv kódja (pl. EN, HU)')
    parser.add_argument('-output_language', required=True, help='A kimeneti nyelv kódja (pl. EN, HU)')
    parser.add_argument('-auth_key', required=True, help='A DeepL API hitelesítési kulcs')

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.input_language, args.output_language, args.auth_key)

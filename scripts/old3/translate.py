import os
import argparse
import deepl
import json

def find_json_file(directory):
    """
    Megkeresi az egyetlen .json fájlt a megadott könyvtárban.
    Visszaadja a fájl nevét, vagy None-t, ha hiba történik.
    """
    try:
        json_files = [f for f in os.listdir(directory) if f.lower().endswith('.json')]
    except FileNotFoundError:
        print(f"Hiba: A bemeneti könyvtár nem található: {directory}")
        return None, "directory_not_found"

    if len(json_files) == 0:
        print(f"Hiba: Nem található JSON fájl a(z) '{directory}' könyvtárban.")
        return None, "no_json_found"
    
    if len(json_files) > 1:
        print(f"Hiba: Több JSON fájl található a(z) '{directory}' könyvtárban. Csak egyet tud feldolgozni.")
        return None, "multiple_jsons_found"
    
    return json_files[0], None

def main(input_dir, output_dir, input_lang, output_lang, auth_key):
    # DeepL fordító inicializálása
    translator = deepl.Translator(auth_key)

    # A bemeneti JSON fájl megkeresése
    input_filename, error = find_json_file(input_dir)
    if error:
        return # Kilépés, ha hiba történt
    
    input_filepath = os.path.join(input_dir, input_filename)
    print(f"Bemeneti fájl feldolgozása: {input_filepath}")

    # A kimeneti könyvtár létrehozása, ha nem létezik
    os.makedirs(output_dir, exist_ok=True)

    # A kimeneti fájl elérési útja (ugyanazzal a névvel, mint a bemeneti)
    output_filepath = os.path.join(output_dir, input_filename)

    # Bemeneti JSON fájl betöltése
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Hiba: A bemeneti fájl nem érvényes JSON formátumú: {input_filepath}")
        return

    # Ellenőrizzük, hogy a JSON tartalmazza-e a 'segments' kulcsot
    if 'segments' not in data or not isinstance(data['segments'], list):
        print("Hiba: A JSON fájlnak tartalmaznia kell egy 'segments' kulcsot, amelynek értéke egy lista.")
        return

    original_segments = data['segments']
    
    # Szövegek kigyűjtése, amelyek már tartalmaznak fordítást, hogy ne fordítsuk le őket újra
    texts_to_translate = []
    indices_to_translate = []
    for i, segment in enumerate(original_segments):
        # Csak akkor fordítjuk, ha van 'text' és nincs még 'translated_text' kulcs, vagy ha a meglévő üres
        if segment.get('text', '').strip() and not segment.get('translated_text', '').strip():
            texts_to_translate.append(segment.get('text').strip() or " ")
            indices_to_translate.append(i)

    # Ha nincs mit fordítani, akkor is elmentjük a fájlt a kimeneti helyre,
    # hogy a pipeline további lépései megkapják a fájlt.
    if not texts_to_translate:
        print("Nincs új, fordítandó szöveg a fájlban.")
        with open(output_filepath, 'w', encoding='utf-8') as f:
            # Hozzáadjuk az üres 'translated_text' mezőt azokhoz, ahol hiányzik
            for segment in original_segments:
                segment.setdefault('translated_text', '')
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"A fájl (módosítás nélkül) átmásolva ide: {output_filepath}")
        return

    # A szegmensek összefűzése egyetlen szöveggé
    text_block = '\n'.join(texts_to_translate)

    print(f"Újonnan fordítandó szegmensek száma: {len(texts_to_translate)}")
    print("A teljes szövegblokk fordítása...")

    # A teljes szövegblokk lefordítása
    try:
        result = translator.translate_text(
            text_block,
            source_lang=input_lang,
            target_lang=output_lang
        )
        translated_lines = result.text.split('\n')
    except deepl.DeepLException as e:
        print(f"Hiba történt a DeepL API hívásakor: {e}")
        return

    # Ellenőrizzük a szegmensek számát
    if len(translated_lines) != len(texts_to_translate):
        print("Figyelmeztetés: A fordított szegmensek száma nem egyezik a fordításra küldöttek számával.")
        print("A fordítás megszakítva a hibás kimenet elkerülése érdekében.")
        return

    # A fordítások beillesztése a 'translated_text' kulcs alá a megfelelő helyre
    for i, translated_line in enumerate(translated_lines):
        original_index = indices_to_translate[i]
        data['segments'][original_index]['translated_text'] = translated_line.strip()
    
    # Hozzáadjuk az üres 'translated_text' mezőt azokhoz a szegmensekhez, ahol nem volt fordítás
    for segment in data['segments']:
        segment.setdefault('translated_text', '')

    # A kiegészített adatok kiírása a kimeneti fájlba
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nFordítás befejezve. A kiegészített fájl a(z) '{output_filepath}' helyre mentve.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Szövegszegmensek fordítása JSON fájlból DeepL segítségével és az eredmény hozzáadása a fájlhoz.')
    parser.add_argument('-input_dir', required=True, help='A könyvtár, amely a fordítandó JSON fájlt tartalmazza.')
    parser.add_argument('-output_dir', required=True, help='A könyvtár, ahová a kiegészített fájl kerül.')
    parser.add_argument('-input_language', required=True, help='A bemeneti nyelv kódja (pl. EN, HU)')
    parser.add_argument('-output_language', required=True, help='A kimeneti nyelv kódja (pl. EN, HU)')
    parser.add_argument('-auth_key', required=True, help='A DeepL API hitelesítési kulcs')

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.input_language, args.output_language, args.auth_key)
import os
import argparse
import deepl
import json  # Új import a JSON kezeléséhez

def translate_file(input_path, output_path, input_lang, output_lang, translator):
    with open(input_path, 'r', encoding='utf-8') as file:
        text = file.read()
        
    translated_text = translator.translate_text(text, source_lang=input_lang, target_lang=output_lang)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(translated_text.text)

def main(input_dir, output_dir, input_lang, output_lang, auth_key):
    # DeepL fordító inicializálása
    translator = deepl.Translator(auth_key)

    # Ellenőrizzük, hogy a kimeneti könyvtár létezik-e, ha nem, létrehozzuk
    os.makedirs(output_dir, exist_ok=True)

    # Az input_dir összes txt fájlán végighaladunk
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Útvonal a megfelelő JSON fájlhoz
            json_filename = os.path.splitext(filename)[0] + '.json'
            json_path = os.path.join(input_dir, 'transcripts_split', json_filename)
            
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                
                json_language = data.get('language', '').lower()
                input_lang_lower = input_lang.lower()
                
                if json_language == input_lang_lower:
                    # Ha a nyelv megegyezik, a .txt fájlból fordít
                    print(f"Fordítás: {filename} -> {output_path}")
                    translate_file(input_path, output_path, input_lang, output_lang, translator)
                else:
                    # Ha más nyelv, a JSON-ból beolvasott szöveget fordít
                    segments = data.get('segments', [])
                    text_to_translate = ' '.join(segment.get('text', '') for segment in segments)
                    
                    if text_to_translate.strip():  # Ellenőrizzük, hogy nem üres-e a szöveg
                        translated_text = translator.translate_text(
                            text_to_translate, 
                            source_lang=input_lang, 
                            target_lang=output_lang
                        )
                        
                        with open(output_path, 'w', encoding='utf-8') as file:
                            file.write(translated_text.text)
                        
                        print(f"Fordítás a JSONból: {filename} -> {output_path}")
                    else:
                        print(f"Üres szöveg a JSON fájlban: {json_path}. Fordítás kihagyva.")
            else:
                # Ha nincs JSON fájl, a .txt fájlból fordít
                print(f"Fordítás: {filename} -> {output_path}")
                translate_file(input_path, output_path, input_lang, output_lang, translator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fordítás DeepL segítségével')
    parser.add_argument('-input_dir', required=True, help='A könyvtár, amely tartalmazza a fordítandó .txt fájlokat')
    parser.add_argument('-output_dir', required=True, help='A könyvtár, ahová a lefordított fájlok kerülnek')
    parser.add_argument('-input_language', required=True, help='A bemeneti nyelv kódja (pl. EN, HU)')
    parser.add_argument('-output_language', required=True, help='A kimeneti nyelv kódja (pl. EN, HU)')
    parser.add_argument('-auth_key', required=True, help='A DeepL API hitelesítési kulcs')

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.input_language, args.output_language, args.auth_key)

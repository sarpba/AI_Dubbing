import os
import argparse
from openai import OpenAI, OpenAIError
import json
import re
import time

# ... (a get_lang_name, find_json_file, create_smart_chunks függvények változatlanok) ...
def get_lang_name(lang_code):
    lang_map = {'EN': 'English', 'HU': 'Hungarian', 'DE': 'German', 'FR': 'French', 'ES': 'Spanish'}
    return lang_map.get(lang_code.upper(), lang_code)

def find_json_file(directory):
    try:
        json_files = [f for f in os.listdir(directory) if f.lower().endswith('.json')]
    except FileNotFoundError:
        print(f"Hiba: A bemeneti könyvtár nem található: {directory}"); return None, "directory_not_found"
    if not json_files:
        print(f"Hiba: Nem található JSON fájl a(z) '{directory}' könyvtárban."); return None, "no_json_found"
    if len(json_files) > 1:
        print(f"Hiba: Több JSON fájl található a(z) '{directory}' könyvtárban."); return None, "multiple_jsons_found"
    return json_files[0], None

def create_smart_chunks(segments, min_size=50, max_size=100, gap_threshold=5.0):
    chunks, current_pos = [], 0
    while current_pos < len(segments):
        chunk_start = current_pos
        if (len(segments) - chunk_start) <= min_size:
            best_split_point = len(segments)
        else:
            best_split_point, search_end = -1, min(chunk_start + max_size, len(segments) - 1)
            for i in range(chunk_start + min_size - 1, search_end):
                try:
                    if segments[i + 1].get('start', 0) - segments[i].get('end', 0) >= gap_threshold:
                        best_split_point = i + 1; break
                except (TypeError, KeyError): continue
            if best_split_point == -1:
                max_gap, best_split_point = -1.0, min(chunk_start + max_size, len(segments))
                for i in range(chunk_start + min_size - 1, search_end):
                    try:
                        gap = segments[i + 1].get('start', 0) - segments[i].get('end', 0)
                        if gap > max_gap: max_gap, best_split_point = gap, i + 1
                    except (TypeError, KeyError): continue
        chunks.append(segments[chunk_start:best_split_point]); current_pos = best_split_point
    return chunks

# *** ÚJ FÜGGVÉNY: A rekurzív "Oszd meg és uralkodj" fordító ***
def translate_or_subdivide_batch(client, batch_segments, lang_from, lang_to, context, model, stream, allow_sensitive, batch_id_str):
    """
    Megpróbál lefordítani egy csoportot. Ha hibázik, kettéosztja és rekurzívan újrapróbálja.
    """
    # Bázis eset: Ha a csoport már csak 1 elemből áll, de mégis hibázik, akkor nagy a baj.
    if not batch_segments:
        return []
    if len(batch_segments) == 1:
        print(f"  [{batch_id_str}] Csoport 1 elemre redukálva, fordítás...")

    numbered_texts = [f"{i+1}. {seg['text']}" for i, seg in enumerate(batch_segments)]
    text_block = '\n'.join(numbered_texts)

    input_lang_name, output_lang_name = get_lang_name(lang_from), get_lang_name(lang_to)
    
    if allow_sensitive:
        system_prompt = (f"You are a professional translator for tasks like film subtitling. Translate the numbered list from {input_lang_name} to {output_lang_name}. "
                         "The source text may contain profanity or sensitive topics. Your primary directive is **faithful and accurate translation**. "
                         "**Do not censor or refuse to translate.** Your role is a neutral translator, not a content filter. "
                         "Your response MUST be a numbered list with the exact same number of items. Format: `1. [translation]`.")
    else:
        system_prompt = (f"You are an expert translator. Translate the numbered list from {input_lang_name} to {output_lang_name}. "
                         "Your response MUST be a numbered list with the exact same number of items. Format: `1. [translation]`.")

    if context: system_prompt += f"\n\nContext: '{context}'"

    translated_lines_raw = []
    try:
        response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text_block}], temperature=0.1)
        translated_lines_raw = [line.strip() for line in response.choices[0].message.content.strip().split('\n') if line.strip()]
    except OpenAIError as e:
        print(f"  HIBA: API hiba történt a(z) [{batch_id_str}] csoportnál: {e}"); return None

    if len(translated_lines_raw) == len(batch_segments):
        # SIKERES ág
        final_translated_lines = [re.sub(r'^\d+\.\s*', '', line) for line in translated_lines_raw]
        if stream: # Vizuális visszajelzés stream módban
            for line in final_translated_lines: print(f"    + {line}")
        return final_translated_lines
    else:
        # SIKERTELEN ág
        print(f"  HIBA: A(z) [{batch_id_str}] csoportnál ({len(batch_segments)} elem) a sorok száma nem egyezik! Várt: {len(batch_segments)}, Kapott: {len(translated_lines_raw)}.")
        
        # *** MÓDOSÍTÁS: Részletes hibadiagnosztika ***
        print("\n" + "="*20 + f" DEBUG START: BATCH [{batch_id_str}] " + "="*20)
        print("--- BEMENET (amit a modell kapott): ---")
        for line in numbered_texts: print(line)
        print("\n--- KIMENET (amit a modell adott): ---")
        for i, line in enumerate(translated_lines_raw): print(f"{i+1}. {line}")
        print("="*20 + f" DEBUG END: BATCH [{batch_id_str}] " + "="*20 + "\n")
        
        if len(batch_segments) <= 1:
            print("  A csoport már nem osztható tovább, a fordítás ennél a pontnál végleg sikertelen.")
            return None # Végleges hiba

        print(f"  A [{batch_id_str}] csoport felosztása és újrapróbálása...")
        mid_point = len(batch_segments) // 2
        first_half = batch_segments[:mid_point]
        second_half = batch_segments[mid_point:]

        # Rekurzív hívás az első felére
        first_half_results = translate_or_subdivide_batch(client, first_half, lang_from, lang_to, context, model, stream, allow_sensitive, f"{batch_id_str}-A")
        if first_half_results is None:
            return None # Ha az al-csoport is hibázik, adjuk tovább a hibát

        # Rekurzív hívás a második felére
        second_half_results = translate_or_subdivide_batch(client, second_half, lang_from, lang_to, context, model, stream, allow_sensitive, f"{batch_id_str}-B")
        if second_half_results is None:
            return None

        return first_half_results + second_half_results

def main(input_dir, output_dir, input_lang, output_lang, auth_key, context, model, stream, allow_sensitive_content):
    client = OpenAI(api_key=auth_key)
    input_filename, error = find_json_file(input_dir)
    if error: return
    
    input_filepath = os.path.join(input_dir, input_filename); print(f"Bemeneti fájl feldolgozása: {input_filepath}")
    os.makedirs(output_dir, exist_ok=True); output_filepath = os.path.join(output_dir, input_filename)

    progress_filename = f"{os.path.splitext(input_filename)[0]}.progress.json"
    progress_filepath = os.path.join(output_dir, progress_filename)
    progress = {}
    if os.path.exists(progress_filepath):
        try:
            with open(progress_filepath, 'r', encoding='utf-8') as f: progress = json.load(f)
            print(f"\nHaladási fájl betöltve: {progress_filepath} ({len(progress)} szegmens már lefordítva).")
        except (json.JSONDecodeError, IOError): progress = {}

    with open(input_filepath, 'r', encoding='utf-8') as f: data = json.load(f)
    if 'segments' not in data or not isinstance(data['segments'], list): print("Hiba: JSON 'segments' kulcs hiányzik."); return

    untranslated_segments = []
    for i, segment in enumerate(data['segments']):
        if segment.get('text', '').strip() and str(i) not in progress:
            segment['original_index'] = i; untranslated_segments.append(segment)

    if not untranslated_segments: print("Nincs új, fordítandó szegmens.")
    else:
        print(f"\n{len(untranslated_segments)} új, fordítandó szegmens található.")
        batches = create_smart_chunks(untranslated_segments)
        total_batches = len(batches)
        print(f"Intelligens csoportképzés befejezve. {total_batches} új csoportot kell feldolgozni.")

        for i, batch in enumerate(batches):
            print(f"\n[{i+1}/{total_batches}] fő csoport feldolgozása...")
            translated_batch_lines = translate_or_subdivide_batch(client, batch, input_lang, output_lang, context, model, stream, allow_sensitive_content, f"{i+1}")
            
            if translated_batch_lines is None:
                print(f"\nA fordítási folyamat megszakadt a(z) [{i+1}] csoportnál egy nem helyreállítható hiba miatt."); return

            for segment_obj, translated_line in zip(batch, translated_batch_lines):
                progress[str(segment_obj['original_index'])] = translated_line
            
            with open(progress_filepath, 'w', encoding='utf-8') as f: json.dump(progress, f, ensure_ascii=False, indent=2)
            print(f"  Haladás elmentve a(z) {progress_filepath} fájlba.")
            time.sleep(1)

    print("\nAz összes szükséges fordítás elkészült.")

    for index_str, line in progress.items(): data['segments'][int(index_str)]['translated_text'] = line
    for segment in data['segments']: segment.setdefault('translated_text', '')

    with open(output_filepath, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nFordítás befejezve. A kiegészített fájl a(z) '{output_filepath}' helyre mentve.")
    
    if os.path.exists(progress_filepath): os.remove(progress_filepath); print(f"Az ideiglenes haladási fájl ({progress_filepath}) törölve.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Szövegszegmensek intelligens, folytatható, rekurzív fordítása.')
    # ... (argumentumok definíciója változatlan) ...
    parser.add_argument('-input_dir', required=True, help='A könyvtár, amely a fordítandó JSON fájlt tartalmazza.')
    parser.add_argument('-output_dir', required=True, help='A könyvtár, ahová a kiegészített fájl kerül.')
    parser.add_argument('-input_language', required=True, help='A bemeneti nyelv kódja (pl. EN, HU)')
    parser.add_argument('-output_language', required=True, help='A kimeneti nyelv kódja (pl. EN, HU)')
    parser.add_argument('-auth_key', required=True, help='Az OpenAI API hitelesítési kulcs.')
    parser.add_argument('-context', required=False, help='Rövid kontextus a fordításhoz.')
    parser.add_argument('-model', default='gpt-4o', required=False, help='A használni kívánt OpenAI modell neve. Alapértelmezett: gpt-4o.')
    parser.add_argument('-stream', action='store_true', help='Bekapcsolja a fordítási folyamat valós idejű, soronkénti megjelenítését a konzolon.')
    parser.add_argument('-allow_sensitive_content', action='store_true', help='Speciális promptot használ, ami a kényes tartalmak (pl. káromkodás) fordítását is megkísérli cenzúrázás nélkül.')

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.input_language, args.output_language, args.auth_key, args.context, args.model, args.stream, args.allow_sensitive_content)
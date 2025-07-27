import os
import argparse
from openai import OpenAI, OpenAIError
import json
import re
import time
import srt # Szükséges: pip install srt
from datetime import timedelta

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

def load_srt_file(filepath):
    if not filepath or not os.path.exists(filepath):
        if filepath: print(f"Figyelmeztetés: Az SRT kontextus fájl nem található: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            subtitles = list(srt.parse(f.read()))
        print(f"SRT fájl sikeresen betöltve és feldolgozva: {filepath} ({len(subtitles)} felirat).")
        return subtitles
    except Exception as e:
        print(f"Figyelmeztetés: Az SRT fájl olvasása sikertelen ({e}). Folytatás SRT kontextus nélkül."); return None

# *** ÚJ FÜGGVÉNY: SRT tartalom tisztítása ***
def clean_srt_content(text):
    """Eltávolítja a formázó tageket (pl. {\an8}, <i>) az SRT szövegből."""
    # {\an8}, {\i1}, stb. tagek eltávolítása
    text = re.sub(r'\{[^\}]*\}', '', text)
    # <i>, <b>, <font>, stb. tagek eltávolítása
    text = re.sub(r'<[^>]*>', '', text)
    return text.strip()

def get_dynamic_srt_context(batch_segments, srt_data, leeway_seconds=15.0, max_chars=2000):
    if not srt_data or not batch_segments: return ""
    
    try:
        batch_start_time = batch_segments[0]['start']
        batch_end_time = batch_segments[-1]['end']
    except (KeyError, IndexError): return ""

    context_start_time = max(0, batch_start_time - leeway_seconds)
    context_end_time = batch_end_time + leeway_seconds

    relevant_subs = []
    for sub in srt_data:
        sub_start_sec = sub.start.total_seconds()
        sub_end_sec = sub.end.total_seconds()
        if sub_end_sec >= context_start_time and sub_start_sec <= context_end_time:
            # *** MÓDOSÍTÁS: A tisztító függvény alkalmazása ***
            cleaned_content = clean_srt_content(sub.content)
            if cleaned_content:
                relevant_subs.append(cleaned_content.replace('\n', ' ').strip())
            
    if relevant_subs:
        full_context = " ".join(relevant_subs)
        sample = full_context[:max_chars].strip()
        print(f"  -> Dinamikus SRT kontextus kikeresve a(z) {time.strftime('%H:%M:%S', time.gmtime(context_start_time))} - "
              f"{time.strftime('%H:%M:%S', time.gmtime(context_end_time))} időablakhoz ({len(relevant_subs)} sor).")
        return sample
    return ""

def translate_or_subdivide_batch(client, batch_segments, lang_from, lang_to, context, srt_data, model, stream, allow_sensitive, batch_id_str):
    if not batch_segments: return []
    if len(batch_segments) == 1: print(f"  [{batch_id_str}] Csoport 1 elemre redukálva, fordítás...")

    numbered_texts = [f"{i+1}. {seg['text']}" for i, seg in enumerate(batch_segments)]
    text_block = '\n'.join(numbered_texts)
    input_lang_name, output_lang_name = get_lang_name(lang_from), get_lang_name(lang_to)
    
    prompt_core = (f"You are a professional translator for film subtitling. Translate the numbered list from {input_lang_name} to {output_lang_name}. "
                   "Your response MUST be a numbered list with the exact same number of items. Format: `1. [translation]`.")
    if allow_sensitive:
        prompt_core += " The source may contain profanity or sensitive topics. Your directive is **faithful translation**, do not censor."
    
    system_prompt = prompt_core
    
    dynamic_srt_context = get_dynamic_srt_context(batch_segments, srt_data)
    if dynamic_srt_context:
        system_prompt += ("\n\nTo ensure consistency, adopt a style and terminology similar to this sample from the same scene:\n"
                          "--- STYLE SAMPLE (from SRT) ---\n"
                          f"{dynamic_srt_context}\n"
                          "--- END STYLE SAMPLE ---")
    
    if context: system_prompt += f"\n\nAdditional context: '{context}'"

    translated_lines_raw = []
    try:
        response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text_block}], temperature=0.1)
        translated_lines_raw = [line.strip() for line in response.choices[0].message.content.strip().split('\n') if line.strip()]
    except OpenAIError as e: print(f"  HIBA: API hiba a(z) [{batch_id_str}] csoportnál: {e}"); return None

    if len(translated_lines_raw) == len(batch_segments):
        final_translated_lines = [re.sub(r'^\d+\.\s*', '', line) for line in translated_lines_raw]
        if stream:
            for line in final_translated_lines: print(f"    + {line}")
        return final_translated_lines
    else:
        print(f"  HIBA: A(z) [{batch_id_str}] csoportnál ({len(batch_segments)} elem) sorszámeltérés! Várt: {len(batch_segments)}, Kapott: {len(translated_lines_raw)}.")
        if len(batch_segments) <= 1: return None
        print(f"  A [{batch_id_str}] csoport felosztása...")
        mid_point = len(batch_segments) // 2
        first_half, second_half = batch_segments[:mid_point], batch_segments[mid_point:]
        first_half_results = translate_or_subdivide_batch(client, first_half, lang_from, lang_to, context, srt_data, model, stream, allow_sensitive, f"{batch_id_str}-A")
        if first_half_results is None: return None
        second_half_results = translate_or_subdivide_batch(client, second_half, lang_from, lang_to, context, srt_data, model, stream, allow_sensitive, f"{batch_id_str}-B")
        if second_half_results is None: return None
        return first_half_results + second_half_results

def main(input_dir, output_dir, input_lang, output_lang, auth_key, context, srt_context_file, model, stream, allow_sensitive_content):
    client = OpenAI(api_key=auth_key)
    input_filename, error = find_json_file(input_dir)
    if error: return
    
    input_filepath = os.path.join(input_dir, input_filename); print(f"Bemeneti fájl feldolgozása: {input_filepath}")
    os.makedirs(output_dir, exist_ok=True); output_filepath = os.path.join(output_dir, input_filename)

    srt_data = load_srt_file(srt_context_file)

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
            translated_batch_lines = translate_or_subdivide_batch(client, batch, input_lang, output_lang, context, srt_data, model, stream, allow_sensitive_content, f"{i+1}")
            
            if translated_batch_lines is None:
                print(f"\nA fordítási folyamat megszakadt."); return

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
    parser = argparse.ArgumentParser(description='Szövegszegmensek intelligens, folytatható, rekurzív fordítása dinamikus SRT stíluskontextussal.')
    parser.add_argument('-input_dir', required=True, help='A könyvtár, amely a fordítandó JSON fájlt tartalmazza.')
    parser.add_argument('-output_dir', required=True, help='A könyvtár, ahová a kiegészített fájl kerül.')
    parser.add_argument('-input_language', required=True, help='A bemeneti nyelv kódja (pl. EN, HU)')
    parser.add_argument('-output_language', required=True, help='A kimeneti nyelv kódja (pl. EN, HU)')
    parser.add_argument('-auth_key', required=True, help='Az OpenAI API hitelesítési kulcs.')
    parser.add_argument('-context', required=False, help='Rövid, általános kontextus a fordításhoz (pl. "sci-fi sorozat").')
    parser.add_argument('-srt_context_file', required=False, help='Opcionális SRT fájl elérési útja a dinamikus, időalapú stíluskontextushoz.')
    parser.add_argument('-model', default='gpt-4o', required=False, help='A használni kívánt OpenAI modell neve.')
    parser.add_argument('-stream', action='store_true', help='Bekapcsolja a valós idejű kimenetet.')
    parser.add_argument('-allow_sensitive_content', action='store_true', help='Speciális promptot használ kényes tartalmak fordításához.')

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.input_language, args.output_language, args.auth_key, args.context, args.srt_context_file, args.model, args.stream, args.allow_sensitive_content)
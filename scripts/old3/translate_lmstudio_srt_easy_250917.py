import os
import argparse
# Az 'openai' könyvtár továbbra is használatban van, mivel az LM Studio kompatibilis API-t biztosít
from openai import OpenAI, OpenAIError
import json
import re
import time
import srt # Szükséges: pip install srt
from datetime import timedelta
import glob
import sys
import base64

# *** FÜGGVÉNYCSOPORT: KONFIGURÁCIÓ ***
def get_project_root():
    """Visszaadja a projekt gyökérkönyvtárát."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    if os.path.basename(project_root) == 'scripts':
         return os.path.abspath(os.path.join(script_dir, '..'))
    return os.getcwd()

def load_config():
    """Betölti a config.json fájlt a projekt gyökeréből."""
    config_path = os.path.join(get_project_root(), 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("Konfigurációs fájl sikeresen betöltve.")
        return config
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"Hiba a konfigurációs fájl betöltése közben ({config_path}): {e}")
        return None

# *** FÜGGVÉNYCSOPORT: SRT ÉS FORDÍTÁSI LOGIKA ***

def find_srt_context_file(upload_dir, lang_code):
    if not os.path.isdir(upload_dir):
        print(f"Figyelmeztetés: Az 'upload' könyvtár nem található: {upload_dir}. Folytatás SRT kontextus nélkül.")
        return None
    search_pattern = os.path.join(upload_dir, f'*{lang_code.lower()}.srt')
    print(f"  -> Kontextusfájl keresése a következő mintával: {search_pattern}")
    matching_files = glob.glob(search_pattern)
    if not matching_files:
        print(f"  -> Nem található a(z) '*{lang_code.lower()}.srt' mintának megfelelő SRT fájl a(z) '{upload_dir}' könyvtárban.")
        return None
    # ... a többi rész változatlan ...
    if len(matching_files) == 1:
        selected_file = matching_files[0]
        print(f"  -> SRT kontextusfájl megtalálva: {os.path.basename(selected_file)}")
        return selected_file
    print(f"  -> Több ({len(matching_files)}) lehetséges SRT fájl található. A legnagyobb kiválasztása...")
    try:
        largest_file = max(matching_files, key=os.path.getsize)
        print(f"  -> A legnagyobb fájl kiválasztva: {os.path.basename(largest_file)} ({os.path.getsize(largest_file) / 1024:.2f} KB)")
        return largest_file
    except Exception as e:
        print(f"Hiba a legnagyobb fájl kiválasztása közben: {e}. Az első találat használata: {os.path.basename(matching_files[0])}")
        return matching_files[0]

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
    """Betölt egy SRT fájlt, visszaadja a feldolgozott felirat listát és a teljes tiszta szöveget."""
    if not filepath or not os.path.exists(filepath):
        if filepath: print(f"  -> Figyelmeztetés: Az SRT fájl nem található a megadott útvonalon.")
        return None, ""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            subtitles = list(srt.parse(content))
        # A teljes szöveg létrehozása, tisztítva
        full_text = " ".join([clean_srt_content(sub.content).replace('\n', ' ') for sub in subtitles])
        print(f"  -> SRT fájl sikeresen betöltve: {os.path.basename(filepath)} ({len(subtitles)} felirat).")
        return subtitles, full_text
    except Exception as e:
        print(f"  -> Figyelmeztetés: Az SRT fájl olvasása sikertelen ({e})."); return None, ""

def clean_srt_content(text):
    text = re.sub(r'\{[^\}]*\}', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    return text.strip()

def get_dynamic_srt_context(batch_segments, srt_data, leeway_seconds=15.0, max_chars=3000):
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

def align_and_segment_batch(client, batch_segments, target_lang_srt_data, model, stream, batch_id_str):
    """
    A meglévő magyar szöveget igazítja az angol ASR szegmentálásához.
    """
    if not batch_segments: return []
    if len(batch_segments) == 1: print(f"  [{batch_id_str}] Csoport 1 elemre redukálva, igazítás...")

    numbered_texts = [f"{i+1}. {seg['text']}" for i, seg in enumerate(batch_segments)]
    source_text_block = '\n'.join(numbered_texts)

    # A célnyelvi (magyar) szöveg kinyerése a megfelelő időablakból
    target_lang_context = get_dynamic_srt_context(batch_segments, target_lang_srt_data)
    if not target_lang_context:
        print(f"  HIBA: Nem található megfelelő magyar SRT kontextus a(z) [{batch_id_str}] csoporthoz. A csoport kihagyása.")
        return [""] * len(batch_segments)

    # Az "igazító" prompt
    system_prompt = (
        "You are an expert subtitle editor. Your task is to re-segment a correct Hungarian translation to perfectly match the timing and structure of an English ASR (Automatic Speech Recognition) transcript.\n\n"
        "INSTRUCTIONS:\n"
        "1. You will be given a numbered list of English text segments from an ASR. This structure is the **target**.\n"
        "2. You will also be given a block of Hungarian text. This is the **correct translation**.\n"
        "3. Your job is to re-format the Hungarian text to match the numbered list structure of the English input. You must use the words from the Hungarian text.\n"
        "4. DO NOT TRANSLATE. Only rearrange and re-segment the provided Hungarian text.\n"
        "5. The output MUST be a numbered list with the exact same number of items as the English input.\n"
        "6. If an English segment is short (e.g., 'Okay, so'), find the corresponding Hungarian word(s) (e.g., 'Rendben, szóval'). If a Hungarian sentence needs to be split across multiple lines to match the English structure, do it.\n"
        "7. Ensure your output is ONLY the numbered list of Hungarian segments."
    )

    user_prompt = (
        "--- ENGLISH ASR SEGMENTS (TARGET STRUCTURE) ---\n"
        f"{source_text_block}\n\n"
        "--- HUNGARIAN TRANSLATION (TEXT SOURCE) ---\n"
        f"{target_lang_context}"
    )

    aligned_lines_raw = []
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        aligned_lines_raw = [line.strip() for line in response.choices[0].message.content.strip().split('\n') if line.strip()]
    except OpenAIError as e:
        print(f"  HIBA: API hiba a(z) [{batch_id_str}] csoportnál: {e}");
        return None

    if len(aligned_lines_raw) == len(batch_segments):
        final_aligned_lines = [re.sub(r'^\d+\.\s*', '', line) for line in aligned_lines_raw]
        if stream:
            for line in final_aligned_lines: print(f"    + {line}")
        return final_aligned_lines
    else:
        print(f"  HIBA: A(z) [{batch_id_str}] csoportnál ({len(batch_segments)} elem) sorszámeltérés! Várt: {len(batch_segments)}, Kapott: {len(aligned_lines_raw)}.")
        if len(batch_segments) <= 1: return None
        print(f"  A [{batch_id_str}] csoport felosztása...")
        mid_point = len(batch_segments) // 2
        first_half, second_half = batch_segments[:mid_point], batch_segments[mid_point:]
        
        first_half_results = align_and_segment_batch(client, first_half, target_lang_srt_data, model, stream, f"{batch_id_str}-A")
        if first_half_results is None: return None
        
        second_half_results = align_and_segment_batch(client, second_half, target_lang_srt_data, model, stream, f"{batch_id_str}-B")
        if second_half_results is None: return None
        
        return first_half_results + second_half_results

# *** FŐ FÜGGVÉNY ***
def main(project_name, input_lang, output_lang, context, model, stream, allow_sensitive_content):
    
    config = load_config()
    if not config: sys.exit(1)
        
    try:
        workdir = config['DIRECTORIES']['workdir']
        subdirs = config['PROJECT_SUBDIRS']
        input_dir = os.path.join(workdir, project_name, subdirs['separated_audio_speech'])
        output_dir = os.path.join(workdir, project_name, subdirs['translated'])
        upload_dir = os.path.join(workdir, project_name, subdirs['upload'])
    except KeyError as e:
        print(f"Hiba: Hiányzó kulcs a config.json fájlban: {e}"); return

    # <<< MÓDOSÍTÁS: OpenAI kliens inicializálása az LM Studio helyi szerveréhez
    try:
        client = OpenAI(base_url="http://127.0.0.1:1235/v1", api_key="not-needed")
        print("LM Studio API kliens sikeresen inicializálva a http://127.0.0.1:1235 címen.")
    except Exception as e:
        print(f"Hiba az LM Studio kliens inicializálása közben: {e}")
        sys.exit(1)

    input_filename, error = find_json_file(input_dir)
    if error: return
    
    input_filepath = os.path.join(input_dir, input_filename); print(f"Bemeneti fájl feldolgozása: {input_filepath}")
    os.makedirs(output_dir, exist_ok=True); output_filepath = os.path.join(output_dir, input_filename)

    # A célnyelvi SRT fájl betöltése. Ez a forrása a szövegnek.
    print(f"\nCélnyelvi ({output_lang.upper()}) SRT keresése a(z) '{upload_dir}' mappában...")
    target_srt_filepath = find_srt_context_file(upload_dir, output_lang)
    target_srt_data, _ = load_srt_file(target_srt_filepath)
    
    if not target_srt_data:
        print("HIBA: A célnyelvi SRT fájl elengedhetetlen ehhez a művelethez, de nem található. A szkript leáll.")
        return

    progress_filename = f"{os.path.splitext(input_filename)[0]}.progress.json"
    progress_filepath = os.path.join(output_dir, progress_filename)
    progress = {}
    if os.path.exists(progress_filepath):
        try:
            with open(progress_filepath, 'r', encoding='utf-8') as f: progress = json.load(f)
            print(f"\nHaladási fájl betöltve: {progress_filepath} ({len(progress)} szegmens már feldolgozva).")
        except (json.JSONDecodeError, IOError): progress = {}

    with open(input_filepath, 'r', encoding='utf-8') as f: data = json.load(f)
    if 'segments' not in data or not isinstance(data['segments'], list): print("Hiba: JSON 'segments' kulcs hiányzik."); return

    unprocessed_segments = []
    for i, segment in enumerate(data['segments']):
        if segment.get('text', '').strip() and str(i) not in progress:
            segment['original_index'] = i; unprocessed_segments.append(segment)

    if not unprocessed_segments: print("Nincs új, feldolgozandó szegmens.")
    else:
        print(f"\n{len(unprocessed_segments)} új, feldolgozandó szegmens található.")
        batches = create_smart_chunks(unprocessed_segments)
        total_batches = len(batches)
        print(f"Intelligens csoportképzés befejezve. {total_batches} új csoportot kell feldolgozni.")

        for i, batch in enumerate(batches):
            print(f"\n[{i+1}/{total_batches}] fő csoport feldolgozása...")
            
            aligned_batch_lines = align_and_segment_batch(client, batch, target_srt_data, model, stream, f"{i+1}")
            
            if aligned_batch_lines is None:
                print(f"\nA feldolgozási folyamat megszakadt."); return

            for segment_obj, aligned_line in zip(batch, aligned_batch_lines):
                progress[str(segment_obj['original_index'])] = aligned_line
            
            with open(progress_filepath, 'w', encoding='utf-8') as f: json.dump(progress, f, ensure_ascii=False, indent=2)
            print(f"  Haladás elmentve a(z) {progress_filepath} fájlba.")
            time.sleep(1)

    print("\nAz összes szükséges szegmens igazítása elkészült.")
    for index_str, line in progress.items(): data['segments'][int(index_str)]['translated_text'] = line
    for segment in data['segments']: segment.setdefault('translated_text', '')
    with open(output_filepath, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nFeldolgozás befejezve. A kiegészített fájl a(z) '{output_filepath}' helyre mentve.")
    
    if os.path.exists(progress_filepath): os.remove(progress_filepath); print(f"Az ideiglenes haladási fájl ({progress_filepath}) törölve.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Meglévő fordítás (SRT) újra-szegmentálása egy ASR JSON időzítése alapján LM Studio segítségével.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('-project_name', required=True, help='A workdir-en belüli projektmappa neve.')
    parser.add_argument('-input_language', default='EN', help='A bemeneti nyelv kódja (pl. EN, HU). Alapértelmezett: EN')
    parser.add_argument('-output_language', default='HU', help='A kimeneti nyelv kódja (pl. EN, HU), aminek az SRT-jét keresi. Alapértelmezett: HU')
    # <<< MÓDOSÍTÁS: Az auth_key argumentum eltávolítva
    parser.add_argument('-context', required=False, help='Ez az argumentum ennél a módszernél figyelmen kívül van hagyva.')
    # <<< MÓDOSÍTÁS: Az alapértelmezett modell megváltoztatva
    parser.add_argument('-model', default='qwen3-coder-30b-a3b-instruct', required=False, help='Az LM Studióban betöltött modell neve. Alapértelmezett: qwen3-coder-30b-a3b-instruct')
    parser.add_argument('-stream', action='store_true', help='Bekapcsolja a valós idejű kimenetet.')
    parser.add_argument('--allow-sensitive-content', action=argparse.BooleanOptionalAction, default=True, help='Ez az argumentum ennél a módszernél figyelmen kívül van hagyva.')

    args = parser.parse_args()
    
    # <<< MÓDOSÍTÁS: Az auth_key argumentum eltávolítva a main hívásból
    main(args.project_name, args.input_language, args.output_language, args.context, args.model, args.stream, args.allow_sensitive_content)
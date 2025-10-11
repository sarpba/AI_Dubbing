import os
import argparse
from openai import OpenAI, OpenAIError
import json
import re
import time
import srt
from datetime import timedelta, datetime
import glob
import sys
import base64
import logging

# ... (a szkript többi része változatlan) ...

def load_config():
    config_path = os.path.join(get_project_root(), 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Hiba a config.json betöltésekor: {e}")
        return None

def get_project_root():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(script_dir, '..')) if os.path.basename(os.path.abspath(os.path.join(script_dir, '..')))!='scripts' else os.getcwd()

def setup_logging(project_name, config):
    try:
        log_dir = os.path.join(config['DIRECTORIES']['workdir'], project_name, config['PROJECT_SUBDIRS']['logs'])
        os.makedirs(log_dir, exist_ok=True)
        log_filepath = os.path.join(log_dir, f"translation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_filepath, encoding='utf-8'), logging.StreamHandler(sys.stdout)])
        logging.info(f"Logolás beállítva: {log_filepath}")
        return True
    except Exception as e:
        print(f"Hiba a logolás beállításakor: {e}")
        return False

def load_api_key():
    path = os.path.join(get_project_root(), 'keyholder.json')
    if not os.path.exists(path): return None
    try:
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
        return base64.b64decode(data.get('chatgpt_api_key', '').encode('utf-8')).decode('utf-8')
    except Exception as e:
        logging.error(f"Hiba az API kulcs betöltésekor: {e}")
        return None

def save_api_key(api_key):
    path = os.path.join(get_project_root(), 'keyholder.json')
    try:
        data = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try: data = json.load(f)
                except json.JSONDecodeError: pass
        data['chatgpt_api_key'] = base64.b64encode(api_key.encode('utf-8')).decode('utf-8')
        with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2)
        logging.info("API kulcs elmentve.")
    except Exception as e:
        logging.error(f"Hiba az API kulcs mentésekor: {e}")

def load_optional_json(file_path):
    if not os.path.exists(file_path): return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Fájl betöltve: {os.path.basename(file_path)}")
        return data
    except Exception as e:
        logging.warning(f"Hiba a(z) {os.path.basename(file_path)} olvasásakor: {e}")
        return {}
        
def clean_srt_content(text):
    return re.sub(r'<[^>]*>', '', re.sub(r'\{[^\}]*\}', '', text)).strip()

# ==============================================================================
# === MÓDOSÍTOTT FÜGGVÉNY ===
# ==============================================================================
def get_dynamic_srt_context(batch_segments, srt_data, leeway_seconds=15.0, max_chars=2000):
    """
    Kikeresi a releváns SRT sorokat a stíluskonzisztencia érdekében.
    A sorokat új sorokkal tagolja a természetesebb párbeszéd-kontextusért.
    """
    if not srt_data or not batch_segments: return ""
    try:
        batch_start_time = batch_segments[0]['start']
        batch_end_time = batch_segments[-1]['end']
    except (KeyError, IndexError): return ""
    
    context_start_time = max(0, batch_start_time - leeway_seconds)
    context_end_time = batch_end_time + leeway_seconds
    
    # A feliratokat most nem fűzzük egybe szóközzel, hanem megtartjuk a sortöréseket
    relevant_subs = [
        clean_srt_content(sub.content)
        for sub in srt_data
        if sub.end.total_seconds() >= context_start_time and sub.start.total_seconds() <= context_end_time and clean_srt_content(sub.content)
    ]
    
    if relevant_subs:
        # A sorokat '\n' karakterrel illesztjük össze
        full_context = "\n".join(relevant_subs)
        sample = full_context[:max_chars].strip()
        
        time_window_str = f"{time.strftime('%H:%M:%S', time.gmtime(context_start_time))} - {time.strftime('%H:%M:%S', time.gmtime(context_end_time))}"
        logging.info(f"Dinamikus, sorokra tagolt SRT kontextus kikeresve a(z) {time_window_str} időablakhoz.")
        return sample
    return ""
# ==============================================================================
# === MÓDOSÍTÁS VÉGE ===
# ==============================================================================

def generate_glossary_from_srt(client, model, source_segments, target_srt_data, upload_dir, lang_from_name, lang_to_name):
    logging.info("Automatikus glosszárium generálás indítása...")
    aligned_pairs = []
    for sub in target_srt_data:
        sub_start, sub_end = sub.start.total_seconds(), sub.end.total_seconds()
        overlapping_segments_text = [
            seg.get('text', '').strip() for seg in source_segments
            if max(seg.get('start', -1), sub_start) < min(seg.get('end', -1), sub_end)
        ]
        if overlapping_segments_text:
            aligned_pairs.append({
                "source": " ".join(overlapping_segments_text),
                "target": clean_srt_content(sub.content).replace('\n', ' ')
            })
    if not aligned_pairs:
        logging.warning("Nem sikerült szövegpárokat igazítani. Glosszárium generálás kihagyva.")
        return {}
    logging.info(f"{len(aligned_pairs)} szövegpár igazítva idő alapján.")
    prompt = (
        f"You are a terminology extraction expert. Analyze the following JSON list of source ({lang_from_name}) and target ({lang_to_name}) subtitle pairs.\n"
        "Identify key, recurring terms (names, jargon, specific concepts) and their translations.\n"
        "Return a single, flat JSON object mapping the source term to its target translation.\n"
        "Example: {\"FTL Drive\": \"Hipertér Hajtómű\", \"Captain Reynolds\": \"Reynolds kapitány\"}\n"
        "Include only high-confidence, important terms. Do not include common words."
    )
    try:
        response = client.chat.completions.create(
            model=model, response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(aligned_pairs, indent=2, ensure_ascii=False)}
            ],
            temperature=0.0
        )
        glossary_data = json.loads(response.choices[0].message.content)
        logging.info(f"AI sikeresen kinyert {len(glossary_data)} terminológiai bejegyzést.")
        glossary_filepath = os.path.join(upload_dir, 'glossary.json')
        with open(glossary_filepath, 'w', encoding='utf-8') as f:
            json.dump(glossary_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Glosszárium elmentve ide: {glossary_filepath}")
        return glossary_data
    except Exception as e:
        logging.error(f"Hiba a glosszárium generálása közben: {e}")
        return {}

def get_lang_name(lang_code):
    return {'EN': 'English', 'HU': 'Hungarian', 'DE': 'German', 'FR': 'French', 'ES': 'Spanish'}.get(lang_code.upper(), lang_code)

def find_json_file(directory):
    try:
        json_files = [f for f in os.listdir(directory) if f.lower().endswith('.json')]
        if not json_files: return None, "no_json_found"
        if len(json_files) > 1: return None, "multiple_jsons_found"
        return json_files[0], None
    except FileNotFoundError:
        return None, "directory_not_found"

def create_smart_chunks(segments, min_size=50, max_size=100, gap_threshold=5.0):
    chunks, pos = [], 0
    while pos < len(segments):
        start = pos
        if (len(segments) - start) <= min_size:
            end = len(segments)
        else:
            search_end = min(start + max_size, len(segments) - 1)
            best_split = -1
            for i in range(start + min_size - 1, search_end):
                if segments[i + 1].get('start', 0) - segments[i].get('end', 0) >= gap_threshold:
                    best_split = i + 1; break
            if best_split == -1:
                end = min(start + max_size, len(segments))
            else:
                end = best_split
        chunks.append(segments[start:end]); pos = end
    return chunks

def load_srt_file(filepath):
    if not filepath or not os.path.exists(filepath): return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            subtitles = list(srt.parse(f.read()))
        logging.info(f"SRT fájl betöltve: {os.path.basename(filepath)} ({len(subtitles)} felirat).")
        return subtitles
    except Exception as e:
        logging.warning(f"SRT fájl olvasása sikertelen ({e})."); return None

def find_relevant_glossary_terms(text_block, glossary_data):
    if not glossary_data: return {}
    return {s: t for s, t in glossary_data.items() if re.search(r'\b' + re.escape(s) + r'\b', text_block, re.IGNORECASE)}

def translate_or_subdivide_batch(client, batch_segments, lang_from, lang_to, context, srt_data, glossary_data, model, stream, allow_sensitive, batch_id_str):
    if not batch_segments: return []
    numbered_texts = [f"{i+1}. {seg['text']}" for i, seg in enumerate(batch_segments)]
    text_block = '\n'.join(numbered_texts)
    
    prompt_parts = [f"You are a professional translator for film subtitling. Translate the numbered list from {get_lang_name(lang_from)} to {get_lang_name(lang_to)}. Your response MUST be a numbered list with the exact same number of items. Format: `1. [translation]`."]
    if allow_sensitive: prompt_parts[0] += " The source may contain profanity or sensitive topics. Your directive is **faithful translation**, do not censor."
    
    relevant_glossary = find_relevant_glossary_terms(text_block, glossary_data)
    if relevant_glossary:
        logging.info(f"Releváns glosszárium-kifejezések a csomagban: {list(relevant_glossary.keys())}")
        prompt_parts.append(f"--- TERMINOLOGY ---\nAdhere to this glossary:\n" + "\n".join([f'- "{s}": must be translated as "{t}"' for s, t in relevant_glossary.items()]) + "\n--- END TERMINOLOGY ---")
    
    dynamic_srt_context = get_dynamic_srt_context(batch_segments, srt_data)
    if dynamic_srt_context: prompt_parts.append(f"\n--- STYLE SAMPLE (from SRT) ---\nAdopt a style similar to this sample:\n{dynamic_srt_context}\n--- END STYLE SAMPLE ---")

    if context: prompt_parts.append(f"\n--- GENERAL CONTEXT ---\n{context}\n--- END GENERAL CONTEXT ---")
    system_prompt = "\n\n".join(prompt_parts)

    logging.info(f"--- API HÍVÁS (Batch: {batch_id_str}) ---"); logging.info(f"--- SYSTEM PROMPT ---\n{system_prompt}"); logging.info(f"--- USER MESSAGE ---\n{text_block}")
    
    try:
        response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text_block}], temperature=0.1)
        raw_response = response.choices[0].message.content.strip()
        logging.info(f"--- RAW API RESPONSE ---\n{raw_response}\n--- API HÍVÁS VÉGE ---")
        translated_lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
    except OpenAIError as e:
        logging.error(f"API hiba a(z) [{batch_id_str}] csoportnál: {e}"); return None

    if len(translated_lines) == len(batch_segments):
        final_lines = [re.sub(r'^\d+\.\s*', '', line) for line in translated_lines]
        if stream: print("\n".join(f"    + {line}" for line in final_lines))
        return final_lines
    else:
        logging.error(f"SORSZÁMELTÉRÉS a(z) [{batch_id_str}] csoportnál! Várt: {len(batch_segments)}, Kapott: {len(translated_lines)}. Felosztás...")
        if len(batch_segments) <= 1: return None
        mid = len(batch_segments) // 2
        first = translate_or_subdivide_batch(client, batch_segments[:mid], lang_from, lang_to, context, srt_data, glossary_data, model, stream, allow_sensitive, f"{batch_id_str}-A")
        if first is None: return None
        second = translate_or_subdivide_batch(client, batch_segments[mid:], lang_from, lang_to, context, srt_data, glossary_data, model, stream, allow_sensitive, f"{batch_id_str}-B")
        if second is None: return None
        return first + second

def main(args):
    config = load_config()
    if not config or not setup_logging(args.project_name, config): sys.exit(1)
        
    logging.info("--- FORDÍTÁSI FOLYAMAT INDÍTÁSA ---"); logging.info(f"Argumentumok: {vars(args)}")

    auth_key = args.auth_key or load_api_key()
    if args.auth_key: save_api_key(args.auth_key)
    if not auth_key: logging.critical("Nincs elérhető OpenAI API kulcs."); sys.exit(1)
    logging.info("OpenAI API kulcs beállítva.")
    
    try:
        workdir, subdirs = config['DIRECTORIES']['workdir'], config['PROJECT_SUBDIRS']
        input_dir = os.path.join(workdir, args.project_name, subdirs['separated_audio_speech'])
        output_dir = os.path.join(workdir, args.project_name, subdirs['translated'])
        upload_dir = os.path.join(workdir, args.project_name, subdirs['upload'])
        os.makedirs(output_dir, exist_ok=True)
    except KeyError as e: logging.critical(f"Hiányzó kulcs a config.json-ban: {e}"); return

    client = OpenAI(api_key=auth_key)
    input_filename, error = find_json_file(input_dir)
    if error: logging.error(f"JSON fájl hiba: {error}"); return
    
    input_filepath = os.path.join(input_dir, input_filename)
    with open(input_filepath, 'r', encoding='utf-8') as f: data = json.load(f)
    if 'segments' not in data: logging.error("A JSON 'segments' kulcsa hiányzik."); return
    
    srt_context_filepath = os.path.join(upload_dir, f'*{args.output_language.lower()}.srt')
    matching_srts = glob.glob(srt_context_filepath)
    srt_data = load_srt_file(max(matching_srts, key=os.path.getsize) if matching_srts else None)
    
    glossary_filepath = os.path.join(upload_dir, 'glossary.json')
    glossary_data = {}
    if os.path.exists(glossary_filepath) and not args.force_glossary:
        logging.info("Meglévő glosszáriumfájl betöltése.")
        glossary_data = load_optional_json(glossary_filepath)
    elif srt_data:
        glossary_data = generate_glossary_from_srt(client, args.model, data['segments'], srt_data, upload_dir, get_lang_name(args.input_language), get_lang_name(args.output_language))
    else:
        logging.warning("Sem glosszárium, sem célnyelvi SRT nem található. Fordítás glosszárium nélkül.")

    progress_filepath = os.path.join(output_dir, f"{os.path.splitext(input_filename)[0]}.progress.json")
    progress = load_optional_json(progress_filepath)
    if progress: logging.info(f"Folyamatban lévő fordítás folytatása ({len(progress)} kész szegmens).")
    
    untranslated = [{**seg, 'original_index': i} for i, seg in enumerate(data['segments']) if seg.get('text','').strip() and str(i) not in progress]
    if not untranslated:
        logging.info("Nincs új, fordítandó szegmens.")
    else:
        logging.info(f"{len(untranslated)} új szegmens fordítása.")
        batches = create_smart_chunks(untranslated)
        logging.info(f"{len(batches)} csoport feldolgozása.")
        for i, batch in enumerate(batches):
            logging.info(f"\n--- Csoport feldolgozása: [{i+1}/{len(batches)}] ---")
            translated = translate_or_subdivide_batch(client, batch, args.input_language, args.output_language, args.context, srt_data, glossary_data, args.model, args.stream, args.allow_sensitive_content, f"{i+1}")
            if translated is None: logging.critical("A fordítás megszakadt."); return
            for seg, line in zip(batch, translated): progress[str(seg['original_index'])] = line
            with open(progress_filepath, 'w', encoding='utf-8') as f: json.dump(progress, f, ensure_ascii=False, indent=2)
            logging.info("Haladás elmentve.")
            time.sleep(1)

    logging.info("\nFordítási folyamat kész.")
    for index, line in progress.items(): data['segments'][int(index)]['translated_text'] = line
    for segment in data['segments']: segment.setdefault('translated_text', '')
    
    output_filepath = os.path.join(output_dir, input_filename)
    with open(output_filepath, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info(f"Kiegészített fájl mentve: '{output_filepath}'")
    
    if os.path.exists(progress_filepath): os.remove(progress_filepath); logging.info("Ideiglenes haladási fájl törölve.")
    logging.info("--- FORDÍTÁSI FOLYAMAT BEFEJEZVE ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Szövegszegmensek intelligens fordítása automatikus glosszárium-generálással és több soros stílusmintával.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-project_name', required=True, help='A workdir-en belüli projektmappa neve.')
    parser.add_argument('-input_language', default='EN', help='Bemeneti nyelv kódja (alap: EN)')
    parser.add_argument('-output_language', default='HU', help='Kimeneti nyelv kódja (alap: HU)')
    parser.add_argument('-auth_key', help='OpenAI API kulcs (mentésre kerül).')
    parser.add_argument('-context', help='Általános kontextus a fordításhoz.')
    parser.add_argument('-model', default='gpt-4o', help='Használandó OpenAI modell (alap: gpt-4o)')
    parser.add_argument('-stream', action='store_true', help='Lefordított sorok valós idejű kiírása.')
    parser.add_argument('--force-glossary', action='store_true', help='Kikényszeríti a glossary.json újra-generálását.')
    parser.add_argument('--allow-sensitive-content', action=argparse.BooleanOptionalAction, default=True, help='Engedélyezi a kényes tartalmak hű fordítását.')
    
    main(parser.parse_args())
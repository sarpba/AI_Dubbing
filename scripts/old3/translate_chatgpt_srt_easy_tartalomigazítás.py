import os
import argparse
from openai import OpenAI, OpenAIError
import json
import re
import time
import srt
from datetime import datetime
import glob
import sys
import base64
import logging

# *** FÜGGVÉNYCSOPORT: KONFIGURÁCIÓ, KULCSKEZELÉS ÉS LOGOLÁS (Változatlan) ***
def get_project_root():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if os.path.basename(script_dir) == 'scripts': return os.path.abspath(os.path.join(script_dir, '..'))
    return os.getcwd()

def setup_logging(project_name, config):
    try:
        workdir = config['DIRECTORIES']['workdir']
        log_subdir = config['PROJECT_SUBDIRS']['logs']
        log_dir = os.path.join(workdir, project_name, log_subdir)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"alignment_{timestamp}.log"
        log_filepath = os.path.join(log_dir, log_filename)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_filepath, encoding='utf-8'), logging.StreamHandler(sys.stdout)])
        logging.info(f"Logolás beállítva. A logfájl helye: {log_filepath}")
        return True
    except Exception as e:
        print(f"KRITIKUS HIBA: A logolás beállítása sikertelen. Hiba: {e}"); return False

def load_config():
    config_path = os.path.join(get_project_root(), 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception as e:
        print(f"Hiba a konfigurációs fájl betöltése közben ({config_path}): {e}"); return None

def save_api_key(api_key):
    path = os.path.join(get_project_root(), 'keyholder.json')
    try:
        data = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try: data = json.load(f)
                except json.JSONDecodeError: data = {}
        encoded_key = base64.b64encode(api_key.encode('utf-8')).decode('utf-8')
        data['chatgpt_api_key'] = encoded_key
        with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2)
        logging.info("API kulcs sikeresen elmentve.")
    except Exception as e:
        logging.error(f"Hiba az API kulcs mentése közben: {e}")

def load_api_key():
    path = os.path.join(get_project_root(), 'keyholder.json')
    if not os.path.exists(path): return None
    try:
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
        encoded_key = data.get('chatgpt_api_key')
        if not encoded_key: return None
        return base64.b64decode(encoded_key.encode('utf-8')).decode('utf-8')
    except Exception as e:
        logging.error(f"Hiba az API kulcs betöltése közben: {e}"); return None

# *** FÜGGVÉNYCSOPORT: FÁJLKEZELÉS ÉS ADAT-ELŐKÉSZÍTÉS ***

def find_file(directory, extension, language_code=None):
    if not os.path.isdir(directory):
        logging.error(f"A keresési könyvtár nem található: {directory}"); return None, "directory_not_found"
    search_pattern = f"*{language_code.lower()}.{extension}" if language_code else f"*.{extension}"
    full_pattern = os.path.join(directory, search_pattern)
    matching_files = glob.glob(full_pattern)
    if not matching_files: return None, "file_not_found"
    largest_file = max(matching_files, key=os.path.getsize)
    if len(matching_files) > 1:
        logging.warning(f"Több fájl található, a legnagyobb kiválasztva: {os.path.basename(largest_file)}")
    return largest_file, None

def load_and_split_srt_file(filepath):
    if not filepath: return None, None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            subtitles = list(srt.parse(f.read()))
        dialogue_subs, narrator_subs = [], []
        for sub in subtitles:
            if sub.content.strip().startswith('{\\an8}'):
                narrator_subs.append(sub)
            else:
                dialogue_subs.append(sub)
        logging.info(f"SRT fájl feldolgozva: {len(dialogue_subs)} párbeszéd, {len(narrator_subs)} narrátor sor.")
        return dialogue_subs, narrator_subs
    except Exception as e:
        logging.error(f"SRT fájl olvasása sikertelen ({filepath}): {e}"); return None, None

def clean_srt_content(text):
    return re.sub(r'<[^>]*>', '', re.sub(r'\{[^\}]*\}', '', text)).strip().replace('\n', ' ')

# *** LOGIKA: HÁROMFÁZISÚ ILLESZTÉS ÉS AI-DARABOLÁS ***

def create_consensus_blocks(source_segments, target_dialogue_data, gap_threshold=1.5):
    logging.info("Konszenzus alapú, szinkronizált blokképítés indítása...")
    if not source_segments or not target_dialogue_data: return []

    json_to_srt_map = { i: frozenset(srt_idx for srt_idx, srt_entry in enumerate(target_dialogue_data) if max(seg.get('start', -1), srt_entry.start.total_seconds()) < min(seg.get('end', -1), srt_entry.end.total_seconds())) for i, seg in enumerate(source_segments)}

    blocks, json_idx = [], 0
    while json_idx < len(source_segments):
        start_segment = source_segments[json_idx]
        base_srt_indices = json_to_srt_map.get(json_idx, frozenset())
        current_speaker = start_segment.get('speaker')
        current_block_indices = [json_idx]

        for next_idx in range(json_idx + 1, len(source_segments)):
            next_segment = source_segments[next_idx]
            gap = next_segment.get('start', 0) - source_segments[next_idx - 1].get('end', 0)
            if (gap < gap_threshold and next_segment.get('speaker') == current_speaker and json_to_srt_map.get(next_idx, frozenset()) == base_srt_indices):
                current_block_indices.append(next_idx)
            else:
                break
        
        if base_srt_indices:
            srt_entries = sorted([target_dialogue_data[i] for i in base_srt_indices], key=lambda s: s.start)
            combined_srt_text = " ".join([clean_srt_content(s.content) for s in srt_entries])
            json_segments_for_block = []
            for idx in current_block_indices:
                segment = source_segments[idx]
                segment['original_index'] = idx
                json_segments_for_block.append(segment)
            blocks.append({'target_srt_text': combined_srt_text, 'source_json_segments': json_segments_for_block})
        
        json_idx += len(current_block_indices)

    logging.info(f"{len(blocks)} szinkronizált blokk sikeresen létrehozva.")
    return blocks

def split_text_with_ai(client, model, target_text, source_segments):
    source_texts_list = [seg['text'] for seg in source_segments]
    if len(source_texts_list) <= 1: return [target_text]
    numbered_source_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(source_texts_list)])
    system_prompt = "You are an expert in audio-visual localization and dubbing script preparation. Your task is to intelligently split a single, complete Hungarian text block into multiple parts for a Text-to-Speech (TTS) engine. You will be given the complete Hungarian text and a numbered list of the original English segments it corresponds to. These original segments represent the precise timing breaks needed for the TTS."
    user_prompt = (f"Split the following Hungarian text into exactly {len(source_texts_list)} parts. The content of each new part should correspond to the content of the similarly numbered original segment. Ensure each new Hungarian part is grammatically correct and flows naturally, and is ready for TTS generation. Your output MUST be a numbered list with the same number of items as the original list. Do not add any other text, apologies, or explanations.\n\n"
                   f"--- HUNGARIAN TEXT (to split) ---\n\"{target_text}\"\n\n"
                   f"--- ORIGINAL ENGLISH SEGMENTS (use as a guide for content and breaks) ---\n{numbered_source_texts}\n\n"
                   f"--- YOUR REQUIRED OUTPUT (numbered list of split Hungarian parts) ---")
    
    logging.info(f"--- AI DARABOLÁSI KÉRÉS --- Cél: \"{target_text}\", Forrás darabszám: {len(source_texts_list)}")
    try:
        response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.0, top_p=0.1)
        raw_response = response.choices[0].message.content.strip()
        logging.info(f"AI válasz (nyers): \n{raw_response}")
        split_lines = [re.sub(r'^\d+\.\s*', '', line).strip() for line in raw_response.split('\n') if line.strip()]
        if len(split_lines) == len(source_texts_list):
            logging.info("AI a szöveget sikeresen a megfelelő darabszámra osztotta.")
            return split_lines
        else:
            logging.error(f"AI HIBA: Sorszámeltérés! Várt: {len(source_texts_list)}, Kapott: {len(split_lines)}. Blokk kihagyva."); return None
    except Exception as e:
        logging.error(f"Váratlan API hiba: {e}"); return None

def main(args):
    config = load_config()
    if not config or not setup_logging(args.project_name, config): sys.exit(1)
        
    logging.info(f"--- TARTALOM-IGAZÍTÁS (v7 - Háromfázisú illesztés) ---")
    logging.info(f"Projekt: {args.project_name}, Célnyelv: {args.output_language}, Modell: {args.model}")

    auth_key = args.auth_key or load_api_key()
    if args.auth_key: save_api_key(args.auth_key)
    if not auth_key: logging.critical("Nincs elérhető OpenAI API kulcs."); sys.exit(1)
    client = OpenAI(api_key=auth_key)
    
    try:
        workdir, subdirs = config['DIRECTORIES']['workdir'], config['PROJECT_SUBDIRS']
        input_dir = os.path.join(workdir, args.project_name, subdirs['separated_audio_speech'])
        output_dir = os.path.join(workdir, args.project_name, subdirs['translated'])
        upload_dir = os.path.join(workdir, args.project_name, subdirs['upload'])
        os.makedirs(output_dir, exist_ok=True)
    except KeyError as e:
        logging.critical(f"Hiányzó kulcs a config.json-ban: {e}"); return

    source_json_path, error = find_file(input_dir, "json")
    if error: return
    
    target_srt_path, error = find_file(upload_dir, "txt", args.output_language)
    if error:
        logging.warning("Nem található .txt fájl, megpróbálkozom .srt kiterjesztéssel...")
        target_srt_path, error = find_file(upload_dir, "srt", args.output_language)
        if error: return

    try:
        with open(source_json_path, 'r', encoding='utf-8') as f: source_data = json.load(f)
        source_segments = source_data.get('segments', [])
        if not source_segments: logging.critical("A forrás JSON üres vagy érvénytelen."); return
    except Exception as e:
        logging.critical(f"Hiba a forrás JSON ({source_json_path}) olvasásakor: {e}"); return

    dialogue_srt_data, narrator_srt_data = load_and_split_srt_file(target_srt_path)
    if dialogue_srt_data is None: return

    if narrator_srt_data:
        narrator_json = [{"start": s.start.total_seconds(), "end": s.end.total_seconds(), "text": re.sub(r'\{\\an8\}', '', s.content).strip()} for s in narrator_srt_data]
        narrator_filepath = os.path.join(output_dir, 'narrator.json')
        with open(narrator_filepath, 'w', encoding='utf-8') as f:
            json.dump(narrator_json, f, ensure_ascii=False, indent=2)
        logging.info(f"Narrátor fájl létrehozva: '{narrator_filepath}'")

    final_segments = source_data['segments']
    json_assigned = [False] * len(source_segments)
    srt_assigned = [False] * len(dialogue_srt_data)
    
    # 1. FÁZIS: NAGY PONTOSSÁGÚ 1:1 PÁROSÍTÁS
    logging.info("\n--- 1. FÁZIS: Nagy pontosságú 1:1 párosítás ---")
    all_pairs = []
    for i, seg in enumerate(source_segments):
        seg_mid = seg['start'] + (seg['end'] - seg['start']) / 2
        for j, srt_entry in enumerate(dialogue_srt_data):
            srt_mid = srt_entry.start.total_seconds() + (srt_entry.end.total_seconds() - srt_entry.start.total_seconds()) / 2
            dist = abs(seg_mid - srt_mid)
            if dist < args.max_midpoint_distance:
                all_pairs.append((dist, i, j))
    all_pairs.sort()
    count_pass1 = 0
    for dist, json_idx, srt_idx in all_pairs:
        if not json_assigned[json_idx] and not srt_assigned[srt_idx]:
            final_segments[json_idx]['translated_text'] = clean_srt_content(dialogue_srt_data[srt_idx].content)
            json_assigned[json_idx] = True
            srt_assigned[srt_idx] = True
            count_pass1 += 1
    logging.info(f"1. Fázis befejezve. {count_pass1} db 1:1 párosítás történt.")

    # 2. FÁZIS: TÖBB JSON -> 1 SRT (AI DARABOLÁS)
    logging.info("\n--- 2. FÁZIS: Több-az-egyhez (N:1) blokkok keresése AI daraboláshoz ---")
    count_pass2 = 0
    for srt_idx, srt_entry in enumerate(dialogue_srt_data):
        if srt_assigned[srt_idx]: continue
        leeway = 0.15
        srt_start, srt_end = srt_entry.start.total_seconds() - leeway, srt_entry.end.total_seconds() + leeway
        candidate_indices = [i for i, seg in enumerate(source_segments) if not json_assigned[i] and max(seg['start'], srt_start) < min(seg['end'], srt_end)]
        if len(candidate_indices) > 1:
            candidate_indices.sort()
            block_segments = [source_segments[candidate_indices[0]]]
            for i in range(1, len(candidate_indices)):
                prev_seg, curr_seg = source_segments[candidate_indices[i-1]], source_segments[candidate_indices[i]]
                if (curr_seg['start'] - prev_seg['end'] < 1.0 and curr_seg.get('speaker') == prev_seg.get('speaker')):
                    block_segments.append(curr_seg)
                else: break
            if len(block_segments) > 1:
                logging.info(f"N:1 blokk található az SRT #{srt_idx+1} sornál.")
                split_texts = split_text_with_ai(client, args.model, clean_srt_content(srt_entry.content), block_segments)
                if split_texts:
                    for seg, text in zip(block_segments, split_texts):
                        json_original_idx = source_segments.index(seg)
                        final_segments[json_original_idx]['translated_text'] = text
                        json_assigned[json_original_idx] = True
                    srt_assigned[srt_idx] = True
                    count_pass2 += 1
    logging.info(f"2. Fázis befejezve. {count_pass2} db N:1 blokk feldolgozva AI-val.")

    # 3. FÁZIS: 1 JSON -> TÖBB SRT (ÖSSZEFŰZÉS)
    logging.info("\n--- 3. FÁZIS: Egy-a-többhöz (1:N) maradék illesztés ---")
    count_pass3 = 0
    for json_idx, seg in enumerate(source_segments):
        if json_assigned[json_idx]: continue
        leeway = 0.15
        seg_start, seg_end = seg['start'] - leeway, seg['end'] + leeway
        unassigned_srt_indices = [i for i, srt_entry in enumerate(dialogue_srt_data) if not srt_assigned[i] and max(seg_start, srt_entry.start.total_seconds()) < min(seg_end, srt_entry.end.total_seconds())]
        if unassigned_srt_indices:
            unassigned_srt_indices.sort()
            combined_text = " ".join([clean_srt_content(dialogue_srt_data[i].content) for i in unassigned_srt_indices])
            final_segments[json_idx]['translated_text'] = combined_text
            json_assigned[json_idx] = True
            for i in unassigned_srt_indices: srt_assigned[i] = True
            count_pass3 += 1
    logging.info(f"3. Fázis befejezve. {count_pass3} db 1:N blokk összefűzve.")

    # Kimeneti fájl írása
    base_filename = os.path.basename(source_json_path)
    output_filename = os.path.splitext(base_filename)[0] + '_aligned_split_v7.json'
    output_filepath = os.path.join(output_dir, output_filename)
    source_data['segments'] = final_segments
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(source_data, f, ensure_ascii=False, indent=2)
    logging.info(f"\nAz igazított és darabolt fájl elmentve: '{output_filepath}'")

    # *** ÚJ: Részletes lista a párosítatlan elemekről ***
    unassigned_json_count = json_assigned.count(False)
    unassigned_srt_count = srt_assigned.count(False)
    if unassigned_json_count > 0 or unassigned_srt_count > 0:
        logging.warning(f"Figyelem: {unassigned_json_count} JSON szegmens és {unassigned_srt_count} SRT sor maradt párosítatlanul.")
        
        unassigned_json_details = [f"  JSON #{i+1} ({s['start']:.2f}-{s['end']:.2f}): '{s['text']}'" for i, s in enumerate(source_segments) if not json_assigned[i]]
        unassigned_srt_details = [f"  SRT #{s.index} ({s.start.total_seconds():.2f}-{s.end.total_seconds():.2f}): '{clean_srt_content(s.content)}'" for i, s in enumerate(dialogue_srt_data) if not srt_assigned[i]]

        if unassigned_json_details:
            logging.info("\n--- PÁROSÍTATLAN JSON SZEGMENSEK ---")
            for detail in unassigned_json_details: logging.info(detail)
        
        if unassigned_srt_details:
            logging.info("\n--- PÁROSÍTATLAN SRT SOROK ---")
            for detail in unassigned_srt_details: logging.info(detail)
    
    logging.info("--- FELDOLGOZÁS BEFEJEZVE ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Forrás JSON időzítésének intelligens, 3-fázisú illesztése egy cél SRT tartalmával.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-project_name', required=True, help='A workdir-en belüli projektmappa neve.')
    parser.add_argument('-output_language', default='HU', help='A célnyelv kódja (pl. HU) a célnyelvi .txt/.srt fájl megkereséséhez.')
    parser.add_argument('-model', default='gpt-4o', help='A daraboláshoz használt OpenAI modell neve. (Alapértelmezett: gpt-4o)')
    parser.add_argument('-auth_key', required=False, help='OpenAI API kulcs. Ha nincs megadva, a keyholder.json-ból olvassa.')
    parser.add_argument('-max_midpoint_distance', type=float, default=1, help='1. FÁZIS: A max. megengedett távolság (mp) a 1:1 párosításhoz.')
    
    main(parser.parse_args())

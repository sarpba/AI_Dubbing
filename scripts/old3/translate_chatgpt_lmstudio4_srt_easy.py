import os
import argparse
from openai import OpenAI, OpenAIError
import json
import re
import time
import glob
import sys
import base64
import binascii
import hashlib
from datetime import timedelta
from pathlib import Path
import random

import srt  # pip install srt

# =========================
# SEGÉDFÜGGVÉNYEK / INFRA
# =========================

def get_project_root():
    p = Path(__file__).resolve().parent
    for _ in range(4):
        if (p / "config.json").exists():
            return str(p)
        p = p.parent
    return str(Path(__file__).resolve().parent.parent)

def load_config():
    config_path = os.path.join(get_project_root(), 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("Konfigurációs fájl sikeresen betöltve.")
        return config
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"Hiba a konfigurációs fájl betöltése közben ({config_path}): {e}")
        return None

def get_keyholder_path():
    return os.path.join(get_project_root(), 'keyholder.json')

def save_api_key(api_key: str):
    path = get_keyholder_path()
    try:
        data = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try: data = json.load(f)
                except json.JSONDecodeError: data = {}
        encoded_key = base64.b64encode(api_key.encode('utf-8')).decode('utf-8')
        data['chatgpt_api_key'] = encoded_key
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        try: os.chmod(path, 0o600)
        except Exception: pass
        print(f"ChatGPT API kulcs sikeresen elmentve a(z) '{path}' fájlba.")
    except Exception as e:
        print(f"Hiba az API kulcs mentése közben: {e}")

def load_api_key():
    path = get_keyholder_path()
    if not os.path.exists(path): return None
    try:
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
        encoded_key = data.get('chatgpt_api_key')
        if not encoded_key: return None
        return base64.b64decode(encoded_key.encode('utf-8')).decode('utf-8')
    except (json.JSONDecodeError, KeyError, binascii.Error, Exception) as e:
        print(f"Hiba az API kulcs betöltése közben: {e}")
        return None

def with_backoff(fn, max_retries=5, base=1.0, cap=8.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except OpenAIError as e:
            if attempt == max_retries - 1: raise
            sleep = min(cap, base * (2 ** attempt)) + random.uniform(0, 0.5)
            print(f"  -> API hiba ({e}). Újrapróbálás {sleep:.1f} mp múlva...")
            time.sleep(sleep)

# =========================
# SRT & FELDOLGOZÁSI LOGIKA
# =========================

def find_srt_context_file(upload_dir, lang_code):
    if not os.path.isdir(upload_dir):
        print(f"Figyelmeztetés: 'upload' könyvtár nem található: {upload_dir}.")
        return None
    search_pattern = os.path.join(upload_dir, f'*{lang_code.lower()}.srt')
    matching_files = glob.glob(search_pattern)
    if not matching_files: return None
    if len(matching_files) == 1:
        print(f"  -> SRT kontextusfájl megtalálva: {os.path.basename(matching_files[0])}")
        return matching_files[0]
    try:
        largest_file = max(matching_files, key=os.path.getsize)
        print(f"  -> Több SRT fájl található, a legnagyobb kiválasztva: {os.path.basename(largest_file)}")
        return largest_file
    except Exception as e:
        print(f"Hiba a legnagyobb fájl kiválasztása közben: {e}. Az első találat használata.")
        return matching_files[0]

def get_lang_name(lang_code):
    lang_map = {'EN': 'English', 'HU': 'Hungarian', 'DE': 'German', 'FR': 'French', 'ES': 'Spanish'}
    return lang_map.get(lang_code.upper(), lang_code)

def find_json_file(directory):
    try:
        json_files = [f for f in os.listdir(directory) if f.lower().endswith('.json')]
    except FileNotFoundError:
        return None, f"Bemeneti könyvtár nem található: {directory}"
    if not json_files: return None, f"Nem található JSON fájl: {directory}"
    if len(json_files) > 1: return None, f"Több JSON fájl található: {directory}"
    return json_files[0], None

def text_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode('utf-8')).hexdigest()[:16]

def create_smart_chunks(segments, min_items=30, max_items=120, gap_threshold=5.0, max_chars_per_batch=8000):
    chunks, i, n = [], 0, len(segments)
    while i < n:
        start, total_chars, best_split = i, 0, None
        end_limit = min(start + max_items, n)
        j = start
        while j < end_limit:
            seg_text = str(segments[j].get('text', ''))
            total_chars += len(seg_text) + 4
            if j - start + 1 >= min_items and total_chars >= max_chars_per_batch and best_split is not None: break
            if j + 1 < n:
                try:
                    if segments[j + 1].get('start', 0) - segments[j].get('end', 0) >= gap_threshold:
                        best_split = j + 1
                except Exception: pass
            j += 1
        best_split_point = best_split if best_split else min(j, n)
        chunks.append(segments[start:best_split_point])
        i = best_split_point
    return chunks

def load_srt_file(filepath):
    if not filepath or not os.path.exists(filepath):
        if filepath: print("Figyelmeztetés: SRT kontextus fájl nem található.")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            subtitles = list(srt.parse(f.read()))
        print(f"SRT fájl sikeresen betöltve: {os.path.basename(filepath)} ({len(subtitles)} felirat).")
        return subtitles
    except Exception as e:
        print(f"Figyelmeztetés: SRT fájl olvasása sikertelen ({e}).")
        return None

def clean_srt_content(text):
    text = re.sub(r'\{[^\}]*\}', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    return ' '.join(text.splitlines()).strip()

def get_srt_candidates_as_flow(batch_segments, srt_data, line_buffer):
    if not srt_data or not batch_segments: return ""
    try:
        batch_start_time, batch_end_time = batch_segments[0]['start'], batch_segments[-1]['end']
    except (KeyError, IndexError): return ""

    overlapping_indices = [i for i, sub in enumerate(srt_data) if sub.end.total_seconds() > batch_start_time and sub.start.total_seconds() < batch_end_time]

    if not overlapping_indices:
        print("  -> Nem található időben átfedő SRT sor a jelöltekhez.")
        return ""

    first_match_index, last_match_index = overlapping_indices[0], overlapping_indices[-1]
    start_index = max(0, first_match_index - line_buffer)
    end_index = min(len(srt_data), last_match_index + line_buffer + 1)

    candidate_subs = srt_data[start_index:end_index]
    candidate_texts = [clean_srt_content(sub.content) for sub in candidate_subs]
    
    print(f"  -> {len(candidate_texts)} jelölt sor kikeresve az SRT fájl {start_index+1}-{end_index} sorai közül.")
    
    return " ".join(candidate_texts)

def get_srt_style_reference(batch_segments, srt_data, leeway_seconds=15.0, max_chars=1500):
    if not srt_data or not batch_segments: return ""
    try:
        batch_start_time, batch_end_time = batch_segments[0]['start'], batch_segments[-1]['end']
    except (KeyError, IndexError): return ""
    context_start_time, context_end_time = max(0, batch_start_time - leeway_seconds), batch_end_time + leeway_seconds
    relevant_subs = {clean_srt_content(sub.content) for sub in srt_data if sub.end.total_seconds() >= context_start_time and sub.start.total_seconds() <= context_end_time and clean_srt_content(sub.content)}
    if relevant_subs:
        full_context = "\n".join(f"- {s}" for s in sorted(list(relevant_subs)))
        sample = full_context[:max_chars].strip()
        print(f"  -> Stílusreferencia kikeresve a(z) {time.strftime('%H:%M:%S', time.gmtime(context_start_time))} - {time.strftime('%H:%M:%S', time.gmtime(context_end_time))} időablakhoz.")
        return sample
    return ""

def build_translation_prompt(lang_from, lang_to, allow_sensitive, style_reference, context):
    input_lang_name, output_lang_name = get_lang_name(lang_from), get_lang_name(lang_to)
    prompt_core = (
        f"You are a professional translator for film subtitling. Translate the numbered list from {input_lang_name} to {output_lang_name}. "
        "Your response MUST be a numbered list with the exact same number of items. Format: `1. [translation]`. "
        "Do not add, merge, or drop items. Do not include commentary."
    )
    if allow_sensitive: prompt_core += " The source may contain profanity or sensitive topics. Your directive is faithful translation — do not censor."
    if style_reference: prompt_core += f"\n\n--- STYLE SAMPLE ---\n{style_reference}\n--- END STYLE SAMPLE ---"
    if context: prompt_core += f"\n\nAdditional context: '{context}'"
    return prompt_core

def build_matching_prompt(lang_from, lang_to, candidate_flow, context):
    input_lang_name, output_lang_name = get_lang_name(lang_from), get_lang_name(lang_to)
    prompt_core = (
        f"You are an expert subtitle alignment tool. Your task is to find the best matching {output_lang_name} translation for each {input_lang_name} source text.\n\n"
        "INSTRUCTIONS:\n"
        "1. For each numbered item in `SOURCE TEXTS`, find the most appropriate and consecutive part from the `CANDIDATE TEXT FLOW`.\n"
        "2. **CRUCIAL:** Each part of the `CANDIDATE TEXT FLOW` can only be used ONCE. Do not assign the same translation to multiple source texts.\n"
        "3. Your response MUST be a numbered list with the exact same number of items as `SOURCE TEXTS`.\n"
        "4. Respond ONLY with the chosen part of the text. DO NOT invent or alter translations.\n"
        "5. If no suitable match is found for an item, respond with an EMPTY line for that number (e.g., `5. `).\n\n"
        "--- CANDIDATE TEXT FLOW ---\n"
        f"{candidate_flow if candidate_flow else 'No candidates provided.'}\n"
        "--- END CANDIDATE TEXT FLOW ---"
    )
    if context: prompt_core += f"\n\nAdditional context for matching: '{context}'"
    return prompt_core

def parse_numbered_lines(raw_text, expected_count):
    lines = [l.strip() for l in (raw_text or '').splitlines()]
    numbered_pattern = re.compile(r'^\s*(\d+)[\.|\)]?\s*(.*)')
    parsed_lines = {}
    for line in lines:
        match = numbered_pattern.match(line)
        if match:
            num, content = int(match.group(1)), match.group(2).strip()
            parsed_lines[num] = content
    result = [parsed_lines.get(i + 1, '') for i in range(expected_count)]
    if len(lines) > 0 and len(parsed_lines) == 0:
        return None
    return result

def process_batch(client, batch_segments, args, srt_data, batch_id_str):
    if not batch_segments: return []
    
    numbered_texts = [f"{i+1}. {seg.get('text', '')}" for i, seg in enumerate(batch_segments)]
    user_prompt_content = '\n'.join(numbered_texts)
    
    system_prompt = ""
    if args.match_from_srt:
        candidate_flow = get_srt_candidates_as_flow(batch_segments, srt_data, args.line_buffer)
        if not candidate_flow:
            print("  -> FIGYELEM: Párosítási mód aktív, de nincsenek SRT jelöltek ehhez a köteghez. A köteg kihagyásra kerül.")
            return [''] * len(batch_segments)
        system_prompt = build_matching_prompt(args.input_language, args.output_language, candidate_flow, args.context)
        user_prompt_content = f"\n--- SOURCE TEXTS ---\n{user_prompt_content}"
    else:
        style_reference = get_srt_style_reference(batch_segments, srt_data)
        system_prompt = build_translation_prompt(args.input_language, args.output_language, args.allow_sensitive_content, style_reference, args.context)

    if args.save_prompt_to_file:
        try:
            full_prompt_for_saving = (
                f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n--- USER PROMPT ---\n{user_prompt_content}\n"
            )
            with open(args.save_prompt_to_file, 'w', encoding='utf-8') as f:
                f.write(full_prompt_for_saving)
            print(f"\n✅ Kérés sikeresen elmentve: '{args.save_prompt_to_file}'. A szkript leáll.")
            sys.exit(0)
        except IOError as e:
            print(f"\nHIBA: A prompt fájl írása sikertelen: {e}")
            sys.exit(1)

    def _call():
        return client.chat.completions.create(model=args.model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_content}], temperature=0.0)
    
    try:
        response = with_backoff(_call)
        raw = response.choices[0].message.content if response and response.choices else ''
        parsed = parse_numbered_lines(raw, len(batch_segments))
        
        if parsed is None:
            print(f"  HIBA: [{batch_id_str}] csoport válaszának feldolgozása sikertelen. Felezés.")
            return None
        
        if args.stream:
            for i, line in enumerate(parsed): print(f"    {i+1}. -> {line if line else '[NINCS TALÁLAT]'}")

        return parsed
    except OpenAIError as e:
        print(f"  HIBA: API hiba [{batch_id_str}]: {e}")
        return None

def pre_translate_from_srt(all_segments, srt_data, threshold):
    if not srt_data:
        print("  -> Nincs SRT adat, az időalapú elő-fordítás kihagyva.")
        return {}, all_segments
    print(f"\nIdőalapú elő-fordítási kísérlet (küszöb: {threshold}s)...")
    srt_midpoints = [((sub.start.total_seconds() + sub.end.total_seconds()) / 2, clean_srt_content(sub.content)) for sub in srt_data if clean_srt_content(sub.content)]
    pretranslated_progress, remaining_segments = {}, []
    available_srt = srt_midpoints[:]
    for segment in all_segments:
        text = (segment.get('text') or '').strip()
        if not text: continue
        seg_start, seg_end = segment.get('start'), segment.get('end')
        if seg_start is None or seg_end is None:
            remaining_segments.append(segment)
            continue
        seg_midpoint = (seg_start + seg_end) / 2
        best_match, smallest_diff = None, float('inf')
        for srt_mid, srt_text in available_srt:
            diff = abs(seg_midpoint - srt_mid)
            if diff < smallest_diff:
                smallest_diff, best_match = diff, (srt_mid, srt_text)
        if best_match and smallest_diff <= threshold:
            _, srt_text_match = best_match
            pretranslated_progress[str(segment['original_index'])] = {"translated_text": srt_text_match, "text_hash": text_hash(text), "source": "pre-translated-from-srt"}
            available_srt.remove(best_match)
        else:
            remaining_segments.append(segment)
    match_count = len(pretranslated_progress)
    if match_count > 0: print(f"  -> Sikeresen párosítva és elő-fordítva {match_count} szegmens időalapon.")
    else: print("  -> Nem található időbélyeg-alapú egyezés az SRT fájllal.")
    return pretranslated_progress, remaining_segments

def main(args):
    # Kliens beállítása
    client = None
    if args.use_openai:
        print("Mód: OpenAI API")
        auth_key = args.auth_key or load_api_key()
        if not auth_key:
            print("\nHIBA: OpenAI használatához API kulcs szükséges.")
            sys.exit(1)
        if args.auth_key: save_api_key(args.auth_key)
        client = OpenAI(api_key=auth_key)
    else:
        print("Mód: Helyi szerver (LM Studio)")
        client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")
    
    # Feladat meghatározása
    if args.match_from_srt: print("Feladat: Szemantikus párosítás SRT alapján")
    else: print("Feladat: Fordítás")
    print(f"Használt modell: {args.model}")

    # Konfig és útvonalak
    config = load_config()
    if not config: sys.exit(1)
    try:
        workdir, subdirs = config['DIRECTORIES']['workdir'], config['PROJECT_SUBDIRS']
        input_dir = os.path.join(workdir, args.project_name, subdirs['separated_audio_speech'])
        output_dir = os.path.join(workdir, args.project_name, subdirs['translated'])
        upload_dir = os.path.join(workdir, args.project_name, subdirs['upload'])
    except KeyError as e:
        print(f"Hiba a config.json-ban: {e}")
        sys.exit(1)

    input_filename, error = find_json_file(input_dir)
    if error:
        print(f"Leállás: {error}")
        sys.exit(1)
    
    input_filepath = os.path.join(input_dir, input_filename)
    output_filepath = os.path.join(output_dir, input_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Bemeneti fájl: {input_filepath}")
    
    srt_context_filepath = find_srt_context_file(upload_dir, args.output_language)
    srt_data = load_srt_file(srt_context_filepath)

    if args.match_from_srt and not srt_data:
        print("\nHIBA: A `--match-from-srt` módhoz kötelező egy SRT fájl a kimeneti nyelven. Nem található.")
        sys.exit(1)
        
    progress_filepath = os.path.join(output_dir, f"{os.path.splitext(input_filename)[0]}.progress.json")
    progress = {}
    if os.path.exists(progress_filepath):
        try:
            with open(progress_filepath, 'r', encoding='utf-8') as f: progress = json.load(f)
            print(f"Haladási fájl betöltve ({len(progress)} elem).")
        except (json.JSONDecodeError, IOError): progress = {}

    with open(input_filepath, 'r', encoding='utf-8') as f: data = json.load(f)
    if 'segments' not in data or not isinstance(data['segments'], list):
        print("Hiba: JSON 'segments' kulcs hiányzik.")
        sys.exit(1)

    all_segments, untranslated_segments = data['segments'], []
    for i, segment in enumerate(all_segments):
        segment['original_index'] = i
        
    if args.pre_translate_from_srt:
        pretranslated_data, segments_to_process_further = pre_translate_from_srt(all_segments, srt_data, args.pre_translate_threshold)
        progress.update(pretranslated_data)
        all_segments = segments_to_process_further

    for segment in all_segments:
        text = (segment.get('text') or '').strip()
        if text and str(segment['original_index']) not in progress:
            segment['_text_hash'] = text_hash(text)
            untranslated_segments.append(segment)

    if not untranslated_segments:
        print("\nNincs új, AI által feldolgozandó szegmens.")
    else:
        print(f"\n{len(untranslated_segments)} új szegmens feldolgozása következik.")
        batches = create_smart_chunks(untranslated_segments)
        print(f"{len(batches)} csoportot kell feldolgozni.")

        for i, batch in enumerate(batches):
            print(f"\n[{i+1}/{len(batches)}] csoport feldolgozása...")
            processed_lines = process_batch(client, batch, args, srt_data, f"{i+1}")
            
            if processed_lines is None:
                if len(batch) <= 1:
                    print("  -> Csoport 1 elemű, hibás. Kihagyva.")
                    progress[str(batch[0]['original_index'])] = {"translated_text": "[HIBA: FELDOLGOZATLAN]", "text_hash": text_hash(batch[0].get('text',''))}
                    continue
                mid = len(batch) // 2
                first_half, second_half = batch[:mid], batch[mid:]
                res1 = process_batch(client, first_half, args, srt_data, f"{i+1}-A")
                res2 = process_batch(client, second_half, args, srt_data, f"{i+1}-B")
                if res1 is None or res2 is None:
                    print("HIBA: A feldolgozás megszakadt a felezés során.")
                    sys.exit(1)
                processed_lines = res1 + res2

            for segment_obj, line in zip(batch, processed_lines):
                progress[str(segment_obj['original_index'])] = {"translated_text": line, "text_hash": segment_obj.get('_text_hash','')}
            
            with open(progress_filepath, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            print(f"Haladás elmentve.")

    print("\nFeldolgozás befejezve.")

    for idx, segment in enumerate(data['segments']):
        entry = progress.get(str(idx))
        if isinstance(entry, dict):
            segment['translated_text'] = entry.get('translated_text', '')
        elif isinstance(entry, str):
            segment['translated_text'] = entry
        else:
            segment.setdefault('translated_text', '')

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nKiegészített fájl mentve: '{output_filepath}'")

    if os.path.exists(progress_filepath):
        try:
            os.remove(progress_filepath)
            print("Ideiglenes haladási fájl törölve.")
        except Exception as e:
            print(f"Figyelmeztetés: haladási fájl törlése sikertelen: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Szövegszegmensek fordítása vagy párosítása SRT-ből.', formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('-project_name', required=True, help='A workdir-en belüli projektmappa neve.')
    parser.add_argument('-input_language', default='EN', help='Bemeneti nyelv kódja.')
    parser.add_argument('-output_language', default='HU', help='Kimeneti nyelv kódja.')
    parser.add_argument('-context', required=False, help='Általános kontextus a feladathoz.')

    mode_group = parser.add_argument_group('Task Mode')
    mode_group.add_argument('--match-from-srt', action='store_true', help='Fordítás helyett szemantikus párosítás a meglévő SRT fájlból.')
    
    api_group = parser.add_argument_group('API and Model Configuration')
    api_group.add_argument('--use-openai', action='store_true', help='Helyi szerver helyett az OpenAI API használata.')
    api_group.add_argument('-auth_key', required=False, help='OpenAI API kulcs.')
    api_group.add_argument('-model', default='unsloth/gpt-oss-20b', required=False, help='Használni kívánt modell neve.')
    
    matching_group = parser.add_argument_group('Matching Mode Options')
    matching_group.add_argument('--line-buffer', type=int, default=2, help='Hány sornyi ráhagyás legyen a párosításhoz az SRT-ből. Alap: 2')
    
    pre_trans_group = parser.add_argument_group('Time-based Pre-translation (Optional)')
    pre_trans_group.add_argument('--pre-translate-from-srt', action='store_true', help='Időbélyeg-alapú elő-fordítás/párosítás bekapcsolása.')
    pre_trans_group.add_argument('--pre-translate-threshold', type=float, default=0.3, help='Időbélyeg-eltérés küszöbe. Alap: 0.3s')

    other_group = parser.add_argument_group('Other and Debugging')
    other_group.add_argument('-stream', action='store_true', help='Soronkénti kimenet a konzolon.')
    other_group.add_argument('--save-prompt-to-file', type=str, metavar='FNAME', help='Elmenti az első AI kérést és kilép.')
    try:
        other_group.add_argument('--allow-sensitive-content', action=argparse.BooleanOptionalAction, default=True, help='Fordítási módban: hű fordítás kényes témáknál is.')
    except Exception:
        other_group.add_argument('--allow-sensitive-content', action='store_true', default=True)

    args = parser.parse_args()
    main(args)
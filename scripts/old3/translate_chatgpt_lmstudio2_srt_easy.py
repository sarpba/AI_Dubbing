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
    """Robusztusabb gyökér-felderítés: a fájlhoz képest megyünk feljebb, és keressük a config.json-t.
    Ha nem találjuk 3 szintenként, fallback: a szkript szülője."""
    p = Path(__file__).resolve().parent
    for _ in range(4):
        if (p / "config.json").exists():
            return str(p)
        p = p.parent
    return str(Path(__file__).resolve().parent.parent)


def load_config():
    """Betölti a config.json-t a projekt gyökeréből."""
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
    """Elmenti az API kulcsot base64 kódolással a keyholder.json-ba."""
    path = get_keyholder_path()
    try:
        data = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print("Figyelmeztetés: A keyholder.json sérült vagy üres, új fájl jön létre.")
                    data = {}

        encoded_key = base64.b64encode(api_key.encode('utf-8')).decode('utf-8')
        data['chatgpt_api_key'] = encoded_key

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        print(f"ChatGPT API kulcs sikeresen elmentve a(z) '{path}' fájlba.")
    except Exception as e:
        print(f"Hiba az API kulcs mentése közben: {e}")


def load_api_key():
    """Betölti és dekódolja az API kulcsot a keyholder.json-ból."""
    path = get_keyholder_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        encoded_key = data.get('chatgpt_api_key')
        if not encoded_key:
            return None
        decoded_key = base64.b64decode(encoded_key.encode('utf-8')).decode('utf-8')
        return decoded_key
    except (json.JSONDecodeError, KeyError, binascii.Error, Exception) as e:
        print(f"Hiba az API kulcs betöltése közben a(z) '{path}' fájlból: {e}")
        return None


def with_backoff(fn, max_retries=5, base=1.0, cap=8.0):
    """Exponenciális backoff wrapper OpenAI hívásokhoz."""
    for attempt in range(max_retries):
        try:
            return fn()
        except OpenAIError as e:
            if attempt == max_retries - 1:
                raise
            sleep = min(cap, base * (2 ** attempt)) + random.uniform(0, 0.5)
            print(f"  -> API hiba ({e}). Újrapróbálás {sleep:.1f} mp múlva...")
            time.sleep(sleep)


# =========================
# SRT & FORDÍTÁSI LOGIKA
# =========================

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
        print(f"Hiba: A bemeneti könyvtár nem található: {directory}")
        return None, "directory_not_found"
    if not json_files:
        print(f"Hiba: Nem található JSON fájl a(z) '{directory}' könyvtárban.")
        return None, "no_json_found"
    if len(json_files) > 1:
        print(f"Hiba: Több JSON fájl található a(z) '{directory}' könyvtárban.")
        return None, "multiple_jsons_found"
    return json_files[0], None

def text_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode('utf-8')).hexdigest()[:16]

def create_smart_chunks(segments, min_items=30, max_items=120, gap_threshold=5.0, max_chars_per_batch=8000):
    chunks = []
    i = 0
    n = len(segments)
    while i < n:
        start = i
        total_chars = 0
        end_limit = min(start + max_items, n)
        j = start
        best_split = None
        while j < end_limit:
            seg = segments[j]
            seg_text = str(seg.get('text', ''))
            total_chars += len(seg_text) + 4
            if j - start + 1 >= min_items and total_chars >= max_chars_per_batch and best_split is not None:
                break
            if j + 1 < n:
                try:
                    gap = segments[j + 1].get('start', 0) - segments[j].get('end', 0)
                    if gap >= gap_threshold:
                        best_split = j + 1
                except Exception:
                    pass
            j += 1
        if best_split is None:
            best_split = min(j, n)
        chunks.append(segments[start:best_split])
        i = best_split
    return chunks

def load_srt_file(filepath):
    if not filepath or not os.path.exists(filepath):
        if filepath:
            print("  -> Figyelmeztetés: Az SRT kontextus fájl nem található a megadott útvonalon.")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            subtitles = list(srt.parse(f.read()))
        print(f"  -> SRT fájl sikeresen betöltve és feldolgozva: {os.path.basename(filepath)} ({len(subtitles)} felirat).")
        return subtitles
    except Exception as e:
        print(f"  -> Figyelmeztetés: Az SRT fájl olvasása sikertelen ({e}). Folytatás SRT kontextus nélkül.")
        return None

def clean_srt_content(text):
    text = re.sub(r'\{[^\}]*\}', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    # Az SRT-ből való párosításkor a sortöréseket is cseréljük szóközre
    return ' '.join(text.splitlines()).strip()

def get_dynamic_srt_context(batch_segments, srt_data, leeway_seconds=15.0, max_chars=1500):
    if not srt_data or not batch_segments:
        return ""
    try:
        batch_start_time = batch_segments[0]['start']
        batch_end_time = batch_segments[-1]['end']
    except (KeyError, IndexError):
        return ""

    context_start_time = max(0, batch_start_time - leeway_seconds)
    context_end_time = batch_end_time + leeway_seconds
    relevant_subs = []
    for sub in srt_data:
        sub_start_sec = sub.start.total_seconds()
        sub_end_sec = sub.end.total_seconds()
        if sub_end_sec >= context_start_time and sub_start_sec <= context_end_time:
            cleaned_content = clean_srt_content(sub.content)
            if cleaned_content:
                relevant_subs.append(cleaned_content)
    if relevant_subs:
        full_context = " ".join(relevant_subs)
        sample = full_context[:max_chars].strip()
        print(
            f"  -> Dinamikus SRT kontextus kikeresve a(z) {time.strftime('%H:%M:%S', time.gmtime(context_start_time))} - "
            f"{time.strftime('%H:%M:%S', time.gmtime(context_end_time))} időablakhoz ({len(relevant_subs)} sor)."
        )
        return sample
    return ""

def build_system_prompt(lang_from, lang_to, allow_sensitive, dynamic_srt_context, context):
    input_lang_name, output_lang_name = get_lang_name(lang_from), get_lang_name(lang_to)
    prompt_core = (
        f"You are a professional translator for film subtitling. Translate the numbered list from {input_lang_name} to {output_lang_name}. "
        "Your response MUST be a numbered list with the exact same number of items. Format: `1. [translation]`."
        " Do not add, merge, or drop items. Do not include commentary."
    )
    if allow_sensitive:
        prompt_core += " The source may contain profanity or sensitive topics. Your directive is faithful translation — do not censor."
    if dynamic_srt_context:
        prompt_core += (
            "\n\nTo ensure consistency, adopt a style and terminology similar to this sample from the same scene.\n"
            "IMPORTANT: Do NOT translate this sample; it's only a style reference.\n"
            "--- STYLE SAMPLE ---\n"
            f"{dynamic_srt_context}\n"
            "--- END STYLE SAMPLE ---"
        )
    if context:
        prompt_core += f"\n\nAdditional context: '{context}'"
    return prompt_core

def parse_numbered_lines(raw_text, expected_count):
    lines = [l.strip() for l in (raw_text or '').splitlines() if l.strip()]
    numbered = [l for l in lines if re.match(r'^\s*\d+[\.|\)]\s+', l)]
    if len(numbered) != expected_count:
        return None
    cleaned = [re.sub(r'^\s*\d+[\.|\)]\s*', '', l).strip() for l in numbered]
    return cleaned

def translate_batch(client, batch_segments, lang_from, lang_to, context, srt_data, model, stream, allow_sensitive, batch_id_str):
    if not batch_segments:
        return []
    numbered_texts = [f"{i+1}. {seg.get('text', '')}" for i, seg in enumerate(batch_segments)]
    text_block = '\n'.join(numbered_texts)
    dynamic_srt_context = get_dynamic_srt_context(batch_segments, srt_data)
    system_prompt = build_system_prompt(lang_from, lang_to, allow_sensitive, dynamic_srt_context, context)

    def _call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text_block}],
            temperature=0.1,
        )
    try:
        response = with_backoff(_call)
        raw = response.choices[0].message.content if response and response.choices else ''
        parsed = parse_numbered_lines(raw, len(batch_segments))
        if parsed is None:
            tightened_prompt = (
                "Return ONLY a numbered list with EXACTLY the same number of lines as the input. "
                "No extra lines, no commentary, no code fences. Format: `1. ...`"
            )
            def _call_tight():
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": tightened_prompt}, {"role": "user", "content": text_block}],
                    temperature=0.0,
                )
            response2 = with_backoff(_call_tight)
            raw2 = response2.choices[0].message.content if response2 and response2.choices else ''
            parsed = parse_numbered_lines(raw2, len(batch_segments))
        if parsed is None:
            print(f"  HIBA: A(z) [{batch_id_str}] csoportnál sorszám-eltérés maradt. Felezés következik.")
            return None
        if stream:
            for line in parsed:
                print(f"    + {line}")
        return parsed
    except OpenAIError as e:
        print(f"  HIBA: API hiba a(z) [{batch_id_str}] csoportnál: {e}")
        return None

# =========================
# ÚJ FUNKCIÓ: ELŐ-FORDÍTÁS
# =========================

def pre_translate_from_srt(all_segments, srt_data, threshold):
    """
    Összepárosítja a JSON szegmenseket és az SRT feliratokat az időbélyegek
    középidejének összehasonlításával. Ha egyezést talál, a haladási
    fájlba írja az SRT szövegét, és a szegmenst eltávolítja a további
    fordítási listából.

    Visszatér:
    - pretranslated_progress: Egy dict a párosított szegmensekkel.
    - remaining_segments: A nem párosított szegmensek listája.
    """
    if not srt_data:
        print("  -> Nincs SRT adat, az elő-fordítási lépés kihagyva.")
        return {}, all_segments

    print(f"\nElő-fordítási kísérlet az SRT fájl alapján (küszöb: {threshold}s)...")
    
    # SRT adatok gyorsítótárazása a hatékony kereséshez
    srt_midpoints = [
        ((sub.start.total_seconds() + sub.end.total_seconds()) / 2, clean_srt_content(sub.content))
        for sub in srt_data if clean_srt_content(sub.content)
    ]

    pretranslated_progress = {}
    remaining_segments = []
    
    # Másolatot készítünk, hogy ne módosítsuk az eredeti SRT listát
    available_srt = srt_midpoints[:]
    
    for i, segment in enumerate(all_segments):
        segment['original_index'] = i
        text = (segment.get('text') or '').strip()
        if not text:
            continue
            
        seg_start = segment.get('start')
        seg_end = segment.get('end')
        
        if seg_start is None or seg_end is None:
            remaining_segments.append(segment)
            continue
            
        seg_midpoint = (seg_start + seg_end) / 2
        
        best_match = None
        smallest_diff = float('inf')
        
        for srt_mid, srt_text in available_srt:
            diff = abs(seg_midpoint - srt_mid)
            if diff < smallest_diff:
                smallest_diff = diff
                best_match = (srt_mid, srt_text)

        if best_match and smallest_diff <= threshold:
            # Találtunk egyezést
            srt_mid_match, srt_text_match = best_match
            pretranslated_progress[str(i)] = {
                "translated_text": srt_text_match,
                "text_hash": text_hash(text),
                "source": "pre-translated-from-srt"
            }
            # Eltávolítjuk a párosított SRT-t, hogy ne használjuk újra
            available_srt.remove(best_match)
        else:
            remaining_segments.append(segment)

    match_count = len(pretranslated_progress)
    if match_count > 0:
        print(f"  -> Sikeresen párosítva és elő-fordítva {match_count} szegmens az SRT fájlból.")
    else:
        print("  -> Nem található időbélyeg-alapú egyezés az SRT fájllal.")
        
    return pretranslated_progress, remaining_segments

# =========================
# FŐ FOLYAMAT
# =========================

def main(args):
    # --- KLIENS BEÁLLÍTÁSA ---
    client = None
    if args.use_openai:
        print("OpenAI API használata kiválasztva (`--use-openai`).")
        auth_key = args.auth_key or load_api_key()
        if not auth_key:
            print("\nHIBA: Az --use-openai használatához API kulcs szükséges.")
            print("Kérjük, adja meg a kulcsot az `-auth_key` argumentummal, vagy mentse el korábban.")
            sys.exit(1)
        if args.auth_key:
            save_api_key(args.auth_key)
        if 'gpt-' not in args.model.lower() and 'dall-e' not in args.model.lower():
             print(f"FIGYELMEZTETÉS: Az OpenAI API-t használja, de a modell neve ('{args.model}') nem tűnik OpenAI modellnek (pl. 'gpt-4o').")
        client = OpenAI(api_key=auth_key)
        print("OpenAI API kliens sikeresen beállítva.")
    else:
        print("Alapértelmezett módban a szkript a helyi LM Studiohoz csatlakozik.")
        print("Cím: http://127.0.0.1:1234")
        client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")
        print("LM Studio kliens beállítva.")
    print(f"Használt modell: {args.model}")

    # --- KONFIG/ÚTVONALAK ---
    config = load_config()
    if not config:
        sys.exit(1)
    try:
        workdir = config['DIRECTORIES']['workdir']
        subdirs = config['PROJECT_SUBDIRS']
        input_dir = os.path.join(workdir, args.project_name, subdirs['separated_audio_speech'])
        output_dir = os.path.join(workdir, args.project_name, subdirs['translated'])
        upload_dir = os.path.join(workdir, args.project_name, subdirs['upload'])
    except KeyError as e:
        print(f"Hiba: Hiányzó kulcs a config.json fájlban: {e}")
        return

    input_filename, error = find_json_file(input_dir)
    if error:
        print(f"Leállás a hiba miatt: {error}")
        return
    input_filepath = os.path.join(input_dir, input_filename)
    print(f"Bemeneti fájl feldolgozása: {input_filepath}")
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, input_filename)

    print(f"\nStíluskontextus keresése a kimeneti nyelv ('{args.output_language.upper()}') alapján a(z) '{upload_dir}' mappában...")
    srt_context_filepath = find_srt_context_file(upload_dir, args.output_language)
    srt_data = load_srt_file(srt_context_filepath)

    progress_filename = f"{os.path.splitext(input_filename)[0]}.progress.json"
    progress_filepath = os.path.join(output_dir, progress_filename)
    progress = {}
    if os.path.exists(progress_filepath):
        try:
            with open(progress_filepath, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            print(f"\nHaladási fájl betöltve: {progress_filepath} ({len(progress)} szegmens már lefordítva).")
        except (json.JSONDecodeError, IOError):
            progress = {}

    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'segments' not in data or not isinstance(data['segments'], list):
        print("Hiba: JSON 'segments' kulcs hiányzik vagy nem lista.")
        return

    all_segments = data['segments']
    untranslated_segments = []

    if args.pre_translate_from_srt:
        pretranslated_data, segments_to_process_further = pre_translate_from_srt(all_segments, srt_data, args.pre_translate_threshold)
        progress.update(pretranslated_data)
        # Azok a szegmensek maradnak, amiket még ellenőrizni kell a progress fájl alapján
        all_segments = segments_to_process_further

    # A maradék szegmensek szűrése a meglévő (nem elő-fordított) haladás alapján
    for segment in all_segments:
        text = (segment.get('text') or '').strip()
        if text and str(segment.get('original_index')) not in progress:
            segment['_text_hash'] = text_hash(text)
            untranslated_segments.append(segment)

    if not untranslated_segments:
        print("\nNincs új, AI által fordítandó szegmens.")
    else:
        print(f"\n{len(untranslated_segments)} új, AI által fordítandó szegmens található.")
        batches = create_smart_chunks(untranslated_segments)
        total_batches = len(batches)
        print(f"Intelligens csoportképzés befejezve. {total_batches} új csoportot kell feldolgozni.")

        for i, batch in enumerate(batches):
            print(f"\n[{i+1}/{total_batches}] fő csoport feldolgozása...")
            translated_batch_lines = translate_batch(client, batch, args.input_language, args.output_language, args.context, srt_data, args.model, args.stream, args.allow_sensitive_content, f"{i+1}")
            if translated_batch_lines is None:
                if len(batch) <= 1:
                    print("  -> A csoport 1 elemű és továbbra is hibás. Kihagyva.")
                    continue
                mid = len(batch) // 2
                first_half, second_half = batch[:mid], batch[mid:]
                first_res = translate_batch(client, first_half, args.input_language, args.output_language, args.context, srt_data, args.model, args.stream, args.allow_sensitive_content, f"{i+1}-A")
                second_res = translate_batch(client, second_half, args.input_language, args.output_language, args.context, srt_data, args.model, args.stream, args.allow_sensitive_content, f"{i+1}-B")
                if first_res is None or second_res is None:
                    print("\nA fordítási folyamat megszakadt ezen a batchen.")
                    return
                translated_batch_lines = first_res + second_res
            for segment_obj, translated_line in zip(batch, translated_batch_lines):
                progress[str(segment_obj['original_index'])] = {
                    "translated_text": translated_line,
                    "text_hash": segment_obj.get('_text_hash') or text_hash(segment_obj.get('text', '')),
                }
            with open(progress_filepath, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            print(f"  Haladás elmentve a(z) {progress_filepath} fájlba.")

    print("\nAz összes szükséges fordítás elkészült.")

    # Kimenet összeállítása
    for idx, segment in enumerate(data['segments']):
        entry = progress.get(str(idx))
        if isinstance(entry, dict):
            segment['translated_text'] = entry.get('translated_text', '')
            if entry.get('source') != 'pre-translated-from-srt':
                old_hash = entry.get('text_hash')
                new_hash = text_hash(segment.get('text', ''))
                if old_hash and new_hash and old_hash != new_hash:
                    print(f"Figyelmeztetés: A(z) {idx}. szegmens forrásszövege megváltozott a fordítás óta (hash mismatch).")
        elif isinstance(entry, str):
            segment['translated_text'] = entry
        else:
            segment.setdefault('translated_text', '')

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nFordítás befejezve. A kiegészített fájl a(z) '{output_filepath}' helyre mentve.")

    if os.path.exists(progress_filepath):
        try:
            os.remove(progress_filepath)
            print(f"Az ideiglenes haladási fájl ({progress_filepath}) törölve.")
        except Exception:
            print(f"Figyelmeztetés: A haladási fájl ({progress_filepath}) törlését nem sikerült végrehajtani.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Szövegszegmensek intelligens, folytatható, rekurzív fordítása dinamikus SRT stíluskontextussal.\n'
            'Alapértelmezetten helyi LM Studio-t használ (http://127.0.0.1:1234).\n'
            'Az `--use-openai` kapcsolóval átváltható a hivatalos OpenAI API használatára.'
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Core arguments
    parser.add_argument('-project_name', required=True, help='A workdir-en belüli projektmappa neve.')
    parser.add_argument('-input_language', default='EN', help='A bemeneti nyelv kódja (pl. EN, HU). Alapértelmezett: EN')
    parser.add_argument('-output_language', default='HU', help='A kimeneti nyelv kódja (pl. EN, HU). Alapértelmezett: HU')
    parser.add_argument('-context', required=False, help='Rövid, általános kontextus a fordításhoz (pl. "sci-fi sorozat").')

    # Model and API arguments
    api_group = parser.add_argument_group('API and Model Configuration')
    api_group.add_argument('--use-openai', action='store_true', help='Helyi LM Studio helyett az OpenAI API használata. Szükséges hozzá API kulcs.')
    api_group.add_argument('-auth_key', required=False, help='Opcionális OpenAI API kulcs. Az `--use-openai` kapcsolóval együtt szükséges.')
    api_group.add_argument('-model', default='openai/gpt-oss-20b', required=False, help='A használni kívánt modell neve. Alapértelmezett: openai/gpt-oss-20b')

    # Pre-translation arguments
    pre_trans_group = parser.add_argument_group('Pre-translation from SRT')
    pre_trans_group.add_argument('--pre-translate-from-srt', action='store_true', help='Bekapcsolja az időbélyeg-alapú elő-fordítást a meglévő SRT fájlból.')
    pre_trans_group.add_argument('--pre-translate-threshold', type=float, default=0.3, help='Az időbélyeg-középeltérés maximuma másodpercben az elő-fordításhoz. Alapértelmezett: 0.3')

    # Other arguments
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument('-stream', action='store_true', help='Bekapcsolja a soronkénti kimenetet a konzolon (nem valódi API stream).')
    try:
        other_group.add_argument(
            '--allow-sensitive-content', action=argparse.BooleanOptionalAction, default=True,
            help='Hű fordítás kényes tartalmaknál is. Kikapcsolás: --no-allow-sensitive-content'
        )
    except Exception:
        other_group.add_argument('--allow-sensitive-content', action='store_true', default=True)

    args = parser.parse_args()
    main(args)

import os
import sys
import json
import srt
import hashlib
import argparse
from datetime import timedelta
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Konfiguráció ---
API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = "deepseek-chat"
CACHE_DIR = ".translation_cache"

# --- Inicializálás ---
if not API_KEY:
    print("Hiba: A DEEPSEEK_API_KEY környezeti változó nincs beállítva.")
    sys.exit(1)

# A DeepSeek kliens inicializálása
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")

# Gyorsítótár mappa létrehozása, ha nem létezik
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(text, context):
    """Egyedi kulcsot generál a kéréshez a tartalom alapján."""
    return hashlib.sha256((text + context).encode('utf-8')).hexdigest()

def load_srt_file(srt_path):
    """Beolvassa az SRT fájlt és visszaadja a feliratokat."""
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return list(srt.parse(content))
    except FileNotFoundError:
        print(f"Hiba: Az SRT fájl nem található: {srt_path}")
        return None
    except Exception as e:
        print(f"Hiba az SRT fájl feldolgozása közben: {e}")
        return None

def find_srt_text_for_segment(segment, subtitles):
    """Megkeresi a JSON szegmens időintervallumának megfelelő SRT feliratokat."""
    segment_start = timedelta(seconds=segment['start'])
    segment_end = timedelta(seconds=segment['end'])
    
    matching_texts = [sub.content.strip() for sub in subtitles if max(segment_start, sub.start) < min(segment_end, sub.end)]
            
    return "\n".join(matching_texts) if matching_texts else None

def get_improved_translation(original_text, existing_translation, srt_context, verbose=False):
    """Meghívja a DeepSeek API-t, hogy javított fordítást kérjen, cache használattal."""
    
    cache_key = get_cache_key(original_text, srt_context)
    cache_file_path = os.path.join(CACHE_DIR, cache_key)

    # 1. Próbálkozás a gyorsítótárból olvasni
    if os.path.exists(cache_file_path):
        if verbose:
            print("[CACHE HIT]")
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    # 2. Ha nincs a cache-ben, hívjuk az API-t
    if verbose:
        print("[API CALL]")
        
    prompt = f"""
    Feladat: Fordítás javítása az SRT kontextus alapján.
    Adott egy eredeti angol szöveg, egy meglévő magyar fordítás, és a pontosabb, időzített magyar felirat (SRT) ugyanahhoz a jelenethez.
    A célod, hogy az SRT kontextust felhasználva adj egy jobb, természetesebb magyar fordítást.

    - Az SRT a mérvadó a stílus, a szóhasználat és a pontosság tekintetében.
    - Ha a meglévő fordítás már jó és megegyezik az SRT-vel, add vissza azt.
    - Ha az SRT tartalma pontosabb, természetesebb vagy jobban illik a kontextusba, használd azt alapul.
    - A válaszod KIZÁRÓLAG a végleges, javított magyar szöveg legyen, mindenféle magyarázat, kommentár vagy formázás nélkül.

    --- Adatok ---
    Eredeti angol: "{original_text}"
    Meglévő fordítás: "{existing_translation}"
    SRT kontextus: "{srt_context}"
    ---

    Javított magyar fordítás:
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt.strip()}],
            max_tokens=200,
            temperature=0.1,
        )
        result = response.choices[0].message.content.strip()
        
        # 3. Eredmény mentése a gyorsítótárba a következő futtatáshoz
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            f.write(result)
            
        return result
    except Exception as e:
        print(f"\nAPI hiba: {e}")
        return f"API_HIBA: {e}"

def process_single_segment(segment, subtitles, verbose):
    """Egyetlen szegmenst dolgoz fel."""
    original_text = segment.get('text', '')
    existing_translation = segment.get('translated_text', '')

    if verbose:
        print("\n" + "="*50)
        print(f"Feldolgozás: {segment['start']:.2f}s - {segment['end']:.2f}s")
        print(f"  Eredeti (EN): {original_text}")
        print(f"  Meglévő (HU): {existing_translation}")

    if not original_text or not existing_translation:
        segment['translated_text_DS'] = existing_translation
        if verbose: print("  -> Kihagyva (nincs eredeti vagy meglévő fordítás).")
        return segment

    srt_context = find_srt_text_for_segment(segment, subtitles)
    
    if srt_context:
        if verbose: print(f"  SRT Kontextus: {srt_context.replace(chr(10), ' / ')}")
        improved_translation = get_improved_translation(original_text, existing_translation, srt_context, verbose)
        segment['translated_text_DS'] = improved_translation
    else:
        segment['translated_text_DS'] = existing_translation
        if verbose: print("  -> Nincs talált SRT kontextus, a meglévő fordítás marad.")
        
    return segment

def process_json_file(json_path, max_workers, verbose):
    """A teljes feldolgozási folyamatot végző fő funkció."""
    base_name = os.path.splitext(json_path)[0]
    srt_path = f"{base_name}.srt"
    output_path = f"{base_name}_DS.json"

    print(f"JSON beolvasása: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Hiba a JSON fájl beolvasása közben: {e}")
        return

    print(f"SRT beolvasása: {srt_path}")
    subtitles = load_srt_file(srt_path)
    if not subtitles:
        return

    print(f"Szegmensek feldolgozása a(z) '{MODEL_NAME}' modellel {max_workers} párhuzamos szállal...")
    
    segments_to_process = data['segments']
    processed_segments = [None] * len(segments_to_process)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # A feladatok beküldése, az eredeti indexszel együtt
        future_to_index = {
            executor.submit(process_single_segment, segment, subtitles, verbose): i
            for i, segment in enumerate(segments_to_process)
        }

        # Eredmények összegyűjtése a tqdm folyamatjelzővel
        for future in tqdm(as_completed(future_to_index), total=len(segments_to_process), desc="Fordítások javítása"):
            index = future_to_index[future]
            try:
                processed_segment = future.result()
                processed_segments[index] = processed_segment
            except Exception as exc:
                print(f'\nA(z) {index}. szegmens hibát generált: {exc}')
                # Hiba esetén is mentsük el az eredetit, hogy ne vesszen el adat
                processed_segments[index] = segments_to_process[index]

    data['segments'] = processed_segments

    print(f"\nFeldolgozás befejezve. Eredmény mentése ide: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("Mentés sikeres!")
    except Exception as e:
        print(f"Hiba a JSON fájl írása közben: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON feliratfájl javítása SRT kontextus alapján a DeepSeek API segítségével.")
    parser.add_argument("json_file", help="A bemeneti JSON fájl elérési útja.")
    parser.add_argument("-w", "--workers", type=int, default=10, help="A párhuzamos API hívások száma (alapértelmezett: 10).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Részletes kimenet a feldolgozás minden lépéséről.")
    
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Hiba: A megadott fájl nem létezik: {args.json_file}")
        sys.exit(1)
        
    process_json_file(args.json_file, args.workers, args.verbose)
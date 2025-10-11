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
CACHE_DIR = ".translation_cache_srt_segments" # Külön cache mappa az új logikához

# --- Inicializálás ---
if not API_KEY:
    print("Hiba: A DEEPSEEK_API_KEY környezeti változó nincs beállítva.")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key_srt(srt_content, json_texts):
    """Egyedi kulcsot generál az SRT felirat és az érintett JSON szövegek alapján."""
    return hashlib.sha256((srt_content + "|||" + "|||".join(json_texts)).encode('utf-8')).hexdigest()

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

def find_json_segments_for_srt(srt_subtitle, json_segments):
    """
    Megkeresi az SRT felirat időintervallumának megfelelő JSON szegmenseket.
    Visszaadja a JSON szegmensek indexeit, az eredeti szövegeket és a meglévő fordításokat.
    """
    srt_start = srt_subtitle.start
    srt_end = srt_subtitle.end
    
    matching_segments_info = [] # (index, original_text, existing_translation)
    
    for i, json_seg in enumerate(json_segments):
        json_start = timedelta(seconds=json_seg['start'])
        json_end = timedelta(seconds=json_seg['end'])
        
        # Átfedés ellenőrzése: max(start1, start2) < min(end1, end2)
        if max(srt_start, json_start) < min(srt_end, json_end):
            matching_segments_info.append(
                (i, json_seg.get('text', ''), json_seg.get('translated_text', ''))
            )
            
    # Rendezés a kezdőidő szerint, hogy a JSON sorrendje megmaradjon
    matching_segments_info.sort(key=lambda x: json_segments[x[0]]['start'])
    
    return matching_segments_info

def get_improved_translation_batch(srt_content, json_original_texts, json_existing_translations, verbose=False):
    """
    Meghívja a DeepSeek API-t, hogy javított fordításokat kérjen egy SRT felirat és több JSON szegmens alapján.
    A modellnek vissza kell adnia a fordításokat sortörésekkel elválasztva,
    az eredeti JSON szegmensek sorrendjében.
    """
    
    cache_key = get_cache_key_srt(srt_content, json_original_texts)
    cache_file_path = os.path.join(CACHE_DIR, cache_key)

    if os.path.exists(cache_file_path):
        if verbose: print("  [CACHE HIT]")
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            return f.read().split('\n')

    if verbose: print("  [API CALL]")
        
    original_json_block = "\n".join([f"- {text}" for text in json_original_texts])
    existing_json_block = "\n".join([f"- {text}" for text in json_existing_translations])

    prompt = f"""
    Feladat: Fordítás javítása SRT kontextus és több JSON szegmens alapján.
    Adott egy eredeti angol SRT felirat, valamint több hozzátartozó angol JSON szövegszegmens, és ezek meglévő magyar fordításai.
    A célod, hogy az SRT feliratot kontextusként felhasználva, a JSON szegmensek számára adj jobb, természetesebb magyar fordításokat.
    A válaszodban minden JSON szegmens fordítása külön sorban legyen, pontosan abban a sorrendben, ahogy az eredeti angol JSON szegmenseket megkaptad.
    Csak a fordításokat add vissza, semmi mást.

    --- Adatok ---
    SRT felirat (kontextus): "{srt_content}"

    Eredeti angol JSON szegmensek (egymás alatt):
    {original_json_block}

    Meglévő magyar fordítások (egymás alatt, az eredeti szegmensekhez):
    {existing_json_block}
    ---

    Javított magyar fordítások (minden szegmens külön sorban):
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt.strip()}],
            max_tokens=500, # Növelve a max_tokens, mivel több sort is várunk
            temperature=0.1,
        )
        result_raw = response.choices[0].message.content.strip()
        
        # Eredmény mentése a gyorsítótárba
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            f.write(result_raw)
            
        return result_raw.split('\n')
    except Exception as e:
        print(f"\nAPI hiba egy SRT batch hívásban: {e}")
        # Ha hiba van, akkor minden fordítási elemhez hibaüzenetet adunk
        return [f"API_HIBA: {e}" for _ in json_original_texts]

def process_json_file(json_path, max_workers, verbose):
    """A teljes feldolgozási folyamatot végző fő funkció."""
    base_name = os.path.splitext(json_path)[0]
    srt_path = f"{base_name}.srt"
    output_path = f"{base_name}_DS_v2.json" # Új kimeneti fájlnév a V2 jelzésére

    print(f"JSON beolvasása: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Hiba a JSON fájl beolvasása közben: {e}")
        return

    # Mivel a JSON szegmensek listája módosulhat a feldolgozás során,
    # készítsünk egy másolatot, vagy direktben indexeljünk.
    # Itt most egy lista referenciát használunk, amit frissítünk.
    json_segments_ref = data['segments']

    print(f"SRT beolvasása: {srt_path}")
    subtitles = load_srt_file(srt_path)
    if not subtitles:
        return

    print(f"SRT feliratok feldolgozása a(z) '{MODEL_NAME}' modellel {max_workers} párhuzamos szállal...")
    
    processed_count = 0
    total_srt_subtitles = len(subtitles)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for srt_sub in subtitles:
            matching_json_info = find_json_segments_for_srt(srt_sub, json_segments_ref)
            if matching_json_info:
                # Elküldjük az SRT felirat tartalmát és az összes érintett JSON szegmens szövegét
                # A futures listában tároljuk a (future, [érintett_json_indexek]) párokat
                future = executor.submit(
                    get_improved_translation_batch,
                    srt_sub.content.strip(),
                    [info[1] for info in matching_json_info], # original_text-ek
                    [info[2] for info in matching_json_info], # existing_translation-ök
                    verbose
                )
                futures.append((future, matching_json_info))
            else:
                # Nincs hozzá JSON szegmens, nem kell API hívás, de beleszámít a teljesbe
                processed_count += 1

        pbar = tqdm(total=total_srt_subtitles, desc="SRT feliratok fordítása")
        for future, matching_json_info in futures:
            try:
                improved_translations = future.result()
                
                # A visszakapott fordításokat szétosztjuk az eredeti JSON szegmensek között
                for i, (json_index, _, _) in enumerate(matching_json_info):
                    if i < len(improved_translations):
                        json_segments_ref[json_index]['translated_text_DS'] = improved_translations[i]
                    else:
                        # Ez ritka, ha a modell kevesebb sort ad vissza
                        json_segments_ref[json_index]['translated_text_DS'] = "HIBA: HIÁNYZÓ FORDÍTÁS"
                
                processed_count += 1 # Növeljük a számlálót, ha az SRT felirat feldolgozva
            except Exception as exc:
                print(f'\nHiba egy SRT felirat feldolgozása közben: {exc}')
                # Hiba esetén a meglévő fordítás marad
                for json_index, _, existing_trans in matching_json_info:
                    json_segments_ref[json_index]['translated_text_DS'] = existing_trans
                processed_count += 1
            finally:
                pbar.update(1)
        pbar.close()

    # Ellenőrizzük azokat a JSON szegmenseket, amikhez nem találtunk SRT feliratot
    # (vagy az SRT feliratokhoz nem találtunk JSON szegmenst, bár a find_json_segments_for_srt ezt kizárja)
    for segment in json_segments_ref:
        if 'translated_text_DS' not in segment:
            segment['translated_text_DS'] = segment.get('translated_text', '') # Meglévő fordítás marad
            if verbose:
                print(f"Figyelem: A JSON szegmenshez {segment['start']:.2f}s - {segment['end']:.2f}s nem társult SRT felirat, a meglévő fordítás marad.")


    print(f"\nFeldolgozás befejezve. Eredmény mentése ide: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("Mentés sikeres!")
    except Exception as e:
        print(f"Hiba a JSON fájl írása közben: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON feliratfájl javítása SRT kontextus alapján a DeepSeek API segítségével (SRT-alapú feldolgozással).")
    parser.add_argument("json_file", help="A bemeneti JSON fájl elérési útja.")
    parser.add_argument("-w", "--workers", type=int, default=10, help="A párhuzamos API hívások száma (alapértelmezett: 10).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Részletes kimenet a feldolgozás minden lépéséről.")
    
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Hiba: A megadott fájl nem létezik: {args.json_file}")
        sys.exit(1)
        
    process_json_file(args.json_file, args.workers, args.verbose)
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
    Visszaadja a JSON szegmensek indexeit, az eredeti szövegeket és a már 'translated_text_deepl'-be másolt fordításokat.
    """
    srt_start = srt_subtitle.start
    srt_end = srt_subtitle.end
    
    matching_segments_info = [] # (index, original_text, existing_deepl_translation)
    
    for i, json_seg in enumerate(json_segments):
        json_start = timedelta(seconds=json_seg['start'])
        json_end = timedelta(seconds=json_seg['end'])
        
        # Átfedés ellenőrzése: max(start1, start2) < min(end1, end2)
        if max(srt_start, json_start) < min(srt_end, json_end):
            # Fontos: itt már a 'translated_text_deepl'-t kell használni az API-nak átadott "meglévő fordítás" paraméterként!
            matching_segments_info.append(
                (i, json_seg.get('text', ''), json_seg.get('translated_text_deepl', '')) 
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
            # Strip potential leading hyphens and whitespace from cached lines
            return [line.lstrip('- ').strip() for line in f.read().split('\n')]

    if verbose: print("  [API CALL]")
        
    original_json_block = "\n".join([f"- {text}" for text in json_original_texts])
    existing_json_block = "\n".join([f"- {text}" for text in json_existing_translations])

    # --- PROMPT MÓDOSÍTÁS ---
    # Utasítjuk a modellt, hogy ne használjon kezdő kötőjelet a válaszban
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

    Javított magyar fordítások (minden szegmens külön sorban, kezdő kötőjel nélkül):
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt.strip()}],
            max_tokens=500,
            temperature=0.1,
        )
        result_raw = response.choices[0].message.content.strip()
        
        # Eredmény mentése a gyorsítótárba
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            f.write(result_raw)
            
        # Strip potential leading hyphens and whitespace from API response lines
        return [line.lstrip('- ').strip() for line in result_raw.split('\n')]
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

    json_segments_ref = data['segments']

    # --- FÁJL KEZDŐ ÁLLAPOT KEZELÉS: translated_text átnevezése translated_text_deepl-re ---
    for segment in json_segments_ref:
        if 'translated_text' in segment:
            # Áthelyezzük az eredeti 'translated_text' értéket a 'translated_text_deepl' kulcsba
            segment['translated_text_deepl'] = segment.pop('translated_text') 
        else:
            # Ha nem volt 'translated_text', akkor is legyen 'translated_text_deepl' (üresen)
            segment['translated_text_deepl'] = "" 
        # Ideiglenes flag, ami jelzi, hogy DeepSeek feldolgozta-e a szegmenst
        segment['_deepseek_processed'] = False 

    print(f"SRT beolvasása: {srt_path}")
    subtitles = load_srt_file(srt_path)
    if not subtitles:
        return

    print(f"SRT feliratok feldolgozása a(z) '{MODEL_NAME}' modellel {max_workers} párhuzamos szállal...")
    
    total_srt_subtitles = len(subtitles)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for srt_sub in subtitles:
            # Fontos: itt már a módosított find_json_segments_for_srt-t használjuk, ami a translated_text_deepl-t olvassa
            matching_json_info = find_json_segments_for_srt(srt_sub, json_segments_ref)
            if matching_json_info:
                future = executor.submit(
                    get_improved_translation_batch,
                    srt_sub.content.strip(),
                    [info[1] for info in matching_json_info], # original_text-ek
                    [info[2] for info in matching_json_info], # existing_translation-ök (már a Deepl-ből)
                    verbose
                )
                futures.append((future, matching_json_info))

        pbar = tqdm(total=total_srt_subtitles, desc="SRT feliratok fordítása")
        for future, matching_json_info in futures:
            try:
                improved_translations = future.result()
                
                for i, (json_index, _, _) in enumerate(matching_json_info):
                    segment = json_segments_ref[json_index]
                    if i < len(improved_translations):
                        # A visszaadott érték kerül a "translated_text"-be
                        segment['translated_text'] = improved_translations[i]
                    else:
                        # Ha a modell kevesebb sort adott vissza, vagy API_HIBA van, akkor ideiglenesen ezt írjuk be
                        segment['translated_text'] = "HIBA: HIÁNYZÓ FORDÍTÁS / API_HIBA"
                    segment['_deepseek_processed'] = True # Megjelöljük, hogy a DeepSeek feldolgozta
            except Exception as exc:
                print(f'\nHiba egy SRT felirat feldolgozása közben: {exc}')
                # Hiba esetén a szegmens _deepseek_processed flag-je True lesz,
                # a 'translated_text' pedig hibaüzenetet kap. Ezt is 'feldolgozottnak' tekintjük,
                # mert megpróbáltuk, de hibával.
                for i, (json_index, _, _) in enumerate(matching_json_info):
                    segment = json_segments_ref[json_index]
                    segment['translated_text'] = f"API_HIBA: {exc}"
                    segment['_deepseek_processed'] = True 
            finally:
                pbar.update(1)
        pbar.close()

    # --- FÁJL VÉGSŐ ÁLLAPOT KEZELÉS: Nem feldolgozott szegmensek és _deepseek_processed eltávolítása ---
    for segment in json_segments_ref:
        if not segment.get('_deepseek_processed', False):
            # Ha a DeepSeek nem dolgozta fel ezt a szegmenst (nincs átfedő SRT, vagy nem volt API hívás),
            # akkor a "translated_text" mezőbe az eredeti, "translated_text_deepl"-ben tárolt értéket tesszük.
            segment['translated_text'] = segment['translated_text_deepl']
        
        # Eltávolítjuk az ideiglenes flag-et
        if '_deepseek_processed' in segment:
            del segment['_deepseek_processed']

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
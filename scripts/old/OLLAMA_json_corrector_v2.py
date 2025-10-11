import os
import sys
import json
import srt
import hashlib
import argparse
from datetime import timedelta
from openai import OpenAI
from tqdm import tqdm # Importáljuk, de lehet, hogy kikapcsoljuk
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Konfiguráció ---
MODEL_NAME = "qwen3:8b" 
OLLAMA_BASE_URL = "http://localhost:11434/v1" 
CACHE_DIR = ".translation_cache_ollama_srt_segments" 

# --- Inicializálás ---
# Az OpenAI kliens konfigurálása az Ollama lokális végpontjához
# Adunk egy dummy API kulcsot, mert az OpenAI kliens elvárja, még ha nem is használja az Ollama
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama_dummy_key") 
os.makedirs(CACHE_DIR, exist_ok=True)

# A korábbi find_srt_text_for_segment és get_cache_key_srt funkciók megegyeznek
def get_cache_key_srt(srt_content, json_texts):
    """Egyedi kulcsot generál az SRT felirat és az érintett JSON szövegek alapján."""
    # Hozzáadjuk a modell nevét is a kulcshoz, hogy modellváltás esetén a cache ne keveredjen
    return hashlib.sha256((srt_content + "|||" + "|||".join(json_texts) + "|||" + MODEL_NAME).encode('utf-8')).hexdigest()

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
    Meghívja az Ollama API-t, hogy javított fordításokat kérjen egy SRT felirat és több JSON szegmens alapján.
    """
    
    cache_key = get_cache_key_srt(srt_content, json_original_texts)
    cache_file_path = os.path.join(CACHE_DIR, cache_key)

    if os.path.exists(cache_file_path):
        if verbose: print("  [CACHE HIT]")
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            return f.read().split('\n')

    if verbose: print("  [API CALL]") # Ezt most a hívás előtt adjuk ki a verbose módban
        
    original_json_block = "\n".join([f"- {text}" for text in json_original_texts])
    existing_json_block = "\n".join([f"- {text}" for text in json_existing_translations])

    prompt = f"""
    /no_think
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
            max_tokens=500, 
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
    output_path = f"{base_name}_Ollama_DS.json" 

    print(f"JSON beolvasása: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Hiba a JSON fájl beolvasása közben: {e}")
        return

    json_segments_ref = data['segments'] 

    print(f"SRT beolvasása: {srt_path}")
    subtitles = load_srt_file(srt_path)
    if not subtitles:
        return

    print(f"SRT feliratok feldolgozása a(z) '{MODEL_NAME}' modellel az Ollama API-n keresztül. Párhuzamos szálak: {max_workers}...")
    
    has_ollama_translation = [False] * len(json_segments_ref)

    # Verbose módban kikapcsoljuk a tqdm-et, hogy a logok ne keveredjenek
    # Nem-verbose módban bekapcsoljuk a tqdm-et
    progress_bar_enabled = not verbose
    pbar = tqdm(total=len(subtitles), desc="SRT feliratok fordítása", disable=not progress_bar_enabled)

    # Számláló a folyamatban levő API kéréseknek
    active_futures = []

    # A ThreadPoolExecutor csak akkor hasznos, ha több, mint 1 worker van.
    # Ha max_workers = 1, akkor sorosan futnak a feladatok.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for srt_sub_idx, srt_sub in enumerate(subtitles):
            matching_json_info = find_json_segments_for_srt(srt_sub, json_segments_ref)
            
            if matching_json_info:
                # API hívás adatai
                srt_content = srt_sub.content.strip()
                json_original_texts = [info[1] for info in matching_json_info]
                json_existing_translations = [info[2] for info in matching_json_info]

                if verbose:
                    print("\n" + "="*80)
                    print(f"SRT Felirat ({srt_sub_idx+1}/{len(subtitles)}) Idő: {srt_sub.start} --> {srt_sub.end}")
                    print(f"  SRT tartalom: {srt_content.replace(chr(10), ' / ')}")
                    print(f"  Érintett JSON szegmensek ({len(matching_json_info)} db):")
                    for info in matching_json_info:
                        idx, orig_text, existing_trans = info
                        print(f"    - Index: {idx}, Eredeti: '{orig_text}', Meglévő: '{existing_trans}'")
                
                # Ha csak 1 worker van, akkor azonnal feldolgozzuk, és nem használunk future-t
                if max_workers == 1:
                    improved_translations = get_improved_translation_batch(
                        srt_content, json_original_texts, json_existing_translations, verbose
                    )
                    
                    # Eredmények szétosztása
                    for i, (json_index, _, _) in enumerate(matching_json_info):
                        if i < len(improved_translations):
                            json_segments_ref[json_index]['translated_text_DS'] = improved_translations[i]
                            has_ollama_translation[json_index] = True
                        else:
                            json_segments_ref[json_index]['translated_text_DS'] = "HIBA: HIÁNYZÓ FORDÍTÁS"
                            if verbose: print(f"  Figyelem: A modell kevesebb fordítást adott vissza a vártnál. JSON szegmens: '{json_original_texts[i]}'")
                    
                    if verbose: 
                        print(f"  Fordítások ({len(improved_translations)} db):")
                        for t in improved_translations: print(f"    '{t}'")
                    
                    pbar.update(1) # Frissítjük a progress bar-t
                else:
                    # Több worker esetén future-t küldünk be
                    future = executor.submit(
                        get_improved_translation_batch,
                        srt_content, json_original_texts, json_existing_translations, verbose
                    )
                    active_futures.append((future, matching_json_info, srt_sub_idx, srt_sub)) # Tároljuk az SRT infókat is
            else:
                pbar.update(1) # Frissítjük a progress bar-t, ha nincs API hívás

        # Várjuk meg a párhuzamosan futó feladatokat, ha vannak
        if max_workers > 1:
            for future, matching_json_info, srt_sub_idx, srt_sub in as_completed(active_futures):
                try:
                    improved_translations = future.result()
                    
                    for i, (json_index, original_text, existing_translation) in enumerate(matching_json_info):
                        if i < len(improved_translations):
                            json_segments_ref[json_index]['translated_text_DS'] = improved_translations[i]
                            has_ollama_translation[json_index] = True
                        else:
                            json_segments_ref[json_index]['translated_text_DS'] = "HIBA: HIÁNYZÓ FORDÍTÁS"
                            if verbose: print(f"\nFigyelem: A modell kevesebb fordítást adott vissza a vártnál. JSON szegmens: '{original_text}'")
                    
                    if verbose: 
                        print(f"\n" + "="*80) # Új szeparáció a párhuzamos kimenetek között
                        print(f"SRT Felirat ({srt_sub_idx+1}/{len(subtitles)}) feldolgozva: {srt_sub.start} --> {srt_sub.end}")
                        print(f"  Fordítások ({len(improved_translations)} db):")
                        for t in improved_translations: print(f"    '{t}'")

                except Exception as exc:
                    print(f'\nHiba egy SRT batch hívás feldolgozása közben az SRT {srt_sub_idx+1} (idő: {srt_sub.start}) feliratnál: {exc}')
                    for json_index, _, existing_trans in matching_json_info:
                        if not has_ollama_translation[json_index]: # Csak akkor írjuk felül, ha még nem fordította le más
                            json_segments_ref[json_index]['translated_text_DS'] = existing_trans
                finally:
                    pbar.update(1)
            
    pbar.close() # Biztosítsuk, hogy a progress bar bezárul

    # Utólagos ellenőrzés: azok a JSON szegmensek, amikhez nem társítottunk fordítást
    for i, segment in enumerate(json_segments_ref):
        if not has_ollama_translation[i]:
            segment['translated_text_DS'] = segment.get('translated_text', '') 
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
    parser = argparse.ArgumentParser(description="JSON feliratfájl javítása SRT kontextus alapján az Ollama API segítségével (SRT-alapú feldolgozással).")
    parser.add_argument("json_file", help="A bemeneti JSON fájl elérési útja.")
    parser.add_argument("-w", "--workers", type=int, default=1, # Alapértelmezett 1 worker
                        help="A párhuzamos API hívások száma (alapértelmezett: 1).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Részletes kimenet a feldolgozás minden lépéséről. Kikapcsolja a progress bar-t.")
    parser.add_argument("-m", "--model", type=str, default=MODEL_NAME, 
                        help=f"Az Ollama modell neve (alapértelmezett: '{MODEL_NAME}'). Győződj meg róla, hogy letöltötted az Ollamával!")
    
    args = parser.parse_args()

    MODEL_NAME = args.model

    if not os.path.exists(args.json_file):
        print(f"Hiba: A megadott fájl nem létezik: {args.json_file}")
        sys.exit(1)
        
    process_json_file(args.json_file, args.workers, args.verbose)

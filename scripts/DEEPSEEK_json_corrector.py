import os
import sys
import json
import srt
from datetime import timedelta
from openai import OpenAI
from tqdm import tqdm

# --- Konfiguráció ---
# A DeepSeek API kulcs beolvasása a környezeti változókból
# Győződj meg róla, hogy beállítottad a 'DEEPSEEK_API_KEY' változót!
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    print("Hiba: A DEEPSEEK_API_KEY környezeti változó nincs beállítva.")
    sys.exit(1)

# A DeepSeek kliens inicializálása
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")
MODEL_NAME = "deepseek-chat" # vagy "deepseek-coder" ha azt preferálod

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
    """
    Megkeresi a JSON szegmens időintervallumának megfelelő SRT feliratokat.
    Több átfedő feliratot is összefűz.
    """
    segment_start = timedelta(seconds=segment['start'])
    segment_end = timedelta(seconds=segment['end'])
    
    matching_texts = []
    for sub in subtitles:
        # Átfedés ellenőrzése: max(start1, start2) < min(end1, end2)
        if max(segment_start, sub.start) < min(segment_end, sub.end):
            matching_texts.append(sub.content.strip())
            
    return "\n".join(matching_texts) if matching_texts else None

def get_improved_translation(original_text, existing_translation, srt_context):
    """
    Meghívja a DeepSeek API-t, hogy javított fordítást kérjen.
    """
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
            temperature=0.1, # Alacsony hőmérséklet a konzisztens, pontos eredményért
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\nAPI hiba: {e}")
        return f"API_HIBA: {e}"

def process_json_file(json_path):
    """A teljes feldolgozási folyamatot végző fő funkció."""
    # Fájlnevek generálása
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

    print(f"Szegmensek feldolgozása a(z) '{MODEL_NAME}' modellel...")
    
    # A tqdm progress bar használata a feldolgozás követéséhez
    for segment in tqdm(data['segments'], desc="Fordítások javítása"):
        original_text = segment.get('text', '')
        existing_translation = segment.get('translated_text', '')

        # Ha nincs eredeti szöveg vagy meglévő fordítás, ugorjunk
        if not original_text or not existing_translation:
            segment['translated_text_DS'] = existing_translation
            continue

        # SRT kontextus keresése
        srt_context = find_srt_text_for_segment(segment, subtitles)
        
        if srt_context:
            improved_translation = get_improved_translation(original_text, existing_translation, srt_context)
            segment['translated_text_DS'] = improved_translation
        else:
            # Ha nincs SRT kontextus, másoljuk a régit és jelezzük
            segment['translated_text_DS'] = existing_translation # Vagy lehetne: "NINCS_SRT_KONTEXTUS"

    print(f"\nFeldolgozás befejezve. Eredmény mentése ide: {output_path}")
    
    # Az eredmény kiírása új fájlba
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False a magyar karakterek helyes mentéséhez
            # indent=2 a szép, olvasható formátumért
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("Mentés sikeres!")
    except Exception as e:
        print(f"Hiba a JSON fájl írása közben: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Használat: python enhance_translation.py <json_fajl_eleresi_utja>")
        sys.exit(1)
        
    input_file_path = sys.argv[1]
    if not os.path.exists(input_file_path):
        print(f"Hiba: A megadott fájl nem létezik: {input_file_path}")
        sys.exit(1)
        
    process_json_file(input_file_path)
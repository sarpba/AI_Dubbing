# scripts/unpack_srt_from_mkv.py

import sys
import subprocess
import json
import os
import argparse
from pathlib import Path

# Lehetséges felirat kiterjesztések a codec ID alapján
CODEC_EXTENSION_MAP = {
    "S_TEXT/UTF8": ".srt",
    "S_TEXT/SRT": ".srt",
    "S_SRT/UTF8": ".srt",
    "S_ASS": ".ass",
    "S_SSA": ".ssa",
    "S_VOBSUB": ".sub",  # Gyakran .idx fájl is tartozik hozzá
    "S_HDMV/PGS": ".sup",
}

# Átfogó leképezés az ISO 639-2 (3 betűs) és ISO 639-1 (2 betűs) nyelvi kódok között.
# Tartalmazza a terminológiai (T) és a bibliográfiai (B) kódokat is, mivel az MKV mindkettőt használhatja.
LANG_CODE_MAP = {
    'aar': 'aa', 'abk': 'ab', 'afr': 'af', 'aka': 'ak', 'alb': 'sq', 'sqi': 'sq',
    'amh': 'am', 'ara': 'ar', 'arg': 'an', 'arm': 'hy', 'hye': 'hy', 'asm': 'as',
    'ava': 'av', 'ave': 'ae', 'aym': 'ay', 'aze': 'az', 'bak': 'ba', 'bam': 'bm',
    'baq': 'eu', 'eus': 'eu', 'bel': 'be', 'ben': 'bn', 'bih': 'bh', 'bis': 'bi',
    'bod': 'bo', 'tib': 'bo', 'bos': 'bs', 'bre': 'br', 'bul': 'bg', 'bur': 'my',
    'mya': 'my', 'cat': 'ca', 'ces': 'cs', 'cze': 'cs', 'cha': 'ch', 'che': 'ce',
    'chi': 'zh', 'zho': 'zh', 'chu': 'cu', 'chv': 'cv', 'cor': 'kw', 'cos': 'co',
    'cre': 'cr', 'cym': 'cy', 'wel': 'cy', 'dan': 'da', 'deu': 'de', 'ger': 'de',
    'div': 'dv', 'dut': 'nl', 'nld': 'nl', 'dzo': 'dz', 'ell': 'el', 'gre': 'el',
    'eng': 'en', 'epo': 'eo', 'est': 'et', 'ewe': 'ee', 'fao': 'fo', 'fas': 'fa',
    'per': 'fa', 'fij': 'fj', 'fin': 'fi', 'fra': 'fr', 'fre': 'fr', 'fry': 'fy',
    'ful': 'ff', 'geo': 'ka', 'kat': 'ka', 'gla': 'gd', 'gle': 'ga', 'glg': 'gl',
    'glv': 'gv', 'grn': 'gn', 'guj': 'gu', 'hat': 'ht', 'hau': 'ha', 'heb': 'he',
    'her': 'hz', 'hin': 'hi', 'hmo': 'ho', 'hrv': 'hr', 'hun': 'hu', 'ibo': 'ig',
    'ice': 'is', 'isl': 'is', 'ido': 'io', 'iii': 'ii', 'iku': 'iu', 'ile': 'ie',
    'ina': 'ia', 'ind': 'id', 'ipk': 'ik', 'ita': 'it', 'jav': 'jv', 'jpn': 'ja',
    'kal': 'kl', 'kan': 'kn', 'kas': 'ks', 'kau': 'kr', 'kaz': 'kk', 'khm': 'km',
    'kik': 'ki', 'kin': 'rw', 'kir': 'ky', 'kom': 'kv', 'kon': 'kg', 'kor': 'ko',
    'kua': 'kj', 'kur': 'ku', 'lao': 'lo', 'lat': 'la', 'lav': 'lv', 'lim': 'li',
    'lin': 'ln', 'lit': 'lt', 'ltz': 'lb', 'lub': 'lu', 'lug': 'lg', 'mac': 'mk',
    'mkd': 'mk', 'mah': 'mh', 'mal': 'ml', 'mao': 'mi', 'mri': 'mi', 'mar': 'mr',
    'may': 'ms', 'msa': 'ms', 'mlg': 'mg', 'mlt': 'mt', 'mon': 'mn', 'nau': 'na',
    'nav': 'nv', 'nbl': 'nr', 'nde': 'nd', 'ndo': 'ng', 'nep': 'ne', 'nno': 'nn',
    'nob': 'nb', 'nor': 'no', 'nya': 'ny', 'oci': 'oc', 'oji': 'oj', 'ori': 'or',
    'orm': 'om', 'oss': 'os', 'pan': 'pa', 'pli': 'pi', 'pol': 'pl', 'por': 'pt',
    'pus': 'ps', 'que': 'qu', 'roh': 'rm', 'ron': 'ro', 'rum': 'ro', 'run': 'rn',
    'rus': 'ru', 'sag': 'sg', 'san': 'sa', 'sin': 'si', 'slk': 'sk', 'slo': 'sk',
    'slv': 'sl', 'sme': 'se', 'smo': 'sm', 'sna': 'sn', 'snd': 'sd', 'som': 'so',
    'sot': 'st', 'spa': 'es', 'srd': 'sc', 'srp': 'sr', 'ssw': 'ss', 'sun': 'su',
    'swa': 'sw', 'swe': 'sv', 'tah': 'ty', 'tam': 'ta', 'tat': 'tt', 'tel': 'te',
    'tgk': 'tg', 'tgl': 'tl', 'tha': 'th', 'tir': 'ti', 'ton': 'to', 'tsn': 'tn',
    'tso': 'ts', 'tuk': 'tk', 'tur': 'tr', 'twi': 'tw', 'uig': 'ug', 'ukr': 'uk',
    'urd': 'ur', 'uzb': 'uz', 'ven': 've', 'vie': 'vi', 'vol': 'vo', 'wln': 'wa',
    'wol': 'wo', 'xho': 'xh', 'yid': 'yi', 'yor': 'yo', 'zha': 'za', 'zul': 'zu'
}

def check_dependencies():
    """Ellenőrzi, hogy az MKVToolNix telepítve van-e és elérhető-e."""
    try:
        subprocess.run(
            ["mkvmerge", "--version"],
            check=True,
            capture_output=True,
            encoding="utf-8"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("HIBA: Az MKVToolNix nincs telepítve vagy nem található a rendszer PATH-jában.")
        print("Kérlek, telepítsd az MKVToolNix-et innen: https://mkvtoolnix.download/")
        sys.exit(1)

def load_config():
    """Beolvassa a konfigurációs fájlt a projekt gyökeréből."""
    try:
        # A szkript a 'scripts' mappában van, a config egy szinttel feljebb
        config_path = Path(__file__).resolve().parent.parent / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"HIBA: A konfigurációs fájl nem található itt: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"HIBA: A konfigurációs fájl ({config_path}) hibás formátumú.")
        sys.exit(1)

def extract_subtitles_from_mkv(mkv_file_path: Path, output_dir: Path):
    """
    Kicsomagolja az összes felirat sávot egy adott MKV fájlból.

    Args:
        mkv_file_path (Path): Az MKV fájl elérési útvonala.
        output_dir (Path): A kimeneti könyvtár, ahova a feliratok mentve lesznek.
    """
    print(f"\n--- Fájl feldolgozása: '{mkv_file_path.name}' ---")

    # 1. Sáv-információk lekérése JSON formátumban
    try:
        mkv_info_process = subprocess.run(
            ["mkvmerge", "-J", str(mkv_file_path)],
            check=True,
            capture_output=True,
            encoding="utf-8"
        )
        mkv_info = json.loads(mkv_info_process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"  HIBA a fájl elemzése közben: {e.stderr}")
        return
    except json.JSONDecodeError:
        print("  HIBA: Nem sikerült feldolgozni az mkvmerge kimenetét.")
        return

    # 2. Felirat sávok kiválasztása
    subtitle_tracks = [
        track for track in mkv_info.get("tracks", []) if track.get("type") == "subtitles"
    ]

    if not subtitle_tracks:
        print("  Nem található felirat sáv a fájlban.")
        return

    print(f"  Talált felirat sávok száma: {len(subtitle_tracks)}")
    
    base_filename = mkv_file_path.stem  # Fájlnév kiterjesztés nélkül
    extracted_count = 0

    # 3. Sávok kicsomagolása
    for track in subtitle_tracks:
        track_id = track["id"]
        codec_id = track["properties"].get("codec_id")
        language_3_letter = track["properties"].get("language", "und") # 'und' = undefined
        
        # Nyelvi kód átalakítása 2 betűsre, ha lehetséges
        lang_code = LANG_CODE_MAP.get(language_3_letter, language_3_letter)
        
        # Fájlkiterjesztés meghatározása
        extension = CODEC_EXTENSION_MAP.get(codec_id, ".txt") # Alapértelmezett: .txt
        
        # Kimeneti fájlnév létrehozása ütközéskezeléssel
        base_output_name = f"{base_filename}_{lang_code}{extension}"
        output_path = output_dir / base_output_name

        counter = 1
        while output_path.exists():
            # Ha a fájl már létezik (pl. több azonos nyelvű sáv), adjunk hozzá egy számlálót
            output_path = output_dir / f"{base_filename}_{lang_code}_{counter}{extension}"
            counter += 1
        
        print(f"\n  -> Sáv kicsomagolása: ID {track_id} ({codec_id}, Nyelv: {language_3_letter} -> {lang_code})")
        print(f"     Kimeneti fájl: {output_path.name}")

        # mkvextract parancs összeállítása és futtatása
        try:
            subprocess.run(
                [
                    "mkvextract",
                    str(mkv_file_path),
                    "tracks",
                    f"{track_id}:{str(output_path)}"
                ],
                check=True,
                capture_output=True,
                encoding="utf-8",
                errors="ignore"
            )
            print("     Kicsomagolás sikeres.")
            extracted_count += 1
        except subprocess.CalledProcessError as e:
            print(f"     HIBA a sáv kicsomagolása közben: {e.stderr}")

    print(f"\nFeldolgozás befejezve. Összesen {extracted_count} feliratfájl lett kicsomagolva a '{mkv_file_path.name}' fájlból.")

def main():
    """A szkript fő belépési pontja."""
    parser = argparse.ArgumentParser(
        description="Python script MKV konténerben lévő feliratfájlok kicsomagolásához egy projekt mappán belül.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("project_name", help="A projekt neve (a 'workdir' alatti mappa neve).")
    args = parser.parse_args()

    print("Függőségek ellenőrzése (MKVToolNix)...")
    check_dependencies()
    print("Függőség rendben.\n")
    
    print("Konfiguráció betöltése...")
    config = load_config()
    print("Konfiguráció betöltve.\n")

    # A projekt 'upload' mappájának meghatározása
    try:
        workdir = Path(config["DIRECTORIES"]["workdir"])
        upload_subdir_name = config["PROJECT_SUBDIRS"]["upload"]
        project_upload_path = workdir / args.project_name / upload_subdir_name
    except KeyError as e:
        print(f"HIBA: Hiányzó kulcs a config.json fájlban: {e}")
        sys.exit(1)


    if not project_upload_path.is_dir():
        print(f"HIBA: A projekt 'upload' mappája nem található itt: {project_upload_path}")
        print("Ellenőrizd, hogy a projekt neve helyes-e, és a mappaszerkezet létezik-e.")
        sys.exit(1)

    print(f"MKV fájlok keresése a következő mappában: {project_upload_path}")
    mkv_files = list(project_upload_path.glob('*.mkv'))

    if not mkv_files:
        print("Nem található .mkv fájl a megadott projekt 'upload' mappájában.")
        return

    print(f"Talált MKV fájlok: {[f.name for f in mkv_files]}")

    for mkv_file in mkv_files:
        extract_subtitles_from_mkv(mkv_file, project_upload_path)

    print("\n======================================")
    print("Minden feladat befejeződött.")
    print("======================================")


if __name__ == "__main__":
    main()
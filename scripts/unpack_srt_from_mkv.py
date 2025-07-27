import sys
import subprocess
import json
import os
import argparse

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

def check_dependencies():
    """Ellenőrzi, hogy az MKVToolNix telepítve van-e és elérhető-e."""
    try:
        # Egy egyszerű parancsot futtatunk, hogy lássuk, létezik-e
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

def extract_subtitles(mkv_file_path):
    """
    Kicsomagolja az összes felirat sávot egy adott MKV fájlból.

    Args:
        mkv_file_path (str): Az MKV fájl elérési útvonala.
    """
    if not os.path.exists(mkv_file_path):
        print(f"HIBA: A megadott fájl nem található: {mkv_file_path}")
        return

    print("MKVToolNix ellenőrzése...")
    check_dependencies()
    print("Függőség rendben.\n")

    print(f"'{os.path.basename(mkv_file_path)}' fájl elemzése...")
    
    # 1. Sáv-információk lekérése JSON formátumban
    try:
        mkv_info_process = subprocess.run(
            ["mkvmerge", "-J", mkv_file_path],
            check=True,
            capture_output=True,
            encoding="utf-8"
        )
        mkv_info = json.loads(mkv_info_process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"HIBA a fájl elemzése közben: {e.stderr}")
        return
    except json.JSONDecodeError:
        print("HIBA: Nem sikerült feldolgozni az mkvmerge kimenetét.")
        return

    # 2. Felirat sávok kiválasztása
    subtitle_tracks = [
        track for track in mkv_info.get("tracks", []) if track.get("type") == "subtitles"
    ]

    if not subtitle_tracks:
        print("Nem található felirat sáv a fájlban.")
        return

    print(f"Talált felirat sávok száma: {len(subtitle_tracks)}")
    
    base_filename = os.path.splitext(mkv_file_path)[0]
    extracted_count = 0

    # 3. Sávok kicsomagolása
    for track in subtitle_tracks:
        track_id = track["id"]
        codec_id = track["properties"].get("codec_id")
        language = track["properties"].get("language", "und") # 'und' = undefined
        
        # Fájlkiterjesztés meghatározása
        extension = CODEC_EXTENSION_MAP.get(codec_id, ".sub") # Alapértelmezett: .sub
        
        # Kimeneti fájlnév létrehozása
        output_filename = f"{base_filename}_track{track_id}_{language}{extension}"
        
        print(f"\n-> Sáv kicsomagolása: ID {track_id} ({codec_id}, Nyelv: {language})")
        print(f"   Kimeneti fájl: {os.path.basename(output_filename)}")

        # mkvextract parancs összeállítása és futtatása
        try:
            subprocess.run(
                [
                    "mkvextract",
                    mkv_file_path,
                    "tracks",
                    f"{track_id}:{output_filename}"
                ],
                check=True,
                capture_output=True
            )
            print("   Kicsomagolás sikeres.")
            extracted_count += 1
        except subprocess.CalledProcessError as e:
            print(f"   HIBA a sáv kicsomagolása közben: {e.stderr.decode('utf-8', errors='ignore')}")

    print(f"\nBefejezve. Összesen {extracted_count} feliratfájl lett kicsomagolva.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python script MKV konténerben lévő feliratfájlok kicsomagolásához.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("mkv_file", help="A feldolgozandó MKV fájl elérési útvonala.")
    
    args = parser.parse_args()
    
    extract_subtitles(args.mkv_file)
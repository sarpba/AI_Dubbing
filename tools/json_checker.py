import json
import argparse
import sys

def json_fajl_ellenorzo(fajl_eleres):
    """
    Ellenőrzi, hogy egy adott, létező fájl érvényes JSON-t tartalmaz-e.

    Args:
        fajl_eleres (str): Az ellenőrizendő JSON fájl elérési útja.

    Returns:
        bool: True, ha a fájl érvényes JSON, egyébként False.
    """
    try:
        # Fájl megnyitása olvasásra, UTF-8 kódolással
        with open(fajl_eleres, 'r', encoding='utf-8') as f:
            # A json.load() megpróbálja beolvasni és feldolgozni a fájl tartalmát
            # Ha a formátum nem megfelelő, json.JSONDecodeError hibát fog dobni
            json.load(f)

    except FileNotFoundError:
        print(f"❌ HIBA: A(z) '{fajl_eleres}' fájl nem található.")
        return False

    except json.JSONDecodeError as e:
        # A hibaüzenet pontosan megmondja, hol van a probléma a fájlban
        print(f"❌ HIBA: A(z) '{fajl_eleres}' fájl nem érvényes JSON.")
        print(f"   Részletek: {e}")
        return False

    except Exception as e:
        print(f"❌ HIBA: Ismeretlen hiba történt a fájl feldolgozása közben: {e}")
        return False

    # Ha nem történt hiba, a JSON érvényes
    print(f"✔️ SIKER: A(z) '{fajl_eleres}' fájl egy érvényes JSON dokumentum.")
    return True

if __name__ == "__main__":
    # Parancssori argumentum-kezelő beállítása
    parser = argparse.ArgumentParser(
        description="Egy meglévő JSON fájl szabványosságának ellenőrzése.",
        epilog="Használat: python validate_json.py /eleresi/ut/a/fajlhoz.json"
    )
    
    # A fájl elérési útját kötelező argumentumként adjuk meg
    parser.add_argument("file_path", help="Az ellenőrizendő JSON fájl elérési útja.")
    
    args = parser.parse_args()

    # Az ellenőrző függvény futtatása a megadott fájlra
    if not json_fajl_ellenorzo(args.file_path):
        # Ha a validáció sikertelen, a szkript 1-es hibakóddal lép ki,
        # ami hasznos lehet automatizált scriptekben (pl. CI/CD)
        sys.exit(1)
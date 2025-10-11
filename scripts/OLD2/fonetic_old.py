# FILE: fonetic.py

import os
import re
import sys
import argparse
import nltk
from functools import lru_cache

# --- Globális változó helyett egy "singleton" getter funkció a szótárnak ---
# Az lru_cache(maxsize=None) biztosítja, hogy a függvény csak egyszer fusson le
# a teljes script élettartama alatt, és az eredményt gyorsítótárazza.
@lru_cache(maxsize=None)
def get_cmudict():
    """
    Ellenőrzi, letölti (ha szükséges) és betölti a CMU kiejtési szótárat.
    A gyorsítótárazás miatt a teljes folyamat csak egyszer fut le.
    """
    try:
        # Ellenőrzés, hogy létezik-e az adatbázis
        nltk.data.find('corpora/cmudict.zip')
    except LookupError:
        print("A CMU kiejtési szótár (nltk:cmudict) nincs telepítve.", file=sys.stderr)
        print("Letöltés... (ehhez internetkapcsolat szükséges, kb. 6-7 MB)", file=sys.stderr)
        try:
            nltk.download('cmudict')
            print("Letöltés sikeres.", file=sys.stderr)
        except Exception as e:
            print(f"\nHIBA: A letöltés sikertelen. Ellenőrizd az internetkapcsolatot vagy a jogosultságokat.", file=sys.stderr)
            print(f"Részletek: {e}", file=sys.stderr)
            sys.exit(1) # Kilépés, mert a script nem tud tovább futni

    print("CMU szótár betöltése a memóriába...", file=sys.stderr)
    cmu_dict = nltk.corpus.cmudict.dict()
    print("Betöltés kész.", file=sys.stderr)
    return cmu_dict

# Arpabet fonémák leképezése magyaros kiejtésre.
ARPABET_TO_MAGYAR = {
    'AA': 'á', 'AE': 'e', 'AH': 'ö', 'AO': 'ó', 'AW': 'au', 'AY': 'áj',
    'B': 'b', 'CH': 'cs', 'D': 'd', 'DH': 'd', 'EH': 'e', 'ER': 'ör',
    'EY': 'éj', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'i', 'IY': 'í',
    'JH': 'dzs', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ng',
    'OW': 'ó', 'OY': 'oj', 'P': 'p', 'R': 'r', 'S': 'sz', 'SH': 's',
    'T': 't', 'TH': 'sz', 'UH': 'u', 'UW': 'ú', 'V': 'v', 'W': 'v',
    'Y': 'j', 'Z': 'z', 'ZH': 'zs'
}

def szabalyalapu_atiras(szo):
    """Az eredeti, egyszerű szabályalapú átíró (tartalék)."""
    replacements = {
        'ough': 'óf', 'augh': 'áf', 'eigh': 'éj', 'igh': 'áj', 'kn': 'n', 'gn': 'n',
        'ph': 'f', 'th': 'sz', 'sh': 's', 'ch': 'cs', 'qu': 'kv', 'ck': 'k',
        'wh': 'v', 'oo': 'ú', 'ee': 'í', 'ie': 'áj', 'ea': 'í', 'ou': 'au',
        'oi': 'oj', 'au': 'ó', 'ei': 'éj', 'ui': 'uj', 'a': 'á', 'b': 'b', 'c': 'k',
        'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g', 'h': 'h', 'i': 'i', 'j': 'dzs',
        'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'ó', 'p': 'p', 'q': 'kv',
        'r': 'r', 's': 'sz', 't': 't', 'u': 'ú', 'v': 'v', 'w': 'v', 'x': 'ksz',
        'y': 'j', 'z': 'z',
    }
    szo = szo.lower()
    sorted_keys = sorted(replacements, key=lambda x: -len(x))
    for angol, magyar in [(k, replacements[k]) for k in sorted_keys]:
        szo = szo.replace(angol, magyar)
    return szo

def fonetikus_atiras(szoveg):
    """
    Hibrid fonetikus átíró. Angol szöveget alakít át magyaros kiejtésűre.
    Automatikusan kezeli a CMU szótár inicializálását az első híváskor.
    """
    cmudict = get_cmudict() # Ez hívja a gyorsítótárazott funkciót

    tokens = re.findall(r"[\w']+|[^\w\s']", szoveg.lower())
    eredmeny = []

    for token in tokens:
        if token in cmudict:
            fonemak = [re.sub(r'\d', '', fonema) for fonema in cmudict[token][0]]
            magyar_szo = ''.join([ARPABET_TO_MAGYAR.get(f, f.lower()) for f in fonemak])
            eredmeny.append(magyar_szo)
        elif token.isalpha():
            eredmeny.append(szabalyalapu_atiras(token))
        else:
            eredmeny.append(token)
    
    teljes_szoveg = ' '.join(eredmeny)
    return re.sub(r'\s+([.,!?;:])', r'\1', teljes_szoveg).strip()

def feldolgoz_rekurzivan(mappa):
    """Rekurzívan feldolgozza a szöveges fájlokat egy mappában."""
    for gyoker, almappak, fajlok in os.walk(mappa):
        print(f"Feldolgozás alatt: {gyoker}")
        for fajlnev in fajlok:
            if fajlnev.lower().endswith(('.txt', '.text')):
                nev, kiterjesztes = os.path.splitext(fajlnev)
                if '_original' in nev:
                    print(f"  Kihagyva (már feldolgozott): {fajlnev}")
                    continue

                eredeti_utvonal = os.path.join(gyoker, fajlnev)
                atnevezett_utvonal = os.path.join(gyoker, f"{nev}_original{kiterjesztes}")
                
                print(f"  Feldolgozás: {fajlnev}")

                try:
                    with open(eredeti_utvonal, 'r', encoding='utf-8') as f_eredeti:
                        sorok = f_eredeti.readlines()

                    fonetikus_tartalom = [fonetikus_atiras(sor) + '\n' for sor in sorok if sor.strip()]
                    
                    if not fonetikus_tartalom:
                        print(f"  Kihagyva (nincs tartalom): {fajlnev}")
                        continue

                    os.rename(eredeti_utvonal, atnevezett_utvonal)
                    print(f"    -> Eredeti átnevezve: {os.path.basename(atnevezett_utvonal)}")

                    with open(eredeti_utvonal, 'w', encoding='utf-8') as f_uj:
                        f_uj.writelines(fonetikus_tartalom)
                    print(f"    -> Fonetikus verzió létrehozva: {fajlnev}")
                
                except Exception as e:
                    print(f"Hiba történt a(z) '{fajlnev}' feldolgozása közben: {e}")

def main():
    """A script fő belépési pontja, ha parancssorból futtatják."""
    parser = argparse.ArgumentParser(
        description="Angol szöveges fájlokat alakít át magyaros fonetikus átírásra.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='A feldolgozandó gyökérkönyvtár elérési útja.'
    )
    args = parser.parse_args()
    
    # Elindítjuk a feldolgozást
    mappa_utvonal = args.input
    if not os.path.isdir(mappa_utvonal):
        print(f"Hiba: A megadott mappa nem létezik: {mappa_utvonal}")
    else:
        # A get_cmudict() első hívása itt történik meg, ami letölti/betölti a szótárat
        get_cmudict()
        feldolgoz_rekurzivan(mappa_utvonal)
        print("\nMinden fájl feldolgozva.")

if __name__ == "__main__":
    main()
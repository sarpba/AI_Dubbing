# FILE: fonetic.py

import os
import re
import sys
import argparse
import nltk
import tempfile
from functools import lru_cache
from typing import List, Tuple, Optional

# -------------------------
# CMU szótár betöltése
# -------------------------

@lru_cache(maxsize=None)
def get_cmudict():
    """
    Ellenőrzi, letölti (ha szükséges) és betölti a CMU kiejtési szótárat.
    A gyorsítótárazás miatt csak egyszer történik meg a script futása alatt.
    """
    try:
        nltk.data.find('corpora/cmudict.zip')
    except LookupError:
        print("A CMU kiejtési szótár (nltk:cmudict) nincs telepítve.", file=sys.stderr)
        print("Letöltés... (ehhez internetkapcsolat szükséges, kb. 6-7 MB)", file=sys.stderr)
        try:
            nltk.download('cmudict')
            print("Letöltés sikeres.", file=sys.stderr)
        except Exception as e:
            print("HIBA: A letöltés sikertelen. Ellenőrizd az internetkapcsolatot vagy a jogosultságokat.", file=sys.stderr)
            print(f"Részletek: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        print("CMU szótár betöltése a memóriába...", file=sys.stderr)
        cmu_dict = nltk.corpus.cmudict.dict()
        print("Betöltés kész.", file=sys.stderr)
        return cmu_dict
    except Exception as e:
        print("HIBA: A CMU szótár betöltése sikertelen.", file=sys.stderr)
        print(f"Részletek: {e}", file=sys.stderr)
        sys.exit(1)

# -------------------------
# ARPABET -> magyar mapping
# -------------------------

ARPABET_TO_MAGYAR = {
    'AA': 'á', 'AE': 'e', 'AH': 'ö', 'AO': 'ó', 'AW': 'au', 'AY': 'áj',
    'B': 'b', 'CH': 'cs', 'D': 'd', 'DH': 'd', 'EH': 'e', 'ER': 'ör',
    'EY': 'éj', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'i', 'IY': 'í',
    'JH': 'dzs', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ng',
    'OW': 'ó', 'OY': 'oj', 'P': 'p', 'R': 'r', 'S': 'sz', 'SH': 's',
    'T': 't', 'TH': 'sz', 'UH': 'u', 'UW': 'ú', 'V': 'v', 'W': 'v',
    'Y': 'j', 'Z': 'z', 'ZH': 'zs'
}

def arpabet_to_magyar(fonemak: List[str]) -> str:
    return ''.join(ARPABET_TO_MAGYAR.get(f, f.lower()) for f in fonemak)

# -------------------------
# Szabályalapú átírás (fallback)
# -------------------------

def szabalyalapu_atiras(szo: str) -> str:
    """
    Konzervatív szabályalapú átírás. Longest-first, nem overlappelő helyettesítések.
    Nem tolja el agresszívan a magánhangzókat (a, e, i, o, u marad),
    mert ezek gyakran félrevezetnének.
    """
    s = szo.lower()
    # Gyakoribb minták előre véve
    rules = [
        ('tion', 'sön'),
        ('sion', 'zson'),
        ('ture', 'csör'),
        ('sure', 'zsör'),
        ('dge', 'dzs'),
        ('ough', 'óf'),
        ('augh', 'áf'),
        ('eigh', 'éj'),
        ('igh', 'áj'),
        ('kn', 'n'),
        ('gn', 'n'),
        ('ph', 'f'),
        ('th', 'sz'),
        ('sh', 's'),
        ('ch', 'cs'),
        ('qu', 'kv'),
        ('ck', 'k'),
        ('wh', 'v'),
        ('oo', 'ú'),
        ('ee', 'í'),
        ('ie', 'áj'),
        ('ea', 'í'),
        ('ou', 'au'),
        ('oi', 'oj'),
        ('au', 'ó'),
        ('ei', 'éj'),
        ('ui', 'uj'),
    ]
    # Hosszabbak előre (bár már így is nagyrészt)
    rules.sort(key=lambda x: -len(x[0]))

    i = 0
    out = []
    L = len(s)
    while i < L:
        matched = False
        for pat, rep in rules:
            if s.startswith(pat, i):
                out.append(rep)
                i += len(pat)
                matched = True
                break
        if matched:
            continue
        ch = s[i]
        base = {
            'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'g', 'h': 'h', 'j': 'dzs', 'k': 'k',
            'l': 'l', 'm': 'm', 'n': 'n', 'p': 'p', 'q': 'kv', 'r': 'r', 's': 'sz', 't': 't',
            'v': 'v', 'w': 'v', 'x': 'ksz', 'y': 'j', 'z': 'z',
            'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u'
        }
        out.append(base.get(ch, ch))
        i += 1

    res = ''.join(out)
    # Némah 'e' a szó végén
    res = re.sub(r'e\b', '', res)
    # -ed végződés egyszerűsítése
    res = re.sub(r'([ptkfsz])ed\b', r'\1t', res)
    res = re.sub(r'ed\b', 'd', res)
    return res

# -------------------------
# Token szintű cache
# -------------------------

def _strip_stress(pron: List[str]) -> List[str]:
    return [re.sub(r'\d', '', f) for f in pron]

@lru_cache(maxsize=100000)
def atiras_tokenre(token_lower: str, cmu_only: bool, fallback_only: bool, all_prons: bool) -> str:
    """
    Egy token (kisbetűs) átírása cache-elve.
    cmu_only: ha True és nincs a CMU-ban, visszaadja az eredetit (vagy üres?) – itt az eredetit adjuk vissza.
    fallback_only: ha True, CMU lookupot sem használunk, csak szabályalapú.
    all_prons: ha True, a CMU összes variánsát |-nel összefűzi.
    """
    if not token_lower:
        return token_lower

    if fallback_only:
        if token_lower.isalpha():
            return szabalyalapu_atiras(token_lower)
        return token_lower

    cmu = get_cmudict()
    if token_lower in cmu:
        variants = []
        for pr in cmu[token_lower]:
            fonemak = _strip_stress(pr)
            variants.append(arpabet_to_magyar(fonemak))
        return '|'.join(variants) if all_prons else variants[0]

    if cmu_only:
        return token_lower

    if token_lower.isalpha():
        return szabalyalapu_atiras(token_lower)
    return token_lower

# -------------------------
# Tokenizálás és átírás
# -------------------------

TOKEN_PATTERN = re.compile(r"[\w']+|[^\w\s']+|\s+")

def tokenizal_spanokkal(szoveg: str) -> List[Tuple[str, int, int]]:
    """
    Szavak, írásjelek és whitespace-ek sorozata, eredeti spacing megőrzésével.
    """
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(szoveg)]

def fonetikus_atiras(szoveg: str, *, all_prons: bool = False, cmu_only: bool = False, fallback_only: bool = False) -> str:
    """
    Hibrid fonetikus átíró. Angol szöveget alakít át magyaros kiejtésűre.
    - all_prons: több CMU kiejtés esetén mindet visszaadja |-del.
    - cmu_only: csak CMU szótárat használ; ha nincs benne, az eredeti token marad.
    - fallback_only: CMU kikapcsolva; csak szabályalapú.
    """
    darabok = tokenizal_spanokkal(szoveg)
    out = []
    for tok, _, _ in darabok:
        # whitespace vagy írásjel/egyéb marad változatlanul
        if tok.strip() == '':
            out.append(tok)
            continue
        if re.fullmatch(r"[\w']+", tok):
            out.append(atiras_tokenre(tok.lower(), cmu_only, fallback_only, all_prons))
        else:
            out.append(tok)
    return ''.join(out)

# -------------------------
# Biztonságos fájlírás
# -------------------------

def ir_biztonsagosan(cel_ut: str, tartalom: str, encoding: str = 'utf-8'):
    """
    Ideiglenes fájlba ír, majd atomikusan cseréli a célt.
    """
    d = os.path.dirname(cel_ut) or '.'
    with tempfile.NamedTemporaryFile('w', delete=False, encoding=encoding, dir=d) as tmp:
        tmp.write(tartalom)
        tmp_name = tmp.name
    os.replace(tmp_name, cel_ut)

# -------------------------
# Mappa feldolgozás
# -------------------------

def feldolgoz_fajl(utvonal: str, *, all_prons: bool, cmu_only: bool, fallback_only: bool, dry_run: bool, encoding: str = 'utf-8') -> Optional[str]:
    """
    Egy fájl feldolgozása. Visszatér hibaüzenettel, ha hiba történt, különben None.
    A forrást átnevezi _original-ra, majd az eredeti nevű fájlba írja a fonetikus verziót.
    """
    try:
        with open(utvonal, 'r', encoding=encoding) as f:
            tartalom = f.read()
        # Soronkénti feldolgozás helyett teljes szöveg megőrzött spacinggel
        atirt = fonetikus_atiras(tartalom, all_prons=all_prons, cmu_only=cmu_only, fallback_only=fallback_only)

        if dry_run:
            return None

        # Eredeti fájl átnevezése
        mappa, fajlnev = os.path.split(utvonal)
        nev, kiterj = os.path.splitext(fajlnev)
        if '_original' in nev:
            # Már feldolgozott
            return None
        original_ut = os.path.join(mappa, f"{nev}_original{kiterj}")

        if os.path.exists(original_ut):
            # Ha már létezik, ne írjuk felül (biztonság)
            return f"Az _original már létezik: {original_ut}"

        # Előbb másoljuk át az eredetit _original-ba, majd írjuk újra az eredeti néven
        # Ehhez: mozgassuk át _original-ba, majd az új tartalmat írjuk az eredeti helyére
        os.replace(utvonal, original_ut)
        ir_biztonsagosan(utvonal, atirt, encoding=encoding)
        return None
    except Exception as e:
        return f"Hiba a fájl feldolgozásakor: {utvonal} -> {e}"

def feldolgoz_rekurzivan(mappa: str, *, all_prons: bool, cmu_only: bool, fallback_only: bool, dry_run: bool, encoding: str = 'utf-8') -> int:
    """
    Rekurzívan feldolgozza a szöveges fájlokat egy mappában.
    Visszatér a hibák számával.
    """
    hibak = 0
    for gyoker, _, fajlok in os.walk(mappa):
        print(f"Feldolgozás alatt: {gyoker}")
        for fajlnev in fajlok:
            if not fajlnev.lower().endswith(('.txt', '.text')):
                continue
            nev, kiterjesztes = os.path.splitext(fajlnev)
            if '_original' in nev:
                print(f"  Kihagyva (már feldolgozott): {fajlnev}")
                continue

            eredeti_utvonal = os.path.join(gyoker, fajlnev)
            print(f"  Feldolgozás: {fajlnev}")
            hiba = feldolgoz_fajl(
                eredeti_utvonal,
                all_prons=all_prons,
                cmu_only=cmu_only,
                fallback_only=fallback_only,
                dry_run=dry_run,
                encoding=encoding
            )
            if hiba:
                hibak += 1
                print(f"    Hiba: {hiba}")
            else:
                if dry_run:
                    print(f"    [Dry-run] OK: {fajlnev}")
                else:
                    print(f"    -> Fonetikus verzió létrehozva: {fajlnev}")
    return hibak

# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Angol szöveges fájlokat alakít át magyaros fonetikus átírásra.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='A feldolgozandó gyökérkönyvtár elérési útja.'
    )
    parser.add_argument(
        '--all-prons',
        action='store_true',
        help='Ha egy szóhoz több CMU kiejtés tartozik, mindet visszaadja | jellel elválasztva.'
    )
    parser.add_argument(
        '--cmu-only',
        action='store_true',
        help='Csak a CMU szótárat használja; ha egy szó nincs benne, változatlanul hagyja.'
    )
    parser.add_argument(
        '--fallback-only',
        action='store_true',
        help='Csak szabályalapú átírás, CMU szótár nélkül.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Nem ír fájlt, csak lefuttatja a feldolgozást jelzésképp.'
    )
    parser.add_argument(
        '--encoding',
        default='utf-8',
        help='Szöveg kódolása (alapértelmezett: utf-8).'
    )

    args = parser.parse_args()

    mappa_utvonal = args.input
    if not os.path.isdir(mappa_utvonal):
        print(f"Hiba: A megadott mappa nem létezik: {mappa_utvonal}", file=sys.stderr)
        sys.exit(2)

    if not args.fallback_only:
        get_cmudict()  # inicializáció, letöltés/betöltés

    hibak = feldolgoz_rekurzivan(
        mappa_utvonal,
        all_prons=args.all_prons,
        cmu_only=args.cmu_only,
        fallback_only=args.fallback_only,
        dry_run=args.dry_run,
        encoding=args.encoding
    )

    if hibak:
        print(f"\nKész. Hibák száma: {hibak}")
        sys.exit(1)
    else:
        print("\nMinden fájl feldolgozva.")
        sys.exit(0)

if __name__ == "__main__":
    main()

import csv
import os
import random
import re
from num2words import num2words

try:
    from phonemizer import phonemize
    from phonemizer.separator import Separator
    HAS_PHONEMIZER = True
    PHONEMIZER_SEPARATOR = Separator(phone=' ', syllable='-', word='|')
except ImportError:  # phonemizer is an optional dependency
    HAS_PHONEMIZER = False
    PHONEMIZER_SEPARATOR = None

HunspellClass = None
HAS_HUNSPELL = False

try:
    from hunspell import Hunspell as HunspellClass  # pyhunspell naming
    HAS_HUNSPELL = True
except ImportError:
    try:
        from hunspell import HunSpell as HunspellClass  # cyhunspell naming
        HAS_HUNSPELL = True
    except ImportError:
        HunspellClass = None
        HAS_HUNSPELL = False

_hunspell_instance = None


def get_hunspell():
    global _hunspell_instance
    if not HAS_HUNSPELL:
        return None
    if _hunspell_instance is not None:
        return _hunspell_instance or None

    dictionary_candidates = ['hu_HU', 'hu']
    hunspell_dirs = []
    env_dir = os.environ.get('HUNSPELL_DICT_PATH')
    if env_dir:
        hunspell_dirs.append(env_dir)
    hunspell_dirs.extend([
        None,  # let Hunspell use its default search path
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hunspell'),
        '/usr/share/hunspell',
        '/usr/local/share/hunspell'
    ])

    def try_init(dic_code, directory):
        try:
            if HunspellClass.__name__ == 'HunSpell':
                dic_filename = f'{dic_code}.dic'
                aff_filename = f'{dic_code}.aff'
                if directory:
                    dic_path = os.path.join(directory, dic_filename)
                    aff_path = os.path.join(directory, aff_filename)
                else:
                    dic_path = dic_filename
                    aff_path = aff_filename
                if os.path.exists(dic_path) and os.path.exists(aff_path):
                    return HunspellClass(dic_path, aff_path)
                # fallback: let HunSpell attempt even if path doesn't exist (may be resolved via default)
                return HunspellClass(dic_path, aff_path)
            else:
                if directory:
                    return HunspellClass(dic_code, hunspell_data_dir=directory)
                return HunspellClass(dic_code)
        except Exception:
            return None

    for dic in dictionary_candidates:
        for directory in hunspell_dirs:
            instance = try_init(dic, directory)
            if instance:
                _hunspell_instance = instance
                return _hunspell_instance

    _hunspell_instance = False
    return None

# Határozzuk meg a normaliser.py könyvtárát
base_dir = os.path.dirname(os.path.abspath(__file__))

# Betűk kiejtése mozaikszavakhoz és alfanumerikus szavakhoz
letter_pronunciations = {
    'A': 'a', 'B': 'bé', 'C': 'cé', 'D': 'dé', 'E': 'e', 'F': 'ef', 'G': 'gé', 'H': 'há',
    'I': 'í', 'J': 'jé', 'K': 'ká', 'L': 'el', 'M': 'em', 'N': 'en', 'O': 'ó', 'P': 'pé',
    'Q': 'kú', 'R': 'er', 'S': 'ess', 'T': 'té', 'U': 'ú', 'V': 'vé', 'W': 'dupla vé',
    'X': 'iksz', 'Y': 'ipszilon', 'Z': 'zé',
    'Á': 'á', 'É': 'é', 'Í': 'í', 'Ó': 'ó', 'Ö': 'ö', 'Ő': 'ő', 'Ú': 'ú', 'Ü': 'ü', 'Ű': 'ű'
}

WORD_TOKEN_PATTERN = re.compile(r"\b[\wÁÉÍÓÖŐÚÜŰáéíóöőúüű'-]+\b")

HUNGARIAN_BASE_CHARS = set('abcdefghijklmnopqrstuvwxyzáéíóöőúüű')
HUNGARIAN_ACCENTED_CHARS = set('áéíóöőúüű')
HUNGARIAN_COMMON_SUFFIXES = (
    't', 'n', 'ban', 'ben', 'nak', 'nek', 'val', 'vel', 'hoz', 'hez', 'höz',
    'ról', 'ről', 'tól', 'től', 'ba', 'be', 'ra', 're',
    'ig', 'ként', 'kkal', 'kkel', 'unk', 'ünk', 'etek', 'tok', 'jük',
    'ások', 'ések', 'áson', 'ésen', 'ásban', 'ésben', 'ással', 'éssel'
)
SORTED_HU_SUFFIXES = tuple(sorted(HUNGARIAN_COMMON_SUFFIXES, key=len, reverse=True))

ESPEAK_HU_PHONEME_MAP = {
    'a': 'a', 'a:': 'á', 'ɑ': 'a', 'ɑ:': 'á', 'ɐ': 'a', 'ɒ': 'a', 'ɒ:': 'á', 'ʌ': 'a',
    'e': 'e', 'e:': 'é', 'ɛ': 'e', 'ɛ:': 'é', 'æ': 'e', 'ə': 'ö', 'ɜ': 'ö', 'ɜ:': 'őr', 'ɚ': 'ör', 'ɝ': 'őr',
    'i': 'i', 'i:': 'í', 'ɪ': 'i', 'ɪ:': 'í',
    'o': 'o', 'o:': 'ó', 'ɔ': 'o', 'ɔ:': 'ó', 'ɞ': 'ö',
    'u': 'u', 'u:': 'ú', 'ʊ': 'u', 'ʊ:': 'ú',
    '2': 'ö', '2:': 'ő', 'y': 'ü', 'y:': 'ű', 'ø': 'ö', 'ø:': 'ő',
    'ju': 'ju', 'ju:': 'jú', 'jʊ': 'jü',
    'aɪ': 'áj', 'aI': 'áj', 'eɪ': 'éj', 'eI': 'éj', 'oʊ': 'ó', 'oU': 'ó', 'uə': 'ua',
    'əʊ': 'ó', 'aʊ': 'au', 'aU': 'au', 'ɔɪ': 'oj', 'OI': 'oj', 'ɪʊ': 'iu',
    'ɪə': 'ia', 'iə': 'ia', 'eə': 'eö', 'ʊə': 'ua',
    'r': 'r', 'ɹ': 'r', 'R': 'r', 'ɾ': 'r',
    'l': 'l', 'ɫ': 'l', 'L': 'ly',
    'j': 'j', 'J': 'j', 'ʎ': 'j',
    'm': 'm', 'n': 'n', 'N': 'ny', 'ŋ': 'ng', 'ɲ': 'ny', 'n̩': 'an', 'm̩': 'm',
    'p': 'p', 'b': 'b', 'pʰ': 'p', 'bʰ': 'b',
    't': 't', 'd': 'd', 'tʰ': 't', 'dʰ': 'd',
    'k': 'k', 'g': 'g', 'ɡ': 'g', 'kʰ': 'k', 'gʰ': 'g',
    'f': 'f', 'v': 'v', 'w': 'v', 'ʍ': 'hu',
    's': 'sz', 'S': 's', 'ʃ': 's', 'ʂ': 's',
    'z': 'z', 'Z': 'zs', 'ʒ': 'zs', 'ʐ': 'zs',
    'ts': 'c', 'dz': 'dz', 'dʒ': 'dzs', 'dZ': 'dzs',
    'tʃ': 'cs', 'tS': 'cs',
    'h': 'h', 'x': 'x', 'χ': 'h',
    'θ': 'sz', 'ð': 'z', 'β': 'v', 'ɣ': 'g',
    'ʔ': 'tt', 'ɸ': 'f', 'ʋ': 'v'
}

PHONEME_MULTI_TOKEN_MAP = {
    ('d', 'ʒ'): 'dzs',
    ('t', 'ʃ'): 'cs',
    ('d', 'Z'): 'dzs',
    ('t', 'S'): 'cs',
    ('j', 'u'): 'ju',
    ('j', 'uː'): 'ju',
    ('j', 'ʊ'): 'jü',
    ('i', 'ə'): 'ia',
    ('i', 'e'): 'ie',
    ('ə', 'ʊ'): 'ó',
    ('ə', 'ɹ'): 'er',
    ('ɑ', 'ɹ'): 'ar',
    ('ɒ', 'ɹ'): 'ar',
    ('ɔ', 'ɹ'): 'or',
    ('ɔ:', 'ɹ'): 'or',
    ('o:', 'ɹ'): 'or',
    ('a:', 'ɹ'): 'ár'
}

PHONEME_SEQUENCE_REPLACEMENTS = [
    (re.compile(r'mju'), 'mu'),
    (re.compile(r'mjú'), 'mu'),
    (re.compile(r'mj'), 'm'),
    (re.compile(r'árk'), 'ark'),
    (re.compile(r'ziam'), 'zeum'),
    (re.compile(r'ziem'), 'zeum'),
    (re.compile(r'ziom'), 'zeum'),
    (re.compile(r'ingtan'), 'ington'),
    (re.compile(r'han$'), 'hatán'),
    #(re.compile(r'manhan'), 'manhattan'),
    (re.compile(r'ávn'), 'ovn'),
    (re.compile(r'ávan'), 'ovn')
]

def pronounce_letters(word):
    return ' '.join(letter_pronunciations.get(char.upper(), char) for char in word)


def is_probably_hungarian(word):
    if not word:
        return False
    speller = get_hunspell()
    if not speller:
        return False
    try:
        return speller.spell(word.lower())
    except Exception:
        return False


def strip_hungarian_suffix(word):
    stripped = word.strip()
    # kezeli a "word-ról" formát
    hyphenated = re.match(r"(.+?)-([a-záéíóöőúüű]+)$", stripped, re.IGNORECASE)
    if hyphenated:
        base, suffix = hyphenated.groups()
        lower_suffix = suffix.lower()
        if lower_suffix in HUNGARIAN_COMMON_SUFFIXES:
            return base, '-' + suffix

    lower = word.lower()
    for suffix in SORTED_HU_SUFFIXES:
        if lower.endswith(suffix) and len(lower) > len(suffix):
            return word[:-len(suffix)], word[-len(suffix):]
    return word, ''

def replace_acronyms(text):
    """
    Mozaikszavak (csak nagybetűkből álló szavak) betűzése.
    """
    # Kizárjuk a római számokat, hogy ne legyenek betűzve
    roman_numerals_pattern = r'[IVXLCDM]+'
    # A pattern biztosítja, hogy csak olyan szavakat célozzon, amelyek legalább egy nem római szám karaktert is tartalmaznak,
    # vagy ha csak római karakterekből állnak, akkor ne legyenek tisztán római számok (pl. MIX, de nem MCM).
    # Ez a feltétel bonyolult lehet, egyszerűbb a római számok explicit kizárása.
    pattern = re.compile(r'\b(?!(?:' + roman_numerals_pattern + r')\b)([A-ZÁÉÍÓÖŐÚÜŰ]{2,})\b') # Legalább két nagybetűs mozaikszavak
    def repl(m):
        acronym = m.group(1)
        # Ellenőrizzük, hogy a mozaikszó nem egyezik-e meg egy ismert római számmal,
        # amit a replace_roman_numerals nem alakított át (mert pl. nincs utána pont).
        # Ez a lépés elhagyható, ha a replace_roman_numerals minden római számot kezelne,
        # vagy ha elfogadjuk, hogy a pont nélküli római számok (pl. "XIV") betűzve lesznek, ha nagybetűsek.
        # A jelenlegi feladatleírás szerint csak a ponttal végződő római számokat kell átírni.
        # Tehát egy "XIV" (pont nélkül) itt mozaikszónak minősülhet, ha a kizárás nem elég erős.
        # A (?!(?:[IVXLCDM]+)\b) kizárásnak ezt kezelnie kellene.
        return pronounce_letters(acronym)
    return pattern.sub(repl, text)

# Római számokat arab számmá alakító segédfüggvény
def roman_to_int(s):
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev = 0
    for char in reversed(s):
        val = roman_map[char]
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total

def replace_roman_numerals(text):
    """
    Minden olyan római számot átír, ami után pont van.
    Pl. 'IX.' -> '9.', 'IV.' -> '4.'
    """
    # Minta: minden római számot keres, ami után pont van
    pattern = re.compile(r'([IVXLCDM]+)\.')
    def repl(m):
        roman = m.group(1)
        try:
            arab = roman_to_int(roman)
            return f"{arab}."
        except KeyError: # Ha nem érvényes római szám, hagyjuk változatlanul
            return m.group(0)
    return pattern.sub(repl, text)

def replace_alphanumeric(text):
    """
    Olyan szavak szétbontása és átírása, amik betűket és számokat egyaránt tartalmaznak.
    A betűket betűzve, a számokat szövegesen írja le.
    Pl. 'A123B' -> 'á százhuszonhárom bé', 'K-9' -> 'ká kilenc'
    """
    # Ez a minta olyan szavakat keres, amelyek tartalmaznak legalább egy betűt és legalább egy számot,
    # és tartalmazhatnak kötőjelet is.
    pattern = re.compile(r'\b([A-Za-zÁÉÍÓÖŐÚÜŰ\d-]*([A-Za-zÁÉÍÓÖŐÚÜŰ]+[A-Za-zÁÉÍÓÖŐÚÜŰ\d-]*\d+|\d+[A-Za-zÁÉÍÓÖŐÚÜŰ\d-]*[A-Za-zÁÉÍÓÖŐÚÜŰ]+)[A-Za-zÁÉÍÓÖŐÚÜŰ\d-]*)\b')

    def repl(m):
        word = m.group(0)
        # Felbontjuk a szót egyes karakterekre
        parts = []
        current = ''
        # Új tokenizáló logika: kötőjel utáni szavakat ne bontsuk fel
        tokens = re.split(r'(-)', word)  # Kötőjelek mentén vág, de megtartja a kötőjeleket
        for token in tokens:
            if not token:
                continue
            if token == '-':
                parts.append(token)
                continue
                
            # Felbontás betű-szám határoknál
            sub_tokens = re.split(r'(\d+)', token)  # Számok mentén vág
            for sub_token in sub_tokens:
                if not sub_token:
                    continue
                parts.append(sub_token)
        if current:
            parts.append(current)
        
        result_parts = []
        for idx, part in enumerate(parts):
            if part == '-':
                # Kötőjel esetén szóközt adunk hozzá
                result_parts.append(' ')
            elif part.isalpha():
                if idx > 0 and parts[idx-1] == '-':
                    # A kötőjel utáni toldalékot hagyjuk érintetlenül
                    result_parts.append(part)
                else:
                    result_parts.append(pronounce_letters(part))
            elif part.isdigit():
                result_parts.append(num2words(int(part), lang='hu'))
            else:
                # Egyéb karakterek megtartása
                result_parts.append(part)
        # Az összes elem szóközzel elválasztva
        return ' '.join(result_parts)
    return pattern.sub(repl, text)

def load_force_changes(filename="force_changes.csv"):
    """
    A force_changes.csv most négy oszlopos:
    key, value, spaces_before, spaces_after
    """
    file_path = os.path.join(base_dir, filename)
    force_changes = {}
    with open(file_path, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            key, value, spaces_before, spaces_after = row
            key = key.strip()
            value = value.strip()
            # számokra castelünk
            before = int(spaces_before.strip())
            after = int(spaces_after.strip())
            force_changes[key] = (value, before, after)
    return force_changes


def load_force_changes_end(filename="force_changes_end.csv"):
    """
    A normalizálás legvégén futó erőltetett cserékhez tartozó CSV két oszlopot tartalmaz:
    key, value
    """
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        return []

    replacements = []
    with open(file_path, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            value = row[1].strip() if len(row) > 1 else ''
            if key:
                replacements.append((key, value))
    return replacements


def apply_force_changes(text, force_changes):
    """
    A CSV-ben megadott szócsere minden előfordulására alkalmazza a változtatást,
    akár szó közepén is.
    """
    for key, (value, before, after) in force_changes.items():
        replacement = ' ' * before + value + ' ' * after
        # Regex használata az összes előfordulás cseréjéhez
        text = re.sub(re.escape(key), replacement, text)
    return text

def load_changes(filenames=("changes.csv", "changes_new.csv")):
    if isinstance(filenames, str):
        filenames = (filenames,)

    changes = {}
    for filename in filenames:
        file_path = os.path.join(base_dir, filename)
        if not os.path.exists(file_path):
            continue
        with open(file_path, encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if not row:
                    continue
                if len(row) < 2:
                    continue
                key = row[0].strip()
                value = row[1].strip()
                if key:
                    changes[key] = value
    return changes

def apply_changes(text, changes):
    # Cserék alkalmazása csak teljes szavakra, betűmérettől függetlenül
    for key, value in changes.items():
        pattern = r'\b{}\b'.format(re.escape(key))
        text = re.sub(pattern, value, text, flags=re.IGNORECASE)
    return text


def apply_force_changes_end(text, replacements):
    """
    Egyszerű (nem regex) cseréket hajt végre a normalizálás legvégén.
    """
    for key, value in replacements:
        text = text.replace(key, value)
    return text


def _map_phoneme_token(token):
    if not token:
        return None
    normalized = token.replace('ː', ':').strip()
    if not normalized:
        return None
    direct = ESPEAK_HU_PHONEME_MAP.get(normalized)
    if direct is not None:
        return direct
    if normalized.endswith(':'):
        base = normalized.rstrip(':')
        long_variant = ESPEAK_HU_PHONEME_MAP.get(base + ':')
        if long_variant is not None:
            return long_variant
        base_variant = ESPEAK_HU_PHONEME_MAP.get(base)
        if base_variant is not None:
            return base_variant + base_variant[-1]
    base_direct = ESPEAK_HU_PHONEME_MAP.get(normalized.lower())
    if base_direct is not None:
        return base_direct
    return normalized.replace(':', '')


def _should_use_english_backend(word):
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", word)) and not any(
        ch in HUNGARIAN_ACCENTED_CHARS for ch in word.lower()
    )


def _map_token_sequence(tokens):
    mapped_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i].strip()
        if not token or token in {'ˈ', 'ˌ', '.'}:
            i += 1
            continue
        normalized = token.replace('|', '').replace('ː', ':')
        if i + 1 < len(tokens):
            next_token = tokens[i + 1].strip().replace('|', '').replace('ː', ':')
            pair = (normalized, next_token)
            if pair in PHONEME_MULTI_TOKEN_MAP:
                mapped_tokens.append(PHONEME_MULTI_TOKEN_MAP[pair])
                i += 2
                continue
        mapped = _map_phoneme_token(normalized)
        if mapped:
            mapped_tokens.append(mapped)
        i += 1
    if not mapped_tokens:
        return None
    text = ''.join(mapped_tokens)
    for pattern, replacement in PHONEME_SEQUENCE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return _finalize_phoneme_text(text.lower())


def _finalize_phoneme_text(text):
    char_replacements = {
        'ɑ': 'a',
        'ɒ': 'a',
        'ɔ': 'o',
        'ɹ': 'r',
        'ʔ': '',
        '̩': '',
        'ɡ': 'g',
        'ʃ': 's',
        'ʒ': 'zs',
        '|': '',
        'ø': 'ö'
    }
    for src, dst in char_replacements.items():
        text = text.replace(src, dst)
    return text


def phonemize_to_hungarian(word):
    if not HAS_PHONEMIZER or not word:
        return None

    stripped = word.strip("'\"")
    if not stripped:
        return None
    base_word, suffix = strip_hungarian_suffix(stripped)
    target_word = base_word if suffix else stripped

    languages = []
    if _should_use_english_backend(target_word):
        languages.append('en-us')
    languages.append('hu')

    for language in languages:
        try:
            phoneme_sequence = phonemize(
                target_word.strip('-'),
                language=language,
                backend='espeak',
                separator=PHONEMIZER_SEPARATOR,
                strip=True,
                preserve_punctuation=False,
                with_stress=False,
                njobs=1
            )
        except TypeError:
            try:
                phoneme_sequence = phonemize(
                    stripped,
                    language=language,
                    backend='espeak',
                    separator=PHONEMIZER_SEPARATOR,
                    strip=True,
                    preserve_punctuation=False,
                    njobs=1
                )
            except Exception:
                continue
        except Exception:
            continue

        if not phoneme_sequence:
            continue

        tokens = [token.replace('|', '') for token in phoneme_sequence.split() if token and token != '|']
        mapped = _map_token_sequence(tokens)
        if mapped:
            raw_phonemes = ' '.join(tokens)
            mapped_text = mapped
            if suffix:
                if suffix.startswith('-'):
                    mapped_text += suffix
                else:
                    mapped_text += suffix.lower()
            return mapped_text, raw_phonemes

    return None


def should_phonemize_word(word, existing_lower):
    if not word:
        return False
    stripped = word.strip("'\"")
    if not stripped:
        return False
    lower = stripped.lower()
    if lower in existing_lower:
        return False
    if is_probably_hungarian(stripped):
        return False
    # Ha a hunspell modul nem elérhető, minden szót phonemizálunk
    if not HAS_HUNSPELL:
        return True
    return True


def collect_new_changes(text, existing_changes, skip_words=None):
    if not HAS_PHONEMIZER:
        return {}

    existing_lower = {key.lower() for key in existing_changes}
    skip_lower = {word.lower() for word in skip_words} if skip_words else set()
    candidates = WORD_TOKEN_PATTERN.findall(text)
    new_entries = {}

    for word in candidates:
        if skip_lower and word.lower() in skip_lower:
            continue
        if not should_phonemize_word(word, existing_lower):
            continue
        phonetic = phonemize_to_hungarian(word)
        if not phonetic:
            continue
        mapped_value, raw_phoneme = phonetic
        if mapped_value.lower() == word.lower():
            continue
        if word not in new_entries:
            new_entries[word] = (mapped_value, raw_phoneme)
            existing_lower.add(word.lower())

    return new_entries


def append_changes_to_file(new_entries, filename="changes_new.csv"):
    if not new_entries:
        return
    file_path = os.path.join(base_dir, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, (value, phoneme) in new_entries.items():
            writer.writerow([key, value, phoneme])


def replace_ordinals(text):
    """
    Bármilyen nagyságú arab számból álló sorszámot (pl. 1233.) 
    átír num2words segítségével magyar ordítóvá.
    A patrón biztosítja, hogy a mondatvégén álló számot ponttal ne bántsa.
    Kizárja az éveket, hogy ne alakítsa át sorszámmá.
    """
    # Kizárjuk az éveket (4 számjegyű számok ponttal)
    pattern = re.compile(r'\b(\d{1,3}|\d{5,})\.(?!\s*$|\s*[\.!\?])')
    def repl(m):
        num = int(m.group(1))
        return num2words(num, to='ordinal', lang='hu')
    return pattern.sub(repl, text)

months = {
    'jan.': 'január',
    'feb.': 'február',
    'márc.': 'március',
    'már.': 'március',
    'ápr.': 'április',
    'máj.': 'május',
    'jún.': 'június',
    'júl.': 'július',
    'aug.': 'augusztus',
    'szept.': 'szeptember',
    'szep.': 'szeptember',
    'okt.': 'október',
    'nov.': 'november',
    'dec.': 'december',
}

months_numbers = {
    1: 'január', 'I': 'január',
    2: 'február', 'II': 'február',
    3: 'március', 'III': 'március',
    4: 'április', 'IV': 'április',
    5: 'május', 'V': 'május',
    6: 'június', 'VI': 'június',
    7: 'július', 'VII': 'július',
    8: 'augusztus', 'VIII': 'augusztus',
    9: 'szeptember', 'IX': 'szeptember',
    10: 'október', 'X': 'október',
    11: 'november', 'XI': 'november',
    12: 'december', 'XII': 'december',
}

day_words = {
    1: 'elseje',
    2: 'másodika',
    3: 'harmadika',
    4: 'negyedike',
    5: 'ötödike',
    6: 'hatodika',
    7: 'hetedike',
    8: 'nyolcadika',
    9: 'kilencedike',
    10: 'tizedike',
    11: 'tizenegyedike',
    12: 'tizenkettedike',
    13: 'tizenharmadika',
    14: 'tizennegyedike',
    15: 'tizenötödike',
    16: 'tizenhatodika',
    17: 'tizenhetedike',
    18: 'tizennyolcadika',
    19: 'tizenkilencedike',
    20: 'huszadika',
    21: 'huszonegyedike',
    22: 'huszonkettedike',
    23: 'huszonharmadika',
    24: 'huszonnegyedike',
    25: 'huszonötödike',
    26: 'huszonhatodika',
    27: 'huszonhetedike',
    28: 'huszonnyolcadika',
    29: 'huszonkilencedike',
    30: 'harmincadika',
    31: 'harmincegyedike',
}

def day_to_text(day_num):
    # Napok átírása szöveges formára, pl. 1 -> elseje
    return day_words.get(day_num, num2words(day_num, lang='hu') + 'ika') # Fallback, bár 1-31 között nem kellene

def format_date_text_new(year_num, month_name_str, day_num):
    """
    Formázza a dátumot szövegesen: "év hónap napadik".
    year_num: int (pl. 2025)
    month_name_str: string (teljes hónapnév, pl. "június")
    day_num: int (pl. 1)
    """
    # Az év kardinális számként (nem sorszám)
    year_text = num2words(year_num, lang='hu')
    # A nap sorszámként (tizedike, elseje stb.)
    day_text = day_to_text(day_num)
    return f'{year_text} {month_name_str} {day_text}'

def replace_dates(text):
    # Segédfüggvény a dátumok szöveges formára alakításához (az új, egységesített formátum_date_text_new-t használva)

    # Kombinált hónap regexek
    # Fontos, hogy a leghosszabb hónapnevek legyenek elöl a regexben, hogy elkerüljük a részleges egyezéseket
    # pl. "júl." vs "július". A re.escape szükséges.
    sorted_month_values = sorted(months.values(), key=len, reverse=True)
    month_names_hu_regex = '|'.join(re.escape(m_val) for m_val in sorted_month_values)
    
    sorted_month_keys = sorted(months.keys(), key=len, reverse=True)
    month_abbrs_hu_regex = '|'.join(re.escape(k) for k in sorted_month_keys)
    
    roman_numerals_regex = r'(M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))' # Római számok regex, capturing group hozzáadva

    # Dátumformátumok kezelése
    patterns = [
        # 1. 2025. június 1. (teljes hónapnévvel)
        (r'(\d{4})\.\s*(' + month_names_hu_regex + r')\s*(\d{1,2})\.(?!\d)', 
         lambda m: format_date_text_new(int(m.group(1)), m.group(2), int(m.group(3)))),
        # 2. 2025. 06. 01. (számmal írt hónap)
        (r'(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.(?!\d)', 
         lambda m: format_date_text_new(int(m.group(1)), months_numbers[int(m.group(2))], int(m.group(3)))),
        # 3. 2025.06.01. (számmal írt hónap, szóköz nélkül)
        (r'(\d{4})\.(\d{2})\.(\d{2})\.(?!\d)', 
         lambda m: format_date_text_new(int(m.group(1)), months_numbers[int(m.group(2))], int(m.group(3)))),
        # 4. 2025. jún. 1. (rövidített hónapnévvel)
        (r'(\d{4})\.\s*(' + month_abbrs_hu_regex + r')\s*(\d{1,2})\.(?!\d)', 
         lambda m: format_date_text_new(int(m.group(1)), months[m.group(2).lower()], int(m.group(3)))),
        # 5. 2025.jún.1. (rövidített hónapnévvel, szóköz nélkül)
        (r'(\d{4})\.(' + month_abbrs_hu_regex + r')(\d{1,2})\.(?!\d)', 
         lambda m: format_date_text_new(int(m.group(1)), months[m.group(2).lower()], int(m.group(3)))),
        # 6. 2025. VI. 1. (római számos hónap)
        (r'(\d{4})\.\s*(' + roman_numerals_regex + r')\s*(\d{1,2})\.(?!\d)',
         lambda m: format_date_text_new(int(m.group(1)), months_numbers[roman_to_int(m.group(2).upper())], int(m.group(3)))),
        # 7. 2025.VI.1. (római számos hónap, szóköz nélkül)
        (r'(\d{4})\.(' + roman_numerals_regex + r')(\d{1,2})\.(?!\d)',
         lambda m: format_date_text_new(int(m.group(1)), months_numbers[roman_to_int(m.group(2).upper())], int(m.group(3)))),
        # 8. 2025-06-01 (kötőjeles formátum)
        (r'(\d{4})-(\d{2})-(\d{2})(?!\d)', 
         lambda m: format_date_text_new(int(m.group(1)), months_numbers[int(m.group(2))], int(m.group(3)))),
        # 9. 2024. aug. 10-én (rövidített hónapnévvel, kötőjeles toldalékkal)
        (r'(\d{4})\.\s*(' + month_abbrs_hu_regex + r')\s*(\d{1,2})-([a-záéíóöőúüű]+)\b', 
         lambda m: format_date_text_new(int(m.group(1)), months[m.group(2).lower()], int(m.group(3))) + m.group(4))
    ]

    for pattern_str, repl_func in patterns:
        text = re.sub(pattern_str, repl_func, text)

    # Külön kezeli az "N-án" vagy "N-én" formátumot (ez már létezett)
    pattern_day_suffix = re.compile(r'\b(\d{1,2})-(án|én)\b')
    def repl_day_suffix(m):
        day = int(m.group(1))
        suffix = m.group(2)
        ordinal = num2words(day, to='ordinal', lang='hu')
        return ordinal + suffix
    text = pattern_day_suffix.sub(repl_day_suffix, text)
    
    # Maradék rövid hónapnevek: dec. -> december stb.
    # (ezeket nem köti nap vagy év, csak önállóan szerepelnek)
    month_abbrs_only = '|'.join(re.escape(k) for k in months.keys())
    pattern_month_only = re.compile(r'(?<!\w)(' + month_abbrs_only + r')(?!\w)')
    def repl_month_only(m):
        abb = m.group(1).lower()
        return months.get(abb, abb)
    text = pattern_month_only.sub(repl_month_only, text)

    # Maradék rövid hónapnevek: dec. -> december stb.
    # (ezeket nem köti nap vagy év, csak önállóan szerepelnek)
    month_abbrs_only = '|'.join(re.escape(k) for k in months.keys())
    pattern_month_only = re.compile(r'(?<!\w)(' + month_abbrs_only + r')(?!\w)')
    def repl_month_only(m):
        abb = m.group(1).lower()
        return months.get(abb, abb)
    text = pattern_month_only.sub(repl_month_only, text)

    return text

def replace_times(text):
    """
    Időpontok átírása két formátum közül véletlenszerűen választva:
    1. "óra perc" forma (pl. "hét óra harminc perc")
    2. "óra perc" forma (pl. "hét harminc")
    A másodperceket külön hozzáadjuk (pl. "15 óra negyvenöt perc harminc másodperc")
    Ha van másodperc, akkor mindig az első formátumot használjuk.
    """
    pattern = re.compile(r'(\d{1,2}):(\d{2})(?::(\d{2}))?(-kor)?\b')
    def repl(match):
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = match.group(3)
        has_kor = match.group(4) == '-kor'
        
        hour_text = num2words(hour, lang='hu')
        minute_text = num2words(minute, lang='hu') if minute != 0 else ""

        # Ha van másodperc, akkor az első formátumot használjuk (óra, perc, másodperc)
        if second:
            second_val = int(second)
            second_text = num2words(second_val, lang='hu')
            time_str = f'{hour_text} óra {minute_text} perc {second_text} másodperc'
        else:
            # Ha a perc 00, akkor csak az órát írjuk ki
            if minute == 0:
                time_str = f'{hour_text} óra'
            else:
                # Véletlenszerű választás a két formátum között
                if random.choice([True, False]):
                    time_str = f'{hour_text} óra {minute_text} perc'
                else:
                    time_str = f'{hour_text} {minute_text}'
        
        # "kor" hozzáadása, ha szükséges
        if has_kor:
            time_str += 'kor'  # Szóköz nélkül csatoljuk
        
        return time_str
    text = pattern.sub(repl, text)
    return text

def replace_numbers(text):
    # Számok átírása szöveges megfelelőjükre
    pattern = r'\b\d+\b'
    def repl(match):
        num = int(match.group(0))
        return num2words(num, lang='hu')
    text = re.sub(pattern, repl, text)
    return text

def remove_duplicate_spaces(text):
    # Többszörös szóközök eltávolítása
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_unwanted_characters(text):
    # Az eltávolítandó karakterek listája (kötőjel kivéve)
    unwanted_characters = r'[*\"\'\:\(\)\/#@\[\]\{\}]'
    # Eltávolítjuk az összes felsorolt karaktert
    return re.sub(unwanted_characters, ' ', text)

def add_prefix(text):
    # Hozzáadja a "... " szöveget a szöveg elejéhez
    return '... ' + text.lstrip()

def convert_to_lowercase(text):
    """Az egész szöveg kisbetűssé alakítása"""
    return text.lower()

def _normalize_pipeline(text, *, include_changes_new=True, enable_phoneme_updates=True):
    # A szöveg normalizálása a megadott lépésekkel
    force_changes = load_force_changes('force_changes.csv')
    force_changes_end = load_force_changes_end('force_changes_end.csv')
    changes_source = ("changes.csv", "changes_new.csv") if include_changes_new else "changes.csv"
    changes = load_changes(changes_source)
    original_tokens_lower = (
        {token.lower() for token in WORD_TOKEN_PATTERN.findall(text)}
        if enable_phoneme_updates else set()
    )

    text = replace_roman_numerals(text)  # Római számok arabra (pl. IV. -> 4.)
    text = replace_times(text)  # Időpontok kezelése előbb, hogy a "-kor" még változatlan legyen
    text = apply_changes(text, changes)
    text = apply_force_changes(text, force_changes)
    #text = replace_acronyms(text)
    text = replace_alphanumeric(text)
    text = replace_dates(text) # Dátumok átalakítása
    text = replace_ordinals(text) # Sorszámok (pl. 4. -> negyedik)
    text = replace_numbers(text) # Számok szöveggé
    
    if enable_phoneme_updates:
        current_tokens = WORD_TOKEN_PATTERN.findall(text)
        skip_for_new_changes = {
            token.lower() for token in current_tokens
            if token.lower() not in original_tokens_lower
        }
        new_changes = collect_new_changes(text, changes, skip_words=skip_for_new_changes)
        if new_changes:
            append_changes_to_file(new_changes)
            new_simple_changes = {key: value for key, (value, _) in new_changes.items()}
            changes.update(new_simple_changes)
            text = apply_changes(text, new_simple_changes)
        
    text = remove_unwanted_characters(text) 
    # Kivételek kezelése kötőjelekkel
    exceptions = {
        "egy-egy": "egy egy",
        "két-két": "két két",
        "három-három": "három három",
        "négy-négy": "négy négy",
        "öt-öt": "öt öt"
    }
    
    for pattern, replacement in exceptions.items():
        text = re.sub(r'\b' + re.escape(pattern) + r'\b', replacement, text)
    
    # Kötőjelek eltávolítása (szóköz helyett egybeírás)
    text = re.sub(r'-', '', text)
    
    text = remove_duplicate_spaces(text)
    text = add_prefix(text)
    text = convert_to_lowercase(text)
    # A végső, TTS-specifikus felülírásoknak minden más módosítás után kell lefutnia.
    text = apply_force_changes_end(text, force_changes_end)

    return text

def normalize_helper(text):
    return _normalize_pipeline(text, include_changes_new=True, enable_phoneme_updates=True)

def normalize(text):
    return _normalize_pipeline(text, include_changes_new=False, enable_phoneme_updates=False)

# Tesztkód eltávolítva, átkerült a normalizer_test.py fájlba

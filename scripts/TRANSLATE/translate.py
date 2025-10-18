import os
import sys
import argparse
import deepl
import json
from pathlib import Path
from typing import Optional, Tuple

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode


def get_project_root() -> Path:
    """Visszaadja a projekt gyökerét, ahol a config.json található."""
    for candidate in Path(__file__).resolve().parents:
        config_candidate = candidate / "config.json"
        if config_candidate.is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")


def load_config() -> Tuple[dict, Path]:
    """Betölti a config.json-t, és visszaadja a konfigurációt a projekt gyökérrel együtt."""
    try:
        project_root = get_project_root()
    except FileNotFoundError as exc:
        print(f"Hiba a projektgyökér meghatározásakor: {exc}")
        sys.exit(1)

    config_path = project_root / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)
        return config, project_root
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Hiba a konfiguráció betöltésekor ({config_path}): {exc}")
        sys.exit(1)


def resolve_project_paths(project_name: str, config: dict, project_root: Path) -> Tuple[Path, Path]:
    """A config alapján meghatározza a projekt bemeneti- és kimeneti mappáit."""
    try:
        workdir = project_root / config["DIRECTORIES"]["workdir"]
        subdirs = config["PROJECT_SUBDIRS"]
        input_dir = workdir / project_name / subdirs["separated_audio_speech"]
        output_dir = workdir / project_name / subdirs["translated"]
    except KeyError as exc:
        print(f"Hiba: Hiányzó kulcs a config.json fájlban: {exc}")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Hiba: A bemeneti mappa nem található: {input_dir}")
        sys.exit(1)

    return input_dir, output_dir


def resolve_language_params(
    config: dict,
    input_language: Optional[str],
    output_language: Optional[str],
) -> Tuple[str, str]:
    """Nyelvi paraméterek előállítása CLI / config alapján."""
    defaults = config.get("CONFIG", {})
    default_input = str(defaults.get("default_source_lang", "en") or "en").upper()
    default_output = str(defaults.get("default_target_lang", "hu") or "hu").upper()

    resolved_input = (input_language or default_input).strip().upper()
    resolved_output = (output_language or default_output).strip().upper()
    return resolved_input, resolved_output

def find_json_file(directory):
    """
    Megkeresi az egyetlen .json fájlt a megadott könyvtárban.
    Visszaadja a fájl nevét, vagy None-t, ha hiba történik.
    """
    try:
        json_files = [f for f in os.listdir(directory) if f.lower().endswith('.json')]
    except FileNotFoundError:
        print(f"Hiba: A bemeneti könyvtár nem található: {directory}")
        return None, "directory_not_found"

    if len(json_files) == 0:
        print(f"Hiba: Nem található JSON fájl a(z) '{directory}' könyvtárban.")
        return None, "no_json_found"
    
    if len(json_files) > 1:
        print(f"Hiba: Több JSON fájl található a(z) '{directory}' könyvtárban. Csak egyet tud feldolgozni.")
        return None, "multiple_jsons_found"
    
    return json_files[0], None

def main(project_name: str, input_lang: Optional[str], output_lang: Optional[str], auth_key: str):
    config, project_root = load_config()
    input_dir_path, output_dir_path = resolve_project_paths(project_name, config, project_root)
    input_dir = str(input_dir_path)
    output_dir = str(output_dir_path)
    input_lang_resolved, output_lang_resolved = resolve_language_params(config, input_lang, output_lang)

    print("Projekt beállítások betöltve:")
    print(f"  - Projekt név:             {project_name}")
    print(f"  - Bemeneti mappa:          {input_dir_path}")
    print(f"  - Kimeneti mappa:          {output_dir_path}")
    print(f"  - Bemeneti nyelv (DeepL):  {input_lang_resolved}")
    print(f"  - Kimeneti nyelv (DeepL):  {output_lang_resolved}")

    # DeepL fordító inicializálása
    translator = deepl.Translator(auth_key)

    # A bemeneti JSON fájl megkeresése
    input_filename, error = find_json_file(input_dir)
    if error:
        return # Kilépés, ha hiba történt
    
    input_filepath = os.path.join(input_dir, input_filename)
    print(f"Bemeneti fájl feldolgozása: {input_filepath}")

    # A kimeneti könyvtár létrehozása, ha nem létezik
    os.makedirs(output_dir, exist_ok=True)

    # A kimeneti fájl elérési útja (ugyanazzal a névvel, mint a bemeneti)
    output_filepath = os.path.join(output_dir, input_filename)

    # Bemeneti JSON fájl betöltése
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Hiba: A bemeneti fájl nem érvényes JSON formátumú: {input_filepath}")
        return

    # Ellenőrizzük, hogy a JSON tartalmazza-e a 'segments' kulcsot
    if 'segments' not in data or not isinstance(data['segments'], list):
        print("Hiba: A JSON fájlnak tartalmaznia kell egy 'segments' kulcsot, amelynek értéke egy lista.")
        return

    original_segments = data['segments']
    
    # Szövegek kigyűjtése, amelyek már tartalmaznak fordítást, hogy ne fordítsuk le őket újra
    texts_to_translate = []
    indices_to_translate = []
    for i, segment in enumerate(original_segments):
        # Csak akkor fordítjuk, ha van 'text' és nincs még 'translated_text' kulcs, vagy ha a meglévő üres
        if segment.get('text', '').strip() and not segment.get('translated_text', '').strip():
            texts_to_translate.append(segment.get('text').strip() or " ")
            indices_to_translate.append(i)

    # Ha nincs mit fordítani, akkor is elmentjük a fájlt a kimeneti helyre,
    # hogy a pipeline további lépései megkapják a fájlt.
    if not texts_to_translate:
        print("Nincs új, fordítandó szöveg a fájlban.")
        with open(output_filepath, 'w', encoding='utf-8') as f:
            # Hozzáadjuk az üres 'translated_text' mezőt azokhoz, ahol hiányzik
            for segment in original_segments:
                segment.setdefault('translated_text', '')
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"A fájl (módosítás nélkül) átmásolva ide: {output_filepath}")
        return

    # A szegmensek összefűzése egyetlen szöveggé
    text_block = '\n'.join(texts_to_translate)

    print(f"Újonnan fordítandó szegmensek száma: {len(texts_to_translate)}")
    print("A teljes szövegblokk fordítása...")

    # A teljes szövegblokk lefordítása
    try:
        result = translator.translate_text(
            text_block,
            source_lang=input_lang_resolved,
            target_lang=output_lang_resolved
        )
        translated_lines = result.text.split('\n')
    except deepl.DeepLException as e:
        print(f"Hiba történt a DeepL API hívásakor: {e}")
        return

    # Ellenőrizzük a szegmensek számát
    if len(translated_lines) != len(texts_to_translate):
        print("Figyelmeztetés: A fordított szegmensek száma nem egyezik a fordításra küldöttek számával.")
        print("A fordítás megszakítva a hibás kimenet elkerülése érdekében.")
        return

    # A fordítások beillesztése a 'translated_text' kulcs alá a megfelelő helyre
    for i, translated_line in enumerate(translated_lines):
        original_index = indices_to_translate[i]
        data['segments'][original_index]['translated_text'] = translated_line.strip()
    
    # Hozzáadjuk az üres 'translated_text' mezőt azokhoz a szegmensekhez, ahol nem volt fordítás
    for segment in data['segments']:
        segment.setdefault('translated_text', '')

    # A kiegészített adatok kiírása a kimeneti fájlba
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nFordítás befejezve. A kiegészített fájl a(z) '{output_filepath}' helyre mentve.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Projekt-alapú JSON fordítás DeepL segítségével. A config.json alapján keresi a feldolgozandó fájlokat.'
    )
    parser.add_argument(
        '-p',
        '--project-name',
        required=True,
        help="A workdir-en belüli projektmappa neve."
    )
    parser.add_argument(
        '-input_language',
        '--input-language',
        dest='input_language',
        default=None,
        help='A bemeneti nyelv kódja (pl. EN, HU). Ha nincs megadva, a config.json default_source_lang értéke lesz használva.'
    )
    parser.add_argument(
        '-output_language',
        '--output-language',
        dest='output_language',
        default=None,
        help='A kimeneti nyelv kódja (pl. EN, HU). Ha nincs megadva, a config.json default_target_lang értéke lesz használva.'
    )
    parser.add_argument(
        '-auth_key',
        '--auth-key',
        required=True,
        help='A DeepL API hitelesítési kulcs.'
    )
    add_debug_argument(parser)

    args = parser.parse_args()
    configure_debug_mode(args.debug)
    main(args.project_name, args.input_language, args.output_language, args.auth_key)

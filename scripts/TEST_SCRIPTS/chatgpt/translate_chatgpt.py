import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI, OpenAIError

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode
from tools.json_sanitizer import sanitize_translation_fields

logger = logging.getLogger("translate_chatgpt")


def get_project_root() -> Path:
    """Locate the repository root based on config.json."""
    for candidate in Path(__file__).resolve().parents:
        config_candidate = candidate / "config.json"
        if config_candidate.is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")


def load_config() -> Tuple[Dict[str, Any], Path]:
    """Load config.json and return it together with the project root."""
    try:
        project_root = get_project_root()
    except FileNotFoundError as exc:
        logger.error("Hiba a projektgyökér meghatározásakor: %s", exc)
        sys.exit(1)

    config_path = project_root / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as fp:
            config: Dict[str, Any] = json.load(fp)
        return config, project_root
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Hiba a konfiguráció betöltésekor (%s): %s", config_path, exc)
        sys.exit(1)


def get_keyholder_path(project_root: Path) -> Path:
    """Return the path to keyholder.json."""
    return project_root / "keyholder.json"


def save_api_key(project_root: Path, api_key: str) -> None:
    """Persist the ChatGPT API key to keyholder.json using base64 encoding."""
    keyholder_path = get_keyholder_path(project_root)
    try:
        data: Dict[str, Any] = {}
        if keyholder_path.exists():
            with open(keyholder_path, "r", encoding="utf-8") as fp:
                try:
                    data = json.load(fp)
                except json.JSONDecodeError:
                    logger.warning("A keyholder.json sérült vagy üres, új fájl készül.")
                    data = {}
        encoded_key = base64.b64encode(api_key.encode("utf-8")).decode("utf-8")
        data["chatgpt_api_key"] = encoded_key
        with open(keyholder_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)
        logger.info("ChatGPT API kulcs elmentve: %s", keyholder_path)
    except Exception as exc:
        logger.error("Hiba az API kulcs mentésekor: %s", exc)


def load_api_key(project_root: Path) -> Optional[str]:
    """Load the ChatGPT API key from keyholder.json."""
    keyholder_path = get_keyholder_path(project_root)
    if not keyholder_path.exists():
        return None
    try:
        with open(keyholder_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        encoded_key = data.get("chatgpt_api_key")
        if not encoded_key:
            return None
        return base64.b64decode(encoded_key.encode("utf-8")).decode("utf-8")
    except (json.JSONDecodeError, KeyError, base64.binascii.Error) as exc:
        logger.error("Hiba az API kulcs betöltésekor (%s): %s", keyholder_path, exc)
        return None
    except Exception as exc:
        logger.error("Váratlan hiba az API kulcs betöltésekor: %s", exc)
        return None


def resolve_project_paths(project_name: str, config: Dict[str, Any], project_root: Path) -> Tuple[Path, Path]:
    """Determine input/output directories for the project."""
    try:
        workdir = project_root / config["DIRECTORIES"]["workdir"]
        subdirs = config["PROJECT_SUBDIRS"]
        input_dir = workdir / project_name / subdirs["separated_audio_speech"]
        output_dir = workdir / project_name / subdirs["translated"]
    except KeyError as exc:
        logger.error("Hiányzó kulcs a config.json fájlban: %s", exc)
        sys.exit(1)

    if not input_dir.is_dir():
        logger.error("A bemeneti mappa nem található: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def resolve_language_params(config: Dict[str, Any], input_language: Optional[str], output_language: Optional[str]) -> Tuple[str, str]:
    """Resolve language parameters using CLI values or config defaults."""
    defaults = config.get("CONFIG", {})
    default_input = str(defaults.get("default_source_lang", "en") or "en").upper()
    default_output = str(defaults.get("default_target_lang", "hu") or "hu").upper()
    resolved_input = (input_language or default_input).strip().upper()
    resolved_output = (output_language or default_output).strip().upper()
    return resolved_input, resolved_output


def get_lang_name(lang_code: str) -> str:
    """Return a human friendly language name."""
    lang_map = {"EN": "English", "HU": "Hungarian", "DE": "German", "FR": "French", "ES": "Spanish"}
    return lang_map.get(lang_code.upper(), lang_code)


def load_glossary(glossary_path: Optional[str], project_root: Path) -> Optional[Dict[str, str]]:
    """Load a glossary JSON file that enforces consistent terminology."""
    if not glossary_path:
        return None

    raw_path = Path(glossary_path).expanduser()
    if not raw_path.is_absolute():
        candidate = project_root / raw_path
        if candidate.exists():
            raw_path = candidate

    if not raw_path.exists():
        logger.warning("A megadott glosszárium fájl nem található: %s", raw_path)
        return None

    try:
        with open(raw_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError as exc:
        logger.warning("A glosszárium fájl nem érvényes JSON: %s", exc)
        return None
    except OSError as exc:
        logger.warning("A glosszárium fájl nem olvasható: %s", exc)
        return None

    if not isinstance(data, dict):
        logger.warning("A glosszárium JSON nem kulcs-érték párokat tartalmaz: %s", raw_path)
        return None

    glossary: Dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            glossary[key.strip()] = value.strip()
    if not glossary:
        logger.warning("A glosszárium fájl nem tartalmaz használható bejegyzést: %s", raw_path)
        return None
    logger.info("Glosszárium betöltve: %s (%s tétel)", raw_path, len(glossary))
    return glossary


def find_json_file(directory: Path) -> Tuple[Optional[str], Optional[str]]:
    """Find the single JSON file in a directory."""
    try:
        json_files = [f for f in os.listdir(directory) if f.lower().endswith(".json")]
    except FileNotFoundError:
        logger.error("Hiba: A bemeneti könyvtár nem található: %s", directory)
        return None, "directory_not_found"
    if not json_files:
        logger.error("Hiba: Nem található JSON fájl a(z) '%s' könyvtárban.", directory)
        return None, "no_json_found"
    if len(json_files) > 1:
        logger.error("Hiba: Több JSON fájl található a(z) '%s' könyvtárban.", directory)
        return None, "multiple_jsons_found"
    return json_files[0], None


def create_smart_chunks(segments: List[Dict[str, Any]], min_size: int = 50, max_size: int = 100, gap_threshold: float = 5.0) -> List[List[Dict[str, Any]]]:
    """Chunk the segments list into smart batches for translation."""
    chunks: List[List[Dict[str, Any]]] = []
    current_pos = 0
    while current_pos < len(segments):
        chunk_start = current_pos
        if (len(segments) - chunk_start) <= min_size:
            best_split_point = len(segments)
        else:
            best_split_point = -1
            search_end = min(chunk_start + max_size, len(segments) - 1)
            for i in range(chunk_start + min_size - 1, search_end):
                try:
                    gap = segments[i + 1].get("start", 0) - segments[i].get("end", 0)
                    if gap >= gap_threshold:
                        best_split_point = i + 1
                        break
                except (TypeError, KeyError):
                    continue
            if best_split_point == -1:
                max_gap = -1.0
                best_split_point = min(chunk_start + max_size, len(segments))
                for i in range(chunk_start + min_size - 1, search_end):
                    try:
                        gap = segments[i + 1].get("start", 0) - segments[i].get("end", 0)
                        if gap > max_gap:
                            max_gap = gap
                            best_split_point = i + 1
                    except (TypeError, KeyError):
                        continue
        chunks.append(segments[chunk_start:best_split_point])
        current_pos = best_split_point
    return chunks


def build_system_prompt(
    lang_from: str,
    lang_to: str,
    allow_sensitive: bool,
    context: Optional[str],
    override: Optional[str],
    tone: Optional[str],
    target_audience: Optional[str],
    platform: Optional[str],
    style_notes: Optional[str],
    glossary: Optional[Dict[str, str]],
) -> str:
    """Construct a strategy-aware system prompt unless the CLI override is used."""

    if override and override.strip():
        return override.strip()

    input_lang_name = get_lang_name(lang_from)
    output_lang_name = get_lang_name(lang_to)
    locale = "HU" if lang_from.strip().upper() == "HU" else "EN"

    def add_glossary_line() -> Optional[str]:
        if not glossary:
            return None
        glossary_pairs = [f"{src} -> {dst}" for src, dst in list(glossary.items())[:20]]
        joined = "; ".join(glossary_pairs)
        if locale == "HU":
            return (
                "Terminológiai emlékeztető: "
                f"{joined}. Kövesd ezeket minden sorban, és ha új megfelelőt alkotsz, maradj következetes."
            )
        return (
            "Terminology reminders: "
            f"{joined}. Always reuse these mappings and stay consistent if you coin new ones."
        )

    instructions: List[str] = []

    if locale == "HU":
        instructions.append(
            (
                "Egy tapasztalt feliratfordító vagy, aki "
                f"{input_lang_name} forrásból {output_lang_name} nyelvre dolgozik. Kezeld a feladatot profi, streamingre kész lokalizációként."
            )
        )
        instructions.append(
            "Fordíts minden számozott sort külön-külön, tartsd meg a számozást és a sorrendet (formátum: `1. fordítás`). Ne vond össze vagy tördeljen máshol a sorokat."
        )
        instructions.append(
            "A fordítás hossza nagyjából egyezzen meg az eredetiével, hogy illeszkedjen az időzítéshez, és maradjon legfeljebb két feliratsor."
        )
        instructions.append(
            f"Használj természetes, beszélt {output_lang_name} nyelvet. Kerüld a tükörfordítást; idiomatikus, hétköznapi megoldásokat válassz."
        )
        instructions.append(
            "A szlenget, kulturális utalásokat és szóvicceket ültesd át magyaros megfelelőre; ha nincs analógia, röviden magyarázd meg, de maradj tömör."
        )
        consistency_line = (
            "Őrizd meg a neveket, formázásokat (pl. [sóhajt]) és a stílust, maradj végig ugyanabban a tegező/magázó formában."
        )
    else:
        instructions.append(
            (
                "You are an experienced subtitle translator working from "
                f"{input_lang_name} into {output_lang_name}. Treat this like a professional streaming/localization brief."
            )
        )
        instructions.append(
            "Translate each numbered subtitle independently and keep the numbering and ordering exactly the same (format: `1. translation`). Do not merge or split subtitles."
        )
        instructions.append(
            "Keep each translation roughly the same length as the source so it fits the timing (max two subtitle lines, comfortable reading speed)."
        )
        instructions.append(
            f"Use natural, spoken {output_lang_name} that sounds like native dialogue. Prefer idiomatic rephrasings over literal copies."
        )
        instructions.append(
            "Adapt slang, dialect, jokes, and cultural references into authentic equivalents; if no analogue exists, add a concise explanation while staying brief."
        )
        consistency_line = (
            "Preserve character names, formatting cues (e.g., [sighs]), and stay consistent with register (informal/formal) throughout."
        )

    instructions.append(consistency_line)

    if allow_sensitive:
        instructions.append(
            (
                "Ne cenzúrázz és ne hagyd ki a durva kifejezéseket, csak akkor enyhíts, ha az eredeti is enyhe." if locale == "HU" else
                "Do not censor or soften profanity unless the source already does; match the original intensity."
            )
        )

    if tone:
        instructions.append(
            ("Elvárt hangnem/stílus: " if locale == "HU" else "Requested tone/register: ") + tone.strip()
        )
    if target_audience:
        instructions.append(
            ("Célközönség: " if locale == "HU" else "Target audience: ") + target_audience.strip()
        )
    if platform:
        platform_note = platform.strip()
        if locale == "HU":
            instructions.append(
                "Platform/specifikus szempontok: "
                f"{platform_note}. Igazítsd a szóhasználatot és a sorhosszakat az adott felület tipikus elvárásaihoz."
            )
        else:
            instructions.append(
                "Platform/style-guide hints: "
                f"{platform_note}. Respect that platform's usual subtitle constraints (two lines, concise phrasing)."
            )
    if context:
        instructions.append(
            ("Kontekstus: " if locale == "HU" else "Context: ") + context.strip()
        )
    if style_notes:
        instructions.append(
            ("További stílusjegyzet: " if locale == "HU" else "Additional style notes: ") + style_notes.strip()
        )

    glossary_line = add_glossary_line()
    if glossary_line:
        instructions.append(glossary_line)

    if locale == "HU":
        instructions.append(
            "A választ csak számozott listaként add vissza, pontosan annyi sorral, ahány forrássor van (pl. `1. ...`)."
        )
    else:
        instructions.append(
            "Return only the numbered list with exactly the same number of items as the input (e.g., `1. ...`)."
        )

    return "\n".join(instructions)


def translate_or_subdivide_batch(
    client: OpenAI,
    batch_segments: List[Dict[str, Any]],
    system_prompt: str,
    model: str,
    stream: bool,
    batch_id_str: str,
) -> Optional[List[str]]:
    """Translate a batch of segments, splitting recursively on failure."""
    if not batch_segments:
        return []
    if len(batch_segments) == 1:
        logger.info("  [%s] Csoport 1 elemre redukálva, fordítás...", batch_id_str)

    numbered_texts = [f"{i + 1}. {seg['text']}" for i, seg in enumerate(batch_segments)]
    text_block = "\n".join(numbered_texts)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text_block}],
            temperature=0.1,
        )
        translated_lines_raw = [
            line.strip()
            for line in response.choices[0].message.content.strip().split("\n")
            if line.strip()
        ]
    except OpenAIError as exc:
        logger.error("  HIBA: API hiba történt a(z) [%s] csoportnál: %s", batch_id_str, exc)
        return None

    if len(translated_lines_raw) == len(batch_segments):
        final_translated_lines = [re.sub(r"^\d+\.\s*", "", line) for line in translated_lines_raw]
        if stream:
            for line in final_translated_lines:
                logger.info("    + %s", line)
        return final_translated_lines

    logger.error(
        "  HIBA: A(z) [%s] csoportnál (%s elem) a sorok száma nem egyezik! Várt: %s, Kapott: %s.",
        batch_id_str,
        len(batch_segments),
        len(batch_segments),
        len(translated_lines_raw),
    )
    logger.info("=" * 20 + f" DEBUG START: BATCH [{batch_id_str}] " + "=" * 20)
    logger.info("--- BEMENET (amit a modell kapott): ---")
    for line in numbered_texts:
        logger.info(line)
    logger.info("\n--- KIMENET (amit a modell adott): ---")
    for idx, line in enumerate(translated_lines_raw, start=1):
        logger.info("%s. %s", idx, line)
    logger.info("=" * 20 + f" DEBUG END: BATCH [{batch_id_str}] " + "=" * 20)

    if len(batch_segments) <= 1:
        logger.error("  A csoport már nem osztható tovább, a fordítás ennél a pontnál végleg sikertelen.")
        return None

    logger.info("  A [%s] csoport felosztása és újrapróbálása...", batch_id_str)
    mid_point = len(batch_segments) // 2
    first_half = batch_segments[:mid_point]
    second_half = batch_segments[mid_point:]

    first_half_results = translate_or_subdivide_batch(
        client, first_half, system_prompt, model, stream, f"{batch_id_str}-A"
    )
    if first_half_results is None:
        return None

    second_half_results = translate_or_subdivide_batch(
        client, second_half, system_prompt, model, stream, f"{batch_id_str}-B"
    )
    if second_half_results is None:
        return None

    return first_half_results + second_half_results


def main(
    project_name: str,
    input_lang: Optional[str],
    output_lang: Optional[str],
    context: Optional[str],
    model: str,
    stream: bool,
    allow_sensitive_content: bool,
    auth_key_arg: Optional[str],
    systemprompt: Optional[str],
    tone: Optional[str],
    target_audience: Optional[str],
    platform: Optional[str],
    style_notes: Optional[str],
    glossary_path: Optional[str],
) -> None:
    """CLI entry point."""
    config, project_root = load_config()

    if auth_key_arg:
        save_api_key(project_root, auth_key_arg)
        auth_key = auth_key_arg
    else:
        auth_key = load_api_key(project_root)

    if not auth_key:
        logger.error("Hiba: Nincs elérhető ChatGPT API kulcs. Add meg a --auth-key paramétert.")
        sys.exit(1)

    client = OpenAI(api_key=auth_key)

    glossary_terms = load_glossary(glossary_path, project_root)

    input_dir_path, output_dir_path = resolve_project_paths(project_name, config, project_root)
    input_lang_resolved, output_lang_resolved = resolve_language_params(config, input_lang, output_lang)

    logger.info("Projekt beállítások:")
    logger.info("  - Projekt név:           %s", project_name)
    logger.info("  - Bemeneti mappa:        %s", input_dir_path)
    logger.info("  - Kimeneti mappa:        %s", output_dir_path)
    logger.info("  - Bemeneti nyelv kód:    %s", input_lang_resolved)
    logger.info("  - Kimeneti nyelv kód:    %s", output_lang_resolved)
    logger.info("  - Modell:                %s", model)
    logger.info("  - Stream mód:            %s", "igen" if stream else "nem")
    logger.info("  - Kényes tartalmak:      %s", "igen" if allow_sensitive_content else "nem")
    if tone:
        logger.info("  - Hangnem:               %s", tone)
    if target_audience:
        logger.info("  - Célközönség:           %s", target_audience)
    if platform:
        logger.info("  - Platform:              %s", platform)

    input_filename, error = find_json_file(input_dir_path)
    if error:
        sys.exit(1)

    input_filepath = input_dir_path / input_filename
    output_filepath = output_dir_path / input_filename
    progress_filepath = output_dir_path / f"{Path(input_filename).stem}.progress.json"

    logger.info("Bemeneti fájl feldolgozása: %s", input_filepath)

    try:
        with open(input_filepath, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError as exc:
        logger.error("Hiba: A bemeneti fájl nem érvényes JSON formátumú: %s", exc)
        sys.exit(1)

    if "segments" not in data or not isinstance(data["segments"], list):
        logger.error("Hiba: JSON 'segments' kulcs hiányzik vagy nem lista.")
        sys.exit(1)

    progress: Dict[str, str] = {}
    if progress_filepath.exists():
        try:
            with open(progress_filepath, "r", encoding="utf-8") as fp:
                progress = json.load(fp)
            logger.info("Haladási fájl betöltve: %s (%s szegmens kész).", progress_filepath, len(progress))
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("A haladási fájl nem olvasható (%s). Újrakezdés.", exc)
            progress = {}

    segments_to_process: List[Dict[str, Any]] = []
    for index, segment in enumerate(data["segments"]):
        if segment.get("text", "").strip() and str(index) not in progress:
            segment["original_index"] = index
            segments_to_process.append(segment)

    if not segments_to_process:
        logger.info("Nincs új, fordítandó szegmens. A meglévő kimeneti fájl frissül.")
    else:
        logger.info("%s új, fordítandó szegmens található.", len(segments_to_process))
        batches = create_smart_chunks(segments_to_process)
        logger.info("Intelligens csoportképzés befejezve. %s csoport feldolgozása szükséges.", len(batches))

        system_prompt_final = build_system_prompt(
            input_lang_resolved,
            output_lang_resolved,
            allow_sensitive_content,
            context,
            systemprompt,
            tone,
            target_audience,
            platform,
            style_notes,
            glossary_terms,
        )
        logger.info(
            "Használt system prompt első 80 karaktere: %s",
            system_prompt_final[:80] + ("..." if len(system_prompt_final) > 80 else ""),
        )

        for idx, batch in enumerate(batches, start=1):
            logger.info("[%s/%s] fő csoport feldolgozása...", idx, len(batches))
            translated_batch_lines = translate_or_subdivide_batch(
                client, batch, system_prompt_final, model, stream, str(idx)
            )
            if translated_batch_lines is None:
                logger.error(
                    "A fordítási folyamat megszakadt a(z) [%s] csoportnál egy nem helyreállítható hiba miatt.",
                    idx,
                )
                sys.exit(1)

            for segment_obj, translated_line in zip(batch, translated_batch_lines):
                progress[str(segment_obj["original_index"])] = translated_line

            with open(progress_filepath, "w", encoding="utf-8") as fp:
                json.dump(progress, fp, ensure_ascii=False, indent=2)
            logger.info("  Haladás elmentve: %s", progress_filepath)
            time.sleep(1)

    if "system_prompt_final" not in locals():
        system_prompt_final = build_system_prompt(
            input_lang_resolved,
            output_lang_resolved,
            allow_sensitive_content,
            context,
            systemprompt,
            tone,
            target_audience,
            platform,
            style_notes,
            glossary_terms,
        )

    logger.info("Az összes szükséges fordítás elkészült.")

    for index_str, line in progress.items():
        data["segments"][int(index_str)]["translated_text"] = line
    for segment in data["segments"]:
        segment.setdefault("translated_text", "")

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["translation_system_prompt"] = system_prompt_final
    metadata["translation_model"] = model
    metadata["translation_language_pair"] = f"{input_lang_resolved}->{output_lang_resolved}"
    data["metadata"] = metadata

    with open(output_filepath, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    logger.info("Fordítás befejezve. A kiegészített fájl ide került: %s", output_filepath)

    try:
        sanitize_translation_fields(output_filepath)
        logger.info("A JSON sanitize lépés sikeresen lefutott: %s", output_filepath)
    except Exception as exc:
        logger.warning("A JSON sanitize lépés sikertelen (%s). Folytatás a nem módosított fájllal.", exc)

    if progress_filepath.exists():
        try:
            progress_filepath.unlink()
            logger.info("Az ideiglenes haladási fájl törölve: %s", progress_filepath)
        except OSError as exc:
            logger.warning("A haladási fájl törlése nem sikerült (%s).", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Projekt-alapú JSON fordítás ChatGPT segítségével. A config.json alapján keresi a feldolgozandó fájlokat."
    )
    parser.add_argument(
        "-p",
        "--project-name",
        required=True,
        help="A workdir-en belüli projektmappa neve.",
    )
    parser.add_argument(
        "-input_language",
        "--input-language",
        dest="input_language",
        default=None,
        help="A bemeneti nyelv kódja (pl. EN, HU). Ha nincs megadva, a config.json default_source_lang értéke lesz használva.",
    )
    parser.add_argument(
        "-output_language",
        "--output-language",
        dest="output_language",
        default=None,
        help="A kimeneti nyelv kódja (pl. EN, HU). Ha nincs megadva, a config.json default_target_lang értéke lesz használva.",
    )
    parser.add_argument(
        "-context",
        "--context",
        dest="context",
        default=None,
        help="Rövid kontextus a fordításhoz.",
    )
    parser.add_argument(
        "-model",
        "--model",
        dest="model",
        default="gpt-4o",
        help="A használni kívánt OpenAI modell neve. Alapértelmezett: gpt-4o.",
    )
    parser.add_argument(
        "-stream",
        "--stream",
        action="store_true",
        help="Bekapcsolja a fordítási folyamat valós idejű, soronkénti megjelenítését a konzolon.",
    )
    parser.add_argument(
        "-allow_sensitive_content",
        "--allow-sensitive-content",
        action="store_true",
        help="Speciális promptot használ, ami a kényes tartalmak fordítását is megkísérli cenzúrázás nélkül.",
    )
    parser.add_argument(
        "-auth_key",
        "--auth-key",
        dest="auth_key",
        default=None,
        help="Az OpenAI API kulcsa. Megadás esetén elmentődik a keyholder.json fájlba.",
    )
    parser.add_argument(
        "-systemprompt",
        "--systemprompt",
        dest="systemprompt",
        default=None,
        help="Egyedi system prompt a fordításhoz. Ha nincs megadva, a script a beépített alapértelmezettet használja.",
    )
    parser.add_argument(
        "--tone",
        dest="tone",
        default=None,
        help="A kívánt hangnem vagy stílus (pl. laza tegező, üzleti formális).",
    )
    parser.add_argument(
        "--target-audience",
        dest="target_audience",
        default=None,
        help="Célközönség rövid leírása a természetes hangvételhez (pl. magyar Netflix nézők, amerikai tinik).",
    )
    parser.add_argument(
        "--platform",
        dest="platform",
        default=None,
        help="Platform vagy formátum (pl. Netflix, YouTube), ami meghatározza a feliratozási elvárásokat.",
    )
    parser.add_argument(
        "--style-notes",
        dest="style_notes",
        default=None,
        help="További stílusutalások vagy instrukciók, amelyeket a promptba szeretnél építeni.",
    )
    parser.add_argument(
        "--glossary",
        dest="glossary",
        default=None,
        help="JSON fájl, amely forrás->cél kifejezéspárokat tartalmaz a következetes terminológiához.",
    )
    add_debug_argument(parser)

    args = parser.parse_args()
    log_level = configure_debug_mode(args.debug)
    logging.basicConfig(level=log_level, format="%(message)s")
    logger.setLevel(log_level)

    main(
        project_name=args.project_name,
        input_lang=args.input_language,
        output_lang=args.output_language,
        context=args.context,
        model=args.model,
        stream=args.stream,
        allow_sensitive_content=args.allow_sensitive_content,
        auth_key_arg=args.auth_key,
        systemprompt=args.systemprompt,
        tone=args.tone,
        target_audience=args.target_audience,
        platform=args.platform,
        style_notes=args.style_notes,
        glossary_path=args.glossary,
    )

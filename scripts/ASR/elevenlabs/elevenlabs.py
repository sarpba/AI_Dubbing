"""
Evenlabs ASR integration – project aware transcription pipeline that stores the raw ElevenLabs JSON response.
Speaker diarizáció támogatása továbbra is elérhető (—diarize / —no-diarize kapcsolók, alapértelmezés: be).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
DEFAULT_API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
DEFAULT_MODEL_ID = "scribe_v1_experimental"
ENV_API_KEY = "EVENLABS_API_KEY"
KEYHOLDER_FIELD = "evenlabs_api_key"


def get_project_root() -> Path:
    """Locate the repository root by walking upwards until config.json is found."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")


def load_config() -> Tuple[dict, Path]:
    """Load config.json and return it together with the project root."""
    project_root = get_project_root()
    config_path = project_root / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            config = json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Hiba a konfiguráció betöltésekor ({config_path}): {exc}")
        sys.exit(1)
    return config, project_root


def resolve_project_input(project_name: str, config: dict, project_root: Path) -> Path:
    """Resolve the directory that contains speech-separated audio for the project."""
    try:
        workdir = project_root / config["DIRECTORIES"]["workdir"]
        input_subdir = config["PROJECT_SUBDIRS"]["separated_audio_speech"]
    except KeyError as exc:
        print(f"Hiba: hiányzó kulcs a config.json-ban: {exc}")
        sys.exit(1)

    input_dir = workdir / project_name / input_subdir
    if not input_dir.is_dir():
        print(f"Hiba: a feldolgozandó mappa nem található: {input_dir}")
        sys.exit(1)
    return input_dir


def get_keyholder_path(project_root: Path) -> Path:
    return project_root / "keyholder.json"


def save_api_key(project_root: Path, api_key: str) -> None:
    keyholder_path = get_keyholder_path(project_root)
    try:
        data: Dict[str, Any] = {}
        if keyholder_path.exists():
            with keyholder_path.open("r", encoding="utf-8") as fp:
                try:
                    data = json.load(fp)
                except json.JSONDecodeError:
                    logging.warning("A keyholder.json sérült, új struktúra létrehozása.")
                    data = {}
        encoded = base64.b64encode(api_key.encode("utf-8")).decode("utf-8")
        data[KEYHOLDER_FIELD] = encoded
        with keyholder_path.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)
        logging.info("Evenlabs API kulcs elmentve a keyholder.json fájlba.")
    except Exception as exc:
        logging.error("Nem sikerült elmenteni az Evenlabs API kulcsot: %s", exc)


def load_api_key(project_root: Path) -> Optional[str]:
    keyholder_path = get_keyholder_path(project_root)
    if not keyholder_path.exists():
        return None
    try:
        with keyholder_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        encoded = data.get(KEYHOLDER_FIELD)
        if not encoded:
            return None
        return base64.b64decode(encoded.encode("utf-8")).decode("utf-8")
    except (json.JSONDecodeError, KeyError, base64.binascii.Error) as exc:
        logging.error("Nem sikerült beolvasni az Evenlabs API kulcsot: %s", exc)
        return None
    except Exception as exc:
        logging.error("Váratlan hiba kulcs betöltésekor: %s", exc)
        return None


def transcribe_with_evenlabs(
    audio_path: Path,
    api_url: str,
    api_key: str,
    *,
    language: Optional[str],
    model_id: Optional[str],
    diarize: bool,
) -> Optional[str]:
    headers = {"xi-api-key": api_key}
    data: Dict[str, Any] = {}
    if language:
        data["language_code"] = language
    if model_id:
        data["model_id"] = model_id
    if diarize:
        data["diarize"] = "true"

    try:
        with audio_path.open("rb") as fp:
            files = {"file": (audio_path.name, fp, "application/octet-stream")}
            response = requests.post(api_url, headers=headers, data=data, files=files, timeout=600)
    except requests.RequestException as exc:
        logging.error("Evenlabs API hívás sikertelen (%s): %s", audio_path.name, exc)
        return None

    if not response.ok:
        logging.error(
            "Evenlabs API hibát jelzett (%s): %s – %s", audio_path.name, response.status_code, response.text[:500]
        )
        return None

    raw_payload = response.text
    try:
        json.loads(raw_payload)
    except ValueError:
        logging.error("Evenlabs API nem JSON választ adott (%s).", audio_path.name)
        return None

    return raw_payload


def process_directory(
    input_dir: Path,
    api_url: str,
    api_key: str,
    *,
    language: Optional[str],
    model_id: Optional[str],
    diarize: bool,
) -> None:
    audio_files = sorted(
        [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    )

    if not audio_files:
        print(f"Nem található támogatott hangfájl a megadott mappában: {input_dir}")
        return

    print(f"{len(audio_files)} hangfájl feldolgozása indul az Evenlabs API-val…")
    for audio_path in audio_files:
        print("-" * 48)
        print(f"▶  Feldolgozás: {audio_path.name}")
        payload = transcribe_with_evenlabs(
            audio_path,
            api_url,
            api_key,
            language=language,
            model_id=model_id,
            diarize=diarize,
        )
        if payload is None:
            print(f"  ✖  Sikertelen API hívás: {audio_path.name}")
            continue

        output_path = audio_path.with_suffix(".json")
        try:
            with output_path.open("w", encoding="utf-8") as fp:
                fp.write(payload)
            print(f"  ✔  Mentve: {output_path.name}")
        except OSError as exc:
            logging.error("Nem sikerült menteni a kimenetet (%s): %s", output_path, exc)
            print(f"  ✖  Mentési hiba: {audio_path.name}")


def resolve_api_key(args: argparse.Namespace, project_root: Path) -> Optional[str]:
    if args.api_key:
        save_api_key(project_root, args.api_key)
        return args.api_key
    env_key = os.environ.get(ENV_API_KEY)
    if env_key:
        logging.info("Evenlabs API kulcs betöltve környezeti változóból (%s).", ENV_API_KEY)
        return env_key
    stored_key = load_api_key(project_root)
    if stored_key:
        logging.info("Evenlabs API kulcs betöltve a keyholder.json fájlból.")
        return stored_key
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Evenlabs ASR – projekt alapú hangtár feldolgozás az ElevenLabs nyers JSON válaszaival.")
    )
    parser.add_argument(
        "-p",
        "--project-name",
        required=True,
        help="A projekt neve (a workdir alatti mappa), amit fel kell dolgozni.",
    )
    parser.add_argument(
        "--language",
        help="ISO nyelvkód (pl. en, hu). Ha nincs megadva, az Evenlabs automatikus felismerését használja.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Evenlabs modell azonosító (alapértelmezett: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Evenlabs API végpont URL-je (alapértelmezett: {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--api-key",
        help=f"Evenlabs API kulcs. Ha megadod, elmenti a keyholder.json fájlba. Környezeti változó: {ENV_API_KEY}.",
    )
    # —diarize / —no-diarize; alapértelmezés: be
    parser.add_argument(
        "--diarize",
        dest="diarize",
        action="store_true",
        default=True,
        help="Speaker diarizáció bekapcsolása (alapértelmezett: be)",
    )
    parser.add_argument(
        "--no-diarize",
        dest="diarize",
        action="store_false",
        help="Speaker diarizáció kikapcsolása",
    )

    add_debug_argument(parser)
    args = parser.parse_args()

    log_level = configure_debug_mode(args.debug)
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    config, project_root = load_config()
    api_key = resolve_api_key(args, project_root)
    if not api_key:
        print(
            "Hiba: Nem található Evenlabs API kulcs. Add meg az --api-key kapcsolóval vagy az EVENLABS_API_KEY környezeti változóval."
        )
        sys.exit(1)

    input_dir = resolve_project_input(args.project_name, config, project_root)
    print("Projekt beállítások betöltve:")
    print(f"  - Projekt név:    {args.project_name}")
    print(f"  - Bemeneti mappa: {input_dir}")
    print(f"  - Evenlabs modell: {args.model}")
    print(f"  - API végpont:     {args.api_url}")
    print(f"  - Diarizáció:      {'BE' if args.diarize else 'KI'}")

    process_directory(
        input_dir=input_dir,
        api_url=args.api_url,
        api_key=api_key,
        language=args.language,
        model_id=args.model,
        diarize=args.diarize,
    )


if __name__ == "__main__":
    main()

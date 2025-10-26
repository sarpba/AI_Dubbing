"""
Audio copy helper script that normalizes uploads to 44.1 kHz and distributes
them into project directories based on CLI switches.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Tuple

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

SUPPORTED_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
)
TARGET_SAMPLE_RATE = 44100
TARGET_SUFFIX = "_44100.wav"


def get_project_root() -> Path:
    """
    Walk upwards from the current file until config.json is located.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")


def load_config() -> Tuple[dict, Path]:
    """
    Load the root config.json and return both the parsed payload and root path.
    """
    project_root = get_project_root()
    config_path = project_root / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle), project_root
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Hiba a konfiguráció betöltésekor ({config_path}): {exc}")
        sys.exit(1)


def ensure_ffmpeg_available() -> str:
    """
    Verify that ffmpeg is accessible on PATH and return its absolute path.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("Hiba: az ffmpeg parancs nem érhető el. Telepítsd vagy add az elérési úthoz.")
        sys.exit(1)
    return ffmpeg_path


def resolve_project_directories(
    project_name: str, config: dict, project_root: Path
) -> dict:
    """
    Resolve all relevant directories for the project based on the config.
    """
    try:
        workdir = project_root / config["DIRECTORIES"]["workdir"]
        project_base = workdir / project_name
        upload_dir = project_base / config["PROJECT_SUBDIRS"]["upload"]
        extracted_dir = project_base / config["PROJECT_SUBDIRS"]["extracted_audio"]
        separated_background_dir = project_base / config["PROJECT_SUBDIRS"][
            "separated_audio_background"
        ]
        separated_speech_dir = project_base / config["PROJECT_SUBDIRS"][
            "separated_audio_speech"
        ]
    except KeyError as exc:
        print(f"Hiba: hiányzó kulcs a config.json-ban: {exc}")
        sys.exit(1)

    if not project_base.is_dir():
        print(f"Hiba: a projekt mappa nem található: {project_base}")
        sys.exit(1)
    if not upload_dir.is_dir():
        print(f"Hiba: az upload mappa nem található: {upload_dir}")
        sys.exit(1)

    return {
        "project_base": project_base,
        "upload": upload_dir,
        "extracted_audio": extracted_dir,
        "separated_audio_background": separated_background_dir,
        "separated_audio_speech": separated_speech_dir,
    }


def discover_audio_files(upload_dir: Path) -> list[Path]:
    """
    Return every supported audio file from the upload directory (non-recursive).
    """
    files: list[Path] = []
    for candidate in upload_dir.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(candidate)
    return sorted(files)


def convert_to_target_rate(
    ffmpeg_path: str, source: Path, destination: Path
) -> subprocess.CompletedProcess:
    """
    Convert the provided audio file to wav with the target sample rate.
    """
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        str(destination),
    ]
    return subprocess.run(command, text=True, capture_output=True, check=False)


def create_zero_volume_copy(
    ffmpeg_path: str, source: Path, destination: Path
) -> subprocess.CompletedProcess:
    """
    Generate a muted copy of the already converted wav file.
    """
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source),
        "-filter:a",
        "volume=0",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        str(destination),
    ]
    return subprocess.run(command, text=True, capture_output=True, check=False)


def ensure_destination(path: Path) -> None:
    """
    Make sure the destination directory exists.
    """
    path.mkdir(parents=True, exist_ok=True)


def copy_audio(
    converted: Path, destination_dir: Path, target_name: str
) -> Path:
    """
    Copy the converted wav into the destination directory.
    """
    ensure_destination(destination_dir)
    target_path = destination_dir / target_name
    shutil.copy2(converted, target_path)
    return target_path


def process_files(
    project_paths: dict,
    ffmpeg_path: str,
    targets: Iterable[str],
) -> None:
    """
    Handle the conversion and conditional copying for every upload file.
    """
    upload_dir = project_paths["upload"]
    audio_files = discover_audio_files(upload_dir)
    if not audio_files:
        logging.info("Nem található feldolgozható audio fájl az upload mappában.")
        return

    logging.info("Feldolgozandó fájlok száma: %d", len(audio_files))
    for source in audio_files:
        target_filename = f"{source.stem}{TARGET_SUFFIX}"
        tmp_path = upload_dir / target_filename

        logging.info("Konvertálás 44.1 kHz-re: %s -> %s", source.name, target_filename)
        conversion = convert_to_target_rate(ffmpeg_path, source, tmp_path)
        if conversion.returncode != 0:
            logging.error(
                "ffmpeg hiba a konvertálás közben (%s): %s",
                source.name,
                conversion.stderr.strip(),
            )
            if tmp_path.exists():
                tmp_path.unlink()
            continue

        for target in targets:
            if target == "separated_audio_background":
                dest_dir = project_paths[target]
                ensure_destination(dest_dir)
                muted_path = dest_dir / target_filename
                logging.info("Néma másolat mentése: %s", muted_path)
                muted = create_zero_volume_copy(ffmpeg_path, tmp_path, muted_path)
                if muted.returncode != 0:
                    logging.error(
                        "ffmpeg hiba a néma másolat készítésénél (%s): %s",
                        muted_path.name,
                        muted.stderr.strip(),
                    )
                    if muted_path.exists():
                        muted_path.unlink()
                continue

            dest_dir = project_paths[target]
            dest_path = copy_audio(tmp_path, dest_dir, target_filename)
            logging.info("Másolat létrehozva: %s", dest_path)

        try:
            tmp_path.unlink()
        except OSError as exc:
            logging.warning("Nem sikerült törölni az ideiglenes fájlt (%s): %s", tmp_path, exc)


def parse_arguments() -> argparse.Namespace:
    """
    Build and parse the CLI arguments for the script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Upload mappa audio fájljainak 44,1 kHz-re konvertálása és opcionális "
            "másolása projekt mappákba."
        )
    )
    parser.add_argument(
        "-p",
        "--project-name",
        dest="project_name",
        required=True,
        help="A feldolgozandó projekt neve.",
    )
    parser.add_argument(
        "--extracted_audio",
        action="store_true",
        help="Másolat készítése a 1.5_extracted_audio mappába.",
    )
    parser.add_argument(
        "--separated_audio_background",
        action="store_true",
        help="Másolat készítése a 2_separated_audio_background mappába (néma verzió).",
    )
    parser.add_argument(
        "--separated_audio_speech",
        action="store_true",
        help="Másolat készítése a 2_separated_audio_speech mappába.",
    )
    add_debug_argument(parser)
    return parser.parse_args()


def determine_targets(args: argparse.Namespace) -> list[str]:
    """
    Collect the destination keys activated by CLI switches.
    """
    selected = []
    if args.extracted_audio:
        selected.append("extracted_audio")
    if args.separated_audio_background:
        selected.append("separated_audio_background")
    if args.separated_audio_speech:
        selected.append("separated_audio_speech")
    return selected


def main() -> None:
    args = parse_arguments()
    log_level = configure_debug_mode(args.debug)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)8s | %(message)s",
    )

    targets = determine_targets(args)
    if not targets:
        logging.error(
            "Legalább egy célmappát aktiválni kell a kapcsolók közül: "
            "--extracted_audio, --separated_audio_background, --separated_audio_speech."
        )
        sys.exit(1)

    config, project_root = load_config()
    project_paths = resolve_project_directories(args.project_name, config, project_root)
    logging.info("Projekt: %s", args.project_name)
    logging.info("Upload mappa: %s", project_paths["upload"])
    logging.info("Aktív célmappák: %s", ", ".join(targets))

    ffmpeg_path = ensure_ffmpeg_available()
    process_files(project_paths, ffmpeg_path, targets)


if __name__ == "__main__":
    main()

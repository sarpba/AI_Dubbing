#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List


def resolve_repo_root() -> Path:
    """
    Locate the repository root by traversing parent directories until config.json is found.
    This allows the script to run correctly even if it is moved deeper into the repo.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "config.json").exists():
            return candidate
    raise SystemExit(
        "Unable to determine repository root. Ensure config.json is present in the repo."
    )


REPO_ROOT = resolve_repo_root()
CONFIG_PATH = REPO_ROOT / "config.json"
NORMALISER_PATH = REPO_ROOT / "normalisers" / "hun" / "normaliser.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect all 'translated_text' entries from the translated JSON files of a "
            "project and run them through the Hungarian normaliser."
        )
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        required=True,
        help=(
            "Project name or path. If a name is provided, it is resolved inside the "
            "configured workdir."
        ),
    )
    return parser.parse_args()


def load_normaliser_module() -> ModuleType:
    if not NORMALISER_PATH.exists():
        raise SystemExit(f"Normaliser module not found: {NORMALISER_PATH}")
    spec = importlib.util.spec_from_file_location("hun_normaliser", NORMALISER_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load module spec from {NORMALISER_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[misc]
    except ModuleNotFoundError as exc:
        missing = exc.name if getattr(exc, "name", None) else str(exc)
        raise SystemExit(
            f"Failed to load normaliser module because dependency '{missing}' is missing. "
            "Install the required dependency (see requirements.txt)."
        ) from exc
    return module


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Config file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as cfg_file:
        return json.load(cfg_file)


def resolve_project_dir(project_arg: str, config: Dict[str, Any]) -> Path:
    candidate = Path(project_arg).expanduser()

    probe_paths = []
    if candidate.is_absolute():
        probe_paths.append(candidate)
    else:
        probe_paths.append((Path.cwd() / candidate).resolve())
        probe_paths.append((REPO_ROOT / candidate).resolve())
        workdir_subdir = config.get("DIRECTORIES", {}).get("workdir", "workdir")
        probe_paths.append((REPO_ROOT / workdir_subdir / project_arg).resolve())

    for path in probe_paths:
        if path.is_dir():
            return path

    raise SystemExit(
        f"Unable to locate project directory for '{project_arg}'. "
        "Ensure the project exists under the configured workdir."
    )


def ensure_project_paths(project_dir: Path, translated_subdir: str) -> Path:
    translated_dir = project_dir / translated_subdir
    if not translated_dir.is_dir():
        raise SystemExit(
            f"Translated directory does not exist: {translated_dir} "
            f"(derived from config key 'PROJECT_SUBDIRS.translated')."
        )
    return translated_dir


def iter_translated_json_files(translated_dir: Path) -> Iterable[Path]:
    for path in sorted(translated_dir.rglob("*.json")):
        if path.is_file():
            yield path


def extract_translated_text_entries(payload: Any) -> List[str]:
    results: List[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "translated_text" and isinstance(value, str):
                results.append(value)
            results.extend(extract_translated_text_entries(value))
    elif isinstance(payload, list):
        for item in payload:
            results.extend(extract_translated_text_entries(item))
    return results


def normalise_file_contents(json_path: Path, normalize_fn: Callable[[str], str]) -> List[Dict[str, Any]]:
    try:
        with json_path.open("r", encoding="utf-8") as json_file:
            payload = json.load(json_file)
    except json.JSONDecodeError as exc:
        print(
            f"Skipping invalid JSON file {json_path}: {exc}",
            file=sys.stderr,
        )
        return []

    collected = extract_translated_text_entries(payload)
    normalised_entries: List[Dict[str, Any]] = []
    for index, original_text in enumerate(collected, start=1):
        normalised_entries.append(
            {
                "file": str(json_path),
                "index": index,
                "original": original_text,
                "normalised": normalize_fn(original_text),
            }
        )
    return normalised_entries


def main() -> None:
    args = parse_args()
    config = load_config()
    normaliser_module = load_normaliser_module()

    if not hasattr(normaliser_module, "normalize"):
        raise SystemExit("The normaliser module does not expose a 'normalize' function.")

    normalize_fn: Callable[[str], str] = getattr(normaliser_module, "normalize")

    try:
        translated_subdir = config["PROJECT_SUBDIRS"]["translated"]
    except KeyError as exc:
        raise SystemExit(
            "Missing 'PROJECT_SUBDIRS.translated' entry in config.json."
        ) from exc
    try:
        logs_subdir = config["PROJECT_SUBDIRS"]["logs"]
    except KeyError as exc:
        raise SystemExit(
            "Missing 'PROJECT_SUBDIRS.logs' entry in config.json."
        ) from exc

    project_dir = resolve_project_dir(args.project_dir, config)
    translated_dir = ensure_project_paths(project_dir, translated_subdir)

    logs_dir = project_dir / logs_subdir
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    for json_file in iter_translated_json_files(translated_dir):
        all_results.extend(normalise_file_contents(json_file, normalize_fn))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"normalised_translations_{timestamp}.log"

    log_lines = [
        f"Project: {project_dir}",
        f"Translated directory: {translated_dir}",
        f"Collected items: {len(all_results)}",
        "",
    ]

    if not all_results:
        log_lines.append("No translated_text entries found.")
    else:
        for entry in all_results:
            log_lines.extend(
                [
                    f"File: {entry['file']}",
                    f"Index: {entry['index']}",
                    f"Original: {entry['original']}",
                    f"Normalised: {entry['normalised']}",
                    "",
                ]
            )

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("\n".join(log_lines))

    print(f"Normalised translations saved to {log_path}")


if __name__ == "__main__":
    main()

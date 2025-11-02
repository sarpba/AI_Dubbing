from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SPECIAL_CHARS = {"\\", "/", '"'}
_TARGET_KEYS = {"text", "translated text"}


def _normalize_key(key: str) -> str:
    """Normalizes keys for comparison."""
    return key.lower().replace("_", " ").strip()


def _sanitize_string(value: str) -> str:
    """Removes the specified special characters from a string."""
    translation_table = {ord(ch): None for ch in _SPECIAL_CHARS}
    return value.translate(translation_table)


def _sanitize_in_place(node: Any) -> None:
    """Recursively walks the JSON-compatible structure and sanitizes in place."""
    if isinstance(node, dict):
        for key, value in node.items():
            normalized = _normalize_key(key)
            if normalized in _TARGET_KEYS and isinstance(value, str):
                node[key] = _sanitize_string(value)
            else:
                _sanitize_in_place(value)
    elif isinstance(node, list):
        for item in node:
            _sanitize_in_place(item)


def sanitize_translation_fields(json_file: str | Path) -> None:
    """
    Removes the specified special characters from `text` and `translated text` fields.

    Parameters
    ----------
    json_file:
        Path to the JSON file that should be sanitized. The file is updated in place.
    """
    path = Path(json_file)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    _sanitize_in_place(payload)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove \\ / \" characters from text fields in a JSON file."
    )
    parser.add_argument("json_file", help="Path to the JSON file to sanitize.")
    args = parser.parse_args()
    sanitize_translation_fields(args.json_file)

"""
Resegment script - JSON szegmensek újraformázása különböző paraméterekkel.
Az ASR script által létrehozott JSON fájlok biztonsági mentése és újraformázása.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

DEFAULT_MAX_PAUSE_S = 0.8
DEFAULT_PADDING_S = 0.1
DEFAULT_MAX_SEGMENT_S = 11.5
PRIMARY_PUNCTUATION = (".", "!", "?")
SECONDARY_PUNCTUATION = (",",)


def _sanitize_text(value: Any) -> str:
    """Remove escaped quote sequences from transcribed text."""
    if not value:
        return ""
    text = str(value)
    cleaned = text.replace('\\"', "").replace('"', "")
    return cleaned.strip()


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
    """Resolve the directory that contains ASR JSON files for the project."""
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


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _extract_speaker(entry: dict) -> Optional[str]:
    """Try to pull a speaker label from various possible fields."""
    for key in ("speaker", "speaker_label", "speaker_id", "speaker_tag", "spk"):
        if key in entry and entry[key] is not None and str(entry[key]) != "":
            return str(entry[key])
    return None


def _normalise_word_entry(entry: dict) -> Optional[dict]:
    text = _sanitize_text(entry.get("word") or entry.get("text") or entry.get("token") or "")
    if not text:
        return None
    start = (
        _safe_float(entry.get("start"))
        or _safe_float(entry.get("start_time"))
        or _safe_float(entry.get("offset_seconds"))
        or _safe_float(entry.get("offset"))
    )
    end = (
        _safe_float(entry.get("end"))
        or _safe_float(entry.get("end_time"))
        or _safe_float(entry.get("offset_seconds_end"))
    )
    if start is None:
        return None
    if end is None:
        duration = _safe_float(entry.get("duration")) or 0.0
        end = start + duration
    confidence = _safe_float(entry.get("confidence") or entry.get("score"))
    speaker = _extract_speaker(entry)
    return {
        "word": text,
        "start": round(start, 3),
        "end": round(end or start, 3),
        "score": round(confidence, 4) if confidence is not None else None,
        "speaker": speaker,
    }


def extract_word_segments(payload: dict) -> List[dict]:
    raw_candidates = []
    if isinstance(payload.get("words"), list):
        raw_candidates = payload["words"]
    elif isinstance(payload.get("word_segments"), list):
        raw_candidates = payload["word_segments"]
    elif isinstance(payload.get("segments"), list):
        collected: List[dict] = []
        for segment in payload["segments"]:
            words = segment.get("words") or []
            if isinstance(words, list):
                collected.extend(words)
        raw_candidates = collected

    normalised: List[dict] = []
    for entry in raw_candidates:
        if isinstance(entry, dict):
            normalised_word = _normalise_word_entry(entry)
            if normalised_word:
                normalised.append(normalised_word)
    normalised.sort(key=lambda item: item["start"])
    return normalised


def adjust_word_timestamps(word_segments: List[dict], padding_s: float) -> List[dict]:
    if not word_segments or padding_s <= 0:
        return word_segments
    adjusted = [dict(word) for word in word_segments]
    for idx in range(len(adjusted) - 1):
        gap = adjusted[idx + 1]["start"] - adjusted[idx]["end"]
        if gap > padding_s * 2:
            delta = min(padding_s, gap / 2.0)
            adjusted[idx]["end"] += delta
            adjusted[idx + 1]["start"] -= delta
    adjusted[0]["start"] = max(0.0, adjusted[0]["start"] - padding_s)
    adjusted[-1]["end"] += padding_s
    for word in adjusted:
        word["start"] = round(word["start"], 3)
        word["end"] = round(word["end"], 3)
    return adjusted


def _create_segment_from_words(words: List[dict]) -> Optional[dict]:
    if not words:
        return None
    text = _sanitize_text(" ".join(word["word"] for word in words))
    seg = {
        "start": round(words[0]["start"], 3),
        "end": round(words[-1]["end"], 3),
        "text": text,
        "words": words,
    }
    # annotate dominant speaker if all the same
    speakers = {w.get("speaker") for w in words if w.get("speaker") is not None}
    if len(speakers) == 1:
        seg["speaker"] = next(iter(speakers))
    elif len(speakers) > 1:
        # Ha több speaker van a szegmensben, akkor a legtöbb szót tartalmazó speaker lesz a domináns
        speaker_counts = {}
        for word in words:
            speaker = word.get("speaker")
            if speaker is not None:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        if speaker_counts:
            dominant_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
            seg["speaker"] = dominant_speaker
            # Jelzés, hogy a szegmens több speakerből áll, de egy domináns speaker van
            seg["mixed_speakers"] = True
    return seg


def sentence_segments_from_words(
    words: List[dict],
    max_pause_s: float,
    *,
    split_on_speaker_change: bool = True,  # Alapértelmezett: mindig szétválasztjuk speaker változásnál
) -> List[dict]:
    if not words:
        return []
    segments: List[dict] = []
    current: List[dict] = []
    for word in words:
        pause_break = current and (word["start"] - current[-1]["end"]) > max_pause_s
        speaker_break = (
            split_on_speaker_change
            and current
            and current[-1].get("speaker") is not None
            and word.get("speaker") is not None
            and word.get("speaker") != current[-1].get("speaker")
        )
        if pause_break or speaker_break:
            segment = _create_segment_from_words(current)
            if segment:
                segments.append(segment)
            current = []
        current.append(word)
    if current:
        segment = _create_segment_from_words(current)
        if segment:
            segments.append(segment)
    return segments


def split_long_segments(segments: List[dict], max_duration_s: float) -> List[dict]:
    if max_duration_s <= 0:
        return segments
    final_segments: List[dict] = []

    def pick_split_index(order_words: List[dict], punctuation: Tuple[str, ...]) -> Optional[int]:
        chosen: Optional[int] = None
        for idx in range(len(order_words) - 1):
            token = order_words[idx]["word"].strip()
            if not token:
                continue
            if token.endswith(punctuation):
                duration = order_words[idx]["end"] - order_words[0]["start"]
                if duration <= max_duration_s:
                    chosen = idx
        return chosen

    def split_by_equal_parts(order_words: List[dict], parts: int) -> List[List[dict]]:
        if parts <= 1 or len(order_words) <= 1:
            return [order_words]
        total_duration = order_words[-1]["end"] - order_words[0]["start"]
        if total_duration <= 0:
            return [order_words]
        target = total_duration / parts
        boundaries = [order_words[0]["start"] + target * k for k in range(1, parts)]
        chunks: List[List[dict]] = []
        start_idx = 0
        for boundary in boundaries:
            idx = start_idx
            while idx < len(order_words) - 1 and order_words[idx]["end"] < boundary:
                idx += 1
            chunk = order_words[start_idx : idx + 1]
            if chunk:
                chunks.append(chunk)
                start_idx = idx + 1
        if start_idx < len(order_words):
            chunks.append(order_words[start_idx:])
        if not chunks:
            return [order_words]
        return chunks

    queue: List[List[dict]] = [segment["words"] for segment in segments if segment.get("words")]

    while queue:
        current_words = queue.pop(0)
        if not current_words:
            continue
        duration = current_words[-1]["end"] - current_words[0]["start"]
        if duration <= max_duration_s or len(current_words) == 1:
            segment = _create_segment_from_words(current_words)
            if segment:
                final_segments.append(segment)
            continue

        split_idx = pick_split_index(current_words, PRIMARY_PUNCTUATION)
        if split_idx is None:
            split_idx = pick_split_index(current_words, SECONDARY_PUNCTUATION)

        if split_idx is not None:
            left = current_words[: split_idx + 1]
            right = current_words[split_idx + 1 :]
            if left:
                queue.insert(0, left)
            if right:
                queue.insert(1, right)
            continue

        parts = max(2, math.ceil(duration / max_duration_s))
        equal_chunks = split_by_equal_parts(current_words, parts)
        if len(equal_chunks) == 1:
            segment = _create_segment_from_words(equal_chunks[0])
            if segment:
                final_segments.append(segment)
        else:
            for chunk in reversed(equal_chunks):
                if chunk:
                    queue.insert(0, chunk)

    return final_segments


def build_segments(
    word_segments: List[dict],
    max_pause_s: float,
    padding_s: float,
    max_segment_s: float,
    *,
    enforce_single_speaker: bool,
) -> List[dict]:
    if not word_segments:
        return []
    adjusted = adjust_word_timestamps(word_segments, padding_s=padding_s)
    initial_segments = sentence_segments_from_words(
        adjusted, max_pause_s=max_pause_s, split_on_speaker_change=enforce_single_speaker
    )
    return split_long_segments(initial_segments, max_segment_s)


def backup_json_file(json_path: Path) -> Path:
    """Create a backup of the JSON file with .json.bak extension."""
    backup_path = json_path.with_suffix(".json.bak")
    try:
        import shutil
        shutil.copy2(json_path, backup_path)
        logging.info(f"Biztonsági mentés létrehozva: {backup_path.name}")
        return backup_path
    except Exception as exc:
        logging.error(f"Hiba a biztonsági mentés létrehozásakor ({json_path}): {exc}")
        raise


def load_json_file(json_path: Path) -> dict:
    """Load JSON file and return its content."""
    try:
        with json_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logging.error(f"Hiba a JSON fájl betöltésekor ({json_path}): {exc}")
        raise


def create_resegmented_json(
    original_data: dict,
    max_pause_s: float,
    padding_s: float,
    max_segment_s: float,
    enforce_single_speaker: bool,
) -> dict:
    """Create new JSON with resegmented data."""
    word_segments = extract_word_segments(original_data)
    
    if not word_segments:
        logging.warning("Nincsenek szó szegmensek a JSON fájlban")
        return original_data

    segments = build_segments(
        word_segments,
        max_pause_s=max_pause_s,
        padding_s=padding_s,
        max_segment_s=max_segment_s,
        enforce_single_speaker=enforce_single_speaker,
    )

    # Preserve original metadata
    result = {
        "segments": segments,
        "word_segments": word_segments,
        "language": original_data.get("language"),
        "provider": original_data.get("provider", "resegment"),
        "diarization": original_data.get("diarization", False),
        "original_provider": original_data.get("provider"),
        "resegment_parameters": {
            "max_pause_s": max_pause_s,
            "padding_s": padding_s,
            "max_segment_s": max_segment_s,
            "enforce_single_speaker": enforce_single_speaker
        }
    }

    return result


def process_json_file(
    json_path: Path,
    max_pause_s: float,
    padding_s: float,
    max_segment_s: float,
    enforce_single_speaker: bool,
    backup: bool,
) -> None:
    """Process a single JSON file: backup, load, resegment, and save."""
    print(f"▶  Feldolgozás: {json_path.name}")
    
    # Create backup if requested
    if backup:
        try:
            backup_json_file(json_path)
        except Exception:
            print(f"  ✖  Sikertelen biztonsági mentés: {json_path.name}")
            return

    # Load original JSON
    try:
        original_data = load_json_file(json_path)
    except Exception:
        print(f"  ✖  Sikertelen JSON betöltés: {json_path.name}")
        return

    # Create resegmented JSON
    try:
        resegmented_data = create_resegmented_json(
            original_data,
            max_pause_s=max_pause_s,
            padding_s=padding_s,
            max_segment_s=max_segment_s,
            enforce_single_speaker=enforce_single_speaker,
        )
    except Exception as exc:
        logging.error(f"Hiba a szegmentálás során ({json_path}): {exc}")
        print(f"  ✖  Szegmentálási hiba: {json_path.name}")
        return

    # Save resegmented JSON - overwrite original file (after backup)
    try:
        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(resegmented_data, fp, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"  ✔  Mentve: {json_path.name}")
    except Exception as exc:
        logging.error(f"Hiba a mentés során ({json_path}): {exc}")
        print(f"  ✖  Mentési hiba: {json_path.name}")


def process_directory(
    input_dir: Path,
    max_pause_s: float,
    padding_s: float,
    max_segment_s: float,
    enforce_single_speaker: bool,
    backup: bool,
) -> None:
    """Process all JSON files in the directory."""
    json_files = sorted([path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json"])

    if not json_files:
        print(f"Nem található JSON fájl a megadott mappában: {input_dir}")
        return

    print(f"{len(json_files)} JSON fájl feldolgozása indul...")
    for json_path in json_files:
        print("-" * 48)
        process_json_file(
            json_path,
            max_pause_s=max_pause_s,
            padding_s=padding_s,
            max_segment_s=max_segment_s,
            enforce_single_speaker=enforce_single_speaker,
            backup=backup,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "JSON szegmensek újraformázása - ASR script által létrehozott JSON fájlok "
            "biztonsági mentése és újraformázása különböző paraméterekkel."
        )
    )
    parser.add_argument(
        "-p",
        "--project-name",
        required=True,
        help="A projekt neve (a workdir alatti mappa), amit fel kell dolgozni.",
    )
    parser.add_argument(
        "--max-pause",
        type=float,
        default=DEFAULT_MAX_PAUSE_S,
        help=f"Mondatszegmensek közti maximális szünet (mp) saját szegmentáláshoz (alapértelmezett: {DEFAULT_MAX_PAUSE_S}).",
    )
    parser.add_argument(
        "--timestamp-padding",
        type=float,
        default=DEFAULT_PADDING_S,
        help=f"Szó időbélyegek bővítése szegmentáláskor (mp, alapértelmezett: {DEFAULT_PADDING_S}).",
    )
    parser.add_argument(
        "--max-segment-duration",
        type=float,
        default=DEFAULT_MAX_SEGMENT_S,
        help=f"Mondatszegmensek maximális hossza (mp, alapértelmezett: {DEFAULT_MAX_SEGMENT_S}).",
    )
    parser.add_argument(
        "--enforce-single-speaker",
        dest="enforce_single_speaker",
        action="store_true",
        default=False,
        help="Speaker diarizáció alapján szegmentálás (alapértelmezett: ki)",
    )
    parser.add_argument(
        "--no-enforce-single-speaker",
        dest="enforce_single_speaker",
        action="store_false",
        help="Speaker diarizáció figyelmen kívül hagyása szegmentáláskor",
    )
    parser.add_argument(
        "--backup",
        dest="backup",
        action="store_true",
        default=True,
        help="JSON fájlok biztonsági mentése .json.bak kiterjesztéssel (alapértelmezett: be)",
    )
    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="JSON fájlok biztonsági mentésének kikapcsolása",
    )

    add_debug_argument(parser)
    args = parser.parse_args()

    log_level = configure_debug_mode(args.debug)
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    config, project_root = load_config()
    input_dir = resolve_project_input(args.project_name, config, project_root)
    
    print("Projekt beállítások betöltve:")
    print(f"  - Projekt név:    {args.project_name}")
    print(f"  - Bemeneti mappa: {input_dir}")
    print(f"  - Max szünet:     {args.max_pause} s")
    print(f"  - Időbélyeg padding: {args.timestamp_padding} s")
    print(f"  - Max szegmens hossz: {args.max_segment_duration} s")
    print(f"  - Speaker szegmentálás: {'BE' if args.enforce_single_speaker else 'KI'}")
    print(f"  - Biztonsági mentés: {'BE' if args.backup else 'KI'}")

    process_directory(
        input_dir=input_dir,
        max_pause_s=args.max_pause,
        padding_s=args.timestamp_padding,
        max_segment_s=args.max_segment_duration,
        enforce_single_speaker=args.enforce_single_speaker,
        backup=args.backup,
    )


if __name__ == "__main__":
    main()

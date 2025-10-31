"""
Resegment script with Montreal Forced Aligner (MFA) integration.
This script refines word timestamps using forced alignment for English audio
and then resegments the transcript based on various parameters.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Attempt to locate the 'textgrid' library, required for MFA output parsing.
try:
    import textgrid
except ImportError:
    print("Error: The 'textgrid' library is not installed.")
    print("Please install it using: pip install textgrid")
    sys.exit(1)

# Ensure the script can find tools from the project root
for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

# --- Constants ---
DEFAULT_MAX_PAUSE_S = 0.8
DEFAULT_PADDING_S = 0.1
DEFAULT_MAX_SEGMENT_S = 11.5
PRIMARY_PUNCTUATION = (".", "!", "?")
SECONDARY_PUNCTUATION = (",",)
SUPPORTED_AUDIO_EXTENSIONS: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
MFA_TARGET_SAMPLE_RATE = 16000  # MFA typically works best with 16kHz audio

# --- Utility Functions ---

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
    raise FileNotFoundError("Could not find config.json in any parent directories.")


def load_config() -> Tuple[dict, Path]:
    """Load config.json and return it together with the project root."""
    project_root = get_project_root()
    config_path = project_root / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            config = json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Error loading configuration ({config_path}): {exc}")
        sys.exit(1)
    return config, project_root


def resolve_project_input(project_name: str, config: dict, project_root: Path) -> Path:
    """Resolve the directory that contains ASR JSON files for the project."""
    try:
        workdir = project_root / config["DIRECTORIES"]["workdir"]
        input_subdir = config["PROJECT_SUBDIRS"]["separated_audio_speech"]
    except KeyError as exc:
        print(f"Error: missing key in config.json: {exc}")
        sys.exit(1)

    input_dir = workdir / project_name / input_subdir
    if not input_dir.is_dir():
        print(f"Error: processing directory not found: {input_dir}")
        sys.exit(1)
    return input_dir


def find_audio_for_json(json_path: Path) -> Optional[Path]:
    """Try to locate the audio file that belongs to the JSON transcript."""
    stem = json_path.stem
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        candidate = json_path.with_suffix(ext)
        if candidate.is_file():
            return candidate
    for candidate in json_path.parent.iterdir():
        if (
            candidate.is_file()
            and candidate.stem == stem
            and candidate.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        ):
            return candidate
    return None

# --- MFA (Montreal Forced Aligner) Integration ---

def _prepare_mfa_input(
    audio_path: Path,
    word_segments: List[dict],
    temp_dir: Path,
) -> Optional[Tuple[Path, Path]]:
    """
    Prepares audio and transcript files for MFA.
    Converts audio to 16kHz mono WAV and creates a .lab transcript file.
    Returns (path_to_wav, path_to_lab) or None on failure.
    """
    # 1. Prepare audio file
    mfa_audio_path = temp_dir / f"{audio_path.stem}.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(MFA_TARGET_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        str(mfa_audio_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        logging.error("ffmpeg is not available. It is required for MFA audio preparation.")
        return None
    except subprocess.CalledProcessError as exc:
        error_message = exc.stderr.decode("utf-8", errors="ignore").strip()
        logging.error(f"FFmpeg conversion failed for MFA ({audio_path}): {error_message}")
        return None

    # 2. Prepare transcript file (.lab)
    transcript = " ".join(word.get("word", "") for word in word_segments)
    mfa_transcript_path = temp_dir / f"{audio_path.stem}.lab"
    try:
        with mfa_transcript_path.open("w", encoding="utf-8") as f:
            f.write(transcript)
    except IOError as exc:
        logging.error(f"Failed to write MFA transcript file: {exc}")
        return None

    return mfa_audio_path, mfa_transcript_path


def _parse_mfa_textgrid(textgrid_path: Path, original_words: List[dict]) -> List[dict]:
    """Parses the TextGrid output from MFA and updates word timestamps."""
    try:
        tg = textgrid.TextGrid.fromFile(str(textgrid_path))
    except Exception as e:
        logging.warning(f"Could not parse TextGrid file '{textgrid_path}': {e}")
        return original_words

    # VÉGLEGES JAVÍTÁS: Verziófüggetlen módszer a sávnevek lekérdezésére.
    # Közvetlenül a 'tiers' listán iterálunk, ami minden verzióban stabil.
    tier_names = [tier.name for tier in tg.tiers]
    if "words" not in tier_names:
        logging.warning(f"TextGrid file '{textgrid_path}' does not contain a 'words' tier.")
        return original_words

    word_tier = tg.getFirst("words")
    aligned_words = [interval for interval in word_tier if interval.mark and interval.mark.strip()]
    
    if len(aligned_words) != len(original_words):
        logging.warning(
            f"MFA alignment mismatch: original had {len(original_words)} words, "
            f"MFA produced {len(aligned_words)}. Timestamps will not be updated."
        )
        return original_words

    refined_segments = []
    for i, original_word in enumerate(original_words):
        aligned_interval = aligned_words[i]
        
        original_alphanum = "".join(filter(str.isalnum, original_word["word"].lower()))
        mfa_alphanum = "".join(filter(str.isalnum, aligned_interval.mark.lower()))

        if original_alphanum != mfa_alphanum:
            logging.debug(
                f"Word mismatch at index {i}: original='{original_word['word']}', "
                f"MFA='{aligned_interval.mark}'. Using MFA timestamps anyway."
            )

        updated_word = dict(original_word)
        updated_word["start"] = round(aligned_interval.minTime, 3)
        updated_word["end"] = round(aligned_interval.maxTime, 3)
        refined_segments.append(updated_word)

    return refined_segments


def refine_word_segments_with_mfa(
    word_segments: List[dict],
    audio_path: Path,
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Adjusts word timestamps using the Montreal Forced Aligner.
    This function assumes MFA is installed and English models are downloaded.
    """
    report = {"status": "skipped", "audio": audio_path.name}

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        mfa_input_dir = temp_dir / "mfa_input"
        mfa_output_dir = temp_dir / "mfa_output"
        mfa_input_dir.mkdir()
        mfa_output_dir.mkdir()

        prepared = _prepare_mfa_input(audio_path, word_segments, mfa_input_dir)
        if not prepared:
            report["status"] = "skipped_ffmpeg_error"
            return word_segments, report
        
        acoustic_model = "english_mfa"
        dictionary = "english_mfa"
        
        command = [
            "mfa", "align", str(mfa_input_dir), dictionary, acoustic_model, str(mfa_output_dir),
            "--clean", "--quiet",
        ]
        
        try:
            logging.info(f"Running MFA on {audio_path.name}...")
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        except FileNotFoundError:
            logging.error("MFA command not found. Is Montreal Forced Aligner installed and in your PATH?")
            report["status"] = "skipped_mfa_not_found"
            return word_segments, report
        except subprocess.CalledProcessError as exc:
            logging.error(f"MFA alignment failed for {audio_path.name}.")
            logging.error(f"MFA STDOUT: {exc.stdout}")
            logging.error(f"MFA STDERR: {exc.stderr}")
            report["status"] = "skipped_mfa_error"
            report["error_log"] = exc.stderr
            return word_segments, report

        output_tg_path = mfa_output_dir / f"{audio_path.stem}.TextGrid"
        if not output_tg_path.is_file():
            logging.warning(f"MFA did not produce an output TextGrid for {audio_path.name}.")
            report["status"] = "skipped_mfa_no_output"
            return word_segments, report
            
        refined_segments = _parse_mfa_textgrid(output_tg_path, word_segments)
        report["status"] = "applied_mfa_alignment"
        logging.info(f"MFA alignment successful for {audio_path.name}.")
        return refined_segments, report

# --- ASR Data Handling & Resegmentation Logic ---

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
    if not text: return None
    start = (_safe_float(entry.get("start")) or _safe_float(entry.get("start_time")) or
             _safe_float(entry.get("offset_seconds")) or _safe_float(entry.get("offset")))
    end = (_safe_float(entry.get("end")) or _safe_float(entry.get("end_time")) or
           _safe_float(entry.get("offset_seconds_end")))
    if start is None: return None
    if end is None: end = start + (_safe_float(entry.get("duration")) or 0.0)
    confidence = _safe_float(entry.get("confidence") or entry.get("score"))
    speaker = _extract_speaker(entry)
    return {"word": text, "start": round(start, 3), "end": round(end or start, 3),
            "score": round(confidence, 4) if confidence is not None else None, "speaker": speaker}


def extract_word_segments(payload: dict) -> List[dict]:
    raw_candidates = []
    if isinstance(payload.get("words"), list):
        raw_candidates = payload["words"]
    elif isinstance(payload.get("word_segments"), list):
        raw_candidates = payload["word_segments"]
    elif isinstance(payload.get("segments"), list):
        collected: List[dict] = []
        for segment in payload["segments"]:
            if isinstance(segment.get("words"), list):
                collected.extend(segment["words"])
        raw_candidates = collected
    normalised = [norm for entry in raw_candidates if isinstance(entry, dict) and (norm := _normalise_word_entry(entry))]
    normalised.sort(key=lambda item: item["start"])
    return normalised


def adjust_word_timestamps(word_segments: List[dict], padding_s: float) -> List[dict]:
    if not word_segments or padding_s <= 0: return word_segments
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
    if not words: return None
    text = _sanitize_text(" ".join(word["word"] for word in words))
    seg = {"start": round(words[0]["start"], 3), "end": round(words[-1]["end"], 3), "text": text, "words": words}
    speakers = {w.get("speaker") for w in words if w.get("speaker") is not None}
    if len(speakers) == 1:
        seg["speaker"] = next(iter(speakers))
    elif len(speakers) > 1:
        speaker_counts = {}
        for spk in (w.get("speaker") for w in words if w.get("speaker") is not None):
            speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
        if speaker_counts:
            dominant_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
            seg["speaker"] = dominant_speaker
            seg["mixed_speakers"] = True
    return seg


def sentence_segments_from_words(words: List[dict], max_pause_s: float, *, split_on_speaker_change: bool = True) -> List[dict]:
    if not words: return []
    segments, current = [], []
    for word in words:
        pause_break = current and (word["start"] - current[-1]["end"]) > max_pause_s
        speaker_break = (split_on_speaker_change and current and current[-1].get("speaker") is not None and
                         word.get("speaker") is not None and word.get("speaker") != current[-1].get("speaker"))
        if pause_break or speaker_break:
            if seg := _create_segment_from_words(current): segments.append(seg)
            current = []
        current.append(word)
    if seg := _create_segment_from_words(current): segments.append(seg)
    return segments


def split_long_segments(segments: List[dict], max_duration_s: float) -> List[dict]:
    if max_duration_s <= 0: return segments
    final_segments = []
    queue = [seg["words"] for seg in segments if seg.get("words")]
    while queue:
        current_words = queue.pop(0)
        if not current_words: continue
        duration = current_words[-1]["end"] - current_words[0]["start"]
        if duration <= max_duration_s or len(current_words) == 1:
            if seg := _create_segment_from_words(current_words): final_segments.append(seg)
            continue
        
        def pick_split_index(words, punc):
            chosen = None
            for i, w in enumerate(words[:-1]):
                if w["word"].strip().endswith(punc) and (w["end"] - words[0]["start"]) <= max_duration_s:
                    chosen = i
            return chosen

        split_idx = pick_split_index(current_words, PRIMARY_PUNCTUATION) or pick_split_index(current_words, SECONDARY_PUNCTUATION)
        if split_idx is not None:
            if left := current_words[: split_idx + 1]: queue.insert(0, left)
            if right := current_words[split_idx + 1 :]: queue.insert(1, right)
            continue
        
        parts = max(2, math.ceil(duration / max_duration_s))
        total_dur = current_words[-1]["end"] - current_words[0]["start"]
        target_dur_per_part = total_dur / parts
        
        start_idx = 0
        for _ in range(parts - 1):
            if start_idx >= len(current_words): break
            split_at = -1
            for i in range(start_idx, len(current_words)):
                if (current_words[i]["end"] - current_words[start_idx]["start"]) >= target_dur_per_part:
                    split_at = i
                    break
            if split_at <= start_idx: split_at = start_idx
            
            chunk = current_words[start_idx : split_at + 1]
            if chunk: queue.append(chunk)
            start_idx = split_at + 1
        
        if final_chunk := current_words[start_idx:]: queue.append(final_chunk)
        
    return final_segments


def build_segments(word_segments: List[dict], max_pause_s: float, padding_s: float, max_segment_s: float, *, enforce_single_speaker: bool) -> List[dict]:
    if not word_segments: return []
    adjusted = adjust_word_timestamps(word_segments, padding_s=padding_s)
    initial = sentence_segments_from_words(adjusted, max_pause_s=max_pause_s, split_on_speaker_change=enforce_single_speaker)
    return split_long_segments(initial, max_segment_s)

# --- Main File Processing Logic ---

def backup_json_file(json_path: Path) -> Path:
    base_backup_str = str(json_path.with_suffix(".json.bak"))
    backup_path = Path(base_backup_str)
    if backup_path.exists():
        counter = 2
        while (numbered_backup_path := Path(f"{base_backup_str}_{counter}")).exists():
            counter += 1
        backup_path = numbered_backup_path
    try:
        shutil.copy2(json_path, backup_path)
        logging.info(f"Backup created: {backup_path.name}")
        return backup_path
    except Exception as exc:
        logging.error(f"Failed to create backup for {json_path}: {exc}")
        raise

def load_json_file(json_path: Path) -> dict:
    try:
        with json_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logging.error(f"Error loading JSON file {json_path}: {exc}")
        raise


def create_resegmented_json(original_data: dict, max_pause_s: float, padding_s: float, max_segment_s: float, enforce_single_speaker: bool, *, source_label: str, audio_path: Optional[Path], mfa_refine: bool, word_by_word: bool) -> dict:
    word_segments = extract_word_segments(original_data)
    if not word_segments:
        logging.warning(f"No word segments found in {source_label}")
        return original_data
    alignment_report = None
    if mfa_refine:
        if audio_path:
            refined_segments, alignment_report = refine_word_segments_with_mfa(word_segments, audio_path)
            if "applied" in alignment_report.get("status", ""):
                word_segments = refined_segments
            else:
                logging.info(f"MFA refinement skipped for {audio_path.name}: {alignment_report.get('status')}")
        else:
            alignment_report = {"status": "skipped_audio_missing", "audio": None}
            logging.warning(f"No associated audio file for MFA refinement (JSON: {source_label}).")
    segments = []
    if word_by_word:
        logging.info("Creating word-by-word segments due to '--word-by-word-segments' flag.")
        for word in word_segments:
            segment = {"start": word["start"], "end": word["end"], "text": word["word"], "words": [word]}
            if word.get("speaker"): segment["speaker"] = word["speaker"]
            segments.append(segment)
    else:
        segments = build_segments(word_segments, max_pause_s, padding_s, max_segment_s, enforce_single_speaker=enforce_single_speaker)
    final_segments = [seg for seg in segments if not (seg.get("text", "").strip().startswith("[") and seg.get("text", "").strip().endswith("]"))]
    result = {
        "segments": final_segments, "word_segments": word_segments,
        "language": original_data.get("language"), "provider": "resegment_mfa",
        "diarization": original_data.get("diarization", False),
        "original_provider": original_data.get("provider"),
        "resegment_parameters": {"max_pause_s": max_pause_s, "padding_s": padding_s, "max_segment_s": max_segment_s,
                                 "enforce_single_speaker": enforce_single_speaker, "mfa_refine": mfa_refine,
                                 "word_by_word_segments": word_by_word}
    }
    if audio_path: result["audio_file"] = audio_path.name
    if alignment_report: result["alignment_adjustment"] = alignment_report
    return result


def process_json_file(json_path: Path, max_pause_s: float, padding_s: float, max_segment_s: float, enforce_single_speaker: bool, backup: bool, mfa_refine: bool, word_by_word: bool) -> None:
    print(f"▶  Processing: {json_path.name}")
    if backup:
        try:
            backup_json_file(json_path)
        except Exception:
            print(f"  ✖  Backup failed: {json_path.name}"); return
    try:
        original_data = load_json_file(json_path)
    except Exception:
        print(f"  ✖  JSON loading failed: {json_path.name}"); return
    audio_path = find_audio_for_json(json_path) if mfa_refine else None
    if mfa_refine and not audio_path:
        logging.warning(f"No audio file found for MFA alignment ({json_path.name}).")
    try:
        resegmented_data = create_resegmented_json(original_data, max_pause_s, padding_s, max_segment_s,
                                                 enforce_single_speaker, source_label=json_path.name,
                                                 audio_path=audio_path, mfa_refine=mfa_refine, word_by_word=word_by_word)
    except Exception as exc:
        logging.error(f"Error during resegmentation of {json_path}: {exc}", exc_info=True)
        print(f"  ✖  Resegmentation error: {json_path.name}"); return
    try:
        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(resegmented_data, fp, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"  ✔  Saved: {json_path.name}")
    except Exception as exc:
        logging.error(f"Error saving file {json_path}: {exc}")
        print(f"  ✖  Save error: {json_path.name}")


def process_directory(input_dir: Path, max_pause_s: float, padding_s: float, max_segment_s: float, enforce_single_speaker: bool, backup: bool, mfa_refine: bool, word_by_word: bool) -> None:
    json_files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    if not json_files:
        print(f"No JSON files found in directory: {input_dir}"); return
    print(f"Found {len(json_files)} JSON files to process...")
    for json_path in json_files:
        print("-" * 48)
        process_json_file(json_path, max_pause_s, padding_s, max_segment_s, enforce_single_speaker, backup, mfa_refine, word_by_word)

# --- Main Execution ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Resegment ASR JSON files with optional MFA refinement for English.")
    parser.add_argument("-p", "--project-name", required=True, help="The project name (subfolder in the workdir) to process.")
    parser.add_argument("--max-pause", type=float, default=DEFAULT_MAX_PAUSE_S, help=f"Max pause between words (default: {DEFAULT_MAX_PAUSE_S}s).")
    parser.add_argument("--timestamp-padding", type=float, default=DEFAULT_PADDING_S, help=f"Padding for timestamps (default: {DEFAULT_PADDING_S}s).")
    parser.add_argument("--max-segment-duration", type=float, default=DEFAULT_MAX_SEGMENT_S, help=f"Max duration of a segment (default: {DEFAULT_MAX_SEGMENT_S}s).")
    parser.add_argument("--enforce-single-speaker", action="store_true", help="Split segments on speaker change.")
    parser.add_argument("--no-backup", dest="backup", action="store_false", help="Disable backup of original JSON files.")
    parser.add_argument("--use-mfa-refine", action="store_true", help="Enable timestamp refinement with MFA.")
    parser.add_argument("--word-by-word-segments", action="store_true", help="Place each word in its own segment.")
    add_debug_argument(parser)
    args = parser.parse_args()
    parser.set_defaults(backup=True)

    log_level = configure_debug_mode(args.debug)
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    config, project_root = load_config()
    input_dir = resolve_project_input(args.project_name, config, project_root)
    
    print("Project settings loaded:")
    print(f"  - Project Name:    {args.project_name}\n  - Input Directory: {input_dir}")
    print(f"  - Max Pause:       {args.max_pause}s\n  - Padding:         {args.timestamp_padding}s")
    print(f"  - Max Segment Len: {args.max_segment_duration}s\n  - Speaker Split:   {'ON' if args.enforce_single_speaker else 'OFF'}")
    print(f"  - Backup:          {'ON' if args.backup else 'OFF'}\n  - MFA Refinement:  {'ON' if args.use_mfa_refine else 'OFF'}")
    print(f"  - Word-by-word:    {'ON' if args.word_by_word_segments else 'OFF'}")

    process_directory(input_dir, args.max_pause, args.timestamp_padding, args.max_segment_duration,
                      args.enforce_single_speaker, args.backup, args.use_mfa_refine, args.word_by_word_segments)


if __name__ == "__main__":
    main()
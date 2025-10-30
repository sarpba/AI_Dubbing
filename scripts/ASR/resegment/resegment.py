"""
Resegment script - JSON szegmensek újraformázása különböző paraméterekkel.
Az ASR script által létrehozott JSON fájlok biztonsági mentése és újraformázása.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import webrtcvad

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
SUPPORTED_AUDIO_EXTENSIONS: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
VAD_ALLOWED_SAMPLE_RATES: Tuple[int, ...] = (8000, 16000, 32000, 48000)
VAD_TARGET_SAMPLE_RATE = 16000
VAD_FRAME_DURATION_MS = 30
VAD_GAP_MERGE_S = 0.05
VAD_TOLERANCE_S = 0.05
MIN_WORD_DURATION_S = 0.02


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


def _convert_to_vad_wav(path: Path) -> Optional[Path]:
    """Convert arbitrary audio to a PCM16 mono WAV accepted by WebRTC VAD."""
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    except Exception as exc:
        logging.warning("Nem sikerült ideiglenes fájlt létrehozni VAD-hoz (%s): %s", path, exc)
        return None
    temp_file.close()
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(VAD_TARGET_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        temp_file.name,
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return Path(temp_file.name)
    except FileNotFoundError:
        logging.warning("Az ffmpeg nem elérhető, a VAD alapú korrekció kihagyva (%s).", path)
    except subprocess.CalledProcessError as exc:
        error_message = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        logging.warning("FFmpeg konverzió sikertelen VAD-hoz (%s): %s", path, error_message.strip())
    try:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
    except OSError:
        pass
    return None


def read_wave_for_vad(path: Path) -> Optional[Tuple[bytes, int, int]]:
    """Load audio data suitable for the VAD. Returns bytes, sample_rate, sample_width."""
    converted_path: Optional[Path] = None
    source_path = path
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as wf:
            sample_rate = wf.getframerate()
            if sample_rate not in VAD_ALLOWED_SAMPLE_RATES or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                raise ValueError("unsupported format")
            frames = wf.readframes(wf.getnframes())
            return frames, sample_rate, wf.getsampwidth()
    except (wave.Error, FileNotFoundError, ValueError):
        converted_path = _convert_to_vad_wav(source_path)
        if not converted_path:
            return None
        try:
            with contextlib.closing(wave.open(str(converted_path), "rb")) as wf:
                frames = wf.readframes(wf.getnframes())
                return frames, wf.getframerate(), wf.getsampwidth()
        except (wave.Error, FileNotFoundError) as exc:
            logging.warning("A konvertált WAV nem tölthető be VAD-hoz (%s): %s", source_path, exc)
            return None
        finally:
            try:
                os.remove(str(converted_path))
            except OSError:
                pass


def _vad_frame_generator(
    frame_duration_ms: int, audio: bytes, sample_rate: int, sample_width: int
) -> Iterable[Tuple[bytes, float]]:
    bytes_per_frame = int(sample_rate * (frame_duration_ms / 1000.0) * sample_width)
    if bytes_per_frame <= 0:
        return []
    offset = 0
    timestamp = 0.0
    frame_duration = frame_duration_ms / 1000.0
    while offset + bytes_per_frame <= len(audio):
        yield audio[offset : offset + bytes_per_frame], timestamp
        timestamp += frame_duration
        offset += bytes_per_frame


def _collect_vad_regions(
    frames: Iterable[Tuple[bytes, float]],
    sample_rate: int,
    aggressiveness: int,
) -> List[Tuple[float, float]]:
    vad = webrtcvad.Vad(int(max(0, min(3, aggressiveness))))
    regions: List[Tuple[float, float]] = []
    in_region = False
    region_start = 0.0
    frame_duration = VAD_FRAME_DURATION_MS / 1000.0
    for frame, timestamp in frames:
        try:
            is_speech = vad.is_speech(frame, sample_rate)
        except Exception as exc:
            logging.debug("VAD feldolgozási hiba %s időbélyegnél: %s", timestamp, exc)
            is_speech = False
        if is_speech:
            if not in_region:
                in_region = True
                region_start = timestamp
        else:
            if in_region:
                regions.append((region_start, timestamp))
                in_region = False
    if in_region:
        regions.append((region_start, timestamp + frame_duration))

    if not regions:
        return []

    # Merge regions separated by tiny gaps
    merged: List[Tuple[float, float]] = []
    current_start, current_end = regions[0]
    for start, end in regions[1:]:
        if start - current_end <= VAD_GAP_MERGE_S:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _find_matching_region(
    regions: Sequence[Tuple[float, float]], start: float, end: float, tolerance: float = VAD_TOLERANCE_S
) -> Optional[Tuple[float, float]]:
    for region_start, region_end in regions:
        if region_end + tolerance < start:
            continue
        if region_start - tolerance > end:
            break
        overlap = min(end, region_end) - max(start, region_start)
        if overlap >= -tolerance:
            return region_start, region_end
    return None


def refine_word_segments_with_vad(
    word_segments: List[dict],
    audio_path: Path,
    *,
    aggressiveness: int = 3,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Use VAD to trim word boundaries closer to actual speech activity."""
    audio_data = read_wave_for_vad(audio_path)
    if not audio_data:
        return word_segments, {
            "status": "skipped_audio_unavailable",
            "audio": audio_path.name,
        }
    audio_bytes, sample_rate, sample_width = audio_data
    frames = list(_vad_frame_generator(VAD_FRAME_DURATION_MS, audio_bytes, sample_rate, sample_width))
    if not frames:
        return word_segments, {
            "status": "skipped_no_frames",
            "audio": audio_path.name,
        }
    regions = _collect_vad_regions(frames, sample_rate, aggressiveness)
    if not regions:
        return word_segments, {
            "status": "skipped_no_regions",
            "audio": audio_path.name,
        }

    refined: List[dict] = []
    for idx, word in enumerate(word_segments):
        start = float(word.get("start", 0.0) or 0.0)
        end = float(word.get("end", start))
        if end <= start:
            end = start + MIN_WORD_DURATION_S
        region = _find_matching_region(regions, start, end)
        adjusted_start, adjusted_end = start, end
        region_end_value: Optional[float] = None
        if region:
            region_start, region_end = region
            region_end_value = region_end
            adjusted_start = max(start, region_start)
            adjusted_end = min(end, region_end)
            if adjusted_end - adjusted_start < MIN_WORD_DURATION_S:
                adjusted_end = min(region_end, adjusted_start + MIN_WORD_DURATION_S)
        if refined:
            previous_end = refined[-1]["end"]
            if adjusted_start < previous_end:
                adjusted_start = previous_end
        if adjusted_end < adjusted_start + MIN_WORD_DURATION_S:
            adjusted_end = adjusted_start + MIN_WORD_DURATION_S
            if region_end_value is not None:
                adjusted_end = min(adjusted_end, region_end_value)
        if region_end_value is not None:
            adjusted_end = min(adjusted_end, region_end_value)
        if adjusted_end > end:
            adjusted_end = end
        if adjusted_end <= adjusted_start:
            adjusted_end = adjusted_start + MIN_WORD_DURATION_S
            if region_end_value is not None:
                adjusted_end = min(adjusted_end, region_end_value)
            if adjusted_end <= adjusted_start:
                adjusted_end = adjusted_start + MIN_WORD_DURATION_S
        refined_word = dict(word)
        refined_word["start"] = round(adjusted_start, 3)
        refined_word["end"] = round(adjusted_end, 3)
        refined.append(refined_word)

    report = {
        "status": "applied",
        "audio": audio_path.name,
        "aggressiveness": int(max(0, min(3, aggressiveness))),
        "frame_duration_ms": VAD_FRAME_DURATION_MS,
        "regions": len(regions),
    }
    return refined, report


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
    *,
    source_label: str,
    audio_path: Optional[Path],
    use_vad: bool,
    vad_aggressiveness: int,
) -> dict:
    """Create new JSON with resegmented data."""
    word_segments = extract_word_segments(original_data)
    
    if not word_segments:
        logging.warning("Nincsenek szó szegmensek a JSON fájlban")
        return original_data

    vad_report: Optional[Dict[str, Any]] = None
    if use_vad:
        if audio_path:
            refined_segments, vad_report = refine_word_segments_with_vad(
                word_segments,
                audio_path,
                aggressiveness=vad_aggressiveness,
            )
            if vad_report.get("status") == "applied":
                word_segments = refined_segments
                logging.debug(
                    "VAD korrekció alkalmazva (%s, régiók: %s)",
                    audio_path.name,
                    vad_report.get("regions"),
                )
            else:
                logging.info(
                    "VAD kihagyva (%s): %s",
                    audio_path.name,
                    vad_report.get("status"),
                )
        else:
            vad_report = {
                "status": "skipped_audio_missing",
                "audio": None,
            }
            logging.warning("Nem található hangfájl VAD korrekcióhoz (JSON: %s).", source_label)

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
            "enforce_single_speaker": enforce_single_speaker,
            "vad_enabled": use_vad,
            "vad_aggressiveness": vad_aggressiveness,
        }
    }
    if audio_path:
        result["audio_file"] = audio_path.name
    if vad_report:
        result["vad_adjustment"] = vad_report

    return result


def process_json_file(
    json_path: Path,
    max_pause_s: float,
    padding_s: float,
    max_segment_s: float,
    enforce_single_speaker: bool,
    backup: bool,
    use_vad: bool,
    vad_aggressiveness: int,
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

    audio_path: Optional[Path] = None
    if use_vad:
        audio_path = find_audio_for_json(json_path)
        if audio_path is None:
            logging.warning("Nem található hozzárendelt hangfájl VAD korrekcióhoz (%s).", json_path.name)

    # Create resegmented JSON
    try:
        resegmented_data = create_resegmented_json(
            original_data,
            max_pause_s=max_pause_s,
            padding_s=padding_s,
            max_segment_s=max_segment_s,
            enforce_single_speaker=enforce_single_speaker,
            source_label=json_path.name,
            audio_path=audio_path,
            use_vad=use_vad,
            vad_aggressiveness=vad_aggressiveness,
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
    use_vad: bool,
    vad_aggressiveness: int,
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
            use_vad=use_vad,
            vad_aggressiveness=vad_aggressiveness,
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
    parser.add_argument(
        "--skip-vad",
        dest="skip_vad",
        action="store_true",
        default=False,
        help="WebRTC VAD alapú szó időbélyeg korrekció kihagyása.",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        choices=(0, 1, 2, 3),
        default=3,
        help="WebRTC VAD agresszivitási szintje (0-3, alapértelmezett: 3).",
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
    vad_enabled = not args.skip_vad
    print(
        f"  - VAD korrekció: {'BE' if vad_enabled else 'KI'}"
        + (f" (aggr.: {args.vad_aggressiveness})" if vad_enabled else "")
    )

    process_directory(
        input_dir=input_dir,
        max_pause_s=args.max_pause,
        padding_s=args.timestamp_padding,
        max_segment_s=args.max_segment_duration,
        enforce_single_speaker=args.enforce_single_speaker,
        backup=args.backup,
        use_vad=vad_enabled,
        vad_aggressiveness=args.vad_aggressiveness,
    )


if __name__ == "__main__":
    main()

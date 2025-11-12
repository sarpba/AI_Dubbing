"""
Soniox ASR integráció – REST alapú aszinkron feldolgozás a projekthez tartozó hangtárra, normalizált szó szintű JSON kimenettel.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from requests import Response, Session

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

SUPPORTED_EXTENSIONS: Sequence[str] = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
ENV_API_KEY = "SONIOX_API_KEY"
KEYHOLDER_FIELD = "soniox_api_key"
DEFAULT_MODEL = "stt-async-v3"
DEFAULT_REFERENCE_PREFIX = "soniox"
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_TIMEOUT = 1800
DEFAULT_CHUNK_SIZE = 131072  # Nem szükséges REST módban, de CLI kompatibilitás miatt marad.
COMPLETED_STATUSES = {"completed"}
FAILED_STATUSES = {"error"}
API_BASE_URL = "https://api.soniox.com"
DEFAULT_SENTENCE_MAX_PAUSE_S = 0.8
DEFAULT_SENTENCE_PADDING_S = 0.1
DEFAULT_SENTENCE_MAX_SEGMENT_S = 11.5
PRIMARY_PUNCTUATION: Tuple[str, ...] = (".", "!", "?")
SECONDARY_PUNCTUATION: Tuple[str, ...] = (",",)
SONIOX_MAX_FILE_SIZE_BYTES = 524_288_000


class ConfigurationError(Exception):
    """Konzisztens hiba jelzés konfigurációs problémákra."""


def get_project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")


def load_config() -> tuple[dict, Path]:
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
        logging.info("Soniox API kulcs elmentve a keyholder.json fájlba.")
    except Exception as exc:
        logging.error("Nem sikerült elmenteni a Soniox API kulcsot: %s", exc)


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
        logging.error("Nem sikerült beolvasni a Soniox API kulcsot: %s", exc)
        return None
    except Exception as exc:
        logging.error("Váratlan hiba kulcs betöltésekor: %s", exc)
        return None


def resolve_api_key(args: argparse.Namespace, project_root: Path) -> Optional[str]:
    if args.api_key:
        save_api_key(project_root, args.api_key)
        return args.api_key
    env_key = os.environ.get(ENV_API_KEY)
    if env_key:
        logging.info("Soniox API kulcs betöltve környezeti változóból (%s).", ENV_API_KEY)
        return env_key
    stored_key = load_api_key(project_root)
    if stored_key:
        logging.info("Soniox API kulcs betöltve a keyholder.json fájlból.")
        return stored_key
    return None


def build_endpoints(api_host: Optional[str]) -> Dict[str, str]:
    base = api_host.rstrip("/") if api_host else API_BASE_URL
    return {
        "base": base,
        "files": f"{base}/v1/files",
        "transcriptions": f"{base}/v1/transcriptions",
    }


def create_session(api_key: str) -> Session:
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {api_key}"
    session.headers["User-Agent"] = "AI_Dubbing-Soniox/1.0"
    return session


def handle_response(response: Response, context: str) -> Response:
    if response.ok:
        return response
    try:
        payload = response.json()
        message = payload.get("error", payload)
    except ValueError:
        message = response.text
    raise RuntimeError(f"Soniox hiba {context}: {response.status_code} – {message}")


def upload_audio(session: Session, endpoints: Dict[str, str], audio_path: Path) -> str:
    logging.info("Soniox fájl feltöltése: %s", audio_path.name)
    with audio_path.open("rb") as fp:
        files = {"file": (audio_path.name, fp, "application/octet-stream")}
        response = handle_response(session.post(endpoints["files"], files=files, timeout=600), "fájl feltöltés")
    payload = response.json()
    file_id = payload.get("id")
    if not file_id:
        raise RuntimeError("A Soniox fájl feltöltés nem adott file_id mezőt.")
    return file_id


def build_transcription_payload(
    *,
    file_id: str,
    args: argparse.Namespace,
    reference_name: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": args.model,
        "file_id": file_id,
        "client_reference_id": reference_name,
    }
    payload["enable_speaker_diarization"] = bool(args.diarize)
    payload["language_hints"] = []
    if args.min_speakers or args.max_speakers:
        logging.warning("A Soniox REST API jelenleg nem támogatja a min/max beszélő beállítást – kihagyva.")
    if args.candidate_speaker:
        logging.warning("A Soniox REST API nem támogatja a candidate speaker listát – kihagyva.")
    if args.speaker_identification:
        logging.warning("A Soniox REST API jelenleg nem támogatja a speaker identification kapcsolót – kihagyva.")
    return payload


def create_transcription(
    session: Session,
    endpoints: Dict[str, str],
    payload: Dict[str, Any],
) -> str:
    response = handle_response(
        session.post(endpoints["transcriptions"], json=payload, timeout=60),
        "transcription létrehozás",
    )
    data = response.json()
    transcription_id = data.get("id")
    if not transcription_id:
        raise RuntimeError("A Soniox nem adott transcription_id mezőt.")
    return transcription_id


def poll_transcription(
    session: Session,
    endpoints: Dict[str, str],
    transcription_id: str,
    *,
    poll_interval: float,
    timeout_seconds: int,
) -> Dict[str, Any]:
    status_url = f"{endpoints['transcriptions']}/{transcription_id}"
    start = time.monotonic()
    last_status = ""
    while True:
        response = handle_response(session.get(status_url, timeout=30), "transcription státusz lekérés")
        payload = response.json()
        status = (payload.get("status") or "").lower()
        if status != last_status:
            logging.info("Soniox státusz (%s): %s", transcription_id, payload.get("status") or "ismeretlen")
            last_status = status
        if status in COMPLETED_STATUSES:
            return payload
        if status in FAILED_STATUSES:
            message = payload.get("error_message") or "Ismeretlen Soniox hiba."
            raise RuntimeError(f"Soniox transcription hiba: {message}")
        if timeout_seconds and (time.monotonic() - start) > timeout_seconds:
            raise TimeoutError(f"Időtúllépés a Soniox transcription befejezésére várva ({transcription_id}).")
        time.sleep(poll_interval)


def download_transcript(session: Session, endpoints: Dict[str, str], transcription_id: str) -> Dict[str, Any]:
    transcript_url = f"{endpoints['transcriptions']}/{transcription_id}/transcript"
    response = handle_response(session.get(transcript_url, timeout=120), "transcript letöltés")
    return response.json()


def delete_transcription(session: Session, endpoints: Dict[str, str], transcription_id: str) -> None:
    try:
        session.delete(f"{endpoints['transcriptions']}/{transcription_id}", timeout=30)
    except requests.RequestException:
        logging.warning("Nem sikerült törölni a Soniox transcription-t (%s).", transcription_id)


def delete_file(session: Session, endpoints: Dict[str, str], file_id: str) -> None:
    try:
        session.delete(f"{endpoints['files']}/{file_id}", timeout=30)
    except requests.RequestException:
        logging.warning("Nem sikerült törölni a Soniox fájlt (%s).", file_id)


def convert_timestamp(value: Any, *, assume_ms: bool = False) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dict):
        seconds = value.get("seconds")
        nanos = value.get("nanos", 0)
        if seconds is None:
            return None
        return float(seconds) + float(nanos) / 1_000_000_000
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if assume_ms:
        return numeric / 1000.0
    return numeric


def extract_timestamp(token: dict, keys: Iterable[Tuple[str, bool]]) -> Optional[float]:
    for key, assume_ms in keys:
        if key in token and token[key] is not None:
            return convert_timestamp(token[key], assume_ms=assume_ms)
    return None


START_KEYS: Tuple[Tuple[str, bool], ...] = (
    ("start_time", False),
    ("start", False),
    ("start_seconds", False),
    ("start_time_s", False),
    ("start_time_ms", True),
    ("start_ms", True),
)
END_KEYS: Tuple[Tuple[str, bool], ...] = (
    ("end_time", False),
    ("end", False),
    ("end_seconds", False),
    ("end_time_s", False),
    ("end_time_ms", True),
    ("end_ms", True),
)
DURATION_KEYS: Tuple[Tuple[str, bool], ...] = (
    ("duration", False),
    ("duration_s", False),
    ("duration_ms", True),
)


def _collapse_whitespace(value: str) -> str:
    if not value:
        return ""
    return "".join(value.split())


def _build_word_entry_from_tokens(word_text: str, tokens: List[dict]) -> Optional[dict]:
    if not tokens:
        return None
    start = extract_timestamp(tokens[0], START_KEYS)
    end = extract_timestamp(tokens[-1], END_KEYS)
    if start is None:
        return None
    if end is None:
        duration = extract_timestamp(tokens[-1], DURATION_KEYS)
        end = start + duration if duration else start
    confidences: List[float] = []
    for token in tokens:
        confidence = token.get("confidence")
        try:
            if confidence is not None:
                confidences.append(float(confidence))
        except (TypeError, ValueError):
            continue
    score = round(sum(confidences) / len(confidences), 4) if confidences else None
    speaker = next((token.get("speaker") for token in tokens if token.get("speaker")), None)
    channel = next((token.get("channel") for token in tokens if token.get("channel")), None)
    return {
        "word": _sanitize_text(word_text),
        "start": round(start, 3),
        "end": round(end if end >= start else start, 3),
        "score": score,
        "speaker": speaker,
        "channel": channel,
    }


def _group_tokens_using_text(tokens: List[dict], transcript_text: str) -> List[dict]:
    if not transcript_text:
        return []
    words = [word for word in transcript_text.replace("\n", " ").split(" ") if word.strip()]
    if not words:
        return []
    grouped: List[dict] = []
    token_index = 0
    total_tokens = len(tokens)
    for raw_word in words:
        target = _collapse_whitespace(raw_word)
        if not target:
            continue
        accumulated_tokens: List[dict] = []
        accumulated_parts: List[str] = []
        while token_index < total_tokens:
            token = tokens[token_index]
            token_index += 1
            if not isinstance(token, dict):
                continue
            token_text = token.get("text")
            collapsed = _collapse_whitespace(token_text or "")
            if not collapsed:
                continue
            accumulated_tokens.append(token)
            accumulated_parts.append(collapsed)
            combined = "".join(accumulated_parts)
            if target.startswith(combined):
                if combined == target:
                    entry = _build_word_entry_from_tokens(raw_word, accumulated_tokens)
                    if entry:
                        grouped.append(entry)
                    break
                continue
            # Overshoot – revert last token and try to finish the word without it
            accumulated_tokens.pop()
            accumulated_parts.pop()
            token_index -= 1
            if "".join(accumulated_parts) == target and accumulated_tokens:
                entry = _build_word_entry_from_tokens(raw_word, accumulated_tokens)
                if entry:
                    grouped.append(entry)
                break
            logging.warning(
                "Soniox tokenek nem illeszthetők a '%s' szóhoz (target=%s, combined=%s).",
                raw_word,
                target,
                combined,
            )
            return []
        else:
            logging.warning("Elfogytak a Soniox tokenek a '%s' szó feldolgozásakor.", raw_word)
            return []
    return grouped


def _normalise_tokens_direct(tokens: List[dict]) -> List[dict]:
    normalised: List[dict] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        text = (token.get("text") or "").strip()
        if not text:
            continue
        start = extract_timestamp(token, START_KEYS)
        end = extract_timestamp(token, END_KEYS)
        if start is None:
            continue
        if end is None:
            duration = extract_timestamp(token, DURATION_KEYS)
            end = start + duration if duration else start
        confidence = token.get("confidence")
        try:
            confidence_val = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_val = None
        normalised.append(
            {
                "word": text,
                "start": round(start, 3),
                "end": round(end if end >= start else start, 3),
                "score": round(confidence_val, 4) if confidence_val is not None else None,
                "speaker": token.get("speaker"),
                "channel": token.get("channel"),
            }
        )
    normalised.sort(key=lambda item: item["start"])
    return normalised


def ensure_uploadable_audio(audio_path: Path) -> Tuple[Path, Optional[Path]]:
    size = audio_path.stat().st_size
    if size <= SONIOX_MAX_FILE_SIZE_BYTES:
        return audio_path, None
    logging.info(
        "A(z) %s fájl mérete %.2f MB, meghaladja a Soniox limitet (500 MB) – MP3 tömörítés indul.",
        audio_path.name,
        size / (1024 * 1024),
    )
    temp_dir = Path(tempfile.mkdtemp(prefix="soniox_compress_"))
    compressed_path = temp_dir / f"{audio_path.stem}_soniox.mp3"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "128k",
        str(compressed_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("Nem található az ffmpeg bináris; telepítsd az ffmpeg-et a nagy fájlok tömörítéséhez.") from exc
    except subprocess.CalledProcessError as exc:
        logging.error("FFmpeg hibát jelzett (exit=%s) – %s", exc.returncode, exc.stderr.decode(errors="ignore")[:4000])
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("Nem sikerült tömöríteni a hangfájlt Soniox feltöltéshez.") from exc
    if not compressed_path.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("Ismeretlen hiba: a tömörített fájl nem jött létre.")
    compressed_size = compressed_path.stat().st_size
    if compressed_size > SONIOX_MAX_FILE_SIZE_BYTES:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(
            f"A tömörített fájl ({compressed_size / (1024 * 1024):.2f} MB) még mindig meghaladja a Soniox limitet."
        )
    logging.info(
        "Tömörítés kész: %s → %.2f MB. A Soniox felé ez a fájl kerül feltöltésre.",
        compressed_path.name,
        compressed_size / (1024 * 1024),
    )
    return compressed_path, temp_dir


def normalise_tokens(tokens: List[dict], transcript_text: Optional[str] = None) -> List[dict]:
    if transcript_text:
        grouped = _group_tokens_using_text(tokens, transcript_text)
        if grouped:
            return grouped
        logging.warning("Nem sikerült a Soniox tokeneket a fő 'text' mező alapján csoportosítani – visszatérés token szintre.")
    return _normalise_tokens_direct(tokens)


def _sanitize_text(value: Any) -> str:
    if not value:
        return ""
    text = str(value)
    cleaned = text.replace('\\"', "").replace('"', "")
    return cleaned.strip()


def adjust_word_timestamps(word_segments: List[dict], padding_s: float) -> List[dict]:
    if not word_segments:
        return []
    adjusted = [word.copy() for word in word_segments]
    if padding_s <= 0:
        return adjusted
    for idx in range(len(adjusted) - 1):
        gap = adjusted[idx + 1]["start"] - adjusted[idx]["end"]
        if gap <= padding_s * 2:
            continue
        adjustment = min(padding_s, gap / 2.0)
        adjusted[idx]["end"] += adjustment
        adjusted[idx + 1]["start"] -= adjustment
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
    segment: dict = {
        "start": round(words[0]["start"], 3),
        "end": round(words[-1]["end"], 3),
        "text": text,
        "words": words,
    }
    speakers = {w.get("speaker") for w in words if w.get("speaker") is not None}
    if len(speakers) == 1:
        segment["speaker"] = next(iter(speakers))
    elif len(speakers) > 1:
        speaker_counts: Dict[Any, int] = {}
        for word in words:
            speaker = word.get("speaker")
            if speaker is None:
                continue
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        if speaker_counts:
            dominant_speaker = max(speaker_counts.items(), key=lambda item: item[1])[0]
            segment["speaker"] = dominant_speaker
            segment["mixed_speakers"] = True
    return segment


def sentence_segments_from_words(
    words: List[dict],
    max_pause_s: float,
    *,
    split_on_speaker_change: bool = True,
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
            and current[-1].get("speaker") != word.get("speaker")
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

    def pick_split_index(words: List[dict], punctuation: Tuple[str, ...]) -> Optional[int]:
        chosen: Optional[int] = None
        for idx in range(len(words) - 1):
            token = words[idx]["word"].strip()
            if not token:
                continue
            if token.endswith(punctuation):
                duration = words[idx]["end"] - words[0]["start"]
                if duration <= max_duration_s:
                    chosen = idx
        return chosen

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
        if split_idx is None:
            split_idx = len(current_words) // 2
        split_idx = max(0, min(split_idx, len(current_words) - 2))
        first_chunk = current_words[: split_idx + 1]
        second_chunk = current_words[split_idx + 1 :]
        if first_chunk:
            queue.append(first_chunk)
        if second_chunk:
            queue.append(second_chunk)
    return final_segments


def build_sentence_segments(word_segments: List[dict]) -> List[dict]:
    if not word_segments:
        return []
    adjusted = adjust_word_timestamps(word_segments, padding_s=DEFAULT_SENTENCE_PADDING_S)
    initial = sentence_segments_from_words(
        adjusted,
        max_pause_s=DEFAULT_SENTENCE_MAX_PAUSE_S,
    )
    return split_long_segments(initial, max_duration_s=DEFAULT_SENTENCE_MAX_SEGMENT_S)


def timestamp_to_iso(timestamp: Any) -> Optional[str]:
    if not timestamp:
        return None
    if isinstance(timestamp, str):
        return timestamp
    seconds = getattr(timestamp, "seconds", None)
    nanos = getattr(timestamp, "nanos", None)
    if seconds is None:
        return None
    nanos = nanos or 0
    dt = datetime.fromtimestamp(seconds + nanos / 1_000_000_000, tz=timezone.utc)
    return dt.isoformat()


def build_output_payload(
    *,
    word_segments: List[dict],
    segments: List[dict],
    diarize: bool,
    model: str,
    transcription_id: str,
    file_id: str,
    status_payload: Dict[str, Any],
) -> dict:
    metadata = {
        "transcription_id": transcription_id,
        "file_id": file_id,
        "status": status_payload.get("status"),
        "created": status_payload.get("created_time") or timestamp_to_iso(status_payload.get("created_timestamp")),
    }
    return {
        "word_segments": word_segments,
        "segments": segments,
        "provider": "soniox",
        "model": model,
        "diarization": diarize,
        "speaker_identification": False,
        "speaker_labels": [],
        "metadata": metadata,
    }


def process_audio_file(
    session: Session,
    endpoints: Dict[str, str],
    audio_path: Path,
    args: argparse.Namespace,
) -> Optional[dict]:
    reference_name = f"{args.reference_prefix}/{audio_path.stem}"
    upload_path, temp_dir = ensure_uploadable_audio(audio_path)
    file_id = upload_audio(session, endpoints, upload_path)
    transcription_id: Optional[str] = None
    try:
        payload = build_transcription_payload(file_id=file_id, args=args, reference_name=reference_name)
        logging.info("Soniox async beküldés: %s -> %s", audio_path.name, reference_name)
        transcription_id = create_transcription(session, endpoints, payload)
        status_payload = poll_transcription(
            session,
            endpoints,
            transcription_id,
            poll_interval=args.poll_interval,
            timeout_seconds=args.timeout,
        )
        transcript = download_transcript(session, endpoints, transcription_id)
        raw_dump_path = audio_path.with_suffix(".soniox_raw.txt")
        try:
            with raw_dump_path.open("w", encoding="utf-8") as raw_fp:
                json.dump(transcript, raw_fp, indent=2, ensure_ascii=False)
        except OSError as exc:
            logging.warning("Nem sikerült a Soniox raw transcript mentése (%s): %s", raw_dump_path.name, exc)
        tokens = transcript.get("tokens") or []
        transcript_text = transcript.get("text")
        word_segments = normalise_tokens(tokens, transcript_text=transcript_text)
        if not word_segments:
            logging.warning("Nem található szó szintű token a Soniox válaszban (%s).", audio_path.name)
            return None
        sentence_segments = build_sentence_segments(word_segments)
        return build_output_payload(
            word_segments=word_segments,
            segments=sentence_segments,
            diarize=args.diarize,
            model=args.model,
            transcription_id=transcription_id,
            file_id=file_id,
            status_payload=status_payload,
        )
    finally:
        if transcription_id:
            delete_transcription(session, endpoints, transcription_id)
        delete_file(session, endpoints, file_id)
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def process_directory(
    input_dir: Path,
    *,
    api_key: str,
    api_host: Optional[str],
    args: argparse.Namespace,
) -> None:
    audio_files = sorted(
        [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    )
    if not audio_files:
        print(f"Nem található támogatott hangfájl a megadott mappában: {input_dir}")
        return

    endpoints = build_endpoints(api_host)
    session = create_session(api_key)

    print(f"{len(audio_files)} hangfájl feldolgozása indul a Soniox REST API-val…")
    for audio_path in audio_files:
        print("-" * 48)
        print(f"▶  Feldolgozás: {audio_path.name}")
        output_path = audio_path.with_suffix(".json")
        if output_path.exists() and not args.overwrite:
            print(f"  ↷  Kihagyva (létező kimenet): {output_path.name}")
            continue
        try:
            payload = process_audio_file(session, endpoints, audio_path, args)
            if not payload:
                print(f"  ✖  Nem sikerült a Soniox transcript normalizálása: {audio_path.name}")
                continue
            with output_path.open("w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2, ensure_ascii=False)
            print(f"  ✔  Mentve: {output_path.name}")
        except Exception as exc:
            logging.exception("Soniox feldolgozási hiba: %s", audio_path.name)
            print(f"  ✖  Hiba: {exc}")


def validate_args(args: argparse.Namespace) -> None:
    if args.poll_interval <= 0:
        raise ConfigurationError("A lekérdezési gyakoriságnak pozitívnak kell lennie.")
    if args.timeout < 0:
        raise ConfigurationError("A timeout érték nem lehet negatív.")
    if args.chunk_size <= 0:
        raise ConfigurationError("A chunk méretének pozitívnak kell lennie (kompatibilitás kedvéért).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Soniox ASR – aszinkron hangtár feldolgozás normalizált szó szintű JSON kimenettel."
    )
    parser.add_argument(
        "-p",
        "--project-name",
        required=True,
        help="A projekt neve (a workdir alatti mappa), amit fel kell dolgozni.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Soniox modell azonosító (alapértelmezett: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=0,
        help="Becsült minimum beszélő szám (0 = automatikus).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=0,
        help="Becsült maximum beszélő szám (0 = automatikus).",
    )
    parser.add_argument(
        "--candidate-speaker",
        action="append",
        help="Ismert beszélő neve (többször is megadható).",
    )
    parser.add_argument(
        "--no-diarize",
        dest="diarize",
        action="store_false",
        default=True,
        help="Globális speaker diarizáció kikapcsolása.",
    )
    parser.add_argument(
        "--speaker-identification",
        dest="speaker_identification",
        action="store_true",
        default=False,
        help="Soniox speaker identification bekapcsolása (REST módban nem támogatott).",
    )
    parser.add_argument(
        "--api-key",
        help=f"Soniox API kulcs. Ha megadod, elmenti a keyholder.json fájlba. Környezeti változó: {ENV_API_KEY}.",
    )
    parser.add_argument(
        "--api-host",
        help="Soniox API host felülbírálása (pl. https://api.soniox.com).",
    )
    parser.add_argument(
        "--reference-prefix",
        default=DEFAULT_REFERENCE_PREFIX,
        help="Async reference név előtag (alapértelmezett: soniox).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Státusz lekérdezési intervallum másodpercben (alapértelmezett: {DEFAULT_POLL_INTERVAL}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Makszimális várakozási idő másodpercben (0 = végtelen, alap: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Kompatibilitási opció (REST módban nem használt). Alapértelmezett: {DEFAULT_CHUNK_SIZE}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Meglévő JSON fájl felülírása. Ha nincs beállítva, a meglévő fájlok kimaradnak.",
    )

    add_debug_argument(parser)
    args = parser.parse_args()

    log_level = configure_debug_mode(args.debug)
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    try:
        validate_args(args)
    except ConfigurationError as exc:
        print(f"Hiba: {exc}")
        sys.exit(1)

    config, project_root = load_config()
    api_key = resolve_api_key(args, project_root)
    if not api_key:
        print(
            "Hiba: Nem található Soniox API kulcs. Add meg az --api-key kapcsolóval vagy a SONIOX_API_KEY környezeti változóval."
        )
        sys.exit(1)

    input_dir = resolve_project_input(args.project_name, config, project_root)

    print("Projekt beállítások betöltve:")
    print(f"  - Projekt név:        {args.project_name}")
    print(f"  - Bemeneti mappa:     {input_dir}")
    print(f"  - Soniox modell:      {args.model}")
    print(f"  - Diarizáció:         {'BE' if args.diarize else 'KI'}")
    print(f"  - Reference prefix:   {args.reference_prefix}")

    process_directory(
        input_dir,
        api_key=api_key,
        api_host=args.api_host,
        args=args,
    )


if __name__ == "__main__":
    main()

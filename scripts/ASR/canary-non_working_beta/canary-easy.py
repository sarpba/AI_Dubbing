# ==============================================================================
#        NVIDIA CANARY ASR SCRIPT (EASY MODE – PROJECT AWARE PROCESSOR)
# ==============================================================================
#  🔊  Rövid és hosszú hangfájlok átírása a Canary modellel automatikus
#     nyelvfelismeréssel és darabolással.
# ------------------------------------------------------------------------------
#  ▸ Projekt-alapú működés: ugyanazt a config.json alapú mappakeresést használja,
#    mint a Parakeet szkript.
#  ▸ Auto-chunk: a hosszabb felvételeket fix hosszú darabokra osztja, így elkerülve
#    a Canary / Lhotse transcribe korlátait.
#  ▸ Kimenet: .json állomány proximitásban az audióval (átírt szöveg +
#    chunkonkénti metaadatok és opcionális alternatívák).
# ==============================================================================

import argparse
import json
import math
import sys
import warnings
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch
from copy import deepcopy

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

try:
    import librosa
    import soundfile as sf
    from nemo.collections.asr.models import ASRModel
    from omegaconf import DictConfig, open_dict
except ImportError as exc:
    print(f"Hiányzó ASR függőség: {exc}")
    print("Aktiváld a Canary‑képes conda/env környezetet (nemo_toolkit, torch, librosa, soundfile).")
    sys.exit(1)


def get_project_root() -> Path:
    """
    Felkeresi a projekt gyökerét a config.json alapján.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")


SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
DEFAULT_MODEL_NAME = "nvidia/canary-1b-v2"
DEFAULT_BATCH_SIZE = 4
DEFAULT_BEAM_SIZE = 5
DEFAULT_LEN_PENALTY = 1.0
DEFAULT_ALT_LIMIT = 2
DEFAULT_CHUNK_S = 30
MIN_CHUNK_S = 10
MAX_CHUNK_S = 120
SAMPLE_RATE = 16000
DEFAULT_MAX_PAUSE_S = 0.6
DEFAULT_PADDING_S = 0.2
DEFAULT_MAX_SEGMENT_S = 11.5


def load_config_and_get_paths(project_name: str) -> str:
    """Config.json alapú projekt feldolgozási útvonal kiválasztása."""
    try:
        project_root = get_project_root()
        config_path = project_root / "config.json"

        if not config_path.is_file():
            raise FileNotFoundError(f"Hiányzó config.json a projekt gyökerében: {project_root}")

        with open(config_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)

        workdir = project_root / config["DIRECTORIES"]["workdir"]
        input_subdir = config["PROJECT_SUBDIRS"]["separated_audio_speech"]
        processing_path = workdir / project_name / input_subdir

        if not processing_path.is_dir():
            raise FileNotFoundError(f"A feldolgozandó mappa nem létezik: {processing_path}")

        print("Projekt beállítások betöltve:")
        print(f"  - Projekt név:  {project_name}")
        print(f"  - Feldolgozandó mappa: {processing_path}")
        return str(processing_path)
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as exc:
        print(f"Hiba a konfiguráció betöltésekor: {exc}")
        sys.exit(1)


T = TypeVar("T")


def chunked(iterable: Sequence[T], size: int) -> Iterable[List[T]]:
    """Darabolja a listát fix méretű csoportokra."""
    if size <= 0:
        raise ValueError("A batch-méretnek pozitívnak kell lennie.")
    for idx in range(0, len(iterable), size):
        yield list(iterable[idx : idx + size])


def _to_serializable(obj: Any) -> Any:
    """Konvertálja az ismeretlen objektumokat JSON-barát struktúrára."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(key): _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "to_dict"):
        try:
            return _to_serializable(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _to_serializable(vars(obj))
        except Exception:
            pass
    return repr(obj)


def build_decoding_cfg(
    asr_model: ASRModel,
    beam_size: int,
    len_pen: float,
    need_full_hypotheses: bool,
) -> DictConfig:
    """Beam-search dekóder konfiguráció létrehozása a modell meglévő beállításainak megőrzésével."""
    base_cfg = getattr(asr_model, "cfg", None)
    decoding_cfg = getattr(base_cfg, "decoding", None) if base_cfg is not None else None
    decoding_cfg = deepcopy(decoding_cfg) if isinstance(decoding_cfg, DictConfig) else DictConfig({})

    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "beam"
        if "beam" not in decoding_cfg or decoding_cfg.beam is None:
            decoding_cfg.beam = DictConfig({})
        with open_dict(decoding_cfg.beam):
            decoding_cfg.beam.beam_size = max(1, beam_size)
            decoding_cfg.beam.len_pen = float(len_pen)
            decoding_cfg.beam.return_best_hypothesis = not need_full_hypotheses
        decoding_cfg.preserve_alignments = True
        decoding_cfg.compute_timestamps = True
        decoding_cfg.word_seperator = " "
        # Biztosítsuk, hogy a confidenciák is elérhetők legyenek szavanként.
        if "confidence_cfg" in decoding_cfg and decoding_cfg.confidence_cfg is not None:
            with open_dict(decoding_cfg.confidence_cfg):
                decoding_cfg.confidence_cfg.preserve_word_confidence = True
                decoding_cfg.confidence_cfg.preserve_token_confidence = True
        decoding_cfg.compute_langs = False
    return decoding_cfg


@dataclass
class TranscriptResult:
    best_text: str
    best_score: Optional[float]
    best_words: List[dict]
    best_segments: List[dict]
    alternatives: List[dict]


def _extract_hypothesis_fields(hyp) -> Tuple[str, Optional[float], List[dict], List[dict]]:
    """Egységesíti a Hypothesis objektumokat könnyen szerializálható dict-re."""
    text = ""
    score: Optional[float] = None
    alt_words: List[dict] = []
    segments: List[dict] = []

    if isinstance(hyp, str):
        text = hyp.strip()
    else:
        text = getattr(hyp, "text", "") or ""
        text = text.strip()
        raw_score = getattr(hyp, "score", None)
        if raw_score is not None and not isinstance(raw_score, (int, float)):
            try:
                raw_score = float(raw_score)
            except (TypeError, ValueError):
                raw_score = None
        score = float(raw_score) if raw_score is not None else None

        raw_words = getattr(hyp, "words", None)
        if raw_words:
            for word in raw_words:
                entry = {"word": getattr(word, "word", "").strip()}
                start = getattr(word, "start_time", None)
                end = getattr(word, "end_time", None)
                conf = getattr(word, "confidence", None)
                if start is not None and not math.isnan(start):
                    entry["start"] = round(float(start), 3)
                if end is not None and not math.isnan(end):
                    entry["end"] = round(float(end), 3)
                if conf is not None and not math.isnan(conf):
                    entry["confidence"] = float(conf)
                alt_words.append(entry)

        timestamps = getattr(hyp, "timestamp", None)
        if isinstance(timestamps, dict):
            word_ts = timestamps.get("word")
            if word_ts:
                alt_words = []
                for item in word_ts:
                    entry = {
                        "word": item.get("word", item.get("text", "")).strip(),
                    }
                    if "start" in item and item["start"] is not None:
                        entry["start"] = round(float(item["start"]), 3)
                    if "end" in item and item["end"] is not None:
                        entry["end"] = round(float(item["end"]), 3)
                    if "conf" in item and item["conf"] is not None:
                        entry["confidence"] = float(item["conf"])
                    elif "confidence" in item and item["confidence"] is not None:
                        entry["confidence"] = float(item["confidence"])
                    alt_words.append(entry)
            segment_ts = timestamps.get("segment")
            if segment_ts:
                for item in segment_ts:
                    entry = {
                        "start": round(float(item.get("start", 0.0)), 3),
                        "end": round(float(item.get("end", 0.0)), 3),
                        "text": item.get("segment") or item.get("text") or "",
                    }
                    segments.append(entry)
    return text, score, alt_words, segments


def select_best_transcript(
    hypotheses,
    *,
    alt_limit: int,
) -> TranscriptResult:
    """Kiválasztja a legjobb hipotézist a modell pontszámai alapján."""
    if not isinstance(hypotheses, (list, tuple)):
        hypotheses = [hypotheses]

    best_text = ""
    best_score_native: Optional[float] = None
    best_words: List[dict] = []
    best_segments: List[dict] = []
    collected_alts: List[dict] = []

    for hyp in hypotheses:
        text, native_score, words, segments = _extract_hypothesis_fields(hyp)
        if not text:
            continue

        score_for_ranking = native_score if native_score is not None else -float("inf")
        if best_text == "" or (native_score is not None and score_for_ranking > (best_score_native or -float("inf"))):
            best_text = text
            best_score_native = native_score
            best_words = words
            best_segments = segments

        alt_entry = {"text": text}
        if native_score is not None:
            alt_entry["model_score"] = native_score
        if words:
            alt_entry["words"] = words
        if segments:
            alt_entry["segments"] = segments
        collected_alts.append(alt_entry)

    alternatives = collected_alts[:alt_limit] if alt_limit > 0 else []
    return TranscriptResult(
        best_text=best_text,
        best_score=best_score_native,
        best_words=best_words,
        best_segments=best_segments,
        alternatives=alternatives,
    )


def prepare_chunks(audio: np.ndarray, chunk_len_s: int) -> List[Tuple[np.ndarray, float]]:
    """Felvétel darabolása fix hosszú (chunk_len_s) darabokra."""
    chunk_len_s = max(MIN_CHUNK_S, min(MAX_CHUNK_S, chunk_len_s))
    if audio.ndim > 1:
        # Librosa mono formátumot ad vissza, de biztos ami biztos.
        audio = np.mean(audio, axis=0)
    total_samples = len(audio)
    chunk_size = int(chunk_len_s * SAMPLE_RATE)
    chunks: List[Tuple[np.ndarray, float]] = []
    for start in range(0, total_samples, chunk_size):
        end = min(total_samples, start + chunk_size)
        segment = audio[start:end]
        if not segment.size:
            continue
        if np.max(np.abs(segment)) < 1e-4:
            # Teljesen csendes rész kihagyható.
            continue
        chunks.append((segment, start / SAMPLE_RATE))
    return chunks


def adjust_word_timestamps(word_segments: List[dict], padding_s: float) -> List[dict]:
    """Finomhangolja a szavak időbélyegeit kis mértékben, hogy jobban fedjék a beszédet."""
    if not word_segments or padding_s <= 0:
        return word_segments

    adjusted_segments = [word.copy() for word in word_segments]
    for idx in range(len(adjusted_segments) - 1):
        current = adjusted_segments[idx]
        nxt = adjusted_segments[idx + 1]
        if "start" not in nxt or "end" not in current:
            continue
        gap = nxt["start"] - current["end"]
        if gap > (padding_s * 2):
            adjustment = min(padding_s, gap / 2.0)
            current["end"] += adjustment
            nxt["start"] -= adjustment

    first = adjusted_segments[0]
    last = adjusted_segments[-1]
    if "start" in first:
        first["start"] = max(0.0, first["start"] - padding_s)
    if "end" in last:
        last["end"] += padding_s

    for word in adjusted_segments:
        if "start" in word:
            word["start"] = round(word["start"], 3)
        if "end" in word:
            word["end"] = round(word["end"], 3)
    return adjusted_segments


def _create_segment_from_words(words: List[dict]) -> Optional[dict]:
    if not words:
        return None
    text = " ".join(w.get("word", "") for w in words).strip()
    text = text.replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return {
        "start": words[0].get("start", 0.0),
        "end": words[-1].get("end", 0.0),
        "text": text,
        "words": words,
    }


def sentence_segments_from_words(words: List[dict], max_pause_s: float) -> List[dict]:
    """Szólistát mondatokra bont a szünetek és írásjelek alapján."""
    if not words:
        return []
    segments: List[dict] = []
    current: List[dict] = []

    for word in words:
        if current:
            prev_end = current[-1].get("end")
            start = word.get("start")
            if prev_end is not None and start is not None and (start - prev_end) > max_pause_s:
                if new_seg := _create_segment_from_words(current):
                    segments.append(new_seg)
                current = []

        current.append(word)

        token = word.get("word", "").strip()
        if token and (token in {".", "!", "?"} or token[-1] in {".", "!", "?"}):
            if new_seg := _create_segment_from_words(current):
                segments.append(new_seg)
            current = []

    if new_seg := _create_segment_from_words(current):
        segments.append(new_seg)
    return segments


def split_long_segments(segments: List[dict], max_duration_s: float) -> List[dict]:
    """Felhasítja a túl hosszú mondatszegmenseket."""
    if max_duration_s <= 0:
        return segments

    final_segments: List[dict] = []
    for segment in segments:
        words_to_process = segment.get("words") or []
        if not words_to_process:
            final_segments.append(segment)
            continue

        remaining = list(words_to_process)
        while remaining:
            first_start = remaining[0].get("start", 0.0)
            last_end = remaining[-1].get("end", first_start)
            duration = last_end - first_start
            if duration <= max_duration_s:
                if new_seg := _create_segment_from_words(remaining):
                    final_segments.append(new_seg)
                break

            candidate_words = [
                w for w in remaining if w.get("end", first_start) - first_start <= max_duration_s
            ]
            if not candidate_words:
                candidate_words = remaining[:1]

            best_split_idx = len(candidate_words) - 1
            if len(candidate_words) > 1:
                max_gap = -1.0
                for idx in range(len(candidate_words) - 1):
                    cur_end = candidate_words[idx].get("end")
                    nxt_start = candidate_words[idx + 1].get("start")
                    if cur_end is None or nxt_start is None:
                        continue
                    gap = nxt_start - cur_end
                    if gap >= max_gap:
                        max_gap = gap
                        best_split_idx = idx

            new_segment_words = remaining[: best_split_idx + 1]
            if new_seg := _create_segment_from_words(new_segment_words):
                final_segments.append(new_seg)
            remaining = remaining[best_split_idx + 1 :]
    return final_segments


def transcribe_audio_file(
    audio_path: Path,
    *,
    asr_model: ASRModel,
    batch_size: int,
    chunk_len_s: int,
    source_lang: Optional[str],
    target_lang: Optional[str],
    max_pause_s: float,
    padding_s: float,
    max_segment_s: float,
    alt_limit: int,
) -> Tuple[Optional[dict], str]:
    """Egyetlen hangfájl feldolgozása Canary modellel."""
    try:
        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        warnings.warn(f"Betöltési hiba '{audio_path.name}': {exc}")
        raw_dump_text = json.dumps(
            {
                "chunks": [],
                "errors": [
                    {
                        "chunk": "n/a",
                        "offset": "n/a",
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        return None, raw_dump_text

    chunks_with_offset = prepare_chunks(audio, chunk_len_s)
    if not chunks_with_offset:
        warnings.warn(f"Nincs feldolgozható rész: {audio_path.name}")
        return None, ""

    print(f"  - {audio_path.name}: {len(chunks_with_offset)} darab, chunk={chunk_len_s}s")

    from tempfile import TemporaryDirectory

    transcribe_kwargs = {
        "batch_size": 1,
        "verbose": False,
        "return_hypotheses": True,
        "timestamps": True,
    }
    if source_lang and source_lang.lower() not in {"", "auto"}:
        transcribe_kwargs["source_lang"] = source_lang
    if target_lang and target_lang.lower() not in {"", "auto"}:
        transcribe_kwargs["target_lang"] = target_lang

    all_words: List[dict] = []
    raw_structured_chunks: List[dict] = []
    raw_error_entries: List[dict] = []

    with TemporaryDirectory() as tmpdir:
        tmp_chunks: List[dict] = []
        for idx, (segment, offset) in enumerate(chunks_with_offset):
            tmp_path = Path(tmpdir) / f"chunk_{idx}.wav"
            try:
                sf.write(tmp_path, segment, SAMPLE_RATE)
            except Exception as exc:
                warnings.warn(f"Iratás sikertelen ideiglenes fájlba ({audio_path.name}, chunk {idx}): {exc}")
                continue
            tmp_chunks.append(
                {
                    "path": tmp_path,
                    "offset": offset,
                    "duration": len(segment) / SAMPLE_RATE,
                }
            )

        if not tmp_chunks:
            warnings.warn(f"Nem sikerült ideiglenes chunkokat létrehozni: {audio_path.name}")
            raw_dump_text = json.dumps({"chunks": [], "errors": raw_error_entries}, ensure_ascii=False, indent=2)
            return None, raw_dump_text

        for batch in chunked(tmp_chunks, transcribe_kwargs["batch_size"]):
            try:
                call_kwargs = dict(transcribe_kwargs)
                # A Canary modell stabilabb időbélyeges kimenetet ad egydarabos batch-sel.
                call_kwargs["batch_size"] = 1
                hypotheses_batch = asr_model.transcribe(
                    audio=[str(item["path"]) for item in batch],
                    **call_kwargs,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                warnings.warn("CUDA memória elfogyott. Csökkentsd a batch-méretet.")
                raw_error_entries.append(
                    {
                        "chunk": batch[0]["path"].name if batch else "n/a",
                        "offset": batch[0]["offset"] if batch else "n/a",
                        "type": "OutOfMemoryError",
                        "message": "CUDA memória elfogyott. Csökkentsd a batch-méretet.",
                    }
                )
                raw_dump_text = json.dumps({"chunks": raw_structured_chunks, "errors": raw_error_entries}, ensure_ascii=False, indent=2)
                return None, raw_dump_text
            except Exception as exc:
                warnings.warn(f"Transzkripciós hiba ({audio_path.name}): {exc}")
                raw_error_entries.append(
                    {
                        "chunk": batch[0]["path"].name if batch else "n/a",
                        "offset": batch[0]["offset"] if batch else "n/a",
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                continue

            if len(hypotheses_batch) != len(batch):
                warnings.warn("A Canary visszatérési hossza eltér a batch méretétől.")

            for item, hypotheses in zip(batch, hypotheses_batch):
                serialized_hyp = _to_serializable(hypotheses)
                raw_structured_chunks.append(
                    {
                        "chunk_index": tmp_chunks.index(item),
                        "offset": item["offset"],
                        "duration": item["duration"],
                        "hypotheses_raw": serialized_hyp,
                    }
                )
                try:
                    result = select_best_transcript(hypotheses, alt_limit=alt_limit)
                except Exception as exc:
                    raw_error_entries.append(
                        {
                            "chunk": Path(item["path"]).name,
                            "offset": item["offset"],
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    continue
                if not result.best_text:
                    continue
                # Adjust word timestamps with chunk offset.
                for word in result.best_words:
                    adjusted = dict(word)
                    start_val = adjusted.get("start")
                    end_val = adjusted.get("end")
                    if start_val is None or end_val is None:
                        continue
                    try:
                        start_f = float(start_val)
                        end_f = float(end_val)
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(start_f) or math.isnan(end_f):
                        continue
                    adjusted["start"] = round(start_f + item["offset"], 3)
                    adjusted["end"] = round(end_f + item["offset"], 3)
                    all_words.append(adjusted)

    raw_dump_text = json.dumps({"chunks": raw_structured_chunks, "errors": raw_error_entries}, ensure_ascii=False, indent=2)

    if not all_words:
        warnings.warn(f"Üres vagy időbélyeg nélküli eredmény: {audio_path.name}")
        return None, raw_dump_text

    all_words.sort(key=lambda w: w.get("start", 0.0))
    processed_words = adjust_word_timestamps(all_words, padding_s=padding_s)
    # Re-sort after padding adjustments.
    processed_words.sort(key=lambda w: w.get("start", 0.0))
    initial_segments = sentence_segments_from_words(processed_words, max_pause_s=max_pause_s)
    final_segments = split_long_segments(initial_segments, max_duration_s=max_segment_s)

    return {
        "word_segments": processed_words,
        "segments": final_segments,
    }, raw_dump_text


def transcribe_directory(
    audio_dir: str,
    *,
    asr_model: ASRModel,
    batch_size: int,
    chunk_len_s: int,
    source_lang: Optional[str],
    target_lang: Optional[str],
    max_pause_s: float,
    padding_s: float,
    max_segment_s: float,
    alt_limit: int,
    overwrite: bool,
) -> None:
    """Hangfájlok átiratait elkészíti és JSON-ba menti."""
    audio_paths = [
        p for p in sorted(Path(audio_dir).iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not audio_paths:
        print(f"Nincs feldolgozható hangfájl ebben a mappában: {audio_dir}")
        return

    print(f"\n{len(audio_paths)} db fájl feldolgozása a Canary modellel…")

    for audio_path in audio_paths:
        output_path = audio_path.with_suffix(".json")
        if output_path.exists() and not overwrite:
            print(f"  ↷ Kihagyva (létezik): {output_path.name}")
            continue

        result, raw_dump = transcribe_audio_file(
            audio_path,
            asr_model=asr_model,
            batch_size=batch_size,
            chunk_len_s=chunk_len_s,
            source_lang=source_lang,
            target_lang=target_lang,
            max_pause_s=max_pause_s,
            padding_s=padding_s,
            max_segment_s=max_segment_s,
            alt_limit=alt_limit,
        )

        raw_output_path = audio_path.with_suffix(".raw.txt")
        try:
            with open(raw_output_path, "w", encoding="utf-8") as raw_fp:
                raw_fp.write(raw_dump or "")
        except Exception as exc:
            warnings.warn(f"Nem sikerült menteni a nyers kimenetet ({raw_output_path.name}): {exc}")

        if not result:
            print(f"  ✖ Sikertelen vagy üres transzkripció: {audio_path.name}")
            continue

        language_hint = target_lang or source_lang or "auto"
        if isinstance(language_hint, str):
            language_hint = language_hint.strip() or "auto"
        else:
            language_hint = "auto"

        payload = {
            "segments": result.get("segments", []),
            "word_segments": result.get("word_segments", []),
            "language": language_hint,
        }

        try:
            with open(output_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=4, ensure_ascii=False)
            print(f"  ✔ Mentve: {output_path.name}")
        except Exception as exc:
            warnings.warn(f"Nem sikerült menteni a(z) {output_path.name} fájlt: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canary ASR alapú hangfájl-átírás (projekt-alapú, chunkolt feldolgozás)."
    )
    parser.add_argument("-p", "--project-name", required=True, help="Feldolgozandó projekt neve a workdir alatt.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="A használni kívánt Canary modell neve.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch méret a transzkripcióhoz.")
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE, help="Beam-search szélessége.")
    parser.add_argument("--len-pen", type=float, default=DEFAULT_LEN_PENALTY, help="Hossz büntetés a dekóderben.")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK_S, help="Chunk hossza másodpercben (10-120).")
    parser.add_argument("--max-pause", type=float, default=DEFAULT_MAX_PAUSE_S, help="Mondatok közti maximális szünet (mp, alapértelmezés: 0.6).")
    parser.add_argument("--timestamp-padding", type=float, default=DEFAULT_PADDING_S, help="Szavak időbélyegének finomhangolása (mp, alapértelmezés: 0.2).")
    parser.add_argument("--max-segment-duration", type=float, default=DEFAULT_MAX_SEGMENT_S, help="Szegmensek maximális hossza (mp, 0 = kikapcsolva, alapértelmezés: 11.5).")
    parser.add_argument("--source-lang", type=str, default="auto", help="Forrásnyelv (auto = automatikus detektálás).")
    parser.add_argument("--target-lang", type=str, default=None, help="Célnyelv (None = csak átírás).")
    parser.add_argument("--keep-alternatives", type=int, default=DEFAULT_ALT_LIMIT, help="Hány alternatív hipotézist mentsen el chunkonként.")
    parser.add_argument("--overwrite", action="store_true", help="Meglévő kimenetek felülírása.")
    add_debug_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_debug_mode(args.debug)

    processing_dir = load_config_and_get_paths(args.project_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nModell betöltése: {args.model_name} ({device})")

    need_full_hypotheses = False  # Csak a legjobb hipotézist kérjük a stabil időbélyegekhez.
    try:
        asr_model = ASRModel.from_pretrained(model_name=args.model_name)
        decoding_cfg = build_decoding_cfg(asr_model, args.beam_size, args.len_pen, need_full_hypotheses)
        asr_model.change_decoding_strategy(decoding_cfg)
        asr_model.to(device)
        asr_model.eval()
        print("  ✔ Canary modell betöltve.")
    except Exception as exc:
        print(f"Nem sikerült betölteni a Canary modellt: {exc}")
        sys.exit(1)

    transcribe_directory(
        processing_dir,
        asr_model=asr_model,
        batch_size=max(1, args.batch_size),
        chunk_len_s=max(MIN_CHUNK_S, min(MAX_CHUNK_S, args.chunk)),
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_pause_s=max(0.0, args.max_pause),
        padding_s=max(0.0, args.timestamp_padding),
        max_segment_s=max(0.0, args.max_segment_duration),
        alt_limit=max(0, args.keep_alternatives),
        overwrite=args.overwrite,
    )
    print("\nKész.")


if __name__ == "__main__":
    main()

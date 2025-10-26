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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch

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
    from omegaconf import DictConfig
except ImportError as exc:
    print(f"Hiányzó ASR függőség: {exc}")
    print("Aktiváld a Canary‑képes conda/env környezetet (nemo_toolkit, torch, librosa, soundfile).")
    sys.exit(1)

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


def load_config_and_get_paths(project_name: str) -> str:
    """Config.json alapú projekt feldolgozási útvonal kiválasztása."""
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
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


def build_decoding_cfg(beam_size: int, len_pen: float, need_full_hypotheses: bool) -> DictConfig:
    """Beam-search dekóder konfiguráció létrehozása."""
    beam_cfg = {
        "strategy": "beam",
        "beam": {
            "beam_size": max(1, beam_size),
            "len_pen": len_pen,
            "return_best_hypothesis": not need_full_hypotheses,
        },
    }
    return DictConfig(beam_cfg)


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


def transcribe_audio_file(
    audio_path: Path,
    *,
    asr_model: ASRModel,
    batch_size: int,
    chunk_len_s: int,
    source_lang: Optional[str],
    target_lang: Optional[str],
    alt_limit: int,
) -> Optional[dict]:
    """Egyetlen hangfájl feldolgozása Canary modellel."""
    try:
        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        warnings.warn(f"Betöltési hiba '{audio_path.name}': {exc}")
        return None

    chunks_with_offset = prepare_chunks(audio, chunk_len_s)
    if not chunks_with_offset:
        warnings.warn(f"Nincs feldolgozható rész: {audio_path.name}")
        return None

    print(f"  - {audio_path.name}: {len(chunks_with_offset)} darab, chunk={chunk_len_s}s")

    from tempfile import TemporaryDirectory

    transcribe_kwargs = {
        "batch_size": min(batch_size, max(1, len(chunks_with_offset))),
        "verbose": False,
        "return_hypotheses": True,
        "timestamps": True,
    }
    if source_lang and source_lang.lower() not in {"", "auto"}:
        transcribe_kwargs["source_lang"] = source_lang
    if target_lang and target_lang.lower() not in {"", "auto"}:
        transcribe_kwargs["target_lang"] = target_lang

    chunk_payloads: List[dict] = []
    texts: List[str] = []
    all_words: List[dict] = []
    all_segments: List[dict] = []

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
            return None

        for batch in chunked(tmp_chunks, transcribe_kwargs["batch_size"]):
            try:
                call_kwargs = dict(transcribe_kwargs)
                call_kwargs["batch_size"] = max(1, min(len(batch), call_kwargs["batch_size"]))
                hypotheses_batch = asr_model.transcribe(
                    audio=[str(item["path"]) for item in batch],
                    **call_kwargs,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                warnings.warn("CUDA memória elfogyott. Csökkentsd a batch-méretet.")
                return None
            except Exception as exc:
                warnings.warn(f"Transzkripciós hiba ({audio_path.name}): {exc}")
                continue

            if len(hypotheses_batch) != len(batch):
                warnings.warn("A Canary visszatérési hossza eltér a batch méretétől.")

            for item, hypotheses in zip(batch, hypotheses_batch):
                result = select_best_transcript(hypotheses, alt_limit=alt_limit)
                if not result.best_text:
                    continue
                texts.append(result.best_text)
                chunk_payload = {
                    "offset_start": round(item["offset"], 3),
                    "offset_end": round(item["offset"] + item["duration"], 3),
                    "transcript": result.best_text,
                }
                # Adjust word timestamps with chunk offset.
                chunk_words: List[dict] = []
                for word in result.best_words:
                    adjusted = dict(word)
                    if "start" in adjusted:
                        adjusted["start"] = round(adjusted["start"] + item["offset"], 3)
                    if "end" in adjusted:
                        adjusted["end"] = round(adjusted["end"] + item["offset"], 3)
                    chunk_words.append(adjusted)
                if chunk_words:
                    chunk_payload["words"] = chunk_words
                    all_words.extend(chunk_words)

                chunk_segments: List[dict] = []
                for segment in result.best_segments:
                    adjusted_seg = dict(segment)
                    adjusted_seg["start"] = round(segment.get("start", 0.0) + item["offset"], 3)
                    adjusted_seg["end"] = round(segment.get("end", 0.0) + item["offset"], 3)
                    chunk_segments.append(adjusted_seg)
                if chunk_segments:
                    chunk_payload["segments"] = chunk_segments
                    all_segments.extend(chunk_segments)

                if result.alternatives:
                    adjusted_alts: List[dict] = []
                    for alt in result.alternatives:
                        new_alt = dict(alt)
                        alt_words = new_alt.get("words")
                        if isinstance(alt_words, list):
                            adjusted_words = []
                            for word in alt_words:
                                adjusted_word = dict(word)
                                if "start" in adjusted_word:
                                    adjusted_word["start"] = round(adjusted_word["start"] + item["offset"], 3)
                                if "end" in adjusted_word:
                                    adjusted_word["end"] = round(adjusted_word["end"] + item["offset"], 3)
                                adjusted_words.append(adjusted_word)
                            new_alt["words"] = adjusted_words
                        alt_segments = new_alt.get("segments")
                        if isinstance(alt_segments, list):
                            adjusted_segments = []
                            for segment in alt_segments:
                                adjusted_segment = dict(segment)
                                adjusted_segment["start"] = round(segment.get("start", 0.0) + item["offset"], 3)
                                adjusted_segment["end"] = round(segment.get("end", 0.0) + item["offset"], 3)
                                adjusted_segments.append(adjusted_segment)
                            new_alt["segments"] = adjusted_segments
                        adjusted_alts.append(new_alt)
                    chunk_payload["alternatives"] = adjusted_alts
                chunk_payloads.append(chunk_payload)

    if not texts:
        warnings.warn(f"Üres eredmény: {audio_path.name}")
        return None

    all_words.sort(key=lambda w: w.get("start", 0.0))
    all_segments.sort(key=lambda s: s.get("start", 0.0))

    combined_text = " ".join(texts)
    return {
        "transcript": combined_text.strip(),
        "chunks": chunk_payloads,
        "word_segments": all_words,
        "segments": all_segments,
    }


def transcribe_directory(
    audio_dir: str,
    *,
    asr_model: ASRModel,
    batch_size: int,
    chunk_len_s: int,
    source_lang: Optional[str],
    target_lang: Optional[str],
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

        result = transcribe_audio_file(
            audio_path,
            asr_model=asr_model,
            batch_size=batch_size,
            chunk_len_s=chunk_len_s,
            source_lang=source_lang,
            target_lang=target_lang,
            alt_limit=alt_limit,
        )

        if not result:
            print(f"  ✖ Sikertelen vagy üres transzkripció: {audio_path.name}")
            continue

        payload = {
            "transcript": result["transcript"],
            "language": target_lang or source_lang or "auto",
            "model": asr_model.cfg.get("init_params", {}).get("model_name", DEFAULT_MODEL_NAME),
            "chunk_length": chunk_len_s,
            "chunks": result.get("chunks", []),
        }
        if result.get("word_segments"):
            payload["word_segments"] = result["word_segments"]
        if result.get("segments"):
            payload["segments"] = result["segments"]

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

    need_full_hypotheses = True  # Alternatívák miatt szükség van a teljes listára.
    decoding_cfg = build_decoding_cfg(args.beam_size, args.len_pen, need_full_hypotheses)

    try:
        asr_model = ASRModel.from_pretrained(model_name=args.model_name)
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
        alt_limit=max(0, args.keep_alternatives),
        overwrite=args.overwrite,
    )
    print("\nKész.")


if __name__ == "__main__":
    main()

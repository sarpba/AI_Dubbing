import argparse
import datetime
import importlib.util
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp
import tqdm
from transformers.utils import logging as transformers_logging

# Ensure project root is importable so we can reuse shared tooling.
for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

try:
    import whisper
except ImportError:  # pragma: no cover
    whisper = None
try:
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None
try:
    from num2words import num2words
except ImportError:  # pragma: no cover
    num2words = None
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:  # pragma: no cover
    levenshtein_distance = None

try:
    from vibevoice.modular.lora_loading import load_lora_assets
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
except ImportError as exc:  # pragma: no cover - environment-specific dependency
    raise ImportError(
        "A 'vibevoice' környezet szükséges a script futtatásához. "
        "Aktiváld a megfelelő környezetet (például: `conda activate vibevoice`)."
    ) from exc

from tools.debug_utils import add_debug_argument, configure_debug_mode

logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """
    Keresi a projekt gyökerét a config.json alapján.
    """
    for candidate in Path(__file__).resolve().parents:
        config_candidate = candidate / "config.json"
        if config_candidate.is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")


PROJECT_ROOT = find_project_root()
DEFAULT_EQ_CONFIG_PATH = PROJECT_ROOT / "scripts" / "TTS" / "EQ.json"
NORMALIZER_TO_WHISPER_LANG = {"hun": "hu", "eng": "en"}
WHISPER_LANG_CODE_TO_NAME = {"hu": "hungarian", "en": "english"}


def extract_cuda_index(device: str) -> int:
    if not device:
        return 0
    if ":" in device:
        _, _, index_str = device.partition(":")
        try:
            return max(0, int(index_str))
        except ValueError:
            logger.warning("Érvénytelen CUDA eszköz megadás: %s -> 0-ra váltunk.", device)
    return 0


def set_random_seeds(seed: int, device: str) -> None:
    if seed <= 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in device:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def time_to_filename_str(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    td = datetime.timedelta(seconds=seconds)
    minutes, secs = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}-{minutes:02d}-{secs:02d}-{milliseconds:03d}"


def normalize_peak(audio: np.ndarray, target_peak: float) -> np.ndarray:
    if not 0.0 < target_peak <= 1.0:
        target_peak = 0.95
    current_peak = np.max(np.abs(audio))
    if current_peak == 0:
        return audio
    return audio * (target_peak / current_peak)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if target_rate <= 0 or source_rate <= 0 or target_rate == source_rate:
        return audio
    if audio.size == 0:
        return audio

    duration_seconds = audio.shape[0] / float(source_rate)
    if duration_seconds == 0.0:
        return audio
    target_length = max(1, int(round(duration_seconds * target_rate)))
    if target_length == audio.shape[0]:
        return audio

    source_times = np.linspace(0.0, duration_seconds, num=audio.shape[0], endpoint=False, dtype=np.float64)
    target_times = np.linspace(0.0, duration_seconds, num=target_length, endpoint=False, dtype=np.float64)
    output_dtype = audio.dtype if np.issubdtype(audio.dtype, np.floating) else np.float32
    audio_float = audio.astype(np.float64, copy=False)

    if audio.ndim == 1:
        resampled = np.interp(target_times, source_times, audio_float)
        return resampled.astype(output_dtype, copy=False)

    resampled = np.empty((target_length, audio.shape[1]), dtype=np.float64)
    for channel_idx in range(audio.shape[1]):
        resampled[:, channel_idx] = np.interp(target_times, source_times, audio_float[:, channel_idx])
    return resampled.astype(output_dtype, copy=False)


@dataclass
class PitchSummary:
    median_hz: float
    voiced_ratio: float


def _prepare_signal_for_pitch(signal: np.ndarray) -> np.ndarray:
    """Convert signal to mono, normalize amplitude, and remove DC offset."""
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    signal = signal.astype(np.float32, copy=False)
    if signal.size == 0:
        return signal
    signal = signal - np.mean(signal)
    max_abs = np.max(np.abs(signal))
    if max_abs > 0:
        signal = signal / max_abs
    return signal


def iterate_frames(signal: np.ndarray, frame_length: int, hop_length: int) -> Iterable[np.ndarray]:
    """Yield overlapping frames from the signal."""
    if frame_length <= 0 or hop_length <= 0 or signal.size == 0:
        return
    total = len(signal)
    if total < frame_length:
        return
    for offset in range(0, total - frame_length + 1, hop_length):
        yield signal[offset : offset + frame_length]


def detect_frame_pitch(
    frame: np.ndarray,
    sample_rate: int,
    min_frequency: float,
    max_frequency: float,
    min_autocorr: float,
) -> float:
    """Estimate dominant pitch in Hz for a single frame via autocorrelation."""
    if sample_rate <= 0:
        return 0.0
    window = np.hanning(len(frame))
    windowed = frame * window
    autocorr = np.correlate(windowed, windowed, mode="full")[len(frame) - 1 :]
    if autocorr[0] <= 1e-9:
        return 0.0
    autocorr = autocorr / autocorr[0]

    min_period = max(1, int(round(sample_rate / max_frequency)))
    max_period = max(min_period + 1, int(round(sample_rate / min_frequency)))
    if max_period >= len(autocorr):
        max_period = len(autocorr) - 1
    if max_period <= min_period:
        return 0.0

    segment = autocorr[min_period : max_period + 1]
    peak_index = int(np.argmax(segment))
    peak_value = segment[peak_index]
    if peak_value < min_autocorr:
        return 0.0

    lag = min_period + peak_index
    if lag <= 0:
        return 0.0
    return float(sample_rate / lag)


def summarize_pitch_signal(
    signal: np.ndarray,
    sample_rate: int,
    min_frequency: float,
    max_frequency: float,
    energy_threshold: float = 0.01,
    min_autocorr: float = 0.1,
    frame_duration: float = 0.05,
    hop_ratio: float = 0.25,
) -> PitchSummary:
    """Compute voiced ratio and median pitch for a signal."""
    if sample_rate <= 0 or signal.size == 0:
        return PitchSummary(median_hz=0.0, voiced_ratio=0.0)

    frame_length = max(1024, int(sample_rate * frame_duration))
    hop_length = max(1, int(frame_length * hop_ratio))

    total_frames = 0
    voiced_frames = 0
    voiced_pitches: List[float] = []

    for frame in iterate_frames(signal, frame_length, hop_length):
        total_frames += 1
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms < energy_threshold:
            continue
        pitch = detect_frame_pitch(
            frame=frame,
            sample_rate=sample_rate,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            min_autocorr=min_autocorr,
        )
        if pitch <= 0:
            continue
        voiced_frames += 1
        voiced_pitches.append(pitch)

    if voiced_pitches:
        pitches = np.array(voiced_pitches, dtype=np.float64)
        median_hz = float(np.median(pitches))
    else:
        median_hz = 0.0

    voiced_ratio = (voiced_frames / total_frames) if total_frames else 0.0
    return PitchSummary(median_hz=median_hz, voiced_ratio=voiced_ratio)


def compute_pitch_summary_from_signal(
    signal: np.ndarray,
    sample_rate: int,
    min_frequency: float,
    max_frequency: float,
) -> PitchSummary:
    prepared = _prepare_signal_for_pitch(signal)
    return summarize_pitch_signal(
        signal=prepared,
        sample_rate=sample_rate,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
    )


def compute_pitch_summary_from_file(
    path: str,
    min_frequency: float,
    max_frequency: float,
) -> PitchSummary:
    try:
        data, sample_rate = sf.read(path)
    except Exception:
        return PitchSummary(median_hz=0.0, voiced_ratio=0.0)
    return compute_pitch_summary_from_signal(
        signal=np.asarray(data),
        sample_rate=sample_rate,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
    )


def load_eq_curve_config(config_path: Optional[str]) -> Optional[Dict[str, object]]:
    if not config_path:
        return None

    eq_path = Path(config_path)
    if not eq_path.exists():
        logger.warning("EQ config path does not exist: %s. EQ kikapcsolva.", eq_path)
        return None

    try:
        with open(eq_path, "r", encoding="utf-8") as fh:
            eq_data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("EQ config betöltése sikertelen (%s): %s", eq_path, exc)
        return None

    points = eq_data.get("points")
    if not isinstance(points, list) or not points:
        logger.warning("Az EQ konfiguráció nem tartalmaz pontokat: %s", eq_path)
        return None

    parsed_points: List[Dict[str, float]] = []
    for raw_point in points:
        try:
            freq = float(raw_point["frequency_hz"])
            gain_db = float(raw_point.get("gain_db", 0.0))
        except (KeyError, TypeError, ValueError):
            logger.warning("Érvénytelen EQ pont: %s", raw_point)
            continue
        parsed_points.append({"frequency_hz": max(0.0, freq), "gain_db": gain_db})

    if not parsed_points:
        logger.warning("Nem maradt használható EQ pont a konfigurációból: %s", eq_path)
        return None

    parsed_points.sort(key=lambda item: item["frequency_hz"])
    global_gain_db = float(eq_data.get("global_gain_db", 0.0))
    logger.info(
        "EQ görbe betöltve (%s pont, globális gain %.2f dB) forrás: %s",
        len(parsed_points),
        global_gain_db,
        eq_path,
    )
    return {"points": parsed_points, "global_gain_db": global_gain_db, "source_path": str(eq_path)}


def apply_eq_curve_to_audio(
    audio: np.ndarray,
    sample_rate: int,
    eq_config: Optional[Dict[str, object]],
) -> np.ndarray:
    if eq_config is None:
        return audio
    points = eq_config.get("points")
    if not points:
        return audio
    if audio.size == 0:
        return audio

    working_audio = audio.astype(np.float32, copy=False)
    n_samples = working_audio.shape[0]
    if n_samples < 2:
        return working_audio

    nyquist = sample_rate / 2.0
    freqs = np.array([min(point["frequency_hz"], nyquist) for point in points], dtype=np.float32)
    gains_db = np.array([point["gain_db"] for point in points], dtype=np.float32)

    unique_freqs: List[float] = []
    unique_gains: List[float] = []
    for freq, gain in zip(freqs, gains_db):
        if unique_freqs and abs(freq - unique_freqs[-1]) < 1e-6:
            unique_gains[-1] = gain
            continue
        unique_freqs.append(freq)
        unique_gains.append(gain)

    if len(unique_freqs) == 1:
        unique_freqs.append(nyquist)
        unique_gains.append(unique_gains[0])

    freq_bins = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    interpolated_gain_db = np.interp(
        freq_bins,
        np.array(unique_freqs, dtype=np.float32),
        np.array(unique_gains, dtype=np.float32),
        left=unique_gains[0],
        right=unique_gains[-1],
    )

    total_gain_db = interpolated_gain_db + float(eq_config.get("global_gain_db", 0.0))
    gain_linear = np.power(10.0, total_gain_db / 20.0).astype(np.float32, copy=False)

    def _apply_single_channel(channel_audio: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(channel_audio)
        equalized_spectrum = spectrum * gain_linear
        equalized = np.fft.irfft(equalized_spectrum, n=n_samples)
        return equalized.astype(np.float32, copy=False)

    if working_audio.ndim == 1:
        equalized_audio = _apply_single_channel(working_audio)
    else:
        equalized_audio = np.empty_like(working_audio)
        for channel_idx in range(working_audio.shape[1]):
            equalized_audio[:, channel_idx] = _apply_single_channel(working_audio[:, channel_idx])

    equalized_audio = np.clip(equalized_audio, -1.0, 1.0)
    return equalized_audio.astype(np.float32, copy=False)


def normalize_text_for_comparison(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("ly", "j")
    text = re.sub(r"[.,?!-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def convert_numbers_to_words(text: str, lang: str) -> str:
    if num2words is None:
        return text

    def replace_match(match: re.Match[str]) -> str:
        try:
            return num2words(int(match.group(0)), lang=lang)
        except Exception:
            return match.group(0)

    return re.sub(r"\b\d+\b", replace_match, text)


def load_normalizer(normaliser_path: Path) -> Optional[Callable[[str], str]]:
    if not normaliser_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("normaliser", normaliser_path)
        if spec is None or spec.loader is None:
            logger.error("Failed to initialize normalizer loader from %s", normaliser_path)
            return None
        normaliser_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(normaliser_module)  # type: ignore[attr-defined]
        if hasattr(normaliser_module, "normalize"):
            return getattr(normaliser_module, "normalize")
        logger.warning("Normalizer at %s does not expose a 'normalize' function.", normaliser_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to load normalizer from %s: %s", normaliser_path, exc, exc_info=True)
    return None


def compute_tolerance(word_count: int, tolerance_factor: float, min_tolerance: int) -> int:
    calculated = math.ceil(word_count * tolerance_factor)
    return max(calculated, min_tolerance)


def increment_stat(stats, key: str, increment: int = 1) -> None:
    current = int(stats.get(key, 0))
    stats[key] = current + increment


def log_generation_attempt(
    worker_label: str,
    filename: str,
    attempt: int,
    max_attempts: int,
    normalized_original: str,
    normalized_transcribed: str,
    distance: int,
    tolerance: int,
    outcome: str,
) -> None:
    message = (
        f"\n--- Generálás Log ({worker_label} | Fájl: {filename}) ---\n"
        f"Generálási próbálkozás: {attempt}/{max_attempts}\n"
        f"Gen_text (normalizált): {normalized_original}\n"
        f"Whisperrel visszaolvasott (normalizált): {normalized_transcribed}\n"
        f"Távolság / Megengedett: {distance} / {tolerance}\n"
        f"{outcome}\n"
        "----------------------------------------------------------"
    )
    logger.info(message)


def save_failed_attempt(
    args: argparse.Namespace,
    filename_stem: str,
    attempt_num: int,
    distance: int,
    tolerance: int,
    temp_gen_path: str,
    original_text: str,
    transcribed_text: Optional[str],
    reference_audio_path: Optional[str],
    failure_type: str = "text",
    extra_metadata: Optional[Dict[str, object]] = None,
) -> None:
    try:
        debug_segment_dir = Path(args.failed_generations_dir) / filename_stem
        debug_segment_dir.mkdir(parents=True, exist_ok=True)

        info_json_path = debug_segment_dir / "info.json"
        info: Dict[str, object] = {}
        if info_json_path.exists():
            with open(info_json_path, "r", encoding="utf-8") as fh:
                info = json.load(fh)

        filename_suffix = f"_dist_{distance}"
        pitch_diff_value: Optional[float] = None
        if failure_type == "pitch" and extra_metadata:
            candidate = extra_metadata.get("pitch_diff_hz") or extra_metadata.get("pitch_diff")
            if isinstance(candidate, (int, float)) and math.isfinite(candidate):
                pitch_diff_value = float(candidate)
                filename_suffix = f"_pitch_{pitch_diff_value:.2f}Hz"
            else:
                filename_suffix = "_pitch_unknown"

        attempt_filename = f"{filename_stem}_attempt_{attempt_num}{filename_suffix}.wav"
        shutil.copy(temp_gen_path, debug_segment_dir / attempt_filename)

        if reference_audio_path and os.path.exists(reference_audio_path):
            reference_filename = "reference.wav"
            reference_target = debug_segment_dir / reference_filename
            if not reference_target.exists():
                shutil.copy(reference_audio_path, reference_target)
            info["reference_audio_filename"] = reference_filename

        failures = info.setdefault("failures", [])
        if isinstance(failures, list):
            failure_entry: Dict[str, object] = {
                "attempt": attempt_num,
                "distance": distance,
                "allowed_tolerance": tolerance,
                "original_text": original_text,
                "transcribed_text": transcribed_text or "",
                "saved_audio_filename": attempt_filename,
                "failure_type": failure_type,
            }
            if extra_metadata:
                failure_entry.update(extra_metadata)
            if pitch_diff_value is not None:
                failure_entry.setdefault("pitch_diff_hz", pitch_diff_value)
            failures.append(failure_entry)
        info["last_update"] = datetime.datetime.utcnow().isoformat()

        with open(info_json_path, "w", encoding="utf-8") as fh:
            json.dump(info, fh, ensure_ascii=False, indent=2)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.error("Failed to save debug information for %s: %s", filename_stem, exc, exc_info=True)


def create_transcriber(args: argparse.Namespace, device: str) -> Tuple[Optional[Callable[[str], str]], Optional[str]]:
    if args.seed != -1:
        return None, None

    if levenshtein_distance is None:
        logger.error("Verification requires the 'python-Levenshtein' package.")
        return None, None
    if num2words is None:
        logger.error("Verification requires the 'num2words' package.")
        return None, None

    whisper_language = NORMALIZER_TO_WHISPER_LANG.get(args.norm.lower())
    if not whisper_language:
        logger.error("No Whisper language mapping found for normalizer '%s'.", args.norm)
        return None, None

    is_hf_model = "/" in args.whisper_model
    if is_hf_model:
        if pipeline is None:
            logger.error(
                "transformers.pipeline is unavailable, cannot load Hugging Face ASR model '%s'.",
                args.whisper_model,
            )
            return None, None

        device_for_pipeline = device
        if device_for_pipeline.startswith("cuda"):
            # pipeline expects device index for cuda
            device_for_pipeline = int(device.split(":")[-1]) if ":" in device else device
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=args.whisper_model,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device=device_for_pipeline,
        )

        def run(audio_path: str) -> str:
            generate_kwargs = {
                "language": WHISPER_LANG_CODE_TO_NAME.get(whisper_language),
                "num_beams": args.beam_size,
            }
            output = transcriber(audio_path, generate_kwargs=generate_kwargs)
            if isinstance(output, dict):
                return output.get("text", "")
            return str(output)

        return run, whisper_language

    if whisper is None:
        logger.error("openai-whisper is unavailable, cannot load model '%s'.", args.whisper_model)
        return None, None

    model_name = args.whisper_model.replace("openai/", "")
    transcriber = whisper.load_model(model_name, device=device)

    def run(audio_path: str) -> str:
        result = transcriber.transcribe(
            audio_path,
            language=whisper_language,
            fp16=torch.cuda.is_available(),
            beam_size=args.beam_size,
        )
        return result.get("text", "")

    return run, whisper_language
def parse_arguments() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="VibeVoice alapú TTS a fordított sávok szegmensenkénti generálásához.",
    )
    parser.add_argument("project_name", type=str, help="A projekt könyvtár neve a 'workdir' mappán belül.")
    parser.add_argument(
        "--norm",
        type=str,
        required=True,
        help="Normalizálási profil azonosító (pl. 'hun', 'eng').",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-1.5b",
        help="A HuggingFace model path vagy lokális modell könyvtár.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Visszafelé kompatibilis alias a --model_path számára.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="LoRA/adapter könyvtár, amelyet a modellre szeretnénk betölteni.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Eszköz a futtatáshoz: cuda | mps | cpu.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.3,
        help="Classifier-Free Guidance skála a generáláshoz.",
    )
    parser.add_argument(
        "--disable_prefill",
        action="store_true",
        help="Ha megadjuk, kikapcsoljuk a voice prefill (klónozás) lépést.",
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=10,
        help="DDPM inference lépések száma.",
    )
    parser.add_argument(
        "--eq_config",
        type=str,
        default=None,
        help="EQ konfiguráció JSON elérési útja.",
    )
    parser.add_argument(
        "--normalize_ref_audio",
        action="store_true",
        help="Referencia audió normalizálása a megadott csúcsértékre.",
    )
    parser.add_argument(
        "--ref_audio_peak",
        type=float,
        default=0.95,
        help="Normalizált referencia audió cél csúcsértéke (0.0-1.0).",
    )
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        default=16000,
        help="Referencia audió újramintavételezés cél frekvenciája Hz-ben (0 vagy negatív érték esetén változatlan).",
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        default="Speaker 1",
        help="A szkript szövegéhez használt beszélő címkéje.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Véletlenszám-generátor magja (-1 esetén véletlen és Whisper ellenőrzés).",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximális generálási kísérletek száma a visszaellenőrzés során.",
    )
    parser.add_argument(
        "--enable_pitch_check",
        action="store_true",
        help="Pitch ellenőrzés bekapcsolása: a referencia és a generált szegmens közötti különbséget is vizsgálja.",
    )
    parser.add_argument(
        "--pitch_tolerance",
        type=float,
        default=20.0,
        help="Megengedett pitch eltérés (Hz) a referencia és a generált szegmens között.",
    )
    parser.add_argument(
        "--pitch_min_frequency",
        type=float,
        default=60.0,
        help="Pitch detektálás minimális frekvencia (Hz).",
    )
    parser.add_argument(
        "--pitch_max_frequency",
        type=float,
        default=400.0,
        help="Pitch detektálás maximális frekvencia (Hz).",
    )
    parser.add_argument(
        "--pitch_retry",
        type=int,
        default=3,
        help="Pitch eltérés miatti újrapróbálkozások száma (alap: 3, azaz összesen 4 kísérlet).",
    )
    parser.add_argument(
        "--tolerance_factor",
        type=float,
        default=1.0,
        help="Levenshtein tolerancia szorzó (szószám * faktor).",
    )
    parser.add_argument(
        "--min_tolerance",
        type=int,
        default=2,
        help="Szövegellenőrzés minimális toleranciája.",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="openai/whisper-large-v3",
        help="Whisper modell az ellenőrzéshez (HF azonosító vagy openai/ prefix).",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Whisper dekódolás nyalábszélessége.",
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=None,
        help="Opcionális limit a feldolgozandó szegmensek számára (debughoz).",
    )
    parser.add_argument(
        "--input_directory_override",
        action="store_true",
        help="A translated JSON mappa helyett a temp alkönyvtárból olvassa be a forrás fájlokat.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Létező kimeneti fájlok felülírása újragenerálás esetén.",
    )
    parser.add_argument(
        "--save_failures",
        action="store_true",
        help="Sikertelen generálási kísérletek mentése diagnosztikához.",
    )
    parser.add_argument(
        "--keep_best_over_tolerance",
        action="store_true",
        help="Megőrzi a legjobb próbálkozást akkor is, ha a toleranciát túllépi.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Párhuzamos GPU workerek maximális száma.",
    )
    add_debug_argument(parser)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    normalized = device.lower()
    if normalized in {"auto", ""}:
        return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if normalized == "mpx":
        logger.info("A 'mpx' eszköznév MPS-ként lesz kezelve.")
        normalized = "mps"
    if normalized == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS nem elérhető. CPU-ra váltunk.")
        return "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA nem elérhető. CPU-ra váltunk.")
        return "cpu"
    return normalized


def prepare_output_directories(args: argparse.Namespace, config: Dict[str, object]) -> None:
    cfg_dirs = config["DIRECTORIES"]
    cfg_subdirs = config["PROJECT_SUBDIRS"]
    full_project_path = (PROJECT_ROOT / cfg_dirs["workdir"]) / args.project_name

    translated_splits_dir = cfg_subdirs["translated_splits"]
    json_subdir = cfg_subdirs["translated"]
    if getattr(args, "input_directory_override", False):
        temp_dir = cfg_subdirs.get("temp")
        if temp_dir:
            json_subdir = temp_dir
            logger.info(
                "A translated JSON mappa helyett a temp (%s) kerül felhasználásra.",
                temp_dir,
            )
        else:
            logger.warning("A config nem tartalmaz 'temp' kulcsot. Marad a translated JSON mappa.")

    args.output_dir = str(full_project_path / translated_splits_dir)
    args.output_dir_noise = str(full_project_path / cfg_subdirs["noice_splits"])
    args.input_wav_dir = full_project_path / cfg_subdirs["separated_audio_speech"]
    args.input_json_dir = full_project_path / json_subdir

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir_noise).mkdir(parents=True, exist_ok=True)


def load_vibevoice_components(
    args: argparse.Namespace,
    device: str,
) -> Tuple[VibeVoiceProcessor, VibeVoiceForConditionalGenerationInference]:
    model_path = args.model_dir or args.model_path
    if not model_path:
        raise ValueError("A VibeVoice modell elérési útja nem lett megadva (--model_path).")

    transformers_logging.set_verbosity_info()

    logger.info("Processor betöltése: %s", model_path)
    processor = VibeVoiceProcessor.from_pretrained(model_path)

    device_lower = device.lower()
    if device_lower == "mps":
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"
    elif device_lower.startswith("cuda"):
        load_dtype = torch.bfloat16
        attn_impl_primary = "flash_attention_2"
    else:
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"

    logger.info(
        "Modell betöltése: %s | device=%s | dtype=%s | attn_impl=%s",
        model_path,
        device,
        load_dtype,
        attn_impl_primary,
    )

    try:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=load_dtype,
            attn_implementation=attn_impl_primary,
        )
        if device_lower.startswith("cuda") or device_lower == "mps":
            model.to(device)
    except Exception as exc:  # pragma: no cover - environment dependent fallback
        if attn_impl_primary == "flash_attention_2":
            logger.warning(
                "flash_attention_2 betöltése sikertelen (%s). SDPA-ra váltunk. Az audió minőség eltérhet.",
                exc,
            )
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                attn_implementation="sdpa",
            )
            if device_lower.startswith("cuda") or device_lower == "mps":
                model.to(device)
        else:
            raise

    if args.checkpoint_path:
        logger.info("LoRA/adapter assetek betöltése: %s", args.checkpoint_path)
        report = load_lora_assets(model, args.checkpoint_path)
        loaded_components = [
            name
            for name, loaded in (
                ("language LoRA", report.language_model),
                ("diffusion head LoRA", report.diffusion_head_lora),
                ("diffusion head weights", report.diffusion_head_full),
                ("acoustic connector", report.acoustic_connector),
                ("semantic connector", report.semantic_connector),
            )
            if loaded
        ]
        if loaded_components:
            logger.info("Betöltött komponensek: %s", ", ".join(loaded_components))
        else:
            logger.warning("Nem sikerült adapter komponenseket betölteni. Ellenőrizd az útvonalat.")
        if report.adapter_root is not None:
            logger.info("Adapter gyökér: %s", report.adapter_root)

    model.eval()
    model.set_ddpm_inference_steps(args.ddpm_steps)
    if hasattr(model.model, "language_model"):
        lm_config = getattr(model.model.language_model, "config", None)
        if lm_config is not None:
            logger.info("Language model attention implementation: %s", lm_config._attn_implementation)
    return processor, model


def prepare_inputs(
    processor: VibeVoiceProcessor,
    text: str,
    voice_sample_paths: List[str],
    device: str,
) -> Dict[str, torch.Tensor]:
    inputs = processor(
        text=[text],
        voice_samples=[voice_sample_paths],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(device)
    return inputs


def ensure_script_format(text: str, speaker_label: str) -> str:
    """
    A VibeVoice processzor legalább egy 'Speaker X:' formátumú sort vár.
    Ha a beérkező szöveg nem tartalmaz ilyet, automatikusan körbetekerjük.
    """
    stripped = text.strip()
    if not stripped:
        return stripped

    first_line = stripped.splitlines()[0]
    if ":" in first_line:
        return stripped
    return f"{speaker_label}: {stripped}"


def save_noise_segments(
    args: argparse.Namespace,
    audio_data: np.ndarray,
    sample_rate: int,
    segments: List[Dict[str, float]],
) -> None:
    noise_dir = Path(args.output_dir_noise)
    noise_dir.mkdir(parents=True, exist_ok=True)

    last_end_time = 0.0
    for segment in segments:
        start_time = segment.get("start")
        if start_time is not None and start_time > last_end_time:
            noise_output_path = noise_dir / f"{time_to_filename_str(last_end_time)}_{time_to_filename_str(start_time)}.wav"
            if not noise_output_path.exists():
                sf.write(
                    noise_output_path,
                    audio_data[int(last_end_time * sample_rate) : int(start_time * sample_rate)],
                    sample_rate,
                )
        last_end_time = segment.get("end", start_time)

    duration_seconds = len(audio_data) / sample_rate
    if duration_seconds > last_end_time:
        noise_output_path = noise_dir / f"{time_to_filename_str(last_end_time)}_{time_to_filename_str(duration_seconds)}.wav"
        if not noise_output_path.exists():
            sf.write(noise_output_path, audio_data[int(last_end_time * sample_rate) :], sample_rate)


def process_segment(
    segment: Dict[str, object],
    args: argparse.Namespace,
    audio_data: np.ndarray,
    sample_rate: int,
    processor: VibeVoiceProcessor,
    model: VibeVoiceForConditionalGenerationInference,
    eq_config: Optional[Dict[str, object]],
    transcribe_fn: Optional[Callable[[str], str]],
    whisper_language: Optional[str],
    normalize_fn: Optional[Callable[[str], str]],
    device: str,
    worker_label: str,
) -> Tuple[bool, Optional[str]]:
    start_time = segment.get("start")
    end_time = segment.get("end")
    original_gen_text = (segment.get("translated_text") or "").strip()

    if not all(isinstance(value, (int, float)) for value in (start_time, end_time)):
        return False, "Hiányzó vagy érvénytelen időbélyeg."
    if not original_gen_text:
        return False, "Üres generálandó szöveg."

    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    end_sample = min(end_sample, len(audio_data))
    if start_sample >= end_sample:
        return False, "Érvénytelen audió intervallum."

    filename = f"{time_to_filename_str(start_time)}_{time_to_filename_str(end_time)}.wav"
    output_path = Path(args.output_dir) / filename
    if output_path.exists() and not args.overwrite:
        logger.debug("Kihagyva (létező fájl): %s", output_path)
        return True, None

    gen_text = original_gen_text
    if normalize_fn:
        try:
            gen_text = normalize_fn(gen_text)
        except Exception as exc:
            logger.error("Szöveg normalizálása sikertelen '%s' szegmensnél: %s", filename, exc, exc_info=True)
            return False, f"Normalizálási hiba: {exc}"

    formatted_text = ensure_script_format(gen_text, args.speaker_name)
    if not formatted_text:
        return False, "Üres generálandó szöveg."

    ref_chunk = audio_data[start_sample:end_sample]
    ref_chunk = apply_eq_curve_to_audio(ref_chunk, sample_rate, eq_config)
    if args.normalize_ref_audio:
        ref_chunk = normalize_peak(ref_chunk.copy(), args.ref_audio_peak)
    ref_sample_rate = sample_rate
    if args.target_sample_rate > 0 and args.target_sample_rate != sample_rate:
        ref_chunk = resample_audio(ref_chunk, sample_rate, args.target_sample_rate)
        ref_sample_rate = args.target_sample_rate

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref_file:
        temp_ref_path = tmp_ref_file.name
    try:
        sf.write(temp_ref_path, ref_chunk, ref_sample_rate)

        inputs = prepare_inputs(
            processor=processor,
            text=formatted_text,
            voice_sample_paths=[temp_ref_path],
            device=device,
        )

        transcribe_enabled = transcribe_fn is not None
        normalized_original = normalize_text_for_comparison(gen_text) if transcribe_enabled else ""
        tolerance_word_count = len(gen_text.split()) if transcribe_enabled else 0
        best_distance = float("inf")
        best_pitch_diff = float("inf")
        best_temp_path: Optional[str] = None
        best_transcribed: str = ""
        last_error: Optional[str] = None
        filename_stem = Path(filename).stem

        reference_pitch_summary: Optional[PitchSummary] = None
        if args.enable_pitch_check:
            reference_pitch_summary = compute_pitch_summary_from_signal(
                signal=ref_chunk,
                sample_rate=ref_sample_rate,
                min_frequency=args.pitch_min_frequency,
                max_frequency=args.pitch_max_frequency,
            )
            if reference_pitch_summary.median_hz <= 0 or reference_pitch_summary.voiced_ratio <= 0:
                logger.warning(
                    "%s: Pitch ellenőrzés kihagyva (referencia szakasz némának tűnik): %s",
                    worker_label,
                    filename,
                )
                reference_pitch_summary = None

        text_limit = max(1, args.max_retries if transcribe_enabled else 1)
        pitch_limit = 1 + max(0, args.pitch_retry) if reference_pitch_summary is not None else 1
        max_attempts = max(text_limit, pitch_limit)
        text_failures = 0
        pitch_failures = 0

        for attempt in range(1, max_attempts + 1):
            temp_gen_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=args.output_dir)
            temp_gen_path = temp_gen_file.name
            temp_gen_file.close()
            keep_temp = False
            try:
                generation_start = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=args.cfg_scale,
                        tokenizer=processor.tokenizer,
                        generation_config={"do_sample": False},
                        verbose=False,
                        is_prefill=not args.disable_prefill,
                    )
                generation_time = time.time() - generation_start
                logger.debug("Generálás kész %.2f mp alatt: %s (attempt %s)", generation_time, filename, attempt)

                speech_outputs = getattr(outputs, "speech_outputs", None)
                if not speech_outputs or speech_outputs[0] is None:
                    raise RuntimeError("A modell nem adott vissza audiót.")

                processor.save_audio(speech_outputs[0], output_path=temp_gen_path)

                converted_transcribed = ""
                normalized_transcribed = ""
                final_tolerance = 0
                distance = 0
                text_pass = True

                if transcribe_enabled:
                    raw_transcribed = transcribe_fn(temp_gen_path)
                    converted_transcribed = convert_numbers_to_words(
                        raw_transcribed,
                        lang=whisper_language or "",
                    )
                    normalized_transcribed = normalize_text_for_comparison(converted_transcribed)
                    final_tolerance = compute_tolerance(
                        tolerance_word_count,
                        args.tolerance_factor,
                        args.min_tolerance,
                    )
                    distance = levenshtein_distance(normalized_original, normalized_transcribed)
                    text_pass = distance <= final_tolerance or distance <= 1
                    outcome = "Sikeres ellenőrzés" if text_pass else "Ellenőrzés sikertelen"
                    log_generation_attempt(
                        worker_label,
                        filename,
                        attempt,
                        max_attempts,
                        normalized_original,
                        normalized_transcribed,
                        distance,
                        final_tolerance,
                        outcome,
                    )
                    if not text_pass:
                        text_failures += 1

                pitch_pass = True
                pitch_diff = 0.0
                gen_pitch_summary: Optional[PitchSummary] = None
                if reference_pitch_summary is not None:
                    gen_pitch_summary = compute_pitch_summary_from_file(
                        temp_gen_path,
                        min_frequency=args.pitch_min_frequency,
                        max_frequency=args.pitch_max_frequency,
                    )
                    if gen_pitch_summary.voiced_ratio <= 0:
                        pitch_pass = False
                        pitch_diff = float("inf")
                        logger.warning(
                            "%s: Pitch ellenőrzés sikertelen (némának tűnik a generált szegmens) -> %s (attempt %s/%s)",
                            worker_label,
                            filename,
                            attempt,
                            max_attempts,
                        )
                    else:
                        pitch_diff = abs(gen_pitch_summary.median_hz - reference_pitch_summary.median_hz)
                        pitch_pass = pitch_diff <= args.pitch_tolerance
                        status = "OK" if pitch_pass else "KINT"
                        logger.info(
                            "%s: Pitch ellenőrzés [%s] %s (attempt %s/%s) | ref: %.2f Hz | gen: %.2f Hz | diff: %.2f Hz | voiced: ref %.0f%% / gen %.0f%%",
                            worker_label,
                            status,
                            filename,
                            attempt,
                            max_attempts,
                            reference_pitch_summary.median_hz,
                            gen_pitch_summary.median_hz,
                            pitch_diff,
                            reference_pitch_summary.voiced_ratio * 100,
                            gen_pitch_summary.voiced_ratio * 100,
                        )
                    if not pitch_pass:
                        pitch_failures += 1

                if text_pass and pitch_pass:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temp_gen_path, output_path)
                    if best_temp_path and os.path.exists(best_temp_path):
                        os.remove(best_temp_path)
                    return True, None

                attempt_reasons: List[str] = []
                if not text_pass and transcribe_enabled:
                    last_error = f"Levenshtein távolság {distance}, tolerancia {final_tolerance}"
                    attempt_reasons.append(last_error)
                    if args.save_failures:
                        save_failed_attempt(
                            args=args,
                            filename_stem=filename_stem,
                            attempt_num=attempt,
                            distance=distance,
                            tolerance=final_tolerance,
                            temp_gen_path=temp_gen_path,
                            original_text=gen_text,
                            transcribed_text=converted_transcribed,
                            reference_audio_path=temp_ref_path,
                            failure_type="text",
                            extra_metadata={"normalized_transcribed": normalized_transcribed},
                        )

                if not pitch_pass and reference_pitch_summary is not None:
                    if math.isfinite(pitch_diff):
                        last_error = (
                            f"Pitch eltérés {pitch_diff:.2f} Hz meghaladja a toleranciát ({args.pitch_tolerance:.2f} Hz)"
                        )
                        attempt_reasons.append(last_error)
                    else:
                        last_error = "Pitch ellenőrzés sikertelen: nincs kimutatható hangmagasság."
                        attempt_reasons.append(last_error)
                    if args.save_failures:
                        diff_value = pitch_diff if math.isfinite(pitch_diff) else -1.0
                        extra_meta = {
                            "pitch_diff_hz": pitch_diff if math.isfinite(pitch_diff) else None,
                            "pitch_tolerance": args.pitch_tolerance,
                            "reference_pitch": reference_pitch_summary.median_hz,
                            "generated_pitch": gen_pitch_summary.median_hz if gen_pitch_summary else None,
                            "generated_voiced_ratio": gen_pitch_summary.voiced_ratio if gen_pitch_summary else None,
                        }
                        save_failed_attempt(
                            args=args,
                            filename_stem=filename_stem,
                            attempt_num=attempt,
                            distance=int(round(diff_value)) if diff_value >= 0 else -1,
                            tolerance=int(round(args.pitch_tolerance)),
                            temp_gen_path=temp_gen_path,
                            original_text=gen_text,
                            transcribed_text=converted_transcribed,
                            reference_audio_path=temp_ref_path,
                            failure_type="pitch",
                            extra_metadata=extra_meta,
                        )

                should_store_best = False
                if args.keep_best_over_tolerance:
                    if transcribe_enabled and not text_pass and distance < best_distance:
                        should_store_best = True
                    elif reference_pitch_summary is not None and not pitch_pass and math.isfinite(pitch_diff):
                        if pitch_diff < best_pitch_diff:
                            should_store_best = True
                    elif not transcribe_enabled and reference_pitch_summary is None and best_temp_path is None:
                        should_store_best = True

                if should_store_best:
                    if best_temp_path and os.path.exists(best_temp_path):
                        os.remove(best_temp_path)
                    best_temp_path = temp_gen_path
                    if transcribe_enabled:
                        best_distance = distance
                        best_transcribed = converted_transcribed
                    if reference_pitch_summary is not None and math.isfinite(pitch_diff):
                        best_pitch_diff = pitch_diff
                    keep_temp = True
                else:
                    keep_temp = False

                if not attempt_reasons and not last_error:
                    last_error = "Generálás ellenőrzési feltételek miatt sikertelen."
                elif attempt_reasons:
                    last_error = " | ".join(attempt_reasons)

            except Exception as exc:
                last_error = f"Generálási hiba: {exc}"
                logger.error("Hiba a(z) %s szegmens generálása közben: %s", filename, exc, exc_info=True)
            finally:
                if not keep_temp and os.path.exists(temp_gen_path):
                    os.remove(temp_gen_path)

            if (transcribe_enabled and text_failures >= text_limit) or (
                reference_pitch_summary is not None and pitch_failures >= pitch_limit
            ):
                logger.debug(
                    "%s: Ellenőrzési limit elérve (%s szöveg, %s pitch) -> %s",
                    worker_label,
                    text_failures,
                    pitch_failures,
                    filename,
                )
                break

        final_tolerance = (
            compute_tolerance(
                tolerance_word_count,
                args.tolerance_factor,
                args.min_tolerance,
            )
            if transcribe_enabled
            else 0
        )
        is_best_available = best_temp_path and os.path.exists(best_temp_path)
        accept_outside_tolerance = (
            args.keep_best_over_tolerance
            and is_best_available
            and best_distance < float("inf")
        )
        pitch_condition = True
        if reference_pitch_summary is not None:
            pitch_condition = best_pitch_diff <= args.pitch_tolerance
        if is_best_available and pitch_condition and (
            not transcribe_enabled or best_distance <= final_tolerance or accept_outside_tolerance
        ):
            if transcribe_enabled and best_distance <= final_tolerance:
                logger.info(
                    "%s: A legjobb próbálkozás elfogadva (distance: %s, tolerance: %s) -> %s",
                    worker_label,
                    best_distance,
                    final_tolerance,
                    filename,
                )
            elif transcribe_enabled and accept_outside_tolerance:
                logger.warning(
                    "%s: A legjobb próbálkozás tolerancia felett is elfogadva (distance: %s, tolerance: %s) -> %s",
                    worker_label,
                    best_distance,
                    final_tolerance,
                    filename,
                )
            else:
                logger.info(
                    "%s: Ellenőrzött kimenet elfogadva pitch tolerancián belül -> %s",
                    worker_label,
                    filename,
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(best_temp_path, output_path)
            return True, None

        if best_temp_path and os.path.exists(best_temp_path):
            os.remove(best_temp_path)

        return False, last_error or "Generálás ellenőrzése sikertelen."
    finally:
        try:
            os.remove(temp_ref_path)
        except OSError:
            pass


def run_segments_on_device(
    device: str,
    worker_label: str,
    args: argparse.Namespace,
    segments: List[Dict[str, object]],
    input_wav_path_str: str,
    eq_config: Optional[Dict[str, object]],
    stats,
    failed_segments_info,
    progress_position: int = 0,
    prefetched_audio: Optional[Tuple[np.ndarray, int]] = None,
) -> None:
    if not segments:
        return

    set_random_seeds(args.seed, device)
    transcribe_fn, whisper_language = create_transcriber(args, device)
    if args.seed == -1 and transcribe_fn is None:
        logger.error("%s: Whisper alapú ellenőrzés szükséges, de nem betölthető.", worker_label)
        return

    normaliser_path = Path(args.normalisers_dir) / args.norm / "normaliser.py"
    normalize_fn = load_normalizer(normaliser_path)
    processor, model = load_vibevoice_components(args, device)

    if prefetched_audio is not None:
        full_audio_data, sample_rate = prefetched_audio
    else:
        full_audio_data, sample_rate = sf.read(input_wav_path_str)

    progress_iter = tqdm.tqdm(
        segments,
        desc=f"Processing on {worker_label}",
        position=progress_position,
        leave=False,
    )
    for segment in progress_iter:
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", 0.0)
        filename = f"{time_to_filename_str(start_time)}_{time_to_filename_str(end_time)}.wav"
        ok, error_msg = process_segment(
            segment=segment,
            args=args,
            audio_data=full_audio_data,
            sample_rate=sample_rate,
            processor=processor,
            model=model,
            eq_config=eq_config,
            transcribe_fn=transcribe_fn,
            whisper_language=whisper_language,
            normalize_fn=normalize_fn,
            device=device,
            worker_label=worker_label,
        )
        if ok:
            increment_stat(stats, "successful")
        else:
            increment_stat(stats, "failed")
            failed_segments_info.append(
                {
                    "filename": filename,
                    "text": segment.get("translated_text") or "",
                    "reason": error_msg or "",
                }
            )


def gpu_worker(
    worker_idx: int,
    args: argparse.Namespace,
    all_chunks: List[List[Dict[str, object]]],
    input_wav_path_str: str,
    eq_config,
    stats,
    failed_segments_info,
) -> None:
    available_gpus = torch.cuda.device_count()
    if worker_idx >= available_gpus:
        logger.error(
            "GPU worker index (%s) nagyobb, mint az elérhető GPU-k száma (%s). A worker leáll.",
            worker_idx,
            available_gpus,
        )
        return

    torch.cuda.set_device(worker_idx)
    device = f"cuda:{worker_idx}"
    worker_label = f"GPU-{worker_idx}"
    run_segments_on_device(
        device=device,
        worker_label=worker_label,
        args=args,
        segments=all_chunks[worker_idx],
        input_wav_path_str=input_wav_path_str,
        eq_config=eq_config,
        stats=stats,
        failed_segments_info=failed_segments_info,
        progress_position=worker_idx,
    )


def process_project(args: argparse.Namespace) -> None:
    config_path = PROJECT_ROOT / "config.json"
    with open(config_path, "r", encoding="utf-8") as fh:
        config_data = json.load(fh)

    prepare_output_directories(args, config_data)

    wav_files = sorted(Path(args.input_wav_dir).glob("*.wav"))
    json_files = sorted(Path(args.input_json_dir).glob("*.json"))
    if not wav_files or not json_files:
        raise FileNotFoundError("Nem található megfelelő .wav vagy .json a projektben.")

    input_wav_path = wav_files[0]
    input_json_path = json_files[0]

    logger.info("Bemeneti wav: %s", input_wav_path)
    logger.info("Bemeneti json: %s", input_json_path)

    with open(input_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    segments = data.get("segments", [])
    segments.sort(key=lambda item: item.get("start", 0.0))

    filtered_segments: List[Dict[str, object]] = []
    skipped_segments = 0
    for segment in segments:
        translated_text = segment.get("translated_text")
        if not translated_text:
            translated_text = segment.get("translates_text")
        if not translated_text or not str(translated_text).strip():
            skipped_segments += 1
            continue
        if "translated_text" not in segment and translated_text is not None:
            segment = dict(segment)
            segment["translated_text"] = translated_text
        filtered_segments.append(segment)

    if skipped_segments:
        logger.info("Fordított szöveg nélküli szegmensek kihagyva: %s db", skipped_segments)

    segments = filtered_segments

    if args.max_segments is not None:
        original_len = len(segments)
        segments = segments[: args.max_segments]
        logger.info("Szegmensek limitálva: %s -> %s (max_segments=%s)", original_len, len(segments), args.max_segments)

    audio_data, sample_rate = sf.read(input_wav_path)
    logger.info(
        "Referencia audió adatai: %s mintavétel, %.2f mp",
        sample_rate,
        len(audio_data) / sample_rate,
    )

    save_noise_segments(args, audio_data, sample_rate, segments)

    eq_config_path = args.eq_config or str(DEFAULT_EQ_CONFIG_PATH)
    if eq_config_path and not Path(eq_config_path).exists():
        logger.warning("EQ konfiguráció nem található: %s. EQ kikapcsolva.", eq_config_path)
        eq_config_path = None
    eq_config = load_eq_curve_config(eq_config_path)

    cfg_dirs = config_data["DIRECTORIES"]
    full_project_path = (PROJECT_ROOT / cfg_dirs["workdir"]) / args.project_name
    args.normalisers_dir = str(PROJECT_ROOT / cfg_dirs["normalisers"])
    args.failed_generations_dir = str(full_project_path / "failed_generations")
    if args.save_failures:
        Path(args.failed_generations_dir).mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(args.device)
    args.device = resolved_device

    total_segments = len(segments)
    stats_summary: Dict[str, int] = {"successful": 0, "failed": 0, "total": total_segments}
    failed_segments: List[Dict[str, str]] = []

    input_wav_path_str = str(input_wav_path)

    if total_segments == 0:
        logger.warning("Nincs feldolgozandó szegmens a projektben.")
        return

    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        max_workers = args.max_workers if args.max_workers is not None else num_gpus
        num_workers = max(1, min(num_gpus, max_workers))

        if num_workers > 1:
            logger.info("Több-GPU feldolgozás %s workerrel.", num_workers)
            chunks: List[List[Dict[str, object]]] = [[] for _ in range(num_workers)]
            for idx, segment in enumerate(segments):
                chunks[idx % num_workers].append(segment)

            with mp.Manager() as manager:
                stats_proxy = manager.dict({"successful": 0, "failed": 0, "total": total_segments})
                failed_proxy = manager.list()
                mp.spawn(
                    gpu_worker,
                    nprocs=num_workers,
                    args=(
                        args,
                        chunks,
                        input_wav_path_str,
                        eq_config,
                        stats_proxy,
                        failed_proxy,
                    ),
                    join=True,
                )
                stats_summary = dict(stats_proxy)
                failed_segments = list(failed_proxy)
        else:
            target_idx = extract_cuda_index(resolved_device)
            available_gpus = torch.cuda.device_count()
            if target_idx >= available_gpus:
                logger.warning(
                    "A kért CUDA index (%s) nem elérhető (%s GPU található). 0-ra váltunk.",
                    target_idx,
                    available_gpus,
                )
                target_idx = 0
            torch.cuda.set_device(target_idx)
            target_device = f"cuda:{target_idx}"
            run_segments_on_device(
                device=target_device,
                worker_label=f"GPU-{target_idx}",
                args=args,
                segments=segments,
                input_wav_path_str=input_wav_path_str,
                eq_config=eq_config,
                stats=stats_summary,
                failed_segments_info=failed_segments,
                progress_position=0,
                prefetched_audio=(audio_data, sample_rate),
            )
    else:
        worker_label = resolved_device.upper()
        run_segments_on_device(
            device=resolved_device,
            worker_label=worker_label,
            args=args,
            segments=segments,
            input_wav_path_str=input_wav_path_str,
            eq_config=eq_config,
            stats=stats_summary,
            failed_segments_info=failed_segments,
            progress_position=0,
            prefetched_audio=(audio_data, sample_rate),
        )

    successful = int(stats_summary.get("successful", 0))
    failed = int(stats_summary.get("failed", 0))
    total = int(stats_summary.get("total", total_segments))
    processed = successful + failed

    logger.info("=" * 50)
    logger.info(
        "Összegzés\n  - Összes szegmens: %s\n  - Feldolgozott: %s\n  - Sikeres: %s\n  - Sikertelen: %s",
        total,
        processed,
        successful,
        failed,
    )
    if failed_segments:
        logger.info("Sikertelen szegmensek listája:")
        for item in sorted(failed_segments, key=lambda x: x.get("filename", "")):
            logger.info("  * %s -> %s", item.get("filename"), item.get("reason"))
    logger.info("=" * 50)


def main() -> None:
    args = parse_arguments()
    log_level = configure_debug_mode(args.debug)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("vibevoice_tts.log", encoding="utf-8"),
        ],
        force=True,
    )
    logger.setLevel(log_level)
    logger.info("VibeVoice TTS script indul. Projekt: %s", args.project_name)

    try:
        process_project(args)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.error("Futás közben hiba történt: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

import argparse
import datetime
import importlib.util
import json
import logging
import math
import os
import random
import shutil
import sys
import tempfile
import time
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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


def resolve_narrator_directory(narrator_argument: str) -> Path:
    """
    Feloldja a narrátor referencia könyvtár elérési útját.
    Abszolút útvonal esetén azt használja, különben először a jelenlegi munkakönyvtárhoz,
    végül a projekt gyökeréhez viszonyítva oldja fel.
    """
    candidate = Path(narrator_argument).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (PROJECT_ROOT / candidate).resolve()


def prepare_narrator_reference_sample(
    narrator_dir: Path,
    eq_config: Optional[Dict[str, object]],
    normalize: bool,
    peak_target: float,
    target_sample_rate: int,
) -> Tuple[str, int, float, Path]:
    """
    Betölti, előfeldolgozza és ideiglenes fájlba menti a narrátor referencia audiót.
    A mappában pontosan egy .wav fájlt várunk, amit opcionálisan EQ-val, normalizálással
    és újramintavételezéssel dolgozunk fel.
    """
    if not narrator_dir.is_dir():
        raise FileNotFoundError(f"A megadott narrátor könyvtár nem található: {narrator_dir}")

    wav_files = sorted(path for path in narrator_dir.iterdir() if path.suffix.lower() == ".wav")
    if not wav_files:
        raise FileNotFoundError(f"Nem található .wav fájl a narrátor könyvtárban: {narrator_dir}")
    if len(wav_files) > 1:
        raise RuntimeError(
            f"A narrátor könyvtár pontosan 1 db .wav fájlt tartalmazhat. Talált fájlok száma: {len(wav_files)} | {narrator_dir}"
        )

    source_wav = wav_files[0]
    audio_data, sample_rate = sf.read(source_wav)
    audio_data = audio_data.astype(np.float32, copy=False)

    processed_audio = apply_eq_curve_to_audio(audio_data, sample_rate, eq_config)
    if normalize:
        processed_audio = normalize_peak(processed_audio.copy(), peak_target)
    if target_sample_rate > 0 and target_sample_rate != sample_rate:
        processed_audio = resample_audio(processed_audio, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    narrator_tmp = tempfile.NamedTemporaryFile(prefix="vibevoice_narrator_", suffix=".wav", delete=False)
    narrator_tmp_path = narrator_tmp.name
    narrator_tmp.close()
    sf.write(narrator_tmp_path, processed_audio, sample_rate)

    duration_seconds = len(processed_audio) / sample_rate if sample_rate else 0.0
    return narrator_tmp_path, sample_rate, duration_seconds, source_wav


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
    transcribed_text: str,
    reference_audio_path: Optional[str],
) -> None:
    try:
        debug_segment_dir = Path(args.failed_generations_dir) / filename_stem
        debug_segment_dir.mkdir(parents=True, exist_ok=True)

        info_json_path = debug_segment_dir / "info.json"
        info: Dict[str, object] = {}
        if info_json_path.exists():
            with open(info_json_path, "r", encoding="utf-8") as fh:
                info = json.load(fh)

        attempt_filename = f"{filename_stem}_attempt_{attempt_num}_dist_{distance}.wav"
        shutil.copy(temp_gen_path, debug_segment_dir / attempt_filename)

        if reference_audio_path and os.path.exists(reference_audio_path):
            reference_filename = "reference.wav"
            reference_target = debug_segment_dir / reference_filename
            if not reference_target.exists():
                shutil.copy(reference_audio_path, reference_target)
            info["reference_audio_filename"] = reference_filename

        failures = info.setdefault("failures", [])
        if isinstance(failures, list):
            failures.append(
                {
                    "attempt": attempt_num,
                    "distance": distance,
                    "allowed_tolerance": tolerance,
                    "original_text": original_text,
                    "transcribed_text": transcribed_text,
                    "saved_audio_filename": attempt_filename,
                }
            )
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
        "--narrator",
        type=str,
        required=True,
        help="Narrátor referencia könyvtár, amely pontosan egy .wav fájlt tartalmaz.",
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

    args.output_dir = str(full_project_path / cfg_subdirs["translated_splits"])
    args.output_dir_noise = str(full_project_path / cfg_subdirs["noice_splits"])
    args.input_wav_dir = full_project_path / cfg_subdirs["separated_audio_speech"]
    args.input_json_dir = full_project_path / cfg_subdirs["translated"]

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
    processor: VibeVoiceProcessor,
    model: VibeVoiceForConditionalGenerationInference,
    transcribe_fn: Optional[Callable[[str], str]],
    whisper_language: Optional[str],
    normalize_fn: Optional[Callable[[str], str]],
    device: str,
    worker_label: str,
    narrator_sample_path: str,
) -> Tuple[bool, Optional[str]]:
    start_time = segment.get("start")
    end_time = segment.get("end")
    original_gen_text = (segment.get("translated_text") or segment.get("text") or "").strip()

    if not all(isinstance(value, (int, float)) for value in (start_time, end_time)):
        return False, "Hiányzó vagy érvénytelen időbélyeg."
    if not original_gen_text:
        return False, "Üres generálandó szöveg."

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
    inputs = prepare_inputs(
        processor=processor,
        text=formatted_text,
        voice_sample_paths=[narrator_sample_path],
        device=device,
    )

    attempts = args.max_retries if transcribe_fn else 1
    normalized_original = normalize_text_for_comparison(gen_text)
    tolerance_word_count = len(gen_text.split())
    best_distance = float("inf")
    best_temp_path: Optional[str] = None
    last_error: Optional[str] = None
    filename_stem = Path(filename).stem

    for attempt in range(1, attempts + 1):
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

            if not transcribe_fn:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temp_gen_path, output_path)
                return True, None

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
            outcome = (
                "Sikeres ellenőrzés"
                if distance <= final_tolerance or distance <= 1
                else "Ellenőrzés sikertelen"
            )
            log_generation_attempt(
                worker_label,
                filename,
                attempt,
                attempts,
                normalized_original,
                normalized_transcribed,
                distance,
                final_tolerance,
                outcome,
            )

            if distance <= final_tolerance or distance <= 1:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temp_gen_path, output_path)
                if best_temp_path and os.path.exists(best_temp_path):
                    os.remove(best_temp_path)
                return True, None

            last_error = f"Levenshtein távolság {distance}, tolerancia {final_tolerance}"
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
                    reference_audio_path=narrator_sample_path,
                )

            if distance < best_distance:
                if best_temp_path and os.path.exists(best_temp_path):
                    os.remove(best_temp_path)
                best_distance = distance
                best_temp_path = temp_gen_path
                keep_temp = True
            else:
                keep_temp = False

        except Exception as exc:
            last_error = f"Generálási hiba: {exc}"
            logger.error("Hiba a(z) %s szegmens generálása közben: %s", filename, exc, exc_info=True)
        finally:
            if not keep_temp and os.path.exists(temp_gen_path):
                os.remove(temp_gen_path)

    final_tolerance = compute_tolerance(
        tolerance_word_count,
        args.tolerance_factor,
        args.min_tolerance,
    )
    is_best_available = best_temp_path and os.path.exists(best_temp_path)
    accept_outside_tolerance = (
        args.keep_best_over_tolerance
        and is_best_available
        and best_distance < float("inf")
    )
    if is_best_available and (best_distance <= final_tolerance or accept_outside_tolerance):
        if best_distance <= final_tolerance:
            logger.info(
                "%s: A legjobb próbálkozás elfogadva (distance: %s, tolerance: %s) -> %s",
                worker_label,
                best_distance,
                final_tolerance,
                filename,
            )
        else:
            logger.warning(
                "%s: A legjobb próbálkozás tolerancia felett is elfogadva (distance: %s, tolerance: %s) -> %s",
                worker_label,
                best_distance,
                final_tolerance,
                filename,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(best_temp_path, output_path)
        return True, None

    if best_temp_path and os.path.exists(best_temp_path):
        os.remove(best_temp_path)

    return False, last_error or "Generálás ismeretlen okból sikertelen."


def run_segments_on_device(
    device: str,
    worker_label: str,
    args: argparse.Namespace,
    segments: List[Dict[str, object]],
    narrator_sample_path: str,
    stats,
    failed_segments_info,
    progress_position: int = 0,
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
            processor=processor,
            model=model,
            transcribe_fn=transcribe_fn,
            whisper_language=whisper_language,
            normalize_fn=normalize_fn,
            device=device,
            worker_label=worker_label,
            narrator_sample_path=narrator_sample_path,
        )
        if ok:
            increment_stat(stats, "successful")
        else:
            increment_stat(stats, "failed")
            failed_segments_info.append(
                {
                    "filename": filename,
                    "text": segment.get("translated_text") or segment.get("text") or "",
                    "reason": error_msg or "",
                }
            )


def gpu_worker(
    worker_idx: int,
    args: argparse.Namespace,
    all_chunks: List[List[Dict[str, object]]],
    narrator_sample_path: str,
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
        narrator_sample_path=narrator_sample_path,
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

    narrator_dir = resolve_narrator_directory(args.narrator)
    narrator_sample_path, narrator_sample_rate, narrator_duration, narrator_source = prepare_narrator_reference_sample(
        narrator_dir=narrator_dir,
        eq_config=eq_config,
        normalize=args.normalize_ref_audio,
        peak_target=args.ref_audio_peak,
        target_sample_rate=args.target_sample_rate,
    )
    logger.info(
        "Narrátor referencia: %s (%.2f mp @ %s Hz)",
        narrator_source,
        narrator_duration,
        narrator_sample_rate,
    )

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

    try:
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
                            narrator_sample_path,
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
                    narrator_sample_path=narrator_sample_path,
                    stats=stats_summary,
                    failed_segments_info=failed_segments,
                    progress_position=0,
                )
        else:
            worker_label = resolved_device.upper()
            run_segments_on_device(
                device=resolved_device,
                worker_label=worker_label,
                args=args,
                segments=segments,
                narrator_sample_path=narrator_sample_path,
                stats=stats_summary,
                failed_segments_info=failed_segments,
                progress_position=0,
            )
    finally:
        try:
            os.remove(narrator_sample_path)
        except OSError:
            pass

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

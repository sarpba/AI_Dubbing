import argparse
import datetime
import importlib.util
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import math
import numpy as np
import torch
import torch.multiprocessing as mp

import soundfile as sf
import tqdm

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from f5_tts.infer.utils_infer import (
    hop_length,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    target_sample_rate,
)
from f5_tts.model import DiT, UNetT
from tools.debug_utils import add_debug_argument, configure_debug_mode

try:
    import whisper
except ImportError:  # pragma: no cover - optional dependency
    whisper = None
try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - optional dependency
    pipeline = None
try:
    from num2words import num2words
except ImportError:  # pragma: no cover - optional dependency
    num2words = None
from Levenshtein import distance as levenshtein_distance

try:
    from fonetic import fonetikus_atiras
except ImportError:  # pragma: no cover - optional dependency
    fonetikus_atiras = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEL_SPEC_TYPE = "vocos"
NORMALIZER_TO_WHISPER_LANG = {"hun": "hu", "eng": "en"}
WHISPER_LANG_CODE_TO_NAME = {"hu": "hungarian", "en": "english"}
DEFAULT_EQ_CONFIG_PATH = PROJECT_ROOT / "scripts" / "EQ.json"

REF_SILENCE_PADDING_SECONDS = 0.4
SHORT_REF_DURATION_SECONDS = 1.5
MAX_EXTENDED_REF_DURATION_SECONDS = 11.5
MIN_VERIFICATION_REF_DURATION_SECONDS = 2.0


def set_random_seeds(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in str(device):
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class F5TTS:
    def __init__(
        self,
        model_cls,
        model_cfg_dict,
        ckpt_file,
        vocab_file,
        vocoder_name: str = DEFAULT_MEL_SPEC_TYPE,
        ode_method: str = "euler",
        use_ema: bool = True,
        local_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.final_wave = None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.seed = -1
        self.mel_spec_type = vocoder_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.load_vocoder_model(vocoder_name, local_path)
        self.load_ema_model(model_cls, model_cfg_dict, ckpt_file, vocab_file, ode_method, use_ema)

    def load_vocoder_model(self, vocoder_name: str, local_path: Optional[str]) -> None:
        self.vocoder = load_vocoder(vocoder_name, local_path is not None, local_path, self.device)

    def load_ema_model(
        self,
        model_cls,
        model_cfg,
        ckpt_file: str,
        vocab_file: str,
        ode_method: str,
        use_ema: bool,
    ) -> None:
        self.ema_model = load_model(
            model_cls,
            model_cfg,
            ckpt_file,
            self.mel_spec_type,
            vocab_file,
            ode_method,
            use_ema,
            self.device,
        )

    def export_wav(self, wav: np.ndarray, file_wave: str, remove_silence: bool = False) -> None:
        sf.write(file_wave, wav, self.target_sample_rate)
        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def infer(
        self,
        ref_file: str,
        ref_text: str,
        gen_text: str,
        file_wave: Optional[str],
        remove_silence: bool = False,
        speed: float = 1.0,
        nfe_step: int = 32,
        seed: int = -1,
    ):
        if not (0.3 <= speed <= 2.0):
            raise ValueError(f"Invalid speed: {speed}")
        if not (16 <= nfe_step <= 64):
            raise ValueError(f"Invalid nfe_step: {nfe_step}")

        numpy_max_seed = 2**32 - 1
        current_seed = seed if seed != -1 else random.randint(0, numpy_max_seed)
        self.seed = current_seed
        set_random_seeds(current_seed, self.device)

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)
        wav, sr, spect = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=logger.info,
            progress=tqdm,
            target_rms=0.1,
            cross_fade_duration=0.15,
            nfe_step=nfe_step,
            cfg_strength=2,
            sway_sampling_coef=-1,
            speed=speed,
            fix_duration=None,
            device=self.device,
        )
        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)
        return wav, sr, spect


def time_to_filename_str(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    td = datetime.timedelta(seconds=seconds)
    minutes, secs = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}-{minutes:02d}-{secs:02d}-{milliseconds:03d}"


def resolve_model_paths_from_dir(model_dir_path: str) -> Tuple[str, str, str]:
    model_dir = Path(model_dir_path)
    if not model_dir.is_dir():
        logger.error("Model directory not found: %s", model_dir)
        sys.exit(1)

    ckpt_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.safetensors"))
    if not ckpt_files:
        logger.error("No checkpoint file found in %s", model_dir)
        sys.exit(1)
    if len(ckpt_files) > 1:
        logger.warning("Multiple checkpoints found, using %s", ckpt_files[0])

    resolved_ckpt_file = ckpt_files[0]
    resolved_vocab_file = model_dir / "vocab.txt"
    if not resolved_vocab_file.exists():
        logger.error("vocab.txt not found in %s", model_dir)
        sys.exit(1)

    json_files = list(model_dir.glob("*.json"))
    if not json_files:
        logger.error("No config .json found in %s", model_dir)
        sys.exit(1)
    if len(json_files) > 1:
        logger.warning("Multiple configs found, using %s", json_files[0])
    resolved_json_file = json_files[0]

    logger.info(
        "Found model files in '%s':\n  - Checkpoint: %s\n  - Vocab: %s\n  - Config: %s",
        model_dir,
        resolved_ckpt_file.name,
        resolved_vocab_file.name,
        resolved_json_file.name,
    )
    return str(resolved_ckpt_file), str(resolved_vocab_file), str(resolved_json_file)


def interactive_model_selection(tts_base_path: Path) -> str:
    logger.info("No model directory specified. Searching for models in: %s", tts_base_path)
    models = sorted([d for d in tts_base_path.iterdir() if d.is_dir()])
    if not models:
        logger.error("No models found in %s", tts_base_path)
        sys.exit(1)

    print("\nPlease select a model from the TTS directory:")
    for i, model_path in enumerate(models):
        print(f"  {i + 1}: {model_path.name}")

    while True:
        try:
            choice = int(input("Enter number: ")) - 1
            if 0 <= choice < len(models):
                selected_path = models[choice]
                logger.info("Model selected: %s", selected_path.name)
                return str(selected_path)
            print("Invalid number, try again.")
        except ValueError:
            print("Invalid input, please enter a number.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="F5-TTS Inference Script with advanced verification and debugging.",
    )
    parser.add_argument("project_name", type=str, help="A projekt könyvtárának neve a 'workdir' mappán belül.")
    parser.add_argument("--norm", type=str, required=True, help="Normalizálás típusa (pl. 'hun', 'eng').")
    parser.add_argument("--model_dir", type=str, default=None, help="Opcionális: a TTS modell könyvtárának elérési útja.")
    parser.add_argument("--speed", type=float, default=1.0, help="A generált hang sebessége (0.3-2.0).")
    parser.add_argument("--nfe_step", type=int, default=32, help="NFE lépések száma (16-64).")
    parser.add_argument("--remove_silence", action="store_true", help="Csend eltávolítása a generált hangból.")
    parser.add_argument(
        "--phonetic-ref",
        action="store_true",
        help="A referencia szöveget (ref_text) fonetikus átírással használja.",
    )
    parser.add_argument(
        "--normalize-ref-audio",
        action="store_true",
        help="Aktiválja a referencia audio csúcshangerejének normalizálását.",
    )
    parser.add_argument(
        "--eq-config",
        type=str,
        default=None,
        help="EQ konfiguráció JSON útvonala, amelyet a referencia audióra alkalmazunk generálás előtt.",
    )
    parser.add_argument(
        "--ref-audio-peak",
        type=float,
        default=0.95,
        help="A normalizált referencia audio cél csúcsértéke (0.0-1.0).",
    )
    parser.add_argument("--max_workers", type=int, default=None, help="Párhuzamos workerek maximális száma.")
    parser.add_argument("--seed", type=int, default=-1, help="Véletlenszám-generátor magja (-1: random és verifikáció).")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximális újragenerálási kísérletek száma.")
    parser.add_argument(
        "--tolerance-factor",
        type=float,
        default=1.0,
        help="Tolerancia szorzó a szavak száma alapján.",
    )
    parser.add_argument(
        "--min-tolerance",
        type=int,
        default=2,
        help="A dinamikusan számított tolerancia minimális értéke.",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="openai/whisper-large-v3",
        help="A verifikációhoz használt Whisper modell.",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="A Whisper dekódoláshoz használt nyalábszélesség.")
    parser.add_argument(
        "--save-failures",
        action="store_true",
        help="Elmenti a hibás generálásokat egy 'failed_generations' mappába.",
    )
    parser.add_argument(
        "--keep-best-over-tolerance",
        action="store_true",
        help="Ha a tolerancia felett is marad a legjobb eredmény, akkor is elmenti a legkisebb távolságú verziót.",
    )
    add_debug_argument(parser)
    return parser.parse_args()


def normalize_peak(audio: np.ndarray, target_peak: float) -> np.ndarray:
    if not 0.0 < target_peak <= 1.0:
        target_peak = 0.95
    current_peak = np.max(np.abs(audio))
    if current_peak == 0:
        return audio
    return audio * (target_peak / current_peak)


def load_eq_curve_config(config_path: Optional[str]) -> Optional[Dict[str, object]]:
    if not config_path:
        return None

    eq_path = Path(config_path)
    if not eq_path.exists():
        logger.warning("EQ config path does not exist: %s. Skipping EQ.", eq_path)
        return None

    try:
        with open(eq_path, "r", encoding="utf-8") as fh:
            eq_data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load EQ config from %s: %s", eq_path, exc)
        return None

    points = eq_data.get("points")
    if not isinstance(points, list) or not points:
        logger.warning("EQ config at %s does not define any points. Skipping EQ.", eq_path)
        return None

    parsed_points: List[Dict[str, float]] = []
    for raw_point in points:
        try:
            freq = float(raw_point["frequency_hz"])
            gain_db = float(raw_point.get("gain_db", 0.0))
        except (KeyError, TypeError, ValueError):
            logger.warning("Invalid EQ point encountered in %s: %s", eq_path, raw_point)
            continue
        parsed_points.append({"frequency_hz": max(0.0, freq), "gain_db": gain_db})

    if not parsed_points:
        logger.warning("No valid EQ points found in %s. Skipping EQ.", eq_path)
        return None

    parsed_points.sort(key=lambda item: item["frequency_hz"])
    global_gain_db = float(eq_data.get("global_gain_db", 0.0))
    logger.info(
        "Loaded EQ curve from %s with %s control points (global gain %.2f dB).",
        eq_path,
        len(parsed_points),
        global_gain_db,
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

    # Remove duplicate frequencies to keep interpolation stable.
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

    def replace_match(match):
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
            logger.error("Failed to prepare normalizer loader from %s", normaliser_path)
            return None
        normaliser_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(normaliser_module)  # type: ignore[attr-defined]
        if hasattr(normaliser_module, "normalize"):
            return getattr(normaliser_module, "normalize")
        logger.warning("Normalizer at %s does not expose a 'normalize' function.", normaliser_path)
    except Exception as exc:
        logger.error("Failed to load normalizer from %s: %s", normaliser_path, exc, exc_info=True)
    return None


def compute_tolerance(word_count: int, tolerance_factor: float, min_tolerance: int) -> int:
    calculated_tolerance = math.ceil(word_count * tolerance_factor)
    return max(calculated_tolerance, min_tolerance)


def expand_reference_audio(
    base_ref_chunk: np.ndarray,
    base_ref_text: str,
    multiplier: int,
    sample_rate: int,
) -> Tuple[np.ndarray, str]:
    if multiplier <= 1:
        return base_ref_chunk, base_ref_text

    silence_samples = int(REF_SILENCE_PADDING_SECONDS * sample_rate)
    silence_shape = base_ref_chunk.shape[1:] if base_ref_chunk.ndim > 1 else ()
    silence_chunk = np.zeros((silence_samples,) + silence_shape, dtype=base_ref_chunk.dtype)

    audio_parts = []
    text_parts = []
    for idx in range(multiplier):
        audio_parts.append(base_ref_chunk)
        text_parts.append(base_ref_text)
        if idx < multiplier - 1:
            audio_parts.append(silence_chunk)

    expanded_audio = np.concatenate(audio_parts)
    expanded_text = " ".join(text_parts)
    return expanded_audio, expanded_text


def create_transcriber(args, device: str) -> Tuple[Optional[Callable[[str], str]], Optional[str]]:
    if args.seed != -1:
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

        transcriber = pipeline(
            "automatic-speech-recognition",
            model=args.whisper_model,
            torch_dtype=torch.float16,
            device=device,
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


def log_generation_attempt(
    gpu: int,
    filename: str,
    attempt: int,
    max_attempts: int,
    ref_multiplier: int,
    ref_duration: float,
    normalized_original: str,
    normalized_transcribed: str,
    distance: int,
    tolerance: int,
    outcome: str,
) -> None:
    message = (
        f"\n--- Generálás Log (Worker {gpu}, Fájl: {filename}) ---\n"
        f"Generálás: | {attempt}/{max_attempts} (Ref. Multiplier: {ref_multiplier}x)\n"
        f"Felhasznált ref audió hossza: | {ref_duration:.2f} sec\n"
        f"Gen_text (normalizált): | {normalized_original}\n"
        f"Whisperrel visszaolvasott (normalizált): | {normalized_transcribed}\n"
        f"Távolság / Megengedett: | {distance} / {tolerance}\n"
        f"{outcome}\n"
        "----------------------------------------------------------"
    )
    logger.info(message)


def save_failed_attempt(
    args,
    filename_stem: str,
    attempt_num: int,
    dist: int,
    final_tolerance: int,
    temp_gen_path: str,
    sample_rate: int,
    original_ref_chunk: np.ndarray,
    base_ref_chunk: np.ndarray,
    current_ref_audio_for_gen: np.ndarray,
    original_ref_text: str,
    gen_text: str,
    raw_transcribed: str,
    converted_transcribed: str,
) -> None:
    """Persist metadata and audio for a failed generation attempt."""
    try:
        debug_segment_dir = Path(args.failed_generations_dir) / filename_stem
        debug_segment_dir.mkdir(parents=True, exist_ok=True)

        info_json_path = debug_segment_dir / "info.json"
        info = {}
        if info_json_path.exists():
            try:
                with open(info_json_path, "r", encoding="utf-8") as fh:
                    info = json.load(fh)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning(
                    "Could not read existing info.json for %s. Reinitialising file.",
                    filename_stem,
                )
                info = {}

        if not info:
            info.update({"original_ref_text": original_ref_text, "gen_text": gen_text})
            original_wav_path = debug_segment_dir / "ref_audio_original.wav"
            if not original_wav_path.exists():
                sf.write(original_wav_path, original_ref_chunk, sample_rate)

            normalized_wav_path = debug_segment_dir / "ref_audio_normalized.wav"
            if args.normalize_ref_audio and not normalized_wav_path.exists():
                sf.write(normalized_wav_path, base_ref_chunk, sample_rate)

        extended_wav_path = debug_segment_dir / "ref_audio_extended.wav"
        if (
            len(current_ref_audio_for_gen) > len(base_ref_chunk)
            and not extended_wav_path.exists()
        ):
            sf.write(extended_wav_path, current_ref_audio_for_gen, sample_rate)

        failed_attempt_filename = f"{filename_stem}_attempt_{attempt_num}_dist_{dist}.wav"
        debug_gen_audio_path = debug_segment_dir / failed_attempt_filename
        shutil.copy(temp_gen_path, debug_gen_audio_path)

        failure_details = {
            "attempt": attempt_num,
            "distance": dist,
            "allowed_tolerance": final_tolerance,
            "raw_transcribed_text": raw_transcribed,
            "converted_transcribed_text": converted_transcribed,
            "saved_audio_filename": failed_attempt_filename,
            "ref_audio_duration_for_this_attempt": len(current_ref_audio_for_gen) / sample_rate,
        }
        info.setdefault("failures", []).append(failure_details)

        with open(info_json_path, "w", encoding="utf-8") as fh:
            json.dump(info, fh, ensure_ascii=False, indent=2)

    except Exception as exc:
        logger.error("Failed to save debug information for %s: %s", filename_stem, exc, exc_info=True)


def main_worker(gpu, args, all_chunks, input_wav_path_str, stats, failed_segments_info):
    tasks_chunk = all_chunks[gpu]
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    logger.info("Worker %s starting on device %s with %s tasks.", gpu, device, len(tasks_chunk))

    phonetic_converter: Optional[Callable[[str], str]] = None
    if args.phonetic_ref:
        if fonetikus_atiras:
            logger.info("Worker %s: Phonetic reference text mode is ON.", gpu)
            phonetic_converter = fonetikus_atiras
            try:
                phonetic_converter("initialize")
            except Exception as exc:
                logger.error(
                    "Worker %s: Failed to initialize phonetic transcriber: %s. Disabling feature.",
                    gpu,
                    exc,
                    exc_info=True,
                )
                phonetic_converter = None
        else:
            logger.warning(
                "Worker %s: --phonetic-ref specified, but 'fonetic.py' not imported. Feature disabled.",
                gpu,
            )

    try:
        transcribe_fn, whisper_language = create_transcriber(args, device)
        if args.seed == -1 and transcribe_fn is None:
            logger.error("Worker %s: Verification requested but no transcriber is available.", gpu)
            return

        normaliser_path = Path(args.normalisers_dir) / args.norm / "normaliser.py"
        normalize_fn = load_normalizer(normaliser_path)

        architecture_class_map = {"DiT": DiT, "UNetT": UNetT}
        resolved_model_cls = architecture_class_map.get(args.resolved_model_architecture)
        if resolved_model_cls is None:
            logger.error(
                "Worker %s: Unsupported model architecture '%s'.",
                gpu,
                args.resolved_model_architecture,
            )
            return

        f5tts = F5TTS(
            model_cls=resolved_model_cls,
            model_cfg_dict=args.resolved_model_params_dict,
            ckpt_file=args.resolved_ckpt_file,
            vocab_file=args.resolved_vocab_file,
            device=device,
        )

        full_audio_data, sample_rate = sf.read(input_wav_path_str)

        for segment in tqdm.tqdm(tasks_chunk, desc=f"Processing on {device}", position=gpu):
            start_time = segment.get("start")
            end_time = segment.get("end")
            original_ref_text = segment.get("text", "").strip()
            original_gen_text = segment.get("translated_text", "").strip()
            if not all(
                [
                    isinstance(start_time, (int, float)),
                    isinstance(end_time, (int, float)),
                    original_ref_text,
                    original_gen_text,
                ]
            ):
                continue

            filename = f"{time_to_filename_str(start_time)}_{time_to_filename_str(end_time)}.wav"
            output_wav_path = Path(args.output_dir) / filename

            if output_wav_path.exists():
                stats["successful"] += 1
                continue

            gen_text = original_gen_text
            if normalize_fn:
                try:
                    gen_text = normalize_fn(gen_text)
                except Exception as exc:
                    logger.error(
                        "Worker %s: Failed to normalize text for '%s': %s",
                        gpu,
                        filename,
                        exc,
                        exc_info=True,
                    )
                    stats["failed"] += 1
                    failed_segments_info.append({"filename": filename, "text": original_gen_text})
                    continue

            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            end_sample = min(end_sample, len(full_audio_data))
            if start_sample >= end_sample:
                continue

            ref_chunk_source = full_audio_data[start_sample:end_sample]
            original_ref_chunk = apply_eq_curve_to_audio(
                ref_chunk_source,
                sample_rate,
                args.eq_curve_settings,
            )
            base_ref_chunk = (
                normalize_peak(original_ref_chunk.copy(), args.ref_audio_peak)
                if args.normalize_ref_audio
                else original_ref_chunk.copy()
            )
            base_ref_text = original_ref_text
            initial_duration = len(base_ref_chunk) / sample_rate

            if args.seed == -1:
                verification_passed = False
                best_distance = float("inf")
                best_attempt_audio_path: Optional[str] = None
                ref_multiplication_factor = 1
                last_ref_audio_duration_used = len(base_ref_chunk) / sample_rate

                try:
                    for attempt in range(1, args.max_retries + 1):
                        temp_gen_path: Optional[str] = None
                        try:
                            processed_ref_chunk_for_gen, processed_ref_text_for_gen = expand_reference_audio(
                                base_ref_chunk,
                                base_ref_text,
                                ref_multiplication_factor,
                                sample_rate,
                            )

                            final_ref_text_for_gen = processed_ref_text_for_gen
                            if phonetic_converter:
                                try:
                                    final_ref_text_for_gen = phonetic_converter(processed_ref_text_for_gen)
                                except Exception as exc:
                                    logger.error(
                                        "Worker %s: Phonetic conversion failed for attempt %s (%s): %s",
                                        gpu,
                                        attempt,
                                        filename,
                                        exc,
                                    )
                                    final_ref_text_for_gen = processed_ref_text_for_gen

                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=args.output_dir) as tmp:
                                temp_gen_path = tmp.name
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_ref:
                                sf.write(tmp_ref.name, processed_ref_chunk_for_gen, sample_rate)
                                f5tts.infer(
                                    ref_file=tmp_ref.name,
                                    ref_text=final_ref_text_for_gen,
                                    gen_text=gen_text,
                                    file_wave=temp_gen_path,
                                    seed=-1,
                                    speed=args.speed,
                                    nfe_step=args.nfe_step,
                                )

                            raw_transcribed_text = transcribe_fn(temp_gen_path) if transcribe_fn else ""
                            converted_transcribed_text = convert_numbers_to_words(
                                raw_transcribed_text,
                                lang=whisper_language,
                            )
                            norm_original = normalize_text_for_comparison(gen_text)
                            norm_transcribed = normalize_text_for_comparison(converted_transcribed_text)
                            dist = levenshtein_distance(norm_original, norm_transcribed)
                            final_tolerance = compute_tolerance(
                                len(gen_text.split()),
                                args.tolerance_factor,
                                args.min_tolerance,
                            )
                            ref_audio_duration_used = len(processed_ref_chunk_for_gen) / sample_rate
                            last_ref_audio_duration_used = ref_audio_duration_used

                            if dist <= 1:
                                outcome = "Generálás sikeres: | Igen (Távolság <= 1)"
                                log_generation_attempt(
                                    gpu,
                                    filename,
                                    attempt,
                                    args.max_retries,
                                    ref_multiplication_factor,
                                    ref_audio_duration_used,
                                    norm_original,
                                    norm_transcribed,
                                    dist,
                                    final_tolerance,
                                    outcome,
                                )
                                os.rename(temp_gen_path, output_wav_path)
                                temp_gen_path = None
                                verification_passed = True
                                break

                            if args.save_failures and temp_gen_path:
                                save_failed_attempt(
                                    args,
                                    output_wav_path.stem,
                                    attempt,
                                    dist,
                                    final_tolerance,
                                    temp_gen_path,
                                    sample_rate,
                                    original_ref_chunk,
                                    base_ref_chunk,
                                    processed_ref_chunk_for_gen,
                                    original_ref_text,
                                    gen_text,
                                    raw_transcribed_text,
                                    converted_transcribed_text,
                                )

                            if dist < best_distance:
                                outcome = f"Generálás sikeres: | Még nem (Új legjobb távolság: {dist})"
                                if best_attempt_audio_path and os.path.exists(best_attempt_audio_path):
                                    os.remove(best_attempt_audio_path)
                                best_distance = dist
                                best_attempt_audio_path = temp_gen_path
                                temp_gen_path = None
                            else:
                                outcome = (
                                    f"Generálás sikeres: | Még nem (Nem jobb, mint az eddigi legjobb: {best_distance})"
                                )

                            log_generation_attempt(
                                gpu,
                                filename,
                                attempt,
                                args.max_retries,
                                ref_multiplication_factor,
                                ref_audio_duration_used,
                                norm_original,
                                norm_transcribed,
                                dist,
                                final_tolerance,
                                outcome,
                            )
                        except Exception as exc:
                            logger.error(
                                "Worker %s: Error during verification attempt %s for '%s': %s",
                                gpu,
                                attempt,
                                filename,
                                exc,
                                exc_info=True,
                            )
                        finally:
                            if temp_gen_path and os.path.exists(temp_gen_path):
                                os.remove(temp_gen_path)

                        if (
                            0 < initial_duration < SHORT_REF_DURATION_SECONDS
                            and last_ref_audio_duration_used < MIN_VERIFICATION_REF_DURATION_SECONDS
                        ):
                            next_potential_duration = (
                                last_ref_audio_duration_used
                                + REF_SILENCE_PADDING_SECONDS
                                + initial_duration
                            )
                            if next_potential_duration <= MAX_EXTENDED_REF_DURATION_SECONDS:
                                ref_multiplication_factor += 1
                                logger.info(
                                    "Worker %s: Increasing reference multiplier to %sx for '%s'.",
                                    gpu,
                                    ref_multiplication_factor,
                                    filename,
                                )
                            else:
                                logger.warning(
                                    "Worker %s: Reference multiplier cannot grow further for '%s' "
                                    "(next duration: %.2fs would exceed the %.1fs limit).",
                                    gpu,
                                    filename,
                                    next_potential_duration,
                                    MAX_EXTENDED_REF_DURATION_SECONDS,
                                )

                    if not verification_passed:
                        final_tolerance = compute_tolerance(
                            len(gen_text.split()),
                            args.tolerance_factor,
                            args.min_tolerance,
                        )
                        is_best_available = best_attempt_audio_path and os.path.exists(best_attempt_audio_path)
                        accept_outside_tolerance = (
                            args.keep_best_over_tolerance
                            and is_best_available
                            and best_distance < float("inf")
                        )
                        if (
                            is_best_available
                            and (
                                best_distance <= final_tolerance
                                or accept_outside_tolerance
                            )
                        ):
                            if best_distance <= final_tolerance:
                                logger.info(
                                    "Worker %s: Accepting best attempt (distance: %s) for '%s' within tolerance (%s).",
                                    gpu,
                                    best_distance,
                                    filename,
                                    final_tolerance,
                                )
                            else:
                                logger.warning(
                                    "Worker %s: Accepting best attempt (distance: %s) for '%s' despite exceeding tolerance (%s) because --keep-best-over-tolerance is set.",
                                    gpu,
                                    best_distance,
                                    filename,
                                    final_tolerance,
                                )
                            os.rename(best_attempt_audio_path, output_wav_path)
                            best_attempt_audio_path = None
                            verification_passed = True

                    if best_attempt_audio_path and os.path.exists(best_attempt_audio_path):
                        os.remove(best_attempt_audio_path)

                    if verification_passed:
                        stats["successful"] += 1
                    else:
                        stats["failed"] += 1
                        failed_segments_info.append({"filename": filename, "text": original_gen_text})
                        logger.warning(
                            "Worker %s: Generation failed for '%s' after %s attempts.",
                            gpu,
                            filename,
                            args.max_retries,
                        )
                except Exception as exc:
                    logger.error(
                        "Worker %s: Verification loop failed for '%s': %s",
                        gpu,
                        filename,
                        exc,
                        exc_info=True,
                    )
                    stats["failed"] += 1
                    failed_segments_info.append({"filename": filename, "text": original_gen_text})
            else:
                try:
                    final_ref_text = base_ref_text
                    if phonetic_converter:
                        try:
                            final_ref_text = phonetic_converter(base_ref_text)
                        except Exception as exc:
                            logger.error(
                                "Worker %s: Phonetic conversion failed for '%s': %s",
                                gpu,
                                filename,
                                exc,
                            )
                            final_ref_text = base_ref_text
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_ref:
                        sf.write(tmp_ref.name, base_ref_chunk, sample_rate)
                        f5tts.infer(
                            ref_file=tmp_ref.name,
                            ref_text=final_ref_text,
                            gen_text=gen_text,
                            file_wave=str(output_wav_path),
                            seed=args.seed,
                            speed=args.speed,
                            nfe_step=args.nfe_step,
                        )
                    stats["successful"] += 1
                except Exception as exc:
                    logger.error(
                        "Worker %s: Failed to generate deterministically for '%s': %s",
                        gpu,
                        filename,
                        exc,
                        exc_info=True,
                    )
                    stats["failed"] += 1
                    failed_segments_info.append({"filename": filename, "text": original_gen_text})
    except Exception as exc:
        logger.critical("Critical error in worker %s: %s", gpu, exc, exc_info=True)


def process_file_pair(args, input_wav_path: Path, input_json_path: Path) -> None:
    with open(input_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    segments = data.get("segments", [])
    segments.sort(key=lambda x: x.get("start", 0))

    full_audio_data, sample_rate = sf.read(input_wav_path)
    duration_seconds = len(full_audio_data) / sample_rate

    noise_dir = Path(args.output_dir_noise)
    last_end_time = 0.0
    for segment in segments:
        start_time = segment.get("start")
        if start_time is not None and start_time > last_end_time:
            noise_output_path = noise_dir / f"{time_to_filename_str(last_end_time)}_{time_to_filename_str(start_time)}.wav"
            if not noise_output_path.exists():
                sf.write(
                    noise_output_path,
                    full_audio_data[int(last_end_time * sample_rate) : int(start_time * sample_rate)],
                    sample_rate,
                )
        last_end_time = segment.get("end", start_time)

    if duration_seconds > last_end_time:
        noise_output_path = noise_dir / f"{time_to_filename_str(last_end_time)}_{time_to_filename_str(duration_seconds)}.wav"
        if not noise_output_path.exists():
            sf.write(noise_output_path, full_audio_data[int(last_end_time * sample_rate) :], sample_rate)

    num_gpus = torch.cuda.device_count()
    max_workers = args.max_workers if args.max_workers is not None else num_gpus
    num_workers = min(num_gpus, max_workers)
    if num_workers == 0:
        logger.error("No GPUs available for processing. Exiting.")
        sys.exit(1)

    chunks = [[] for _ in range(num_workers)]
    for idx, segment in enumerate(segments):
        chunks[idx % num_workers].append(segment)

    with mp.Manager() as manager:
        stats = manager.dict({"successful": 0, "failed": 0, "total": len(segments)})
        failed_segments_info = manager.list()
        spawn_args = (args, chunks, str(input_wav_path), stats, failed_segments_info)
        mp.spawn(main_worker, nprocs=num_workers, args=spawn_args)

        successful = stats["successful"]
        failed = stats["failed"]
        total = stats["total"]
        processed = successful + failed

        logger.info(
            "=" * 50
            + f"\nVÉGSŐ STATISZTIKA\n  - Összes szegmens: {total}\n  - Feldolgozott: {processed}\n  - Sikeres: {successful}\n  - Sikertelen: {failed}\n"
            + "=" * 50
        )

        if failed_segments_info:
            logger.info("\n" + "=" * 50 + "\nSIKERTELEN SZEGMENSEK JELENTÉSE")
            sorted_failures = sorted(list(failed_segments_info), key=lambda x: x["filename"])
            for failure in sorted_failures:
                logger.info("  - %s: %s", failure["filename"], failure["text"])
            logger.info("=" * 50)


def main() -> None:
    args = parse_arguments()
    log_level = configure_debug_mode(args.debug)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] [%(processName)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("infer_batch.log", encoding="utf-8"),
        ],
        force=True,
    )
    logger.setLevel(log_level)

    config_path = PROJECT_ROOT / "config.json"
    with open(config_path, "r", encoding="utf-8") as fh:
        config_data = json.load(fh)

    cfg_dirs = config_data["DIRECTORIES"]
    cfg_subdirs = config_data["PROJECT_SUBDIRS"]
    full_project_path = (PROJECT_ROOT / cfg_dirs["workdir"]) / args.project_name

    eq_config_path = args.eq_config or str(DEFAULT_EQ_CONFIG_PATH)
    if eq_config_path and not Path(eq_config_path).exists():
        logger.warning("Specified EQ config not found at %s. EQ will be disabled.", eq_config_path)
        eq_config_path = None
    args.eq_config_path = eq_config_path
    args.eq_curve_settings = load_eq_curve_config(eq_config_path)

    args.output_dir = str(full_project_path / cfg_subdirs['translated_splits'])
    args.output_dir_noise = str(full_project_path / cfg_subdirs['noice_splits'])
    args.normalisers_dir = str(PROJECT_ROOT / cfg_dirs['normalisers'])
    args.failed_generations_dir = str(full_project_path / "failed_generations")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir_noise).mkdir(parents=True, exist_ok=True)
    if args.save_failures:
        Path(args.failed_generations_dir).mkdir(parents=True, exist_ok=True)

    if args.model_dir:
        model_dir_path = args.model_dir
    else:
        tts_base_path = PROJECT_ROOT / cfg_dirs['TTS']
        model_dir_path = interactive_model_selection(tts_base_path)

    ckpt_path, vocab_path, json_path = resolve_model_paths_from_dir(model_dir_path)
    args.resolved_ckpt_file = ckpt_path
    args.resolved_vocab_file = vocab_path
    with open(json_path, 'r', encoding='utf-8') as fh:
        model_config_data = json.load(fh)
    args.resolved_model_architecture = model_config_data["model_architecture"]
    args.resolved_model_params_dict = model_config_data["model_params"]

    input_wav_dir = full_project_path / cfg_subdirs['separated_audio_speech']
    input_json_dir = full_project_path / cfg_subdirs['translated']
    wav_files = list(input_wav_dir.glob('*.wav'))
    json_files = list(input_json_dir.glob('*.json'))

    if not wav_files or not json_files:
        logger.error("Could not find required .wav or .json files in '%s'", args.project_name)
        sys.exit(1)

    input_wav_path = wav_files[0]
    input_json_path = json_files[0]
    process_file_pair(args, input_wav_path, input_json_path)
    logger.info("Script finished.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

import argparse
import datetime
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import soundfile as sf

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "A 'torch' csomag szükséges a Higgs TTS script futtatásához. "
        "Aktiváld a megfelelő környezetet (például: `conda activate higgs-tts-gradio`)."
    ) from exc

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "A 'transformers' csomag szükséges a Higgs TTS script futtatásához. "
        "Aktiváld a megfelelő környezetet (például: `conda activate higgs-tts-gradio`)."
    ) from exc

from tools.debug_utils import add_debug_argument, configure_debug_mode

logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        config_candidate = candidate / "config.json"
        if config_candidate.is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")


PROJECT_ROOT = find_project_root()
DEFAULT_EQ_CONFIG_PATH = PROJECT_ROOT / "scripts" / "TTS" / "EQ.json"
DEFAULT_MODEL_ALIAS = "bosonai/higgs-tts-3-4b"
DEFAULT_LOCAL_MODEL = "multimodalart/higgs-audio-v3-tts-4b-transformers"
REFERENCE_EXPAND_IF_SHORTER_SECONDS = 2.0
REFERENCE_EXPAND_TARGET_SECONDS = 4.0
DIRECT_MODELS: Dict[Tuple[str, str, int], Tuple[Any, Any]] = {}


def time_to_filename_str(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    td = datetime.timedelta(seconds=seconds)
    minutes, secs = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}-{minutes:02d}-{secs:02d}-{milliseconds:03d}"


def best_cuda_device_index() -> int:
    if not torch.cuda.is_available():
        return -1

    supported: List[Tuple[int, int]] = []
    fallback: List[Tuple[int, int]] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        memory = int(props.total_memory)
        if (major, minor) >= (7, 5):
            supported.append((memory, idx))
        else:
            fallback.append((memory, idx))

    if supported:
        return max(supported)[1]
    if fallback:
        return max(fallback)[1]
    return -1


def resolve_device(device: str) -> str:
    normalized = (device or "auto").strip().lower()
    if normalized == "auto":
        cuda_index = best_cuda_device_index()
        if cuda_index >= 0:
            return f"cuda:{cuda_index}"
        return "cpu"

    if normalized == "cuda":
        cuda_index = best_cuda_device_index()
        if cuda_index >= 0:
            return f"cuda:{cuda_index}"
        logger.warning("CUDA nem elérhető. CPU-ra váltunk.")
        return "cpu"

    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            logger.warning("CUDA nem elérhető. CPU-ra váltunk.")
            return "cpu"
        try:
            index = int(normalized.split(":", 1)[1])
        except ValueError:
            logger.warning("Érvénytelen CUDA eszközmegadás: %s. CPU-ra váltunk.", device)
            return "cpu"
        if index < 0 or index >= torch.cuda.device_count():
            logger.warning(
                "A kért CUDA index (%s) nem elérhető (%s GPU található). CPU-ra váltunk.",
                index,
                torch.cuda.device_count(),
            )
            return "cpu"
        return normalized

    if normalized in {"cpu", "mps"}:
        return normalized

    logger.warning("Ismeretlen eszközmegadás: %s. CPU-ra váltunk.", device)
    return "cpu"


def resolve_cuda_dtype(device: str):
    if not device.startswith("cuda:"):
        return torch.float32
    try:
        index = int(device.split(":", 1)[1])
        props = torch.cuda.get_device_properties(index)
    except Exception:
        return torch.float16
    return torch.bfloat16 if props.major >= 8 else torch.float16


def local_model_id(model_path: str) -> str:
    model_id = (model_path or "").strip() or DEFAULT_MODEL_ALIAS
    if model_id in {"bosonai/higgs-tts-3-4b", "bosonai/higgs-audio-v3-tts-4b"}:
        return DEFAULT_LOCAL_MODEL
    return model_id


def get_direct_model(model_path: str, device: str) -> Tuple[Any, Any, str]:
    resolved_model = local_model_id(model_path)
    mode = (device or "cpu").strip().lower()
    if mode.startswith("cuda:"):
        device_index = int(mode.split(":", 1)[1])
    else:
        device_index = -1

    cache_key = (resolved_model, mode, device_index)
    if cache_key in DIRECT_MODELS:
        model_obj, tokenizer = DIRECT_MODELS[cache_key]
        return model_obj, tokenizer, resolved_model

    tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True)
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if mode.startswith("cuda"):
        kwargs["dtype"] = resolve_cuda_dtype(mode)
    else:
        kwargs["dtype"] = torch.float32

    model_obj = AutoModelForCausalLM.from_pretrained(resolved_model, **kwargs).eval()
    if mode.startswith("cuda") or mode == "mps":
        model_obj = model_obj.to(device)

    DIRECT_MODELS[cache_key] = (model_obj, tokenizer)
    return model_obj, tokenizer, resolved_model


def load_normalizer(normaliser_path: Path) -> Optional[Callable[[str], str]]:
    if not normaliser_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("normaliser", normaliser_path)
        if spec is None or spec.loader is None:
            logger.error("Nem sikerült a normalizáló betöltőjét előkészíteni: %s", normaliser_path)
            return None
        normaliser_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(normaliser_module)  # type: ignore[attr-defined]
        if hasattr(normaliser_module, "normalize"):
            return getattr(normaliser_module, "normalize")
        logger.warning("A normalizáló nem exportál 'normalize' függvényt: %s", normaliser_path)
    except Exception as exc:
        logger.error("Normalizáló betöltése sikertelen (%s): %s", normaliser_path, exc, exc_info=True)
    return None


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
        logger.warning("EQ konfiguráció nem található: %s. EQ kihagyva.", eq_path)
        return None

    try:
        with open(eq_path, "r", encoding="utf-8") as handle:
            eq_data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("EQ konfiguráció beolvasása sikertelen (%s): %s", eq_path, exc)
        return None

    points = eq_data.get("points")
    if not isinstance(points, list) or not points:
        logger.warning("EQ konfiguráció nem tartalmaz érvényes pontokat: %s", eq_path)
        return None

    parsed_points: List[Dict[str, float]] = []
    for raw_point in points:
        try:
            freq = float(raw_point["frequency_hz"])
            gain_db = float(raw_point.get("gain_db", 0.0))
        except (KeyError, TypeError, ValueError):
            logger.warning("Hibás EQ pont kihagyva: %s", raw_point)
            continue
        parsed_points.append({"frequency_hz": max(0.0, freq), "gain_db": gain_db})

    if not parsed_points:
        logger.warning("Nincs felhasználható EQ pont: %s", eq_path)
        return None

    parsed_points.sort(key=lambda item: item["frequency_hz"])
    return {
        "points": parsed_points,
        "global_gain_db": float(eq_data.get("global_gain_db", 0.0)),
        "source_path": str(eq_path),
    }


def apply_eq_curve_to_audio(
    audio: np.ndarray,
    sample_rate: int,
    eq_config: Optional[Dict[str, object]],
) -> np.ndarray:
    if eq_config is None or audio.size == 0:
        return audio

    points = eq_config.get("points")
    if not points:
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
        unique_freqs.append(float(freq))
        unique_gains.append(float(gain))

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

    return np.clip(equalized_audio, -1.0, 1.0).astype(np.float32, copy=False)


def token(category: str, value: str) -> str:
    return f"<|{category}:{value}|>"


def compose_prompt(
    text: str,
    emotion: str,
    style: str,
    speed: str,
    pitch: str,
    expressive: str,
    manual_prefix: str,
) -> str:
    delivery_tokens: List[str] = []
    if emotion and emotion != "none":
        delivery_tokens.append(token("emotion", emotion))
    if style and style.strip():
        delivery_tokens.append(token("style", style.strip()))
    for value in (speed, pitch, expressive):
        if value and value != "none":
            delivery_tokens.append(token("prosody", value))
    if manual_prefix and manual_prefix.strip():
        delivery_tokens.append(manual_prefix.strip())
    return f"{''.join(delivery_tokens)}{text.strip()}".strip()


def prepare_reference_array(
    audio: np.ndarray,
    sample_rate: int,
    expand_if_shorter_seconds: float,
    expand_target_seconds: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if array.ndim != 1:
        raise ValueError(f"A referenciahang shape-je nem támogatott: {array.shape}")

    original_samples = int(array.shape[0])
    original_seconds = original_samples / float(sample_rate) if sample_rate else 0.0
    repeats = 1
    expanded = False

    if sample_rate > 0 and 0 < original_seconds < expand_if_shorter_seconds:
        target_samples = int(expand_target_seconds * sample_rate)
        repeats = max(1, int(np.ceil(target_samples / max(original_samples, 1))))
        array = np.tile(array, repeats)
        expanded = True

    metadata = {
        "original_seconds": round(original_seconds, 4),
        "processed_seconds": round(array.shape[0] / float(sample_rate), 4) if sample_rate else 0.0,
        "sample_rate": int(sample_rate),
        "expanded": expanded,
        "repeats": repeats,
        "expand_if_shorter_seconds": expand_if_shorter_seconds,
        "expand_target_seconds": expand_target_seconds,
    }
    return array, metadata


def prepare_output_directories(args: argparse.Namespace, config: Dict[str, object]) -> None:
    cfg_dirs = config["DIRECTORIES"]
    cfg_subdirs = config["PROJECT_SUBDIRS"]
    full_project_path = (PROJECT_ROOT / cfg_dirs["workdir"]) / args.project_name

    translated_splits_dir = cfg_subdirs["translated_splits"]
    json_subdir = cfg_subdirs["translated"]
    if args.input_directory_override:
        temp_dir = cfg_subdirs.get("temp")
        if temp_dir:
            json_subdir = temp_dir
            logger.info("A translated JSON mappa helyett a temp (%s) kerül felhasználásra.", temp_dir)
        else:
            logger.warning("A config nem tartalmaz 'temp' kulcsot. Marad a translated JSON mappa.")

    args.full_project_path = full_project_path
    args.output_dir = str(full_project_path / translated_splits_dir)
    args.output_dir_noise = str(full_project_path / cfg_subdirs["noice_splits"])
    args.input_wav_dir = full_project_path / cfg_subdirs["separated_audio_speech"]
    args.input_json_dir = full_project_path / json_subdir

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir_noise).mkdir(parents=True, exist_ok=True)


def save_noise_segments(
    args: argparse.Namespace,
    audio_data: np.ndarray,
    sample_rate: int,
    segments: List[Dict[str, object]],
) -> None:
    noise_dir = Path(args.output_dir_noise)
    noise_dir.mkdir(parents=True, exist_ok=True)

    last_end_time = 0.0
    for segment in segments:
        start_time = segment.get("start")
        if isinstance(start_time, (int, float)) and start_time > last_end_time:
            noise_output_path = noise_dir / f"{time_to_filename_str(last_end_time)}_{time_to_filename_str(start_time)}.wav"
            if not noise_output_path.exists():
                sf.write(
                    noise_output_path,
                    audio_data[int(last_end_time * sample_rate) : int(start_time * sample_rate)],
                    sample_rate,
                )

        end_time = segment.get("end")
        if isinstance(end_time, (int, float)):
            last_end_time = max(last_end_time, float(end_time))

    duration_seconds = len(audio_data) / sample_rate
    if duration_seconds > last_end_time:
        noise_output_path = noise_dir / f"{time_to_filename_str(last_end_time)}_{time_to_filename_str(duration_seconds)}.wav"
        if not noise_output_path.exists():
            sf.write(noise_output_path, audio_data[int(last_end_time * sample_rate) :], sample_rate)


def parse_generation_overrides(raw_json: str) -> Dict[str, Any]:
    if not raw_json.strip():
        return {}
    parsed = json.loads(raw_json)
    if not isinstance(parsed, dict):
        raise ValueError("A raw_json csak JSON objektum lehet.")
    generation = parsed.pop("generation", None)
    if generation is not None:
        if not isinstance(generation, dict):
            raise ValueError("A raw_json 'generation' mezője csak objektum lehet.")
        parsed.update(generation)
    return parsed


def build_generation_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens) if args.max_new_tokens > 0 else 1024,
        "temperature": float(args.temperature) if args.temperature >= 0 else 0.8,
        "top_p": None if args.top_p in ("", None, 0) else float(args.top_p),
        "top_k": None if args.top_k in ("", None, 0) else int(args.top_k),
    }
    overrides = parse_generation_overrides(args.raw_json)
    supported_keys = {"max_new_tokens", "temperature", "top_p", "top_k"}
    unknown_keys = sorted(set(overrides) - supported_keys)
    if unknown_keys:
        raise ValueError(f"Nem támogatott raw_json mezők: {', '.join(unknown_keys)}")
    generation_kwargs.update({key: overrides[key] for key in supported_keys if key in overrides})
    generation_kwargs["max_new_tokens"] = int(generation_kwargs["max_new_tokens"])
    generation_kwargs["temperature"] = float(generation_kwargs["temperature"])
    generation_kwargs["top_p"] = None if generation_kwargs["top_p"] in ("", None, 0) else float(generation_kwargs["top_p"])
    generation_kwargs["top_k"] = None if generation_kwargs["top_k"] in ("", None, 0) else int(generation_kwargs["top_k"])
    return generation_kwargs


def generate_audio_for_segment(
    model: Any,
    tokenizer: Any,
    prompt: str,
    reference_audio: np.ndarray,
    reference_sample_rate: int,
    reference_text: str,
    generation_kwargs: Dict[str, Any],
    seed: int,
) -> np.ndarray:
    if seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    output = model.generate_speech(
        prompt.strip(),
        tokenizer,
        reference_audio=torch.from_numpy(reference_audio),
        reference_sample_rate=int(reference_sample_rate),
        reference_text=reference_text.strip() or None,
        **generation_kwargs,
    )
    audio = output.detach().cpu().numpy() if hasattr(output, "detach") else np.asarray(output)
    return np.squeeze(np.asarray(audio, dtype=np.float32))


def process_segment(
    segment: Dict[str, object],
    args: argparse.Namespace,
    audio_data: np.ndarray,
    sample_rate: int,
    model: Any,
    tokenizer: Any,
    normalize_fn: Optional[Callable[[str], str]],
    eq_config: Optional[Dict[str, object]],
    generation_kwargs: Dict[str, Any],
) -> Tuple[bool, str]:
    start_time = segment.get("start")
    end_time = segment.get("end")
    translated_text_raw = segment.get("translated_text") or segment.get("translates_text") or ""
    translated_text = str(translated_text_raw).strip()
    reference_text = (
        str(segment.get("text") or segment.get("source_text") or segment.get("original_text") or "").strip()
    )

    if not all(isinstance(value, (int, float)) for value in (start_time, end_time)):
        return False, "Hiányzó vagy érvénytelen időbélyeg."
    if not translated_text:
        return False, "Üres translated_text."

    start_sample = int(float(start_time) * sample_rate)
    end_sample = int(float(end_time) * sample_rate)
    end_sample = min(end_sample, len(audio_data))
    if start_sample >= end_sample:
        return False, "Érvénytelen audió intervallum."

    filename = f"{time_to_filename_str(float(start_time))}_{time_to_filename_str(float(end_time))}.wav"
    output_path = Path(args.output_dir) / filename
    if output_path.exists() and not args.overwrite:
        logger.debug("Kihagyva (létező fájl): %s", output_path)
        return True, filename

    gen_text = translated_text
    if normalize_fn is not None:
        gen_text = normalize_fn(gen_text)
    prompt = compose_prompt(
        text=gen_text,
        emotion=args.emotion,
        style=args.style,
        speed=args.speed,
        pitch=args.pitch,
        expressive=args.expressive,
        manual_prefix=args.manual_prefix,
    )
    if not prompt:
        return False, "Üres prompt a normalizálás után."

    ref_chunk = audio_data[start_sample:end_sample]
    ref_chunk = apply_eq_curve_to_audio(ref_chunk, sample_rate, eq_config)
    if args.normalize_ref_audio:
        ref_chunk = normalize_peak(ref_chunk.copy(), args.ref_audio_peak)

    short_ref_threshold = 0.0 if args.disable_short_ref_expansion else float(args.short_ref_threshold)
    short_ref_target = float(args.short_ref_target_seconds)
    prepared_ref, ref_metadata = prepare_reference_array(
        audio=ref_chunk,
        sample_rate=sample_rate,
        expand_if_shorter_seconds=short_ref_threshold,
        expand_target_seconds=short_ref_target,
    )

    logger.debug(
        "Generálás: %s | ref %.3fs -> %.3fs | repeats=%s | expanded=%s",
        filename,
        ref_metadata["original_seconds"],
        ref_metadata["processed_seconds"],
        ref_metadata["repeats"],
        ref_metadata["expanded"],
    )

    generated_audio = generate_audio_for_segment(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        reference_audio=prepared_ref,
        reference_sample_rate=sample_rate,
        reference_text=reference_text,
        generation_kwargs=generation_kwargs,
        seed=args.seed,
    )

    output_sample_rate = int(getattr(model.config, "sample_rate", 24000))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, generated_audio, output_sample_rate)
    return True, filename


def process_project(args: argparse.Namespace) -> None:
    with open(PROJECT_ROOT / "config.json", "r", encoding="utf-8") as handle:
        config_data = json.load(handle)

    prepare_output_directories(args, config_data)

    wav_files = sorted(Path(args.input_wav_dir).glob("*.wav"))
    json_files = sorted(Path(args.input_json_dir).glob("*.json"))
    if not wav_files or not json_files:
        raise FileNotFoundError("Nem található megfelelő .wav vagy .json a projektben.")

    input_wav_path = wav_files[0]
    input_json_path = json_files[0]
    logger.info("Bemeneti wav: %s", input_wav_path)
    logger.info("Bemeneti json: %s", input_json_path)

    with open(input_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    segments = data.get("segments", [])
    segments.sort(key=lambda item: item.get("start", 0.0))

    filtered_segments: List[Dict[str, object]] = []
    skipped_segments = 0
    for segment in segments:
        translated_text = segment.get("translated_text") or segment.get("translates_text")
        if not translated_text or not str(translated_text).strip():
            skipped_segments += 1
            continue
        if "translated_text" not in segment:
            segment = dict(segment)
            segment["translated_text"] = translated_text
        filtered_segments.append(segment)

    if skipped_segments:
        logger.info("Fordított szöveg nélküli szegmensek kihagyva: %s db", skipped_segments)

    if args.max_segments is not None:
        filtered_segments = filtered_segments[: args.max_segments]
        logger.info("Szegmensek limitálva: %s", len(filtered_segments))

    if not filtered_segments:
        logger.warning("Nincs feldolgozható szegmens a projektben.")
        return

    audio_data, sample_rate = sf.read(input_wav_path, always_2d=False)
    logger.info("Referencia audió: %s Hz, %.2f mp", sample_rate, len(audio_data) / sample_rate)
    save_noise_segments(args, audio_data, sample_rate, filtered_segments)

    eq_config_path = args.eq_config or str(DEFAULT_EQ_CONFIG_PATH)
    if eq_config_path and not Path(eq_config_path).exists():
        logger.warning("EQ konfiguráció nem található: %s. EQ kikapcsolva.", eq_config_path)
        eq_config_path = None
    eq_config = load_eq_curve_config(eq_config_path)

    cfg_dirs = config_data["DIRECTORIES"]
    normaliser_path = Path(PROJECT_ROOT / cfg_dirs["normalisers"]) / args.norm / "normaliser.py"
    normalize_fn = load_normalizer(normaliser_path)
    if normalize_fn is None:
        logger.warning("A normalizáló nem tölthető be, nyers translated_text kerül felhasználásra: %s", normaliser_path)

    resolved_device = resolve_device(args.device)
    args.device = resolved_device
    logger.info("Eszköz: %s", resolved_device)

    model_source = args.model_dir or args.model_path
    model, tokenizer, resolved_model = get_direct_model(model_source, resolved_device)
    logger.info("Betöltött modell: %s", resolved_model)

    generation_kwargs = build_generation_kwargs(args)
    logger.info(
        "Generálási paraméterek: temperature=%s, top_p=%s, top_k=%s, max_new_tokens=%s",
        generation_kwargs["temperature"],
        generation_kwargs["top_p"],
        generation_kwargs["top_k"],
        generation_kwargs["max_new_tokens"],
    )

    successful = 0
    failed = 0
    failed_segments: List[Tuple[str, str]] = []
    for segment in filtered_segments:
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", 0.0)
        filename = f"{time_to_filename_str(float(start_time))}_{time_to_filename_str(float(end_time))}.wav"
        try:
            ok, message = process_segment(
                segment=segment,
                args=args,
                audio_data=audio_data,
                sample_rate=sample_rate,
                model=model,
                tokenizer=tokenizer,
                normalize_fn=normalize_fn,
                eq_config=eq_config,
                generation_kwargs=generation_kwargs,
            )
            if ok:
                successful += 1
                logger.info("Kész: %s", message)
            else:
                failed += 1
                failed_segments.append((filename, message))
                logger.error("Sikertelen: %s -> %s", filename, message)
        except Exception as exc:
            failed += 1
            failed_segments.append((filename, str(exc)))
            logger.error("Hiba a(z) %s szegmens feldolgozása közben: %s", filename, exc, exc_info=True)

    print()
    print("Végső statisztika:", flush=True)
    print(f"Összes szegmens: {len(filtered_segments)}", flush=True)
    print(f"Sikeres: {successful}", flush=True)
    print(f"Sikertelen: {failed}", flush=True)
    if failed_segments:
        print("Sikertelen szegmensek:", flush=True)
        for segment_name, reason in failed_segments:
            print(f"- {segment_name}: {reason}", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Higgs TTS script fordított szegmensek generálásához, a projekt pipeline-jába illesztve.",
    )
    parser.add_argument("project_name", type=str, help="A projekt könyvtárának neve a workdir alatt.")
    parser.add_argument("--norm", type=str, required=True, help="A használt normalizálási profil neve.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_ALIAS, help="HF modellazonosító vagy alias.")
    parser.add_argument("--model_dir", type=str, default=None, help="Lokális modellkönyvtár vagy snapshot útvonala.")
    parser.add_argument("--device", type=str, default="auto", help="Eszköz: auto, cpu, cuda, cuda:0 vagy mps.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Mintavételi hőmérséklet.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p mintavétel.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k mintavétel.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximális generált tokenek száma.")
    parser.add_argument("--seed", type=int, default=-1, help="Véletlenmag. -1 esetén nincs fixálva.")
    parser.add_argument("--eq_config", type=str, default=None, help="EQ konfiguráció JSON útvonala.")
    parser.add_argument(
        "--normalize_ref_audio",
        action="store_true",
        help="A referenciahangot generálás előtt cél peak szintre normalizálja.",
    )
    parser.add_argument(
        "--ref_audio_peak",
        type=float,
        default=0.95,
        help="A normalizált referenciahang cél peak értéke.",
    )
    parser.add_argument(
        "--disable_short_ref_expansion",
        action="store_true",
        help="Kikapcsolja a rövid referenciahangok automatikus ismétlését.",
    )
    parser.add_argument(
        "--short_ref_threshold",
        type=float,
        default=REFERENCE_EXPAND_IF_SHORTER_SECONDS,
        help="Ez alatti referenciahangokat ismétli a script.",
    )
    parser.add_argument(
        "--short_ref_target_seconds",
        type=float,
        default=REFERENCE_EXPAND_TARGET_SECONDS,
        help="A rövid referenciahang bővítésének célhossza másodpercben.",
    )
    parser.add_argument("--emotion", type=str, default="none", help="Globális emotion token érték.")
    parser.add_argument("--style", type=str, default="", help="Globális style token érték.")
    parser.add_argument("--speed", type=str, default="none", help="Globális prosody speed token.")
    parser.add_argument("--pitch", type=str, default="none", help="Globális prosody pitch token.")
    parser.add_argument("--expressive", type=str, default="none", help="Globális prosody expressive token.")
    parser.add_argument("--manual_prefix", type=str, default="", help="Kézzel megadott prompt prefix tokenek.")
    parser.add_argument(
        "--raw_json",
        type=str,
        default="",
        help="Opcionális JSON felülírás a generation paraméterekhez.",
    )
    parser.add_argument(
        "--input_directory_override",
        action="store_true",
        help="A translated JSON mappa helyett a temp alkönyvtárból olvas.",
    )
    parser.add_argument("--max_segments", type=int, default=None, help="Szegmenslimit teszteléshez.")
    parser.add_argument("--overwrite", action="store_true", help="Felülírja a meglévő kimeneti wav fájlokat.")
    add_debug_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    log_level = configure_debug_mode(args.debug, default_level=logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("higgs_tts.log", encoding="utf-8"),
        ],
        force=True,
    )
    logger.setLevel(log_level)
    logger.info("Higgs TTS script indul. Projekt: %s", args.project_name)

    try:
        process_project(args)
    except Exception as exc:  # pragma: no cover
        logger.error("Futás közben hiba történt: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()

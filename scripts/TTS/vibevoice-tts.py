import argparse
import datetime
import json
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers.utils import logging as transformers_logging

# Ensure project root is importable so we can reuse shared tooling.
for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

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


def parse_arguments() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="VibeVoice alapú TTS a fordított sávok szegmensenkénti generálásához.",
    )
    parser.add_argument("project_name", type=str, help="A projekt könyvtár neve a 'workdir' mappán belül.")
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
        "--speaker_name",
        type=str,
        default="Speaker 1",
        help="A szkript szövegéhez használt beszélő címkéje.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Véletlenszám-generátor magja (0 esetén nincs fixálás).",
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


def load_vibevoice_components(args: argparse.Namespace) -> Tuple[VibeVoiceProcessor, VibeVoiceForConditionalGenerationInference]:
    model_path = args.model_dir or args.model_path
    if not model_path:
        raise ValueError("A VibeVoice modell elérési útja nem lett megadva (--model_path).")

    transformers_logging.set_verbosity_info()

    logger.info("Processor betöltése: %s", model_path)
    processor = VibeVoiceProcessor.from_pretrained(model_path)

    device = args.device
    if device == "mps":
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"
        device_map = None
    elif device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl_primary = "flash_attention_2"
        device_map = "cuda"
    else:
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"
        device_map = "cpu"

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
            device_map=device_map,
            attn_implementation=attn_impl_primary,
        )
        if device == "mps":
            model.to("mps")
    except Exception as exc:  # pragma: no cover - environment dependent fallback
        if attn_impl_primary == "flash_attention_2":
            logger.warning(
                "flash_attention_2 betöltése sikertelen (%s). SDPA-ra váltunk. Az audió minőség eltérhet.",
                exc,
            )
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map=("cuda" if device == "cuda" else "cpu"),
                attn_implementation="sdpa",
            )
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
) -> Tuple[bool, Optional[str]]:
    start_time = segment.get("start")
    end_time = segment.get("end")
    original_gen_text = (segment.get("translated_text") or segment.get("text") or "").strip()

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

    ref_chunk = audio_data[start_sample:end_sample]
    ref_chunk = apply_eq_curve_to_audio(ref_chunk, sample_rate, eq_config)
    if args.normalize_ref_audio:
        ref_chunk = normalize_peak(ref_chunk.copy(), args.ref_audio_peak)

    # VibeVoice a referenciamintát fájlból várja, ezért ideiglenes fájlba mentünk.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref_file:
        temp_ref_path = tmp_ref_file.name
    try:
        sf.write(temp_ref_path, ref_chunk, sample_rate)

        formatted_text = ensure_script_format(original_gen_text, args.speaker_name)
        if not formatted_text:
            return False, "Üres generálandó szöveg."

        inputs = prepare_inputs(
            processor=processor,
            text=formatted_text,
            voice_sample_paths=[temp_ref_path],
            device=args.device,
        )

        logger.debug("Generálás indul: %s", filename)
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
        logger.debug("Generálás kész %.2f mp alatt: %s", generation_time, filename)

        speech_outputs = getattr(outputs, "speech_outputs", None)
        if not speech_outputs or speech_outputs[0] is None:
            return False, "A modell nem adott vissza audiót."

        output_path.parent.mkdir(parents=True, exist_ok=True)
        processor.save_audio(speech_outputs[0], output_path=str(output_path))
        return True, None
    except Exception as exc:  # pragma: no cover - runtime dependent
        return False, f"Generálási hiba: {exc}"
    finally:
        try:
            os.remove(temp_ref_path)
        except OSError:
            pass


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

    args.device = resolve_device(args.device)
    set_random_seeds(args.seed, args.device)

    processor, model = load_vibevoice_components(args)

    stats = {
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "total": len(segments),
        "errors": [],
    }

    for index, segment in enumerate(segments, start=1):
        if args.max_segments is not None and stats["successful"] + stats["failed"] >= args.max_segments:
            logger.info("Elérte a megadott max szegmens limitet (%s).", args.max_segments)
            break

        ok, error_msg = process_segment(
            segment=segment,
            args=args,
            audio_data=audio_data,
            sample_rate=sample_rate,
            processor=processor,
            model=model,
            eq_config=eq_config,
        )
        if ok:
            stats["successful"] += 1
        else:
            stats["failed"] += 1
            filename = (
                f"{time_to_filename_str(segment.get('start', 0.0))}_"
                f"{time_to_filename_str(segment.get('end', 0.0))}.wav"
            )
            stats["errors"].append({"filename": filename, "reason": error_msg})
            logger.error("Szegmens feldolgozása sikertelen (%s): %s", filename, error_msg)

        if index % 10 == 0 or index == len(segments):
            logger.info(
                "Haladás: %s/%s | sikeres: %s | sikertelen: %s",
                index,
                len(segments),
                stats["successful"],
                stats["failed"],
            )

    logger.info("=" * 50)
    logger.info(
        "Összegzés\n  - Összes szegmens: %s\n  - Sikeres: %s\n  - Sikertelen: %s",
        stats["total"],
        stats["successful"],
        stats["failed"],
    )
    if stats["errors"]:
        logger.info("Sikertelen szegmensek:")
        for item in stats["errors"]:
            logger.info("  * %s -> %s", item["filename"], item["reason"])
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
    main()

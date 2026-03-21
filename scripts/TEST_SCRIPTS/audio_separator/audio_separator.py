#!/usr/bin/env python3

import argparse
import importlib
import json
import logging
import multiprocessing
import os
import re
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode


DEFAULT_SPEECH_MODELS = "htdemucs_ft.yaml,UVR-MDX-NET-Voc_FT.onnx"
DEFAULT_BACKGROUND_MODELS = "htdemucs_ft.yaml"
DEFAULT_SPEECH_ENSEMBLE = "avg_wave"
DEFAULT_BACKGROUND_ENSEMBLE = "avg_wave"
DEFAULT_BACKGROUND_BLEND = 0.2
CHANNEL_SUFFIXES = ["_FC", "_FL", "_FR", "_SL", "_SR", "_RL", "_RR", "_LFE", "_C", "_L", "_R"]
VOCAL_STEM_PATTERN = re.compile(r"vocal", re.IGNORECASE)
STEM_NAME_PATTERN = re.compile(r"_\(([^)]+)\)")
SUPPORTED_INPUT_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".ac3")


def get_project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        config_candidate = candidate / "config.json"
        if config_candidate.is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")


def parse_model_list(model_arg: str) -> List[str]:
    if not model_arg:
        return []
    return [item.strip() for item in model_arg.split(",") if item.strip()]


def ensure_audio_separator_available() -> None:
    try:
        import_audio_separator_class()
    except ImportError as exc:
        raise RuntimeError(
            "A python-audio-separator csomag nem érhető el ebben a környezetben. "
            "Telepítsd például: pip install \"audio-separator[gpu]\" vagy pip install \"audio-separator[cpu]\""
        ) from exc


def import_audio_separator_class():
    script_dir = str(Path(__file__).resolve().parent)
    original_sys_path = list(sys.path)
    existing_module = sys.modules.get("audio_separator")

    try:
        sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(script_dir).resolve()]
        if existing_module is not None:
            module_file = getattr(existing_module, "__file__", None)
            if module_file and Path(module_file).resolve() == Path(__file__).resolve():
                sys.modules.pop("audio_separator", None)

        separator_module = importlib.import_module("audio_separator.separator")
        return separator_module.Separator
    finally:
        sys.path = original_sys_path


def load_config(project_root: Path) -> Dict[str, object]:
    config_path = project_root / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"A config.json nem található itt: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"A config.json hibás formátumú ({config_path}): {exc}") from exc


def collect_grouped_inputs(input_dir: Path) -> Dict[str, List[str]]:
    files_by_group: Dict[str, List[str]] = {}
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(SUPPORTED_INPUT_EXTENSIONS):
            continue
        base = Path(filename).stem
        group_name = base
        for suffix in CHANNEL_SUFFIXES:
            if base.upper().endswith(suffix):
                group_name = base[: -len(suffix)]
                break
        files_by_group.setdefault(group_name, []).append(filename)
    return files_by_group


def choose_group_representative(group_name: str, file_list: Sequence[str]) -> str:
    for filename in file_list:
        if Path(filename).stem.upper().endswith("_FC"):
            print(f"\nCenter sáv (_FC) kiválasztva a '{group_name}' csoporthoz: {filename}")
            return filename
    selected = file_list[0]
    if len(file_list) > 1:
        print(f"\nNem található '_FC' sáv a '{group_name}' csoportban. Automatikus választás: {selected}")
    return selected


def sanitize_stem_base_name(stem_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem_name.strip()).strip("_") or "stem"


def build_jobs(
    input_dir: Path,
    speech_output_dir: Path,
    background_output_dir: Path,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    speech_jobs: List[Dict[str, str]] = []
    background_jobs: List[Dict[str, str]] = []
    files_by_group = collect_grouped_inputs(input_dir)

    for group_name, file_list in files_by_group.items():
        filename_to_process = choose_group_representative(group_name, file_list)
        input_path = input_dir / filename_to_process
        base_name = Path(filename_to_process).stem

        speech_output_path = speech_output_dir / f"{base_name}_speech.wav"
        background_output_path = background_output_dir / f"{base_name}_non_speech.wav"

        if speech_output_path.exists():
            print(f"Speech kimenet már létezik, kihagyva: {speech_output_path}")
        else:
            speech_jobs.append(
                {
                    "input_path": str(input_path),
                    "base_name": base_name,
                    "output_path": str(speech_output_path),
                }
            )

        if background_output_path.exists():
            print(f"Background kimenet már létezik, kihagyva: {background_output_path}")
        else:
            background_jobs.append(
                {
                    "input_path": str(input_path),
                    "base_name": base_name,
                    "output_path": str(background_output_path),
                    "speech_output_path": str(speech_output_path),
                }
            )

    return speech_jobs, background_jobs


def configure_worker_acceleration(acceleration: str) -> None:
    normalized = acceleration.strip().lower()
    if normalized == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif normalized == "cuda":
        pass
    else:
        raise ValueError(f"Ismeretlen gyorsítási mód: {acceleration}. Használható értékek: auto, cpu")


def extract_stem_name(output_path: str) -> str:
    match = STEM_NAME_PATTERN.search(Path(output_path).name)
    if match:
        return match.group(1)
    return Path(output_path).stem


def is_vocal_stem(stem_name: str) -> bool:
    return VOCAL_STEM_PATTERN.search(stem_name) is not None


def is_pure_speech(
    vocals_np: np.ndarray,
    non_speech_np: np.ndarray,
    eps: float = 1e-9,
    rel_thresh_db: float = -25.0,
    abs_thresh: float = -50.0,
) -> bool:
    def rms_db(x: np.ndarray) -> float:
        rms = np.sqrt(np.mean(x**2) + eps)
        return 20 * np.log10(rms + eps)

    v_db = rms_db(vocals_np)
    n_db = rms_db(non_speech_np)
    return (n_db - v_db) <= rel_thresh_db or n_db <= abs_thresh


def ensure_stereo_array(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=1)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio.astype(np.float32)


def align_audio_arrays(*arrays: np.ndarray) -> List[np.ndarray]:
    prepared = [ensure_stereo_array(array) for array in arrays]
    min_length = min(array.shape[0] for array in prepared)
    return [array[:min_length] for array in prepared]


def blend_background_signal(
    mix: np.ndarray,
    vocals: np.ndarray,
    predicted_background: np.ndarray,
    blend_ratio: float,
) -> np.ndarray:
    residual_background = mix - vocals
    residual_background = np.clip(residual_background, -1.0, 1.0)
    blend_ratio = float(np.clip(blend_ratio, 0.0, 1.0))
    return np.clip(
        blend_ratio * predicted_background + (1.0 - blend_ratio) * residual_background,
        -1.0,
        1.0,
    )


def combine_background_stems(stem_paths: Sequence[str], target_output_path: Path) -> None:
    instrumental_path: Optional[str] = None
    non_vocal_sources: List[np.ndarray] = []
    target_sample_rate: Optional[int] = None
    target_subtype: Optional[str] = None

    for stem_path in stem_paths:
        stem_name = extract_stem_name(stem_path)
        audio, sample_rate = sf.read(stem_path, always_2d=True)
        info = sf.info(stem_path)
        if target_sample_rate is None:
            target_sample_rate = sample_rate
        elif sample_rate != target_sample_rate:
            raise RuntimeError(
                f"Eltérő mintavételi frekvencia a background stemek között: {stem_path} ({sample_rate}) "
                f"!= {target_sample_rate}"
            )
        if target_subtype is None:
            target_subtype = info.subtype

        if stem_name.lower() == "instrumental":
            instrumental_path = stem_path
            continue
        if not is_vocal_stem(stem_name):
            non_vocal_sources.append(audio.astype(np.float32))

    if instrumental_path:
        shutil.copy2(instrumental_path, target_output_path)
        return

    if not non_vocal_sources:
        raise RuntimeError("Nem található nem-vokál stem a háttérsáv összerakásához.")

    combined = np.sum(non_vocal_sources, axis=0, dtype=np.float32)
    combined = np.clip(combined, -1.0, 1.0)
    sf.write(str(target_output_path), combined, target_sample_rate or 44100, subtype=target_subtype)


def resolve_generated_paths(generated_files: Sequence[str], base_dir: Path) -> List[Path]:
    resolved: List[Path] = []
    for item in generated_files:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        resolved.append(candidate)
    return resolved


def postprocess_background_outputs(
    background_jobs: Sequence[Dict[str, str]],
    background_blend: float,
    non_speech_silence: bool,
) -> None:
    for job in background_jobs:
        input_path = Path(job["input_path"])
        speech_path = Path(job["speech_output_path"])
        background_path = Path(job["output_path"])

        if not input_path.exists() or not speech_path.exists() or not background_path.exists():
            continue

        mix, mix_sr = sf.read(str(input_path), always_2d=True)
        speech, speech_sr = sf.read(str(speech_path), always_2d=True)
        predicted_background, background_sr = sf.read(str(background_path), always_2d=True)

        if mix_sr != speech_sr or mix_sr != background_sr:
            logging.warning(
                "Mintavételezési eltérés miatt background blend kihagyva: %s (mix=%s speech=%s bg=%s)",
                input_path.name,
                mix_sr,
                speech_sr,
                background_sr,
            )
            continue

        mix, speech, predicted_background = align_audio_arrays(mix, speech, predicted_background)
        auto_non_speech_silence = is_pure_speech(speech, predicted_background)

        if non_speech_silence or auto_non_speech_silence:
            final_background = np.zeros_like(predicted_background)
        else:
            final_background = blend_background_signal(
                mix=mix,
                vocals=speech,
                predicted_background=predicted_background,
                blend_ratio=background_blend,
            )

        subtype = sf.info(str(background_path)).subtype
        sf.write(str(background_path), final_background, mix_sr, subtype=subtype)


def create_separator(
    *,
    output_dir: Path,
    model_file_dir: Path,
    output_single_stem: Optional[str],
    ensemble_algorithm: str,
    sample_rate: int,
    use_soundfile: bool,
    use_autocast: bool,
    chunk_duration: Optional[float],
    mdx_segment_size: int,
    mdx_overlap: float,
    mdx_batch_size: int,
    mdx_enable_denoise: bool,
    vr_batch_size: int,
    vr_window_size: int,
    vr_aggression: int,
    vr_enable_tta: bool,
    demucs_segment_size: str,
    demucs_shifts: int,
    demucs_overlap: float,
    demucs_segments_enabled: bool,
    mdxc_segment_size: int,
    mdxc_overlap: int,
    mdxc_batch_size: int,
    log_level: int,
):
    Separator = import_audio_separator_class()

    separator = Separator(
        log_level=log_level,
        model_file_dir=str(model_file_dir),
        output_dir=str(output_dir),
        output_format="WAV",
        output_single_stem=output_single_stem,
        sample_rate=sample_rate,
        use_soundfile=use_soundfile,
        use_autocast=use_autocast,
        chunk_duration=chunk_duration,
        ensemble_algorithm=ensemble_algorithm,
        mdx_params={
            "hop_length": 1024,
            "segment_size": mdx_segment_size,
            "overlap": mdx_overlap,
            "batch_size": mdx_batch_size,
            "enable_denoise": mdx_enable_denoise,
        },
        vr_params={
            "batch_size": vr_batch_size,
            "window_size": vr_window_size,
            "aggression": vr_aggression,
            "enable_tta": vr_enable_tta,
            "enable_post_process": False,
            "post_process_threshold": 0.2,
            "high_end_process": False,
        },
        demucs_params={
            "segment_size": demucs_segment_size,
            "shifts": demucs_shifts,
            "overlap": demucs_overlap,
            "segments_enabled": demucs_segments_enabled,
        },
        mdxc_params={
            "segment_size": mdxc_segment_size,
            "override_model_segment_size": False,
            "batch_size": mdxc_batch_size,
            "overlap": mdxc_overlap,
            "pitch_shift": 0,
        },
    )
    return separator


def detect_cuda_device_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    try:
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def resolve_workflow_devices(
    speech_preference: str,
    background_preference: str,
    gpu_count: int,
) -> Dict[str, Optional[object]]:
    speech_pref = speech_preference.strip().lower()
    background_pref = background_preference.strip().lower()

    if speech_pref not in {"auto", "cpu"}:
        raise ValueError(f"Ismeretlen speech gyorsítási mód: {speech_preference}")
    if background_pref not in {"auto", "cpu"}:
        raise ValueError(f"Ismeretlen background gyorsítási mód: {background_preference}")

    speech_mode = "cpu"
    speech_visible_gpu: Optional[str] = None
    if speech_pref == "auto" and gpu_count >= 1:
        speech_mode = "cuda"
        speech_visible_gpu = "0"

    background_mode = "cpu"
    background_visible_gpu: Optional[str] = None
    if background_pref == "auto" and gpu_count >= 2:
        background_mode = "cuda"
        background_visible_gpu = "1"

    return {
        "speech_mode": speech_mode,
        "speech_visible_gpu": speech_visible_gpu,
        "background_mode": background_mode,
        "background_visible_gpu": background_visible_gpu,
    }


def run_speech_workflow(jobs: Sequence[Dict[str, str]], worker_config: Dict[str, object]) -> Dict[str, object]:
    configure_worker_acceleration(str(worker_config["speech_mode"]))
    visible_gpu = worker_config.get("speech_visible_gpu")
    if visible_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpu)
    ensure_audio_separator_available()

    output_dir = Path(str(worker_config["speech_output_dir"]))
    model_dir = Path(str(worker_config["model_file_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    separator = create_separator(
        output_dir=output_dir,
        model_file_dir=model_dir,
        output_single_stem="Vocals",
        ensemble_algorithm=str(worker_config["speech_ensemble_algorithm"]),
        sample_rate=int(worker_config["sample_rate"]),
        use_soundfile=bool(worker_config["use_soundfile"]),
        use_autocast=bool(worker_config["use_autocast"]) and str(worker_config["speech_mode"]) == "cuda",
        chunk_duration=float(worker_config["chunk_duration"]) if worker_config["chunk_duration"] is not None else None,
        mdx_segment_size=int(worker_config["mdx_segment_size"]),
        mdx_overlap=float(worker_config["mdx_overlap"]),
        mdx_batch_size=int(worker_config["mdx_batch_size"]),
        mdx_enable_denoise=bool(worker_config["mdx_enable_denoise"]),
        vr_batch_size=int(worker_config["vr_batch_size"]),
        vr_window_size=int(worker_config["vr_window_size"]),
        vr_aggression=int(worker_config["vr_aggression"]),
        vr_enable_tta=bool(worker_config["vr_enable_tta"]),
        demucs_segment_size=str(worker_config["demucs_segment_size"]),
        demucs_shifts=int(worker_config["demucs_shifts"]),
        demucs_overlap=float(worker_config["demucs_overlap"]),
        demucs_segments_enabled=bool(worker_config["demucs_segments_enabled"]),
        mdxc_segment_size=int(worker_config["mdxc_segment_size"]),
        mdxc_overlap=int(worker_config["mdxc_overlap"]),
        mdxc_batch_size=int(worker_config["mdxc_batch_size"]),
        log_level=int(worker_config["log_level"]),
    )
    speech_models = parse_model_list(str(worker_config["speech_models"]))
    separator.load_model(speech_models if len(speech_models) > 1 else speech_models[0])

    processed: List[str] = []
    for job in jobs:
        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        custom_output_names = {"Vocals": output_path.stem}
        generated_files = separator.separate(job["input_path"], custom_output_names=custom_output_names)
        if not output_path.exists():
            if len(generated_files) == 1 and Path(generated_files[0]).exists():
                generated_path = Path(generated_files[0])
                if generated_path.resolve() != output_path.resolve():
                    if output_path.exists():
                        output_path.unlink()
                    shutil.move(str(generated_path), str(output_path))
            elif generated_files:
                existing_generated = [Path(item) for item in generated_files if Path(item).exists()]
                vocal_like = [item for item in existing_generated if is_vocal_stem(extract_stem_name(str(item))) or "unknown" in item.name.lower()]
                if len(vocal_like) == 1:
                    source_path = vocal_like[0]
                    if output_path.exists():
                        output_path.unlink()
                    shutil.move(str(source_path), str(output_path))
        if not output_path.exists():
            raise RuntimeError(f"A speech workflow nem hozta létre a várt fájlt: {output_path}. Kimenetek: {generated_files}")
        processed.append(str(output_path))

    return {
        "workflow": "speech",
        "processed": processed,
        "count": len(processed),
        "device": f"cuda:{visible_gpu}" if visible_gpu is not None else "cpu",
    }


def run_background_workflow(jobs: Sequence[Dict[str, str]], worker_config: Dict[str, object]) -> Dict[str, object]:
    configure_worker_acceleration(str(worker_config["background_mode"]))
    visible_gpu = worker_config.get("background_visible_gpu")
    if visible_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpu)
    ensure_audio_separator_available()

    output_dir = Path(str(worker_config["background_output_dir"]))
    temp_root = Path(str(worker_config["temp_dir"])) / "audio_separator_background"
    model_dir = Path(str(worker_config["model_file_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    separator = create_separator(
        output_dir=temp_root,
        model_file_dir=model_dir,
        output_single_stem=None,
        ensemble_algorithm=str(worker_config["background_ensemble_algorithm"]),
        sample_rate=int(worker_config["sample_rate"]),
        use_soundfile=bool(worker_config["use_soundfile"]),
        use_autocast=bool(worker_config["use_autocast"]) and str(worker_config["background_mode"]) == "cuda",
        chunk_duration=float(worker_config["chunk_duration"]) if worker_config["chunk_duration"] is not None else None,
        mdx_segment_size=int(worker_config["mdx_segment_size"]),
        mdx_overlap=float(worker_config["mdx_overlap"]),
        mdx_batch_size=int(worker_config["mdx_batch_size"]),
        mdx_enable_denoise=bool(worker_config["mdx_enable_denoise"]),
        vr_batch_size=int(worker_config["vr_batch_size"]),
        vr_window_size=int(worker_config["vr_window_size"]),
        vr_aggression=int(worker_config["vr_aggression"]),
        vr_enable_tta=bool(worker_config["vr_enable_tta"]),
        demucs_segment_size=str(worker_config["demucs_segment_size"]),
        demucs_shifts=int(worker_config["demucs_shifts"]),
        demucs_overlap=float(worker_config["demucs_overlap"]),
        demucs_segments_enabled=bool(worker_config["demucs_segments_enabled"]),
        mdxc_segment_size=int(worker_config["mdxc_segment_size"]),
        mdxc_overlap=int(worker_config["mdxc_overlap"]),
        mdxc_batch_size=int(worker_config["mdxc_batch_size"]),
        log_level=int(worker_config["log_level"]),
    )
    background_models = parse_model_list(str(worker_config["background_models"]))
    separator.load_model(background_models if len(background_models) > 1 else background_models[0])

    processed: List[str] = []
    keep_stems = bool(worker_config["keep_intermediate_stems"])
    for job in jobs:
        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scratch_dir = temp_root / sanitize_stem_base_name(job["base_name"])
        scratch_dir.mkdir(parents=True, exist_ok=True)

        separator.output_dir = str(scratch_dir)
        if separator.model_instance is not None:
            separator.model_instance.output_dir = str(scratch_dir)

        generated_files = separator.separate(job["input_path"])
        resolved_generated_files = resolve_generated_paths(generated_files, scratch_dir)
        combine_background_stems([str(path) for path in resolved_generated_files], output_path)
        processed.append(str(output_path))

        if not keep_stems:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    return {
        "workflow": "background",
        "processed": processed,
        "count": len(processed),
        "device": f"cuda:{visible_gpu}" if visible_gpu is not None else "cpu",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Beszéd és háttér audió szétválasztása python-audio-separator modellekkel, külön workflow-kban."
    )
    parser.add_argument("-p", "--project", required=True, help='A projekt neve a "workdir" könyvtáron belül.')
    parser.add_argument(
        "--speech_models",
        default=DEFAULT_SPEECH_MODELS,
        help=(
            "A speaker workflow modelljei vesszővel elválasztva. "
            f"Alapértelmezés: {DEFAULT_SPEECH_MODELS}"
        ),
    )
    parser.add_argument(
        "--speech_ensemble_algorithm",
        default=DEFAULT_SPEECH_ENSEMBLE,
        help=f"A speaker workflow ensemble algoritmusa. Alapértelmezés: {DEFAULT_SPEECH_ENSEMBLE}",
    )
    parser.add_argument(
        "--background_models",
        default=DEFAULT_BACKGROUND_MODELS,
        help=(
            "A background workflow modelljei vesszővel elválasztva. "
            f"Alapértelmezés: {DEFAULT_BACKGROUND_MODELS}"
        ),
    )
    parser.add_argument(
        "--background_ensemble_algorithm",
        default=DEFAULT_BACKGROUND_ENSEMBLE,
        help=f"A background workflow ensemble algoritmusa. Alapértelmezés: {DEFAULT_BACKGROUND_ENSEMBLE}",
    )
    parser.add_argument(
        "--background_blend",
        type=float,
        default=DEFAULT_BACKGROUND_BLEND,
        help=f"0-1 közötti keverési arány a modell háttérsávja és a mix-vocals residual között. Alapértelmezés: {DEFAULT_BACKGROUND_BLEND}",
    )
    parser.add_argument(
        "--non_speech_silence",
        action="store_true",
        help="Ha aktiválva, a háttérsáv tiszta beszédnél vagy mindig csendesítve lesz.",
    )
    parser.add_argument(
        "--speech_acceleration",
        choices=["auto", "cpu"],
        default="auto",
        help='A speaker workflow gyorsítása. "auto" engedi a GPU használatát, "cpu" letiltja.',
    )
    parser.add_argument(
        "--background_acceleration",
        choices=["auto", "cpu"],
        default="auto",
        help='A background workflow gyorsítása. "auto" esetén a 2. GPU-t használja, ha elérhető, különben CPU-ra esik vissza.',
    )
    parser.add_argument(
        "--run_sequentially",
        action="store_true",
        help="A két workflow ne párhuzamosan, hanem egymás után fusson.",
    )
    parser.add_argument(
        "--model_file_dir",
        default=None,
        help="Opcionális modell-cache könyvtár. Ha nincs megadva, a projekt temp könyvtárában jön létre.",
    )
    parser.add_argument(
        "--chunk_duration",
        type=float,
        default=None,
        help="Hosszú fájlok chunkolt feldolgozása másodpercben. Ha nincs megadva, nincs extra chunkolás.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Kimeneti mintavételi frekvencia. Alapértelmezés: 44100",
    )
    parser.add_argument(
        "--use_soundfile",
        action="store_true",
        help="Kimeneti fájlíráshoz a soundfile backend használata.",
    )
    parser.add_argument(
        "--use_autocast",
        action="store_true",
        help="Ha a workflow GPU-t használ, bekapcsolja a PyTorch autocastot.",
    )
    parser.add_argument(
        "--keep_intermediate_stems",
        action="store_true",
        help="Megtartja a background workflow köztes stem-fájljait a temp mappában.",
    )
    parser.add_argument("--mdx_segment_size", type=int, default=256, help="MDX segment size.")
    parser.add_argument("--mdx_overlap", type=float, default=0.25, help="MDX overlap.")
    parser.add_argument("--mdx_batch_size", type=int, default=1, help="MDX batch size.")
    parser.add_argument("--mdx_enable_denoise", action="store_true", help="MDX denoise bekapcsolása.")
    parser.add_argument("--vr_batch_size", type=int, default=1, help="VR batch size.")
    parser.add_argument("--vr_window_size", type=int, default=512, help="VR window size.")
    parser.add_argument("--vr_aggression", type=int, default=5, help="VR aggression.")
    parser.add_argument("--vr_enable_tta", action="store_true", help="VR TTA bekapcsolása.")
    parser.add_argument(
        "--demucs_segment_size",
        default="Default",
        help='Demucs segment size. Példák: "Default", "10", "50".',
    )
    parser.add_argument("--demucs_shifts", type=int, default=2, help="Demucs shifts.")
    parser.add_argument("--demucs_overlap", type=float, default=0.25, help="Demucs overlap.")
    parser.add_argument(
        "--no_demucs_segments",
        action="store_true",
        help="Letiltja a Demucs split/segments módját.",
    )
    parser.add_argument("--mdxc_segment_size", type=int, default=256, help="MDXC segment size.")
    parser.add_argument("--mdxc_overlap", type=int, default=8, help="MDXC overlap.")
    parser.add_argument("--mdxc_batch_size", type=int, default=1, help="MDXC batch size.")
    add_debug_argument(parser)
    return parser.parse_args()


def run_parallel_workflows(
    *,
    speech_jobs: Sequence[Dict[str, str]],
    background_jobs: Sequence[Dict[str, str]],
    worker_config: Dict[str, object],
    run_sequentially: bool,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []

    if run_sequentially:
        if speech_jobs:
            results.append(run_speech_workflow(speech_jobs, worker_config))
        if background_jobs:
            results.append(run_background_workflow(background_jobs, worker_config))
        return results

    spawn_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=spawn_context) as executor:
        futures = []
        if speech_jobs:
            futures.append(executor.submit(run_speech_workflow, speech_jobs, worker_config))
        if background_jobs:
            futures.append(executor.submit(run_background_workflow, background_jobs, worker_config))
        for future in futures:
            results.append(future.result())
    return results


def main() -> None:
    args = parse_args()
    log_level = configure_debug_mode(args.debug)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        ensure_audio_separator_available()
        project_root = get_project_root()
        config = load_config(project_root)
    except Exception as exc:
        print(f"Hiba: {exc}")
        sys.exit(1)

    try:
        directories = config["DIRECTORIES"]
        project_subdirs = config["PROJECT_SUBDIRS"]
        workdir_name = directories["workdir"]
        project_dir = project_root / workdir_name / args.project
        input_dir = project_dir / project_subdirs["extracted_audio"]
        speech_output_dir = project_dir / project_subdirs["separated_audio_speech"]
        background_output_dir = project_dir / project_subdirs["separated_audio_background"]
        temp_dir = project_dir / project_subdirs["temp"]
    except KeyError as exc:
        print(f"Hiba: Hiányzó kulcs a config.json-ban: {exc}")
        sys.exit(1)

    if not project_dir.is_dir():
        print(f"Hiba: A projekt könyvtár nem létezik: {project_dir}")
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"Hiba: A bemeneti könyvtár nem létezik: {input_dir}")
        sys.exit(1)

    speech_output_dir.mkdir(parents=True, exist_ok=True)
    background_output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    speech_models = parse_model_list(args.speech_models)
    background_models = parse_model_list(args.background_models)
    if not speech_models:
        print("Hiba: Legalább egy speech modellt meg kell adni.")
        sys.exit(1)
    if not background_models:
        print("Hiba: Legalább egy background modellt meg kell adni.")
        sys.exit(1)

    speech_jobs, background_jobs = build_jobs(input_dir, speech_output_dir, background_output_dir)
    if not speech_jobs and not background_jobs:
        print("Nincs feldolgozandó fájl: minden elvárt kimenet már létezik.")
        return

    gpu_count = detect_cuda_device_count()
    try:
        resolved_devices = resolve_workflow_devices(
            speech_preference=args.speech_acceleration,
            background_preference=args.background_acceleration,
            gpu_count=gpu_count,
        )
    except ValueError as exc:
        print(f"Hiba: {exc}")
        sys.exit(1)

    print(
        "GPU detektálás: "
        f"{gpu_count} CUDA eszköz. "
        f"Speech workflow: {resolved_devices['speech_mode']}"
        + (f" (GPU {resolved_devices['speech_visible_gpu']})" if resolved_devices["speech_visible_gpu"] is not None else "")
        + ", "
        f"background workflow: {resolved_devices['background_mode']}"
        + (f" (GPU {resolved_devices['background_visible_gpu']})" if resolved_devices["background_visible_gpu"] is not None else "")
    )

    model_file_dir = Path(args.model_file_dir) if args.model_file_dir else temp_dir / "audio_separator_models"
    worker_config: Dict[str, object] = {
        "speech_models": args.speech_models,
        "background_models": args.background_models,
        "speech_ensemble_algorithm": args.speech_ensemble_algorithm,
        "background_ensemble_algorithm": args.background_ensemble_algorithm,
        "background_blend": args.background_blend,
        "non_speech_silence": args.non_speech_silence,
        "speech_acceleration": args.speech_acceleration,
        "background_acceleration": args.background_acceleration,
        "speech_mode": resolved_devices["speech_mode"],
        "speech_visible_gpu": resolved_devices["speech_visible_gpu"],
        "background_mode": resolved_devices["background_mode"],
        "background_visible_gpu": resolved_devices["background_visible_gpu"],
        "speech_output_dir": str(speech_output_dir),
        "background_output_dir": str(background_output_dir),
        "temp_dir": str(temp_dir),
        "model_file_dir": str(model_file_dir),
        "sample_rate": args.sample_rate,
        "use_soundfile": args.use_soundfile,
        "use_autocast": args.use_autocast,
        "keep_intermediate_stems": args.keep_intermediate_stems,
        "chunk_duration": args.chunk_duration,
        "mdx_segment_size": args.mdx_segment_size,
        "mdx_overlap": args.mdx_overlap,
        "mdx_batch_size": args.mdx_batch_size,
        "mdx_enable_denoise": args.mdx_enable_denoise,
        "vr_batch_size": args.vr_batch_size,
        "vr_window_size": args.vr_window_size,
        "vr_aggression": args.vr_aggression,
        "vr_enable_tta": args.vr_enable_tta,
        "demucs_segment_size": args.demucs_segment_size,
        "demucs_shifts": args.demucs_shifts,
        "demucs_overlap": args.demucs_overlap,
        "demucs_segments_enabled": not args.no_demucs_segments,
        "mdxc_segment_size": args.mdxc_segment_size,
        "mdxc_overlap": args.mdxc_overlap,
        "mdxc_batch_size": args.mdxc_batch_size,
        "log_level": log_level,
    }

    try:
        results = run_parallel_workflows(
            speech_jobs=speech_jobs,
            background_jobs=background_jobs,
            worker_config=worker_config,
            run_sequentially=args.run_sequentially,
        )
        if background_jobs:
            postprocess_background_outputs(
                background_jobs=background_jobs,
                background_blend=args.background_blend,
                non_speech_silence=args.non_speech_silence,
            )
    except Exception:
        print("Hiba történt a szeparálás közben:")
        traceback.print_exc()
        sys.exit(1)

    for result in results:
        print(f"{result['workflow']} workflow kész [{result['device']}], feldolgozott fájlok: {result['count']}")
        for output_path in result["processed"]:
            print(f" - {output_path}")


if __name__ == "__main__":
    main()

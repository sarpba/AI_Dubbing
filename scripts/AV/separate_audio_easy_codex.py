#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode


def extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract stereo PCM audio from a media file using ffmpeg."""
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ac",
        "2",
        "-ar",
        "44100",
        "-vn",
        audio_path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Hiba történt az audio kivonása közben: {video_path}")
        print(result.stderr.decode())
        return False
    if not os.path.exists(audio_path):
        print(f"Az audio fájl nem jött létre: {audio_path}")
        return False
    if os.path.getsize(audio_path) == 0:
        print(f"Az audio fájl üres: {audio_path}")
        return False
    return True


def save_audio_torchaudio(audio_data: np.ndarray, path: str, sample_rate: int = 44100) -> None:
    """Save audio as 16-bit PCM WAV using torchaudio with basic normalization."""
    if audio_data.ndim == 1:
        audio_data = np.expand_dims(audio_data, axis=0)

    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    audio_data = audio_data.astype(np.float32)
    tensor = torch.from_numpy(audio_data)

    if tensor.shape[0] == 1:
        tensor = tensor.repeat(2, 1)

    torchaudio.save(path, tensor, sample_rate, encoding="PCM_S", bits_per_sample=16)


def is_pure_speech(
    vocals_np: np.ndarray,
    non_speech_np: np.ndarray,
    eps: float = 1e-9,
    rel_thresh_db: float = -25.0,
    abs_thresh: float = -50.0,
) -> bool:
    """Basic pure speech heuristic based on RMS ratios between Demucs stems."""

    def rms_db(x: np.ndarray) -> float:
        rms = np.sqrt(np.mean(x**2) + eps)
        return 20 * np.log10(rms + eps)

    v_db = rms_db(vocals_np)
    n_db = rms_db(non_speech_np)

    rel_ok = (n_db - v_db) <= rel_thresh_db
    abs_ok = n_db <= abs_thresh

    return rel_ok or abs_ok


def ensure_stereo(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[0] == 1:
        print("Mono bemenet észlelve, sztereóvá alakítás.")
        waveform = waveform.repeat(2, 1)
    return waveform


def resample_if_needed(waveform: torch.Tensor, sample_rate: int, target_rate: int = 44100) -> Tuple[torch.Tensor, int]:
    if sample_rate == target_rate:
        return waveform, sample_rate
    print(f"Mintavételezés átalakítása {sample_rate} Hz-ről {target_rate} Hz-re.")
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)
    return resampler(waveform), target_rate


def build_chunk_weight(
    chunk_length: int,
    start: int,
    end: int,
    total: int,
    overlap: int,
    device: torch.device,
) -> torch.Tensor:
    weight = torch.ones(1, chunk_length, dtype=torch.float32, device=device)
    if overlap <= 0 or chunk_length == 0:
        return weight

    min_gain = 1e-3
    overlap = min(overlap, chunk_length)

    if start > 0:
        fade_in = torch.linspace(min_gain, 1.0, steps=overlap, device=device)
        weight[:, :overlap] *= fade_in
    if end < total:
        fade_out = torch.linspace(1.0, min_gain, steps=overlap, device=device)
        weight[:, -overlap:] *= fade_out

    return weight


def separate_with_model(
    model,
    waveform: torch.Tensor,
    sample_rate: int,
    device: torch.device,
    chunk_size_min: float,
    overlap_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    waveform = waveform.to(device)
    total_samples = waveform.shape[1]

    chunk_samples = int(chunk_size_min * 60 * sample_rate) if chunk_size_min > 0 else 0
    overlap_samples = int(overlap_sec * sample_rate)

    if chunk_samples <= 0 or chunk_samples >= total_samples:
        with torch.no_grad():
            estimates = apply_model(model, waveform.unsqueeze(0), device=device)
        sources = model.sources
        if "vocals" not in sources:
            raise RuntimeError("A 'vocals' forrás nem található a modell kimenetében.")

        vocals_index = sources.index("vocals")
        vocals = estimates[0, vocals_index].cpu().numpy()
        non_speech = estimates[0, [i for i in range(len(sources)) if i != vocals_index]].sum(dim=0).cpu().numpy()
        return vocals, non_speech

    if overlap_samples >= chunk_samples:
        overlap_samples = max(chunk_samples // 4, 0)
        print(f"Figyelmeztetés: túl nagy átfedés, módosítva: {overlap_samples} minta.")

    step = max(chunk_samples - overlap_samples, 1)
    acc_vocals = torch.zeros(2, total_samples, dtype=torch.float32)
    acc_non_speech = torch.zeros(2, total_samples, dtype=torch.float32)
    acc_weights = torch.zeros(1, total_samples, dtype=torch.float32)

    for start in range(0, total_samples, step):
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, start:end]
        if chunk.shape[1] == 0:
            continue
        with torch.no_grad():
            estimates = apply_model(model, chunk.unsqueeze(0), device=device)
        sources = model.sources
        if "vocals" not in sources:
            raise RuntimeError("A 'vocals' forrás nem található a modell kimenetében.")
        vocals_index = sources.index("vocals")
        vocals_chunk = estimates[0, vocals_index].cpu()
        other_indices = [i for i in range(len(sources)) if i != vocals_index]
        non_speech_chunk = estimates[0, other_indices].sum(dim=0).cpu()

        chunk_weight = build_chunk_weight(vocals_chunk.shape[1], start, end, total_samples, overlap_samples, vocals_chunk.device)
        chunk_weight = chunk_weight.clamp_min(1e-3)

        acc_vocals[:, start:end] += vocals_chunk * chunk_weight
        acc_non_speech[:, start:end] += non_speech_chunk * chunk_weight
        acc_weights[:, start:end] += chunk_weight

        print(f"Chunk feldolgozva: {start}-{end} (minták)")

        if end == total_samples:
            break

    acc_weights = acc_weights.clamp_min(1e-3)
    vocals = (acc_vocals / acc_weights).numpy()
    non_speech = (acc_non_speech / acc_weights).numpy()
    return vocals, non_speech


def compute_model_weights(
    vocals_list: Sequence[np.ndarray],
    non_speech_list: Sequence[np.ndarray],
) -> np.ndarray:
    scores: List[float] = []
    eps = 1e-9
    for vocals, non_speech in zip(vocals_list, non_speech_list):
        speech_power = np.mean(vocals**2)
        leak_power = np.mean(non_speech**2)
        score = speech_power / (leak_power + eps)
        scores.append(score)
    weights = np.array(scores, dtype=np.float32)
    if np.all(weights == 0):
        weights = np.ones_like(weights)
    weights /= np.sum(weights)
    return weights


def blend_background(
    mix: np.ndarray,
    vocals: np.ndarray,
    predicted_background: np.ndarray,
    blend_ratio: float,
) -> np.ndarray:
    mix = mix.copy()
    vocals = vocals.copy()
    predicted_background = predicted_background.copy()
    residual_background = mix - vocals
    residual_background = np.clip(residual_background, -1.0, 1.0)
    blend_ratio = float(np.clip(blend_ratio, 0.0, 1.0))
    return blend_ratio * predicted_background + (1.0 - blend_ratio) * residual_background


def parse_model_list(model_arg: str) -> List[str]:
    if not model_arg:
        return ["htdemucs"]
    if "," in model_arg:
        models = [item.strip() for item in model_arg.split(",") if item.strip()]
    else:
        models = [model_arg.strip()]
    if not models:
        raise ValueError("Legalább egy modellt meg kell adni.")
    return models


def load_models(model_names: Sequence[str], device: torch.device):
    models: Dict[str, any] = {}
    for name in model_names:
        if name in models:
            continue
        print(f"Demucs/MDX modell betöltése: {name}")
        model = get_model(name)
        model.to(device)
        model.eval()
        models[name] = model
    return models


def process_file(
    base_name: str,
    speech_output_dir: str,
    background_output_dir: str,
    mix_waveform: torch.Tensor,
    mix_sample_rate: int,
    models: Dict[str, any],
    model_order: Sequence[str],
    device: torch.device,
    chunk_size_min: float,
    overlap_sec: float,
    non_speech_silence: bool,
    background_blend: float,
) -> bool:
    vocals_results: List[np.ndarray] = []
    non_speech_results: List[np.ndarray] = []

    for model_name in model_order:
        model = models[model_name]
        print(f"--- Modell futtatása: {model_name} ---")
        vocals, non_speech = separate_with_model(
            model=model,
            waveform=mix_waveform.clone(),
            sample_rate=mix_sample_rate,
            device=device,
            chunk_size_min=chunk_size_min,
            overlap_sec=overlap_sec,
        )
        vocals_results.append(vocals)
        non_speech_results.append(non_speech)

    weights = compute_model_weights(vocals_results, non_speech_results)
    print(f"Modell súlyok: {dict(zip(model_order, weights.round(4)))}")

    combined_vocals = np.zeros_like(vocals_results[0])
    combined_background = np.zeros_like(non_speech_results[0])

    for w, vocals, non_speech in zip(weights, vocals_results, non_speech_results):
        combined_vocals += w * vocals
        combined_background += w * non_speech

    auto_non_speech_silence = is_pure_speech(combined_vocals, combined_background)
    if non_speech_silence or auto_non_speech_silence:
        if auto_non_speech_silence and not non_speech_silence:
            print("Automatikus tiszta beszéd detektálva, non_speech csendesítése.")
        combined_background = np.zeros_like(combined_background)

    mix_np = mix_waveform.cpu().numpy()
    final_background = blend_background(
        mix=mix_np,
        vocals=combined_vocals,
        predicted_background=combined_background,
        blend_ratio=background_blend,
    )

    vocals_path = os.path.join(speech_output_dir, f"{base_name}_speech.wav")
    background_path = os.path.join(background_output_dir, f"{base_name}_non_speech.wav")

    save_audio_torchaudio(combined_vocals, vocals_path, mix_sample_rate)
    save_audio_torchaudio(final_background, background_path, mix_sample_rate)

    print(f"Szétválasztott fájlok mentve: {vocals_path}, {background_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audiófájlok szétválasztása fejlettebb Demucs/MDX munkafolyamattal."
    )
    parser.add_argument("-p", "--project", required=True, help='A projekt neve a "workdir" könyvtáron belül.')
    parser.add_argument(
        "--device",
        default="cuda",
        help='Eszköz: "cuda" vagy "cpu". Automatikusan CPU-ra vált, ha GPU nem érhető el.',
    )
    parser.add_argument(
        "--models",
        type=str,
        default="htdemucs,mdx_extra",
        help='Használandó modellek vesszővel elválasztva (pl. "htdemucs,mdx_extra").',
    )
    parser.add_argument(
        "--chunk_size",
        type=float,
        default=5.0,
        help="Darabolás hossza percben. 0 érték esetén a teljes fájl egyszerre lesz feldolgozva.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=float,
        default=10.0,
        help="Átfedés hossza másodpercben a chunk alapú feldolgozáshoz.",
    )
    parser.add_argument(
        "--non_speech_silence",
        action="store_true",
        help="Ha aktiválva, a non_speech fájlban csak csend lesz.",
    )
    parser.add_argument(
        "--background_blend",
        type=float,
        default=0.5,
        help="0-1 közötti érték, amely megadja a háttér sáv keverésének arányát a modell és az eredeti audio között.",
    )
    parser.add_argument(
        "--keep_full_audio",
        action="store_true",
        help='A konvertált teljes audio fájl megtartása a "separated_audio_speech" mappában.',
    )
    add_debug_argument(parser)
    args = parser.parse_args()
    configure_debug_mode(args.debug)

    device_str = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    try:
        model_list = parse_model_list(args.models)
    except ValueError as exc:
        print(f"Hiba: {exc}")
        sys.exit(1)

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        workdir_name = config["DIRECTORIES"]["workdir"]
        project_dir = os.path.join(project_root, workdir_name, args.project)

        if not os.path.isdir(project_dir):
            print(f"Hiba: A projekt könyvtár nem létezik: {project_dir}")
            sys.exit(1)

        input_dir = os.path.join(project_dir, config["PROJECT_SUBDIRS"]["extracted_audio"])
        speech_output_dir = os.path.join(project_dir, config["PROJECT_SUBDIRS"]["separated_audio_speech"])
        background_output_dir = os.path.join(
            project_dir, config["PROJECT_SUBDIRS"]["separated_audio_background"]
        )

    except FileNotFoundError:
        print("Hiba: A config.json nem található. A szkriptnek a megfelelő könyvtárszerkezetben kell lennie.")
        sys.exit(1)
    except KeyError as e:
        print(f"Hiba: A config.json feldolgozása közben. Hiányzó kulcs: {e}")
        sys.exit(1)

    if not os.path.isdir(input_dir):
        print(f"Hiba: A bemeneti könyvtár nem létezik: {input_dir}")
        sys.exit(1)

    os.makedirs(speech_output_dir, exist_ok=True)
    os.makedirs(background_output_dir, exist_ok=True)

    models = load_models(model_list, device)

    files_by_group: Dict[str, List[str]] = {}
    suffixes = ["_FC", "_FL", "_FR", "_SL", "_SR", "_RL", "_RR", "_LFE", "_C", "_L", "_R"]

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            continue
        base = os.path.splitext(filename)[0]
        group_name = base
        for suffix in suffixes:
            if base.upper().endswith(suffix):
                group_name = base[: -len(suffix)]
                break
        files_by_group.setdefault(group_name, []).append(filename)

    for group_name, file_list in files_by_group.items():
        filename_to_process = None
        for filename in file_list:
            if os.path.splitext(filename)[0].upper().endswith("_FC"):
                filename_to_process = filename
                print(f"\nCenter sáv (_FC) kiválasztva a '{group_name}' csoporthoz: {filename}")
                break
        if filename_to_process is None:
            filename_to_process = file_list[0]
            if len(file_list) > 1:
                print(f"\nNem található '_FC' sáv a '{group_name}' csoportban. Automatikus választás: {filename_to_process}")

        audio_path = os.path.join(input_dir, filename_to_process)
        base_name = os.path.splitext(filename_to_process)[0]
        temp_audio_path = os.path.join(speech_output_dir, f"{base_name}_temp.wav")

        print(f"--- Feldolgozás: {audio_path} ---")

        if filename_to_process.lower().endswith(".wav"):
            temp_audio_path = audio_path
            was_converted = False
        else:
            print(f"Konvertálás WAV formátumba: {temp_audio_path}")
            if not extract_audio(audio_path, temp_audio_path):
                continue
            was_converted = True

        try:
            waveform, sample_rate = torchaudio.load(temp_audio_path)
        except Exception:
            print(f"Hiba a hang fájl betöltése közben: {temp_audio_path}")
            continue

        waveform = ensure_stereo(waveform)
        waveform, sample_rate = resample_if_needed(waveform, sample_rate, 44100)

        if args.keep_full_audio:
            full_audio_path = os.path.join(speech_output_dir, f"{base_name}_full.wav")
            try:
                shutil.copy(temp_audio_path, full_audio_path)
                print(f"Teljes audio mentve: {full_audio_path}")
            except Exception as exc:
                print(f"Hiba a teljes audio mentése közben: {exc}")

        try:
            success = process_file(
                base_name=base_name,
                speech_output_dir=speech_output_dir,
                background_output_dir=background_output_dir,
                mix_waveform=waveform,
                mix_sample_rate=sample_rate,
                models=models,
                model_order=model_list,
                device=device,
                chunk_size_min=args.chunk_size,
                overlap_sec=args.chunk_overlap,
                non_speech_silence=args.non_speech_silence,
                background_blend=args.background_blend,
            )
        except Exception:
            print(f"Hiba történt a szétválasztás közben: {audio_path}")
            traceback.print_exc()
            success = False

        if was_converted and os.path.exists(temp_audio_path) and temp_audio_path != audio_path:
            os.remove(temp_audio_path)

        if not success:
            print(f"A szétválasztás sikertelen volt a következő fájlnál: {filename_to_process}")

    print("\n--- Feldolgozás befejezve ---")
    print(f"A beszéd sávok a következő könyvtárba kerültek: {speech_output_dir}")
    print(f"A háttérzaj sávok a következő könyvtárba kerültek: {background_output_dir}")


if __name__ == "__main__":
    main()

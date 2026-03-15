#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma")


def get_project_root() -> Path:
    """
    Felkeresi a projekt gyökerét a config.json alapján.
    """
    for candidate in Path(__file__).resolve().parents:
        config_candidate = candidate / "config.json"
        if config_candidate.is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")


def load_config() -> Tuple[dict, Path]:
    """
    Betölti a config.json-t, és visszaadja a konfigurációt és a projekt gyökerét.
    """
    project_root = get_project_root()
    config_path = project_root / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle), project_root
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Hiba a konfiguráció betöltésekor ({config_path}): {exc}")
        sys.exit(1)


def ensure_ffmpeg_available() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("Hiba: az ffmpeg parancs nem érhető el. Telepítsd vagy add az elérési úthoz.")
        sys.exit(1)
    return ffmpeg_path


def resolve_project_directories(
    project_name: str, config: dict, project_root: Path
) -> Dict[str, Path]:
    try:
        workdir = project_root / config["DIRECTORIES"]["workdir"]
        project_dir = workdir / project_name
        input_dir = project_dir / config["PROJECT_SUBDIRS"]["extracted_audio"]
        speech_dir = project_dir / config["PROJECT_SUBDIRS"]["separated_audio_speech"]
        background_dir = project_dir / config["PROJECT_SUBDIRS"]["separated_audio_background"]
        temp_dir = project_dir / config["PROJECT_SUBDIRS"]["temp"]
    except KeyError as exc:
        print(f"Hiba: hiányzó kulcs a config.json-ban: {exc}")
        sys.exit(1)

    if not project_dir.is_dir():
        print(f"Hiba: a projekt mappa nem található: {project_dir}")
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"Hiba: a bemeneti mappa nem található: {input_dir}")
        sys.exit(1)

    speech_dir.mkdir(parents=True, exist_ok=True)
    background_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_dir": project_dir,
        "input_dir": input_dir,
        "speech_dir": speech_dir,
        "background_dir": background_dir,
        "temp_dir": temp_dir,
    }


def discover_audio_files(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for candidate in input_dir.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(candidate)
    return sorted(files)


def convert_to_wav(
    ffmpeg_path: str,
    source_path: Path,
    target_path: Path,
    sample_rate: Optional[int],
    mono: bool,
) -> bool:
    command = [ffmpeg_path, "-y", "-i", str(source_path)]
    command.extend(["-ac", "1" if mono else "2"])
    if sample_rate:
        command.extend(["-ar", str(sample_rate)])
    command.append(str(target_path))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.error("FFmpeg konverzió sikertelen: %s", source_path)
        logging.error(result.stderr.decode(errors="ignore"))
        return False
    if not target_path.exists() or target_path.stat().st_size == 0:
        logging.error("A konvertált WAV fájl nem jött létre: %s", target_path)
        return False
    return True


def load_onnx_interface(script_path: Path):
    if not script_path.is_file():
        print(f"Hiba: az ONNX interface fájl nem található: {script_path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("sam_audio_onnx_interface", script_path)
    if spec is None or spec.loader is None:
        print(f"Hiba: nem sikerült betölteni az ONNX interface modult: {script_path}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    sys.modules["sam_audio_onnx_interface"] = module
    spec.loader.exec_module(module)

    for attr in ("SAMAudioONNXPipeline", "load_audio", "save_audio"):
        if not hasattr(module, attr):
            print(f"Hiba: hiányzó '{attr}' az ONNX interface modulban: {script_path}")
            sys.exit(1)

    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Beszéd/háttér szétválasztás SAM-Audio backenddel (wrapper)."
    )
    parser.add_argument(
        "-p",
        "--project-name",
        required=True,
        help='A projekt neve a "workdir" könyvtáron belül.',
    )
    parser.add_argument(
        "--prompt",
        default="speech",
        help="Text prompt, amely a kiválasztandó hangot írja le.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Eszköz: "cuda" vagy "cpu".',
    )
    parser.add_argument(
        "--onnx-script",
        default="onnx_inference.py",
        help="Az ONNX interface (onnx_inference.py) elérési útja.",
    )
    parser.add_argument(
        "--onnx-model-dir",
        default="onnx_models",
        help="Az ONNX modelleket tartalmazó mappa.",
    )
    parser.add_argument(
        "--onnx-steps",
        type=int,
        default=16,
        help="ODE lépések száma az ONNX pipeline-ban.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Konverziós cél mintavételezési frekvencia (Hz).",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Ha aktív, a konvertált WAV fájlok mono csatornára kerülnek.",
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Ha aktív, minden bemeneti fájlt WAV-ra konvertál az ffmpeg.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Ideiglenes WAV fájlok megtartása a temp mappában.",
    )
    parser.add_argument(
        "--predict-spans",
        action="store_true",
        help="SAM-Audio automatikus span predikció bekapcsolása.",
    )
    parser.add_argument(
        "--reranking-candidates",
        type=int,
        default=1,
        help="Reranking jelöltek száma (nagyobb érték jobb minőség, nagyobb memória).",
    )
    parser.add_argument(
        "--span-threshold",
        type=float,
        default=0.3,
        help="Span predikció küszöbértéke (ONNX PEAFrame).",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=60,
        help="Feldolgozás darabolása másodpercben (0 = teljes fájl).",
    )
    add_debug_argument(parser)
    args = parser.parse_args()
    log_level = configure_debug_mode(args.debug)
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    config, project_root = load_config()
    dirs = resolve_project_directories(args.project_name, config, project_root)
    input_dir = dirs["input_dir"]
    speech_dir = dirs["speech_dir"]
    background_dir = dirs["background_dir"]
    temp_dir = dirs["temp_dir"]

    logging.info("Projekt: %s", args.project_name)
    logging.info("Bemenet: %s", input_dir)
    logging.info("Kimenet (speech): %s", speech_dir)
    logging.info("Kimenet (background): %s", background_dir)
    logging.info("Mintavétel: %s Hz, mono: %s", args.sample_rate, args.mono)
    logging.info("Prompt: %s", args.prompt)
    logging.info("Predict spans: %s", args.predict_spans)
    logging.info("Reranking candidates: %s", args.reranking_candidates)
    logging.info("ONNX script: %s", args.onnx_script)
    logging.info("ONNX model dir: %s", args.onnx_model_dir)
    logging.info("ONNX steps: %s", args.onnx_steps)

    ffmpeg_path = ensure_ffmpeg_available()
    onnx_module = load_onnx_interface(Path(args.onnx_script).expanduser())
    pipeline = onnx_module.SAMAudioONNXPipeline(
        model_dir=str(Path(args.onnx_model_dir).expanduser()),
        device="cuda" if args.device == "cuda" else "cpu",
        num_ode_steps=args.onnx_steps,
    )

    predict_spans = args.predict_spans
    if predict_spans and getattr(pipeline, "peaframe", None) is None:
        logging.warning(
            "A PEAFrame model nincs betöltve, a span predikció kikapcsolva."
        )
        predict_spans = False

    reranking_candidates = max(1, int(args.reranking_candidates))
    if reranking_candidates > 1 and getattr(pipeline, "clap_audio_encoder", None) is None:
        logging.warning("A CLAP model nincs betöltve, a rerank kikapcsolva.")
        reranking_candidates = 1

    audio_files = discover_audio_files(input_dir)
    if not audio_files:
        print(f"Nincsenek feldolgozható audio fájlok a bemeneti mappában: {input_dir}")
        sys.exit(1)

    overall_success = True

    for audio_path in audio_files:
        base_name = audio_path.stem
        speech_path = speech_dir / f"{base_name}_speech.wav"
        background_path = background_dir / f"{base_name}_non_speech.wav"

        if speech_path.exists() and background_path.exists():
            logging.info("Kihagyva (már létezik): %s", audio_path.name)
            continue

        temp_input = audio_path
        converted = False
        if args.force_convert or audio_path.suffix.lower() != ".wav":
            temp_input = temp_dir / f"{base_name}_sam_audio_input.wav"
            if not convert_to_wav(
                ffmpeg_path,
                audio_path,
                temp_input,
                args.sample_rate,
                args.mono,
            ):
                overall_success = False
                continue
            converted = True

        logging.info("SAM-Audio futtatása: %s", audio_path.name)
        try:
            target_rate = 48000
            if args.sample_rate != target_rate:
                logging.warning(
                    "SAM-Audio ONNX csak 48kHz-et támogat. A megadott %s Hz felülírva 48kHz-re.",
                    args.sample_rate,
                )
            audio = onnx_module.load_audio(str(temp_input), target_sr=target_rate)
            chunk_seconds = max(int(args.chunk_seconds), 0)
            chunk_samples = int(chunk_seconds * target_rate) if chunk_seconds > 0 else 0
            total_samples = audio.shape[0]
            if chunk_samples <= 0 or chunk_samples >= total_samples:
                chunk_ranges = [(0, total_samples)]
            else:
                chunk_ranges = [
                    (start, min(start + chunk_samples, total_samples))
                    for start in range(0, total_samples, chunk_samples)
                ]

            target_chunks = []
            residual_chunks = []

            for index, (start, end) in enumerate(chunk_ranges, start=1):
                chunk = audio[start:end]
                if chunk.size == 0:
                    continue
                logging.info(
                    "Chunk feldolgozás: %s (%s/%s)",
                    audio_path.name,
                    index,
                    len(chunk_ranges),
                )
                rerank = reranking_candidates > 1
                target, residual, _, _ = pipeline.separate(
                    chunk,
                    args.prompt,
                    predict_spans=predict_spans,
                    span_threshold=args.span_threshold,
                    rerank=rerank,
                    num_candidates=reranking_candidates,
                )
                target_chunks.append(target)
                residual_chunks.append(residual)

            if not target_chunks or not residual_chunks:
                raise RuntimeError("Üres kimenet a chunk feldolgozás után.")

            import numpy as np

            target_full = np.concatenate(target_chunks, axis=0)
            residual_full = np.concatenate(residual_chunks, axis=0)
            onnx_module.save_audio(target_full, str(speech_path), sample_rate=target_rate)
            onnx_module.save_audio(
                residual_full, str(background_path), sample_rate=target_rate
            )
            success = True
        except Exception as exc:
            logging.error("SAM-Audio feldolgozási hiba: %s", exc)
            success = False

        if converted and not args.keep_temp and temp_input.exists():
            try:
                temp_input.unlink()
            except OSError as exc:
                logging.warning("Nem sikerült törölni: %s (%s)", temp_input, exc)

        if not success:
            overall_success = False
            continue

        if not speech_path.exists() or not background_path.exists():
            logging.error("Hiányzó kimeneti fájlok: %s", audio_path.name)
            overall_success = False

    if overall_success:
        print("Feldolgozás kész. A beszéd és háttér fájlok elkészültek.")
        sys.exit(0)
    print("Feldolgozás kész, de hibák történtek. Kérlek, ellenőrizd a logokat.")
    sys.exit(1)


if __name__ == "__main__":
    main()

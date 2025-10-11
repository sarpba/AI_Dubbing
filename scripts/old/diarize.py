#!/usr/bin/env python3
"""Add speaker labels to Whisper JSON using pyannote speaker-diarization.

CUDA / CPU switch
=================
The script automatically chooses **CUDA** if available, else CPU, but you can
force either with `--device cuda|cpu|auto`.

Example
-------
```bash
python diarize_add_speakers.py -i movie.wav        # auto (CUDA if possible)
python diarize_add_speakers.py -i movie.wav --device cuda
python diarize_add_speakers.py -i movie.wav --device cpu

# Example with advanced tuning for better accuracy:
python diarize_add_speakers.py -i movie.wav --min-speakers 2 --max-speakers 3 \
       --segmentation-threshold 0.4 --clustering-threshold 0.6 \
       --segmentation-min-duration-off 0.2
```

Workflow
--------
1. Convert the input audio (`-i`) to 16 kHz / mono WAV (if needed).
2. Run pyannote speaker-diarization on the selected device.
3. Merge speaker labels into the Whisper JSON (same basename).
4. Create a UTC-timestamped backup of the original JSON.
5. Overwrite the JSON with the speaker-annotated version.

Dependencies
------------
```bash
pip install pyannote.audio==3.* torch soundfile rich tqdm
# CUDA build, e.g. (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from rich.console import Console

console = Console()

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def convert_audio(src: Path) -> Path:
    """Convert *src* to 16 kHz mono WAV next to it and return new path."""
    dst = src.with_name(f"{src.stem}_16k.wav")
    if dst.exists():
        console.print(f"[bold yellow]ℹ️  Reusing existing converted file {dst.name}")
        return dst

    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i",
        str(src),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(dst),
    ]
    console.print("[cyan]🔄 Converting audio to 16 kHz mono WAV …")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        console.print(result.stderr.decode(), style="red")
        raise RuntimeError("FFmpeg conversion failed")
    return dst


def diarize(
    wav: Path,
    model: str,
    hf_token: str | None,
    device: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    segmentation_threshold: float | None = None,
    clustering_threshold: float | None = None,
    segmentation_min_duration_off: float | None = None,
) -> List[Tuple[float, float, str]]:
    """Run pyannote speaker-diarization and return list of (start, end, speaker)."""
    console.print(f"[cyan]🗣️  Running diarization on [b]{device.upper()}[/] …")

    from pyannote.audio import Pipeline  # heavy import → delayed
    import torch

    pipeline = Pipeline.from_pretrained(model, **({"use_auth_token": hf_token} if hf_token else {}))

    # move to device -----------------------------------------------------------
    try:
        torch_device = torch.device(device)
        pipeline.to(torch_device)
    except Exception as exc:
        console.print(
            f"[red]⚠ Could not move pipeline to {device}: {exc!s}\n   Falling back to CPU."
        )
        torch_device = torch.device("cpu")
        pipeline.to(torch_device)
        device = "cpu"
    # -------------------------------------------------------------------------

    # Apply custom hyperparameters if provided
    custom_hyperparameters = {}
    seg_overrides = {}
    if segmentation_threshold is not None:
        seg_overrides["threshold"] = segmentation_threshold
    if segmentation_min_duration_off is not None:
        seg_overrides["min_duration_off"] = segmentation_min_duration_off
    if seg_overrides:
        custom_hyperparameters["segmentation"] = seg_overrides

    clus_overrides = {}
    if clustering_threshold is not None:
        clus_overrides["threshold"] = clustering_threshold
    if clus_overrides:
        custom_hyperparameters["clustering"] = clus_overrides
    
    if custom_hyperparameters:
        console.print(f"[cyan]🔧 Applying custom hyperparameters: {custom_hyperparameters}")
        pipeline = pipeline.with_parameters(custom_hyperparameters, partial=True)

    # Prepare arguments for the pipeline call (e.g., number of speakers)
    pipeline_call_kwargs = {}
    if min_speakers is not None:
        pipeline_call_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        pipeline_call_kwargs["max_speakers"] = max_speakers
    if min_speakers is not None or max_speakers is not None:
        console.print(f"[cyan]🔧 Applying speaker count constraints: {pipeline_call_kwargs}")


    diarization = pipeline(str(wav), **pipeline_call_kwargs)
    tracks = [
        (segment.start, segment.end, label)
        for segment, _, label in diarization.itertracks(yield_label=True)
    ]
    console.print(
        f"[green]✔ Diarization finished on {device}. Found {len(set(label for *_, label in tracks))} speakers."
    )
    return tracks


def speaker_for_time(tracks: List[Tuple[float, float, str]], t: float) -> str:
    for start, end, label in tracks:
        if start <= t < end:
            return label
    return "UNKNOWN"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Add speaker labels to Whisper JSON")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to input audio file")
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="HuggingFace model name or local path",
    )
    parser.add_argument("--hf-token", help="HuggingFace access token")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection (default: auto)",
    )
    # Arguments for tuning diarization accuracy
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum number of speakers (optional)")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum number of speakers (optional)")
    parser.add_argument(
        "--segmentation-threshold", type=float, default=None,
        help="Speech segmentation threshold (0.0-1.0). Default: model's own."
    )
    parser.add_argument(
        "--clustering-threshold", type=float, default=None,
        help="Speaker clustering threshold. Default: model's own."
    )
    parser.add_argument(
        "--segmentation-min-duration-off", type=float, default=None,
        help="Minimum duration of silence (seconds) for segmentation. Default: model's own."
    )
    args = parser.parse_args()

    if args.min_speakers is not None and args.max_speakers is not None and args.min_speakers > args.max_speakers:
        console.print("[red]✖ Error: --min-speakers cannot be greater than --max-speakers.")
        sys.exit(1)

    audio_path = args.input.resolve()
    if not audio_path.exists():
        console.print(f"[red]✖ Audio file not found: {audio_path}")
        sys.exit(1)

    json_path = audio_path.with_suffix(".json")
    if not json_path.exists():
        console.print(f"[red]✖ JSON file not found: {json_path}")
        sys.exit(1)

    # 1) Convert audio ---------------------------------------------------------
    wav_path = convert_audio(audio_path)

    # 2) Decide device ---------------------------------------------------------
    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            console.print("[yellow]⚠ CUDA requested but not available; switching to CPU")
            device = "cpu"

    # 3) Diarize ---------------------------------------------------------------
    tracks = diarize(
        wav_path,
        args.model,
        args.hf_token or os.getenv("HUGGINGFACE_TOKEN"),
        device,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        segmentation_threshold=args.segmentation_threshold,
        clustering_threshold=args.clustering_threshold,
        segmentation_min_duration_off=args.segmentation_min_duration_off,
    )

    # 4) Load JSON -------------------------------------------------------------
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # 5) Merge labels ----------------------------------------------------------
    for seg in data.get("segments", []):
        seg_spk = speaker_for_time(tracks, float(seg["start"]))
        seg["speaker"] = seg_spk
        for w in seg.get("words", []):
            w["speaker"] = speaker_for_time(tracks, float(w["start"]))

    # 6) Backup & overwrite ----------------------------------------------------
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_path = json_path.with_name(f"{json_path.stem}_{timestamp}.bak.json")
    shutil.copy2(json_path, backup_path)
    console.print(f"[blue]💾 Backup written to {backup_path.name}")

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]✅ Speaker labels added. File updated: {json_path.name}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
High-accuracy speaker-change detector focused on change points and overlap boundaries.

Design goals
------------
- offline/local execution
- optimized for long-form WAV input (up to ~2 hours)
- prioritizes recall of real speaker changes
- outputs only temporal regions of type: "single" or "overlap"
- optional evaluation against ground-truth JSON
- preset-based speed/accuracy tradeoff

Recommended models
------------------
- pyannote/speaker-diarization-community-1
- pyannote/segmentation-3.0

Example
-------
python diarize_changes.py input.wav output.json --hf-token YOUR_TOKEN --preset ultra

Ground truth format for --evaluate
----------------------------------
[
  {"start": 12.340, "end": 18.120, "type": "single"},
  {"start": 18.120, "end": 18.740, "type": "overlap"},
  {"start": 18.740, "end": 26.510, "type": "single"}
]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from pyannote.audio import Inference, Model, Pipeline
from pyannote.core import Annotation, Segment, Timeline


# -----------------------------
# Presets
# -----------------------------

PRESETS = {
    "fast": {
        "refine_radius": 0.18,
        "seg_batch_size": 16,
        "chunk_seconds": 1800.0,
        "chunk_overlap": 4.0,
        "min_region": 0.08,
        "boundary_snap_max_shift": 0.18,
        "fallback_variation_smooth": 3,
        "dedup_tolerance": 0.08,
    },
    "balanced": {
        "refine_radius": 0.25,
        "seg_batch_size": 8,
        "chunk_seconds": 1500.0,
        "chunk_overlap": 6.0,
        "min_region": 0.06,
        "boundary_snap_max_shift": 0.25,
        "fallback_variation_smooth": 5,
        "dedup_tolerance": 0.06,
    },
    "ultra": {
        "refine_radius": 0.35,
        "seg_batch_size": 4,
        "chunk_seconds": 1200.0,
        "chunk_overlap": 8.0,
        "min_region": 0.04,
        "boundary_snap_max_shift": 0.35,
        "fallback_variation_smooth": 7,
        "dedup_tolerance": 0.04,
    },
}


# -----------------------------
# Data classes
# -----------------------------

@dataclass
class Region:
    start: float
    end: float
    type: str  # "single" | "overlap"

    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, object]:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "type": self.type,
        }


@dataclass
class ChangePoint:
    time: float
    left_type: str
    right_type: str


# -----------------------------
# Audio utils
# -----------------------------

def load_audio_mono_16k(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.ndim != 2:
        raise RuntimeError(f"Unexpected waveform shape: {tuple(wav.shape)}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    wav = wav.contiguous()
    return wav, sr


def audio_duration_seconds(path: str) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def save_temp_wav(waveform: torch.Tensor, sample_rate: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="spkchg_")
    os.close(fd)
    torchaudio.save(path, waveform.cpu(), sample_rate)
    return path


def crop_waveform(
    waveform: torch.Tensor,
    sr: int,
    start_sec: float,
    end_sec: float,
) -> torch.Tensor:
    start_i = max(0, int(round(start_sec * sr)))
    end_i = min(waveform.shape[1], int(round(end_sec * sr)))
    if end_i <= start_i:
        return torch.zeros((1, 1), dtype=waveform.dtype)
    return waveform[:, start_i:end_i]


# -----------------------------
# Diarization core
# -----------------------------

def build_pipeline(hf_token: str, device: torch.device) -> Pipeline:
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )
    pipe.to(device)
    return pipe


def build_segmentation_inference(
    hf_token: str,
    device: torch.device,
    batch_size: int,
) -> Inference:
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=hf_token)
    model.to(device)
    return Inference(model, window="sliding", duration=10.0, step=2.5, batch_size=batch_size)


def run_diarization_on_chunk(
    pipeline: Pipeline,
    chunk_path: str,
    max_speakers: Optional[int],
) -> Annotation:
    kwargs = {}
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    output = pipeline(chunk_path, **kwargs)
    return output.speaker_diarization if hasattr(output, "speaker_diarization") else output


def shift_annotation(annotation: Annotation, offset: float) -> Annotation:
    shifted = Annotation()
    for seg, track, label in annotation.itertracks(yield_label=True):
        shifted[Segment(seg.start + offset, seg.end + offset), track] = label
    return shifted


def trim_annotation(annotation: Annotation, start: float, end: float) -> Annotation:
    cropped = Annotation()
    focus = Segment(start, end)
    for seg, track, label in annotation.itertracks(yield_label=True):
        inter = seg & focus
        if inter and inter.duration > 0:
            cropped[inter, track] = label
    return cropped


def merge_annotations(annotations: Sequence[Annotation]) -> Annotation:
    merged = Annotation()
    for ann in annotations:
        for seg, track, label in ann.itertracks(yield_label=True):
            merged[seg, f"{track}_{len(merged)}"] = label
    return merged


def diarize_long_audio(
    input_path: str,
    pipeline: Pipeline,
    waveform: torch.Tensor,
    sr: int,
    max_speakers: Optional[int],
    preset: Dict[str, object],
) -> Annotation:
    duration = waveform.shape[1] / sr
    chunk_seconds = float(preset["chunk_seconds"])
    chunk_overlap = float(preset["chunk_overlap"])

    if duration <= chunk_seconds:
        return run_diarization_on_chunk(pipeline, input_path, max_speakers)

    annotations = []
    starts = []
    t = 0.0
    while t < duration:
        starts.append(t)
        t += chunk_seconds - chunk_overlap

    for idx, start in enumerate(starts):
        end = min(duration, start + chunk_seconds)
        chunk = crop_waveform(waveform, sr, start, end)
        tmp = save_temp_wav(chunk, sr)
        try:
            ann = run_diarization_on_chunk(pipeline, tmp, max_speakers)
            ann = shift_annotation(ann, start)

            if idx == 0:
                keep_start = 0.0
            else:
                keep_start = start + chunk_overlap / 2.0

            if idx == len(starts) - 1:
                keep_end = duration
            else:
                keep_end = end - chunk_overlap / 2.0

            ann = trim_annotation(ann, keep_start, keep_end)
            annotations.append(ann)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    return merge_annotations(annotations)


# -----------------------------
# Annotation -> regions
# -----------------------------

def annotation_to_atomic_regions(annotation: Annotation, duration: float) -> List[Region]:
    boundaries = {0.0, duration}
    for seg in annotation.itersegments():
        boundaries.add(float(seg.start))
        boundaries.add(float(seg.end))

    times = sorted(boundaries)
    regions: List[Region] = []

    for a, b in zip(times[:-1], times[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2.0
        active = 0
        for seg in annotation.itersegments():
            if seg.start <= mid < seg.end:
                active += 1
        if active <= 0:
            continue
        typ = "overlap" if active >= 2 else "single"
        regions.append(Region(a, b, typ))

    return merge_adjacent_same_type(regions)


def merge_adjacent_same_type(regions: Sequence[Region], gap_tolerance: float = 1e-6) -> List[Region]:
    if not regions:
        return []
    out = [Region(regions[0].start, regions[0].end, regions[0].type)]
    for r in regions[1:]:
        last = out[-1]
        if r.type == last.type and r.start <= last.end + gap_tolerance:
            last.end = max(last.end, r.end)
        else:
            out.append(Region(r.start, r.end, r.type))
    return out


def filter_short_regions(
    regions: Sequence[Region],
    min_region: float,
    preserve_edges: bool = True,
) -> List[Region]:
    if not regions:
        return []

    regions = [Region(r.start, r.end, r.type) for r in regions]
    changed = True
    while changed and len(regions) > 1:
        changed = False
        for i, r in enumerate(list(regions)):
            if r.duration() >= min_region:
                continue
            if len(regions) == 1:
                break

            if i == 0 and preserve_edges:
                regions[1].start = regions[0].start
                del regions[0]
                changed = True
                break
            elif i == len(regions) - 1 and preserve_edges:
                regions[-2].end = regions[-1].end
                del regions[-1]
                changed = True
                break
            else:
                left = regions[i - 1]
                right = regions[i + 1]
                if left.type == right.type:
                    merged = Region(left.start, right.end, left.type)
                    regions[i - 1:i + 2] = [merged]
                elif left.duration() >= right.duration():
                    left.end = r.end
                    del regions[i]
                else:
                    right.start = r.start
                    del regions[i]
                changed = True
                break

    return merge_adjacent_same_type(regions)


def regions_to_change_points(regions: Sequence[Region]) -> List[ChangePoint]:
    cps = []
    for left, right in zip(regions[:-1], regions[1:]):
        cps.append(ChangePoint(time=left.end, left_type=left.type, right_type=right.type))
    return cps


# -----------------------------
# Boundary refinement with segmentation-3.0
# -----------------------------

STATE_MAP = {
    0: "nonspeech",
    1: "single",
    2: "single",
    3: "single",
    4: "overlap",
    5: "overlap",
    6: "overlap",
}


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(xp, kernel, mode="valid")


def infer_local_segmentation_scores(
    inference: Inference,
    waveform_16k: torch.Tensor,
    sr: int,
    center_time: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    start = max(0.0, center_time - 5.0)
    end = start + 10.0
    whole_dur = waveform_16k.shape[1] / sr
    if end > whole_dur:
        end = whole_dur
        start = max(0.0, end - 10.0)

    crop = crop_waveform(waveform_16k, sr, start, end)
    if crop.shape[1] < int(10.0 * sr):
        pad = int(10.0 * sr) - crop.shape[1]
        crop = torch.nn.functional.pad(crop, (0, pad))

    with torch.inference_mode():
        scores = inference({"waveform": crop.to(device), "sample_rate": sr})

    data = scores.data
    sw = scores.sliding_window
    times = np.array([start + sw[i].middle for i in range(data.shape[0])], dtype=np.float64)
    return times, np.asarray(data, dtype=np.float32)


def refine_boundary_with_segmentation(
    cp: ChangePoint,
    inference: Inference,
    waveform_16k: torch.Tensor,
    sr: int,
    device: torch.device,
    refine_radius: float,
    boundary_snap_max_shift: float,
    smooth: int,
) -> float:
    try:
        times, scores = infer_local_segmentation_scores(
            inference=inference,
            waveform_16k=waveform_16k,
            sr=sr,
            center_time=cp.time,
            device=device,
        )
    except Exception:
        return cp.time

    if scores.ndim != 2 or scores.shape[1] < 7 or scores.shape[0] < 3:
        return cp.time

    labels = scores.argmax(axis=1)
    states = np.array([STATE_MAP.get(int(lbl), "single") for lbl in labels], dtype=object)

    low = cp.time - refine_radius
    high = cp.time + refine_radius
    idx = np.where((times >= low) & (times <= high))[0]
    if idx.size < 2:
        return cp.time

    target = None
    for i in idx[:-1]:
        if states[i] == cp.left_type and states[i + 1] == cp.right_type:
            target = float((times[i] + times[i + 1]) / 2.0)
            break

    if target is None:
        # fallback: largest posterior change near coarse boundary
        diff = np.abs(np.diff(scores, axis=0)).sum(axis=1)
        diff = _moving_average(diff.astype(np.float32), smooth)
        didx = np.where((times[:-1] >= low) & (times[:-1] <= high))[0]
        if didx.size:
            best = int(didx[np.argmax(diff[didx])])
            target = float((times[best] + times[best + 1]) / 2.0)
        else:
            target = cp.time

    shift = target - cp.time
    if abs(shift) > boundary_snap_max_shift:
        return cp.time
    return target


def refine_regions(
    regions: Sequence[Region],
    inference: Inference,
    waveform_16k: torch.Tensor,
    sr: int,
    device: torch.device,
    preset: Dict[str, object],
) -> List[Region]:
    cps = regions_to_change_points(regions)
    if not cps:
        return list(regions)

    refined_times = []
    for cp in cps:
        rt = refine_boundary_with_segmentation(
            cp=cp,
            inference=inference,
            waveform_16k=waveform_16k,
            sr=sr,
            device=device,
            refine_radius=float(preset["refine_radius"]),
            boundary_snap_max_shift=float(preset["boundary_snap_max_shift"]),
            smooth=int(preset["fallback_variation_smooth"]),
        )
        refined_times.append(rt)

    refined = []
    left = regions[0].start
    for i, r in enumerate(regions):
        if i < len(refined_times):
            boundary = max(left, refined_times[i])
            if boundary < r.end:
                refined.append(Region(left, boundary, r.type))
                left = boundary
            else:
                refined.append(Region(left, r.end, r.type))
                left = r.end
        else:
            refined.append(Region(left, r.end, r.type))

    refined = merge_adjacent_same_type(refined)
    refined = filter_short_regions(refined, float(preset["min_region"]))
    return refined


# -----------------------------
# Export / evaluation
# -----------------------------

def write_regions_json(path: str, regions: Sequence[Region]) -> None:
    payload = [r.to_dict() for r in regions]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_change_points_json(path: str, regions: Sequence[Region]) -> None:
    cps = regions_to_change_points(regions)
    payload = [
        {
            "time": round(cp.time, 3),
            "left_type": cp.left_type,
            "right_type": cp.right_type,
        }
        for cp in cps
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_regions_json(path: str) -> List[Region]:
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    out = []
    for item in arr:
        out.append(Region(float(item["start"]), float(item["end"]), str(item["type"])))
    return out


def change_times_from_regions(regions: Sequence[Region]) -> List[float]:
    return [r.end for r in regions[:-1]]


def greedy_match_changes(
    truth: Sequence[float],
    pred: Sequence[float],
    tol: float,
) -> Tuple[int, List[float]]:
    used = [False] * len(pred)
    matched = 0
    errors = []
    for t in truth:
        best_j = None
        best_err = None
        for j, p in enumerate(pred):
            if used[j]:
                continue
            err = abs(p - t)
            if err <= tol and (best_err is None or err < best_err):
                best_err = err
                best_j = j
        if best_j is not None:
            used[best_j] = True
            matched += 1
            errors.append(best_err)
    return matched, errors


def evaluate_regions(
    truth_regions: Sequence[Region],
    pred_regions: Sequence[Region],
    tolerance: float = 0.1,
) -> Dict[str, object]:
    truth = change_times_from_regions(truth_regions)
    pred = change_times_from_regions(pred_regions)
    matched, errors = greedy_match_changes(truth, pred, tol=tolerance)

    precision = matched / len(pred) if pred else 0.0
    recall = matched / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tolerance_seconds": tolerance,
        "truth_change_count": len(truth),
        "pred_change_count": len(pred),
        "matched_within_tolerance": matched,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "mean_abs_error_ms": round((statistics.mean(errors) * 1000.0) if errors else math.nan, 3),
        "median_abs_error_ms": round((statistics.median(errors) * 1000.0) if errors else math.nan, 3),
    }


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="High-accuracy speaker-change detector")
    ap.add_argument("input_wav", help="input WAV file")
    ap.add_argument("output_json", help="output JSON file with regions")
    ap.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""), help="Hugging Face token")
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), default="ultra")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max-speakers", type=int, default=50)
    ap.add_argument("--min-region", type=float, default=None, help="override preset minimum region length in seconds")
    ap.add_argument("--output-change-points-json", default=None, help="optional JSON for boundary-only output")
    ap.add_argument("--evaluate", default=None, help="ground truth regions JSON")
    ap.add_argument("--eval-tol-ms", type=float, default=100.0)
    ap.add_argument("--no-refine", action="store_true", help="skip segmentation-based boundary refinement")
    ap.add_argument("--print-regions", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not args.hf_token:
        print("ERROR: missing Hugging Face token. Pass --hf-token or set HF_TOKEN.", file=sys.stderr)
        return 2

    input_path = str(Path(args.input_wav).expanduser().resolve())
    output_path = str(Path(args.output_json).expanduser().resolve())
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    preset = dict(PRESETS[args.preset])
    if args.min_region is not None:
        preset["min_region"] = float(args.min_region)

    print(f"[info] loading audio: {input_path}")
    waveform, sr = load_audio_mono_16k(input_path)
    duration = waveform.shape[1] / sr
    print(f"[info] duration: {duration:.2f}s | sr={sr} | device={device}")

    print("[info] loading diarization pipeline...")
    pipeline = build_pipeline(args.hf_token, device)

    print("[info] running coarse diarization...")
    annotation = diarize_long_audio(
        input_path=input_path,
        pipeline=pipeline,
        waveform=waveform,
        sr=sr,
        max_speakers=args.max_speakers,
        preset=preset,
    )

    print("[info] converting to atomic regions...")
    regions = annotation_to_atomic_regions(annotation, duration)
    regions = filter_short_regions(regions, float(preset["min_region"]))

    if not args.no_refine:
        print("[info] loading segmentation model for boundary refinement...")
        inference = build_segmentation_inference(
            hf_token=args.hf_token,
            device=device,
            batch_size=int(preset["seg_batch_size"]),
        )
        print("[info] refining boundaries...")
        regions = refine_regions(
            regions=regions,
            inference=inference,
            waveform_16k=waveform,
            sr=sr,
            device=device,
            preset=preset,
        )

    regions = merge_adjacent_same_type(regions)
    regions = filter_short_regions(regions, float(preset["min_region"]))

    print(f"[info] writing regions json: {output_path}")
    write_regions_json(output_path, regions)

    if args.output_change_points_json:
        cp_path = str(Path(args.output_change_points_json).expanduser().resolve())
        print(f"[info] writing change points json: {cp_path}")
        write_change_points_json(cp_path, regions)

    if args.print_regions:
        for r in regions:
            print(json.dumps(r.to_dict(), ensure_ascii=False))

    if args.evaluate:
        gt = load_regions_json(args.evaluate)
        metrics = evaluate_regions(gt, regions, tolerance=float(args.eval_tol_ms) / 1000.0)
        print("[eval]")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

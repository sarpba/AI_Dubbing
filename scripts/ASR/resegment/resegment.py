"""
Resegment script - JSON szegmensek újraformázása különböző paraméterekkel.
Az ASR script által létrehozott JSON fájlok biztonsági mentése és újraformázása.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

DEFAULT_MAX_PAUSE_S = 0.3
DEFAULT_PADDING_S = 0.1
DEFAULT_MAX_SEGMENT_S = 11.5
PRIMARY_PUNCTUATION = (".", "!", "?")
SECONDARY_PUNCTUATION = (",",)
SUPPORTED_AUDIO_EXTENSIONS: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
ENERGY_ALLOWED_SAMPLE_RATES: Tuple[int, ...] = (8000, 16000, 22050, 24000, 32000, 44100, 48000)
ENERGY_TARGET_SAMPLE_RATE = 16000
ENERGY_WINDOW_MS = 10
ENERGY_BACKTRACK_LIMIT_S = 0.5
ENERGY_THRESHOLD_FLOOR = 150.0
ENERGY_FORWARD_WINDOW_S = 0.3
MIN_WORD_DURATION_S = 0.02
MIN_SILENCE_FRAMES = 3
INTER_WORD_SEARCH_WINDOW_S = 0.1
DEFAULT_SPEAKER_CHANGE_PRESET = "balanced"
DEFAULT_SPEAKER_CHANGE_DEVICE = "cuda"

SPEAKER_CHANGE_PRESETS: Dict[str, Dict[str, float]] = {
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


@dataclass
class SpeakerChangeRegion:
    start: float
    end: float
    region_type: str

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SpeakerChangePoint:
    time: float
    left_type: str
    right_type: str


def _sanitize_text(value: Any) -> str:
    """Remove escaped quote sequences from transcribed text."""
    if not value:
        return ""
    text = str(value)
    cleaned = text.replace('\\"', "").replace('"', "")
    return cleaned.strip()


def get_project_root() -> Path:
    """Locate the repository root by walking upwards until config.json is found."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")


def load_config() -> Tuple[dict, Path]:
    """Load config.json and return it together with the project root."""
    project_root = get_project_root()
    config_path = project_root / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            config = json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Hiba a konfiguráció betöltésekor ({config_path}): {exc}")
        sys.exit(1)
    return config, project_root


def resolve_project_input(project_name: str, config: dict, project_root: Path) -> Path:
    """Resolve the directory that contains ASR JSON files for the project."""
    try:
        workdir = project_root / config["DIRECTORIES"]["workdir"]
        input_subdir = config["PROJECT_SUBDIRS"]["separated_audio_speech"]
    except KeyError as exc:
        print(f"Hiba: hiányzó kulcs a config.json-ban: {exc}")
        sys.exit(1)

    input_dir = workdir / project_name / input_subdir
    if not input_dir.is_dir():
        print(f"Hiba: a feldolgozandó mappa nem található: {input_dir}")
        sys.exit(1)
    return input_dir


def find_audio_for_json(json_path: Path) -> Optional[Path]:
    """Try to locate the audio file that belongs to the JSON transcript."""
    stem = json_path.stem
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        candidate = json_path.with_suffix(ext)
        if candidate.is_file():
            return candidate
    for candidate in json_path.parent.iterdir():
        if (
            candidate.is_file()
            and candidate.stem == stem
            and candidate.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        ):
            return candidate
    return None


def _convert_to_energy_wav(path: Path) -> Optional[Path]:
    """Convert arbitrary audio to a PCM16 mono WAV accepted by the energy analysis."""
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    except Exception as exc:
        logging.warning("Nem sikerült ideiglenes fájlt létrehozni energiabecsléshez (%s): %s", path, exc)
        return None
    temp_file.close()
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(ENERGY_TARGET_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        temp_file.name,
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return Path(temp_file.name)
    except FileNotFoundError:
        logging.warning("Az ffmpeg nem elérhető, az energiabecsléses korrekció kihagyva (%s).", path)
    except subprocess.CalledProcessError as exc:
        error_message = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        logging.warning("FFmpeg konverzió sikertelen energiabecsléshez (%s): %s", path, error_message.strip())
    try:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
    except OSError:
        pass
    return None


def read_wave_for_energy(path: Path) -> Optional[Tuple[array, int]]:
    """Load audio data suitable for energy analysis. Returns PCM samples and sample_rate."""
    converted_path: Optional[Path] = None
    source_path = path
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
            if (
                sample_rate not in ENERGY_ALLOWED_SAMPLE_RATES
                or sample_width != 2
                or channels != 1
            ):
                raise ValueError("unsupported format")
            frames = wf.readframes(wf.getnframes())
            samples = array("h")
            samples.frombytes(frames)
            return samples, sample_rate
    except (wave.Error, FileNotFoundError, ValueError):
        converted_path = _convert_to_energy_wav(source_path)
        if not converted_path:
            return None
        try:
            with contextlib.closing(wave.open(str(converted_path), "rb")) as wf:
                frames = wf.readframes(wf.getnframes())
                samples = array("h")
                samples.frombytes(frames)
                return samples, wf.getframerate()
        except (wave.Error, FileNotFoundError) as exc:
            logging.warning("A konvertált WAV nem tölthető be energiabecsléshez (%s): %s", source_path, exc)
            return None
        finally:
            try:
                os.remove(str(converted_path))
            except OSError:
                pass


def _compute_energy_frames(
    samples: array,
    sample_rate: int,
    frame_ms: int = ENERGY_WINDOW_MS,
) -> Tuple[List[float], float]:
    """Return RMS energies per frame and the frame duration in seconds."""
    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    if frame_size <= 0:
        frame_size = 1
    energies: List[float] = []
    total_samples = len(samples)
    if total_samples == 0:
        return energies, frame_size / sample_rate if sample_rate else 0.0
    for offset in range(0, total_samples, frame_size):
        chunk = samples[offset : offset + frame_size]
        if not chunk:
            continue
        # RMS energy
        sumsq = 0
        for value in chunk:
            sumsq += value * value
        rms = math.sqrt(sumsq / len(chunk))
        energies.append(rms)
    frame_duration = frame_size / sample_rate if sample_rate else 0.0
    return energies, frame_duration


def _calculate_energy_thresholds(energies: Sequence[float]) -> Tuple[float, float]:
    positives = [e for e in energies if e > 0]
    if not positives:
        return 0.0, 0.0
    positives.sort()
    median_val = statistics.median(positives)
    ninety_idx = min(int(round(0.9 * (len(positives) - 1))), len(positives) - 1)
    p90 = positives[ninety_idx]
    max_val = positives[-1]
    threshold_high = max(median_val * 0.8, p90 * 0.6)
    if max_val > ENERGY_THRESHOLD_FLOOR:
        threshold_high = max(threshold_high, ENERGY_THRESHOLD_FLOOR)
    threshold_high = min(threshold_high, max_val)
    if threshold_high <= 0:
        threshold_high = max_val
    threshold_low = threshold_high * 0.5
    return threshold_high, threshold_low


def refine_word_segments_with_energy(
    word_segments: List[dict],
    audio_path: Path,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Adjust word timestamps using internal content analysis to handle ASR inaccuracies."""
    audio_data = read_wave_for_energy(audio_path)
    if not audio_data:
        return word_segments, {"status": "skipped_audio_unavailable", "audio": audio_path.name}
    
    samples, sample_rate = audio_data
    energies, frame_duration = _compute_energy_frames(samples, sample_rate)
    if not energies or frame_duration <= 0:
        return word_segments, {"status": "skipped_no_energy_frames", "audio": audio_path.name}

    threshold_high, threshold_low = _calculate_energy_thresholds(energies)
    if threshold_high == 0:
        return word_segments, {"status": "skipped_low_energy", "audio": audio_path.name}

    audio_length = len(samples) / sample_rate if sample_rate else 0.0
    num_frames = len(energies)
    refined: List[dict] = []

    for idx, word in enumerate(word_segments):
        original_start = float(word.get("start") or 0.0)
        original_end = float(word.get("end") or original_start)
        
        start_frame = min(max(int(original_start / frame_duration), 0), num_frames - 1)
        end_frame = min(max(int(original_end / frame_duration), 0), num_frames - 1)

        first_active_frame = -1
        last_active_frame = -1

        if start_frame <= end_frame:
            for i in range(start_frame, end_frame + 1):
                if energies[i] > threshold_high:
                    first_active_frame = i
                    break
            
            for i in range(end_frame, start_frame - 1, -1):
                if energies[i] > threshold_high:
                    last_active_frame = i
                    break
        
        adjusted_start_frame: int
        adjusted_end_frame: int

        if first_active_frame != -1 and last_active_frame != -1:
            adjusted_start_frame = first_active_frame
            adjusted_end_frame = last_active_frame
        else:
            adjusted_start_frame = start_frame
            adjusted_end_frame = end_frame

        adjusted_start = adjusted_start_frame * frame_duration
        adjusted_end = (adjusted_end_frame + 1) * frame_duration

        if refined:
            adjusted_start = max(adjusted_start, refined[-1]["end"])
        
        if adjusted_end - adjusted_start < MIN_WORD_DURATION_S:
            adjusted_end = adjusted_start + MIN_WORD_DURATION_S

        adjusted_start = max(0.0, adjusted_start)
        adjusted_end = min(audio_length, adjusted_end)

        refined_word = dict(word)
        refined_word["start"] = round(adjusted_start, 3)
        refined_word["end"] = round(adjusted_end, 3)
        refined.append(refined_word)

    report = {
        "status": "applied_internal_scan",
        "audio": audio_path.name,
        "frame_duration_ms": ENERGY_WINDOW_MS,
        "threshold_high": threshold_high,
        "threshold_low": threshold_low,
        "frames": len(energies),
    }
    return refined, report


class SpeakerChangeDetector:
    """Detect additional speaker-change boundaries from audio using pyannote."""

    def __init__(
        self,
        hf_token: str,
        *,
        preset_name: str,
        device_name: str,
        max_speakers: int,
        refine: bool,
        min_region_override: Optional[float],
    ) -> None:
        self.hf_token = hf_token
        self.preset_name = preset_name
        self.device_name = device_name
        self.max_speakers = max_speakers
        self.refine = refine
        self.min_region_override = min_region_override
        self._runtime: Optional[Dict[str, Any]] = None

    def _ensure_runtime(self) -> Dict[str, Any]:
        if self._runtime is not None:
            return self._runtime

        try:
            import numpy as np
            import soundfile as sf
            import torch
            import torchaudio
            from pyannote.audio import Inference, Model, Pipeline
            from pyannote.core import Annotation, Segment
        except ImportError as exc:
            raise RuntimeError(
                "A speaker change detektáláshoz hiányzó csomag szükséges "
                f"(numpy/soundfile/torch/torchaudio/pyannote): {exc}"
            ) from exc

        if not self.hf_token:
            raise RuntimeError("A speaker change detektáláshoz hiányzik a Hugging Face token.")

        device = torch.device(
            "cuda" if self.device_name == "cuda" and torch.cuda.is_available() else "cpu"
        )
        preset = dict(SPEAKER_CHANGE_PRESETS[self.preset_name])
        if self.min_region_override is not None:
            preset["min_region"] = float(self.min_region_override)

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=self.hf_token,
        )
        pipeline.to(device)

        inference = None
        if self.refine:
            model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=self.hf_token)
            model.to(device)
            inference = Inference(
                model,
                window="sliding",
                duration=10.0,
                step=2.5,
                batch_size=int(preset["seg_batch_size"]),
            )

        self._runtime = {
            "np": np,
            "sf": sf,
            "torch": torch,
            "torchaudio": torchaudio,
            "Inference": Inference,
            "Annotation": Annotation,
            "Segment": Segment,
            "device": device,
            "pipeline": pipeline,
            "inference": inference,
            "preset": preset,
        }
        return self._runtime

    def _load_audio_mono_16k(self, path: Path) -> Tuple[Any, int]:
        runtime = self._ensure_runtime()
        torchaudio = runtime["torchaudio"]

        wav, sr = torchaudio.load(str(path))
        if wav.ndim != 2:
            raise RuntimeError(f"Nem várt waveform alak: {tuple(wav.shape)}")
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000
        return wav.contiguous(), sr

    def _crop_waveform(self, waveform: Any, sr: int, start_sec: float, end_sec: float) -> Any:
        runtime = self._ensure_runtime()
        torch = runtime["torch"]

        start_i = max(0, int(round(start_sec * sr)))
        end_i = min(waveform.shape[1], int(round(end_sec * sr)))
        if end_i <= start_i:
            return torch.zeros((1, 1), dtype=waveform.dtype)
        return waveform[:, start_i:end_i]

    def _save_temp_wav(self, waveform: Any, sample_rate: int) -> str:
        runtime = self._ensure_runtime()
        torchaudio = runtime["torchaudio"]

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="spkchg_")
        os.close(fd)
        torchaudio.save(path, waveform.cpu(), sample_rate)
        return path

    def _audio_duration_seconds(self, path: Path) -> float:
        runtime = self._ensure_runtime()
        sf = runtime["sf"]
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)

    def _run_diarization_on_chunk(self, chunk_path: str) -> Any:
        runtime = self._ensure_runtime()
        pipeline = runtime["pipeline"]
        kwargs: Dict[str, Any] = {}
        if self.max_speakers is not None and self.max_speakers > 0:
            kwargs["max_speakers"] = self.max_speakers
        output = pipeline(chunk_path, **kwargs)
        return output.speaker_diarization if hasattr(output, "speaker_diarization") else output

    def _shift_annotation(self, annotation: Any, offset: float) -> Any:
        runtime = self._ensure_runtime()
        Annotation = runtime["Annotation"]
        Segment = runtime["Segment"]

        shifted = Annotation()
        for seg, track, label in annotation.itertracks(yield_label=True):
            shifted[Segment(seg.start + offset, seg.end + offset), track] = label
        return shifted

    def _trim_annotation(self, annotation: Any, start: float, end: float) -> Any:
        runtime = self._ensure_runtime()
        Annotation = runtime["Annotation"]
        Segment = runtime["Segment"]

        cropped = Annotation()
        focus = Segment(start, end)
        for seg, track, label in annotation.itertracks(yield_label=True):
            inter = seg & focus
            if inter and inter.duration > 0:
                cropped[inter, track] = label
        return cropped

    def _merge_annotations(self, annotations: Sequence[Any]) -> Any:
        runtime = self._ensure_runtime()
        Annotation = runtime["Annotation"]

        merged = Annotation()
        for ann in annotations:
            for seg, track, label in ann.itertracks(yield_label=True):
                merged[seg, f"{track}_{len(merged)}"] = label
        return merged

    def _diarize_long_audio(self, input_path: Path, waveform: Any, sr: int) -> Any:
        runtime = self._ensure_runtime()
        preset = runtime["preset"]

        duration = waveform.shape[1] / sr
        chunk_seconds = float(preset["chunk_seconds"])
        chunk_overlap = float(preset["chunk_overlap"])

        if duration <= chunk_seconds:
            return self._run_diarization_on_chunk(str(input_path))

        annotations = []
        starts: List[float] = []
        current = 0.0
        while current < duration:
            starts.append(current)
            current += chunk_seconds - chunk_overlap

        for idx, start in enumerate(starts):
            end = min(duration, start + chunk_seconds)
            chunk = self._crop_waveform(waveform, sr, start, end)
            temp_path = self._save_temp_wav(chunk, sr)
            try:
                ann = self._run_diarization_on_chunk(temp_path)
                ann = self._shift_annotation(ann, start)
                keep_start = 0.0 if idx == 0 else start + chunk_overlap / 2.0
                keep_end = duration if idx == len(starts) - 1 else end - chunk_overlap / 2.0
                annotations.append(self._trim_annotation(ann, keep_start, keep_end))
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        return self._merge_annotations(annotations)

    def _merge_adjacent_same_type(
        self,
        regions: Sequence[SpeakerChangeRegion],
        gap_tolerance: float = 1e-6,
    ) -> List[SpeakerChangeRegion]:
        if not regions:
            return []
        out = [SpeakerChangeRegion(regions[0].start, regions[0].end, regions[0].region_type)]
        for region in regions[1:]:
            last = out[-1]
            if region.region_type == last.region_type and region.start <= last.end + gap_tolerance:
                last.end = max(last.end, region.end)
            else:
                out.append(SpeakerChangeRegion(region.start, region.end, region.region_type))
        return out

    def _annotation_to_atomic_regions(self, annotation: Any, duration: float) -> List[SpeakerChangeRegion]:
        boundaries = {0.0, duration}
        for seg in annotation.itersegments():
            boundaries.add(float(seg.start))
            boundaries.add(float(seg.end))

        regions: List[SpeakerChangeRegion] = []
        times = sorted(boundaries)
        segments = list(annotation.itersegments())
        for start, end in zip(times[:-1], times[1:]):
            if end <= start:
                continue
            mid = (start + end) / 2.0
            active = 0
            for seg in segments:
                if seg.start <= mid < seg.end:
                    active += 1
            if active <= 0:
                continue
            region_type = "overlap" if active >= 2 else "single"
            regions.append(SpeakerChangeRegion(start, end, region_type))
        return self._merge_adjacent_same_type(regions)

    def _filter_short_regions(
        self,
        regions: Sequence[SpeakerChangeRegion],
        min_region: float,
        preserve_edges: bool = True,
    ) -> List[SpeakerChangeRegion]:
        if not regions:
            return []

        filtered = [SpeakerChangeRegion(r.start, r.end, r.region_type) for r in regions]
        changed = True
        while changed and len(filtered) > 1:
            changed = False
            for idx, region in enumerate(list(filtered)):
                if region.duration() >= min_region:
                    continue
                if idx == 0 and preserve_edges:
                    filtered[1].start = filtered[0].start
                    del filtered[0]
                    changed = True
                    break
                if idx == len(filtered) - 1 and preserve_edges:
                    filtered[-2].end = filtered[-1].end
                    del filtered[-1]
                    changed = True
                    break

                left = filtered[idx - 1]
                right = filtered[idx + 1]
                if left.region_type == right.region_type:
                    filtered[idx - 1 : idx + 2] = [
                        SpeakerChangeRegion(left.start, right.end, left.region_type)
                    ]
                elif left.duration() >= right.duration():
                    left.end = region.end
                    del filtered[idx]
                else:
                    right.start = region.start
                    del filtered[idx]
                changed = True
                break

        return self._merge_adjacent_same_type(filtered)

    def _regions_to_change_points(
        self,
        regions: Sequence[SpeakerChangeRegion],
    ) -> List[SpeakerChangePoint]:
        return [
            SpeakerChangePoint(time=left.end, left_type=left.region_type, right_type=right.region_type)
            for left, right in zip(regions[:-1], regions[1:])
        ]

    def _moving_average(self, values: Any, window_size: int) -> Any:
        runtime = self._ensure_runtime()
        np = runtime["np"]

        if window_size <= 1:
            return values
        pad = window_size // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
        return np.convolve(padded, kernel, mode="valid")

    def _infer_local_segmentation_scores(
        self,
        waveform_16k: Any,
        sr: int,
        center_time: float,
    ) -> Tuple[Any, Any]:
        runtime = self._ensure_runtime()
        np = runtime["np"]
        torch = runtime["torch"]
        inference = runtime["inference"]
        device = runtime["device"]

        start = max(0.0, center_time - 5.0)
        end = start + 10.0
        whole_duration = waveform_16k.shape[1] / sr
        if end > whole_duration:
            end = whole_duration
            start = max(0.0, end - 10.0)

        crop = self._crop_waveform(waveform_16k, sr, start, end)
        if crop.shape[1] < int(10.0 * sr):
            crop = torch.nn.functional.pad(crop, (0, int(10.0 * sr) - crop.shape[1]))

        with torch.inference_mode():
            scores = inference({"waveform": crop.to(device), "sample_rate": sr})

        data = scores.data
        sliding_window = scores.sliding_window
        times = np.array(
            [start + sliding_window[i].middle for i in range(data.shape[0])],
            dtype=np.float64,
        )
        return times, np.asarray(data, dtype=np.float32)

    def _refine_boundary_with_segmentation(
        self,
        change_point: SpeakerChangePoint,
        waveform_16k: Any,
        sr: int,
    ) -> float:
        runtime = self._ensure_runtime()
        np = runtime["np"]
        preset = runtime["preset"]

        try:
            times, scores = self._infer_local_segmentation_scores(waveform_16k, sr, change_point.time)
        except Exception:
            return change_point.time

        if scores.ndim != 2 or scores.shape[1] < 7 or scores.shape[0] < 3:
            return change_point.time

        state_map = {
            0: "nonspeech",
            1: "single",
            2: "single",
            3: "single",
            4: "overlap",
            5: "overlap",
            6: "overlap",
        }
        labels = scores.argmax(axis=1)
        states = np.array([state_map.get(int(label), "single") for label in labels], dtype=object)

        low = change_point.time - float(preset["refine_radius"])
        high = change_point.time + float(preset["refine_radius"])
        idx = np.where((times >= low) & (times <= high))[0]
        if idx.size < 2:
            return change_point.time

        target = None
        for position in idx[:-1]:
            if (
                states[position] == change_point.left_type
                and states[position + 1] == change_point.right_type
            ):
                target = float((times[position] + times[position + 1]) / 2.0)
                break

        if target is None:
            diff = np.abs(np.diff(scores, axis=0)).sum(axis=1)
            diff = self._moving_average(
                diff.astype(np.float32),
                int(preset["fallback_variation_smooth"]),
            )
            diff_idx = np.where((times[:-1] >= low) & (times[:-1] <= high))[0]
            if diff_idx.size:
                best = int(diff_idx[np.argmax(diff[diff_idx])])
                target = float((times[best] + times[best + 1]) / 2.0)
            else:
                target = change_point.time

        if abs(target - change_point.time) > float(preset["boundary_snap_max_shift"]):
            return change_point.time
        return target

    def _refine_regions(self, regions: Sequence[SpeakerChangeRegion], waveform_16k: Any, sr: int) -> List[SpeakerChangeRegion]:
        runtime = self._ensure_runtime()
        preset = runtime["preset"]

        change_points = self._regions_to_change_points(regions)
        if not change_points:
            return list(regions)

        refined_times = [
            self._refine_boundary_with_segmentation(change_point, waveform_16k, sr)
            for change_point in change_points
        ]

        refined: List[SpeakerChangeRegion] = []
        left = regions[0].start
        for idx, region in enumerate(regions):
            if idx < len(refined_times):
                boundary = max(left, refined_times[idx])
                if boundary < region.end:
                    refined.append(SpeakerChangeRegion(left, boundary, region.region_type))
                    left = boundary
                else:
                    refined.append(SpeakerChangeRegion(left, region.end, region.region_type))
                    left = region.end
            else:
                refined.append(SpeakerChangeRegion(left, region.end, region.region_type))

        refined = self._merge_adjacent_same_type(refined)
        return self._filter_short_regions(refined, float(preset["min_region"]))

    def detect_change_points(self, audio_path: Path) -> Tuple[List[float], Dict[str, Any]]:
        runtime = self._ensure_runtime()
        preset = runtime["preset"]

        waveform, sr = self._load_audio_mono_16k(audio_path)
        duration = self._audio_duration_seconds(audio_path)
        annotation = self._diarize_long_audio(audio_path, waveform, sr)

        regions = self._annotation_to_atomic_regions(annotation, duration)
        regions = self._filter_short_regions(regions, float(preset["min_region"]))
        if self.refine and runtime["inference"] is not None:
            regions = self._refine_regions(regions, waveform, sr)

        regions = self._merge_adjacent_same_type(regions)
        regions = self._filter_short_regions(regions, float(preset["min_region"]))
        change_points = self._regions_to_change_points(regions)

        dedup_tolerance = float(preset["dedup_tolerance"])
        deduped: List[float] = []
        for change_point in change_points:
            if deduped and abs(change_point.time - deduped[-1]) <= dedup_tolerance:
                deduped[-1] = round((deduped[-1] + change_point.time) / 2.0, 3)
            else:
                deduped.append(round(change_point.time, 3))

        report = {
            "status": "applied",
            "audio": audio_path.name,
            "preset": self.preset_name,
            "device": str(runtime["device"]),
            "refined": self.refine,
            "regions": len(regions),
            "change_points": len(deduped),
            "min_region": float(preset["min_region"]),
        }
        return deduped, report


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _extract_speaker(entry: dict) -> Optional[str]:
    """Try to pull a speaker label from various possible fields."""
    for key in ("speaker", "speaker_label", "speaker_id", "speaker_tag", "spk"):
        if key in entry and entry[key] is not None and str(entry[key]) != "":
            return str(entry[key])
    return None


def _normalise_word_entry(entry: dict) -> Optional[dict]:
    text = _sanitize_text(entry.get("word") or entry.get("text") or entry.get("token") or "")
    if not text:
        return None
    start = (
        _safe_float(entry.get("start"))
        or _safe_float(entry.get("start_time"))
        or _safe_float(entry.get("offset_seconds"))
        or _safe_float(entry.get("offset"))
    )
    end = (
        _safe_float(entry.get("end"))
        or _safe_float(entry.get("end_time"))
        or _safe_float(entry.get("offset_seconds_end"))
    )
    if start is None:
        return None
    if end is None:
        duration = _safe_float(entry.get("duration")) or 0.0
        end = start + duration
    confidence = _safe_float(entry.get("confidence") or entry.get("score"))
    speaker = _extract_speaker(entry)
    return {
        "word": text,
        "start": round(start, 3),
        "end": round(end or start, 3),
        "score": round(confidence, 4) if confidence is not None else None,
        "speaker": speaker,
    }


def extract_word_segments(payload: dict) -> List[dict]:
    raw_candidates = []
    if isinstance(payload.get("words"), list):
        raw_candidates = payload["words"]
    elif isinstance(payload.get("word_segments"), list):
        raw_candidates = payload["word_segments"]
    elif isinstance(payload.get("segments"), list):
        collected: List[dict] = []
        for segment in payload["segments"]:
            words = segment.get("words") or []
            if isinstance(words, list):
                collected.extend(words)
        raw_candidates = collected

    normalised: List[dict] = []
    for entry in raw_candidates:
        if isinstance(entry, dict):
            normalised_word = _normalise_word_entry(entry)
            if normalised_word:
                normalised.append(normalised_word)
    normalised.sort(key=lambda item: item["start"])
    return normalised


def adjust_word_timestamps(word_segments: List[dict], padding_s: float) -> List[dict]:
    if not word_segments or padding_s <= 0:
        return word_segments
    adjusted = [dict(word) for word in word_segments]
    for idx in range(len(adjusted) - 1):
        gap = adjusted[idx + 1]["start"] - adjusted[idx]["end"]
        if gap > padding_s * 2:
            delta = min(padding_s, gap / 2.0)
            adjusted[idx]["end"] += delta
            adjusted[idx + 1]["start"] -= delta
    adjusted[0]["start"] = max(0.0, adjusted[0]["start"] - padding_s)
    adjusted[-1]["end"] += padding_s
    for word in adjusted:
        word["start"] = round(word["start"], 3)
        word["end"] = round(word["end"], 3)
    return adjusted


def _create_segment_from_words(words: List[dict]) -> Optional[dict]:
    if not words:
        return None
    text = _sanitize_text(" ".join(word["word"] for word in words))
    seg = {
        "start": round(words[0]["start"], 3),
        "end": round(words[-1]["end"], 3),
        "text": text,
        "words": words,
    }
    speakers = {w.get("speaker") for w in words if w.get("speaker") is not None}
    if len(speakers) == 1:
        seg["speaker"] = next(iter(speakers))
    elif len(speakers) > 1:
        speaker_counts = {}
        for word in words:
            speaker = word.get("speaker")
            if speaker is not None:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        if speaker_counts:
            dominant_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
            seg["speaker"] = dominant_speaker
            seg["mixed_speakers"] = True
    return seg


def map_change_points_to_word_boundaries(
    words: Sequence[dict],
    change_points: Sequence[float],
    max_shift_s: float,
) -> Set[int]:
    """Map continuous change-point timestamps to discrete word-boundary indices."""
    if len(words) < 2 or not change_points:
        return set()

    boundaries: List[Tuple[int, float]] = []
    for idx in range(1, len(words)):
        prev_word = words[idx - 1]
        next_word = words[idx]
        boundary_time = (float(prev_word["end"]) + float(next_word["start"])) / 2.0
        boundaries.append((idx, boundary_time))

    forced_indices: Set[int] = set()
    for change_time in sorted(change_points):
        best_idx: Optional[int] = None
        best_distance: Optional[float] = None
        for idx, boundary_time in boundaries:
            distance = abs(boundary_time - change_time)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_idx = idx
        if best_idx is not None and best_distance is not None and best_distance <= max_shift_s:
            forced_indices.add(best_idx)
    return forced_indices


def sentence_segments_from_words(
    words: List[dict],
    max_pause_s: float,
    *,
    split_on_speaker_change: bool = True,
    forced_split_indices: Optional[Set[int]] = None,
) -> List[dict]:
    if not words:
        return []
    segments: List[dict] = []
    current: List[dict] = []
    forced_split_indices = forced_split_indices or set()
    for idx, word in enumerate(words):
        forced_break = idx in forced_split_indices and bool(current)
        pause_break = current and (word["start"] - current[-1]["end"]) > max_pause_s
        speaker_break = (
            split_on_speaker_change
            and current
            and current[-1].get("speaker") is not None
            and word.get("speaker") is not None
            and word.get("speaker") != current[-1].get("speaker")
        )
        if forced_break or pause_break or speaker_break:
            segment = _create_segment_from_words(current)
            if segment:
                segments.append(segment)
            current = []
        current.append(word)
    if current:
        segment = _create_segment_from_words(current)
        if segment:
            segments.append(segment)
    return segments


def split_long_segments(segments: List[dict], max_duration_s: float) -> List[dict]:
    if max_duration_s <= 0:
        return segments
    final_segments: List[dict] = []

    def pick_split_index(order_words: List[dict], punctuation: Tuple[str, ...]) -> Optional[int]:
        chosen: Optional[int] = None
        for idx in range(len(order_words) - 1):
            token = order_words[idx]["word"].strip()
            if not token:
                continue
            if token.endswith(punctuation):
                duration = order_words[idx]["end"] - order_words[0]["start"]
                if duration <= max_duration_s:
                    chosen = idx
        return chosen

    def split_by_equal_parts(order_words: List[dict], parts: int) -> List[List[dict]]:
        if parts <= 1 or len(order_words) <= 1:
            return [order_words]
        total_duration = order_words[-1]["end"] - order_words[0]["start"]
        if total_duration <= 0:
            return [order_words]
        target = total_duration / parts
        boundaries = [order_words[0]["start"] + target * k for k in range(1, parts)]
        chunks: List[List[dict]] = []
        start_idx = 0
        for boundary in boundaries:
            idx = start_idx
            while idx < len(order_words) - 1 and order_words[idx]["end"] < boundary:
                idx += 1
            chunk = order_words[start_idx : idx + 1]
            if chunk:
                chunks.append(chunk)
                start_idx = idx + 1
        if start_idx < len(order_words):
            chunks.append(order_words[start_idx:])
        if not chunks:
            return [order_words]
        return chunks

    queue: List[List[dict]] = [segment["words"] for segment in segments if segment.get("words")]

    while queue:
        current_words = queue.pop(0)
        if not current_words:
            continue
        duration = current_words[-1]["end"] - current_words[0]["start"]
        if duration <= max_duration_s or len(current_words) == 1:
            segment = _create_segment_from_words(current_words)
            if segment:
                final_segments.append(segment)
            continue

        split_idx = pick_split_index(current_words, PRIMARY_PUNCTUATION)
        if split_idx is None:
            split_idx = pick_split_index(current_words, SECONDARY_PUNCTUATION)

        if split_idx is not None:
            left = current_words[: split_idx + 1]
            right = current_words[split_idx + 1 :]
            if left:
                queue.insert(0, left)
            if right:
                queue.insert(1, right)
            continue

        parts = max(2, math.ceil(duration / max_duration_s))
        equal_chunks = split_by_equal_parts(current_words, parts)
        if len(equal_chunks) == 1:
            segment = _create_segment_from_words(equal_chunks[0])
            if segment:
                final_segments.append(segment)
        else:
            for chunk in reversed(equal_chunks):
                if chunk:
                    queue.insert(0, chunk)

    return final_segments


def build_segments(
    word_segments: List[dict],
    max_pause_s: float,
    padding_s: float,
    max_segment_s: float,
    *,
    enforce_single_speaker: bool,
    speaker_change_points: Optional[Sequence[float]] = None,
    speaker_change_max_shift_s: float = 0.25,
) -> List[dict]:
    if not word_segments:
        return []
    adjusted = adjust_word_timestamps(word_segments, padding_s=padding_s)
    forced_split_indices = map_change_points_to_word_boundaries(
        adjusted,
        speaker_change_points or [],
        max_shift_s=speaker_change_max_shift_s,
    )
    initial_segments = sentence_segments_from_words(
        adjusted,
        max_pause_s=max_pause_s,
        split_on_speaker_change=enforce_single_speaker,
        forced_split_indices=forced_split_indices,
    )
    return split_long_segments(initial_segments, max_segment_s)


def backup_json_file(json_path: Path) -> Path:
    """
    Create a backup of the JSON file with .json.bak extension.
    If a backup already exists, create a numbered version (.bak_2, .bak_3, etc.).
    """
    base_backup_str = str(json_path.with_suffix(".json.bak"))
    backup_path = Path(base_backup_str)

    if backup_path.exists():
        counter = 2
        while True:
            numbered_backup_path = Path(f"{base_backup_str}_{counter}")
            if not numbered_backup_path.exists():
                backup_path = numbered_backup_path
                break
            counter += 1
            
    try:
        shutil.copy2(json_path, backup_path)
        logging.info(f"Biztonsági mentés létrehozva: {backup_path.name}")
        return backup_path
    except Exception as exc:
        logging.error(f"Hiba a biztonsági mentés létrehozásakor ({json_path}): {exc}")
        raise


def load_json_file(json_path: Path) -> dict:
    """Load JSON file and return its content."""
    try:
        with json_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logging.error(f"Hiba a JSON fájl betöltésekor ({json_path}): {exc}")
        raise


def create_resegmented_json(
    original_data: dict,
    max_pause_s: float,
    padding_s: float,
    max_segment_s: float,
    enforce_single_speaker: bool,
    *,
    source_label: str,
    audio_path: Optional[Path],
    energy_refine: bool,
    word_by_word: bool,
    speaker_change_detector: Optional[SpeakerChangeDetector],
) -> dict:
    """Create new JSON with resegmented data."""
    word_segments = extract_word_segments(original_data)
    
    if not word_segments:
        logging.warning("Nincsenek szó szegmensek a JSON fájlban")
        return original_data

    energy_report: Optional[Dict[str, Any]] = None
    speaker_change_report: Optional[Dict[str, Any]] = None
    speaker_change_points: List[float] = []
    if energy_refine:
        if audio_path:
            refined_segments, energy_report = refine_word_segments_with_energy(
                word_segments,
                audio_path,
            )
            if "applied" in energy_report.get("status", ""):
                word_segments = refined_segments
                logging.debug(
                    "Energia alapú korrekció alkalmazva (%s)",
                    audio_path.name,
                )
            else:
                logging.info(
                    "Energia alapú korrekció kihagyva (%s): %s",
                    audio_path.name,
                    energy_report.get("status"),
                )
        else:
            energy_report = {
                "status": "skipped_audio_missing",
                "audio": None,
            }
            logging.warning("Nem található hangfájl energiabecsléshez (JSON: %s).", source_label)

    if speaker_change_detector:
        if audio_path:
            try:
                speaker_change_points, speaker_change_report = speaker_change_detector.detect_change_points(
                    audio_path
                )
            except Exception as exc:
                speaker_change_report = {
                    "status": "failed",
                    "audio": audio_path.name,
                    "error": str(exc),
                }
                logging.warning(
                    "Speaker change detektálás sikertelen (%s): %s",
                    audio_path.name,
                    exc,
                )
        else:
            speaker_change_report = {
                "status": "skipped_audio_missing",
                "audio": None,
            }
            logging.warning("Nem található hangfájl speaker change detektáláshoz (JSON: %s).", source_label)

    segments: List[dict]
    if word_by_word:
        logging.info("Egyszavas szegmensek létrehozása a '--word-by-word-segments' kapcsoló miatt.")
        segments = []
        for word in word_segments:
            segment = {
                "start": word["start"],
                "end": word["end"],
                "text": word["word"],
                "words": [word],
            }
            if word.get("speaker"):
                segment["speaker"] = word["speaker"]
            segments.append(segment)
    else:
        segments = build_segments(
            word_segments,
            max_pause_s=max_pause_s,
            padding_s=padding_s,
            max_segment_s=max_segment_s,
            enforce_single_speaker=enforce_single_speaker,
            speaker_change_points=speaker_change_points,
            speaker_change_max_shift_s=(
                float(SPEAKER_CHANGE_PRESETS[speaker_change_detector.preset_name]["boundary_snap_max_shift"])
                if speaker_change_detector
                else 0.25
            ),
        )

    ### ÚJ SZŰRÉSI LOGIKA ###
    # A zárójeles (pl. [zene]) szegmensek eltávolítása a végső kimenetből.
    # A word_segments listában megmaradnak.
    final_segments = [
        seg for seg in segments 
        if not (seg.get("text", "").strip().startswith("[") and seg.get("text", "").strip().endswith("]"))
    ]

    result = {
        "segments": final_segments,
        "word_segments": word_segments,
        "language": original_data.get("language"),
        "provider": original_data.get("provider", "resegment"),
        "diarization": original_data.get("diarization", False),
        "original_provider": original_data.get("provider"),
        "resegment_parameters": {
            "max_pause_s": max_pause_s,
            "padding_s": padding_s,
            "max_segment_s": max_segment_s,
            "enforce_single_speaker": enforce_single_speaker,
            "energy_refine": energy_refine,
            "energy_window_ms": ENERGY_WINDOW_MS,
            "word_by_word_segments": word_by_word,
            "speaker_change_detection": bool(speaker_change_detector),
        }
    }
    if audio_path:
        result["audio_file"] = audio_path.name
    if energy_report:
        result["energy_adjustment"] = energy_report
    if speaker_change_report:
        result["speaker_change_detection"] = speaker_change_report
    if speaker_change_points:
        result["speaker_change_points"] = speaker_change_points

    return result


def process_json_file(
    json_path: Path,
    max_pause_s: float,
    padding_s: float,
    max_segment_s: float,
    enforce_single_speaker: bool,
    backup: bool,
    energy_refine: bool,
    word_by_word: bool,
    speaker_change_detector: Optional[SpeakerChangeDetector],
) -> None:
    """Process a single JSON file: backup, load, resegment, and save."""
    print(f"▶  Feldolgozás: {json_path.name}")
    
    if backup:
        try:
            backup_json_file(json_path)
        except Exception:
            print(f"  ✖  Sikertelen biztonsági mentés: {json_path.name}")
            return

    try:
        original_data = load_json_file(json_path)
    except Exception:
        print(f"  ✖  Sikertelen JSON betöltés: {json_path.name}")
        return

    audio_path: Optional[Path] = None
    if energy_refine or speaker_change_detector:
        audio_path = find_audio_for_json(json_path)
        if audio_path is None:
            if energy_refine:
                logging.warning("Nem található hozzárendelt hangfájl energiabecsléshez (%s).", json_path.name)
            if speaker_change_detector:
                logging.warning("Nem található hozzárendelt hangfájl speaker change detektáláshoz (%s).", json_path.name)

    try:
        resegmented_data = create_resegmented_json(
            original_data,
            max_pause_s=max_pause_s,
            padding_s=padding_s,
            max_segment_s=max_segment_s,
            enforce_single_speaker=enforce_single_speaker,
            source_label=json_path.name,
            audio_path=audio_path,
            energy_refine=energy_refine,
            word_by_word=word_by_word,
            speaker_change_detector=speaker_change_detector,
        )
    except Exception as exc:
        logging.error(f"Hiba a szegmentálás során ({json_path}): {exc}", exc_info=True)
        print(f"  ✖  Szegmentálási hiba: {json_path.name}")
        return

    try:
        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(resegmented_data, fp, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"  ✔  Mentve: {json_path.name}")
    except Exception as exc:
        logging.error(f"Hiba a mentés során ({json_path}): {exc}")
        print(f"  ✖  Mentési hiba: {json_path.name}")


def process_directory(
    input_dir: Path,
    max_pause_s: float,
    padding_s: float,
    max_segment_s: float,
    enforce_single_speaker: bool,
    backup: bool,
    energy_refine: bool,
    word_by_word: bool,
    speaker_change_detector: Optional[SpeakerChangeDetector],
) -> None:
    """Process all JSON files in the directory."""
    json_files = sorted([path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json"])

    if not json_files:
        print(f"Nem található JSON fájl a megadott mappában: {input_dir}")
        return

    print(f"{len(json_files)} JSON fájl feldolgozása indul...")
    for json_path in json_files:
        print("-" * 48)
        process_json_file(
            json_path,
            max_pause_s=max_pause_s,
            padding_s=padding_s,
            max_segment_s=max_segment_s,
            enforce_single_speaker=enforce_single_speaker,
            backup=backup,
            energy_refine=energy_refine,
            word_by_word=word_by_word,
            speaker_change_detector=speaker_change_detector,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "JSON szegmensek újraformázása - ASR script által létrehozott JSON fájlok "
            "biztonsági mentése és újraformázása különböző paraméterekkel."
        )
    )
    parser.add_argument(
        "-p",
        "--project-name",
        required=True,
        help="A projekt neve (a workdir alatti mappa), amit fel kell dolgozni.",
    )
    parser.add_argument(
        "--max-pause",
        type=float,
        default=DEFAULT_MAX_PAUSE_S,
        help=f"Mondatszegmensek közti maximális szünet (mp) saját szegmentáláshoz (alapértelmezett: {DEFAULT_MAX_PAUSE_S}).",
    )
    parser.add_argument(
        "--timestamp-padding",
        type=float,
        default=DEFAULT_PADDING_S,
        help=f"Szó időbélyegek bővítése szegmentáláskor (mp, alapértelmezett: {DEFAULT_PADDING_S}).",
    )
    parser.add_argument(
        "--max-segment-duration",
        type=float,
        default=DEFAULT_MAX_SEGMENT_S,
        help=f"Mondatszegmensek maximális hossza (mp, alapértelmezett: {DEFAULT_MAX_SEGMENT_S}).",
    )
    parser.add_argument(
        "--enforce-single-speaker",
        dest="enforce_single_speaker",
        action="store_true",
        default=False,
        help="Speaker diarizáció alapján szegmentálás (alapértelmezett: ki)",
    )
    parser.add_argument(
        "--no-enforce-single-speaker",
        dest="enforce_single_speaker",
        action="store_false",
        help="Speaker diarizáció figyelmen kívül hagyása szegmentáláskor",
    )
    parser.add_argument(
        "--backup_disable",
        action="store_true",
        default=False,
        help="JSON fájlok biztonsági mentésének kikapcsolása (alapértelmezett: ki).",
    )
    parser.add_argument(
        "--skip-energy-refine",
        dest="skip_energy_refine",
        action="store_true",
        default=False,
        help="Szó időbélyegek energiabecslésen alapuló korrekciójának kihagyása.",
    )
    parser.add_argument(
        "--word-by-word-segments",
        action="store_true",
        default=False,
        help="Minden szót külön szegmensbe helyez, az intelligens újraegyesítés kihagyásával.",
    )
    parser.add_argument(
        "--detect-speaker-changes",
        action="store_true",
        default=False,
        help=(
            "Pyannote-alapú extra speaker change pontokat detektál a hangból, "
            "és ezeknél is új szegmenst hoz létre."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN", ""),
        help=(
            "Hugging Face token a speaker change detektáláshoz. "
            "A --detect-speaker-changes használatakor kötelező. "
            "Alapértelmezés: HF_TOKEN env."
        ),
    )
    parser.add_argument(
        "--speaker-change-preset",
        choices=sorted(SPEAKER_CHANGE_PRESETS.keys()),
        default=DEFAULT_SPEAKER_CHANGE_PRESET,
        help=f"Speaker change detektálási preset (alapértelmezett: {DEFAULT_SPEAKER_CHANGE_PRESET}).",
    )
    parser.add_argument(
        "--speaker-change-device",
        choices=["cuda", "cpu"],
        default=DEFAULT_SPEAKER_CHANGE_DEVICE,
        help=f"Eszköz a speaker change detektáláshoz (alapértelmezett: {DEFAULT_SPEAKER_CHANGE_DEVICE}).",
    )
    parser.add_argument(
        "--speaker-change-max-speakers",
        type=int,
        default=50,
        help="Maximális speakerszám a speaker change detektáláshoz (alapértelmezett: 50).",
    )
    parser.add_argument(
        "--speaker-change-min-region",
        type=float,
        default=None,
        help="Felülírja a preset minimum régióhosszát speaker change detektáláskor.",
    )
    parser.add_argument(
        "--speaker-change-no-refine",
        action="store_true",
        default=False,
        help="Kikapcsolja a segmentation-3.0 alapú boundary refine lépést.",
    )

    add_debug_argument(parser)
    args = parser.parse_args()

    if args.detect_speaker_changes and not args.hf_token:
        parser.error(
            "A --detect-speaker-changes használatához kötelező a --hf-token "
            "kapcsoló vagy a HF_TOKEN env változó, mert a pyannote modellek "
            "Hugging Face hitelesítést igényelnek."
        )

    log_level = configure_debug_mode(args.debug)
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    config, project_root = load_config()
    input_dir = resolve_project_input(args.project_name, config, project_root)
    
    print("Projekt beállítások betöltve:")
    print(f"  - Projekt név:    {args.project_name}")
    print(f"  - Bemeneti mappa: {input_dir}")
    print(f"  - Max szünet:     {args.max_pause} s")
    print(f"  - Időbélyeg padding: {args.timestamp_padding} s")
    print(f"  - Max szegmens hossz: {args.max_segment_duration} s")
    print(f"  - Speaker szegmentálás: {'BE' if args.enforce_single_speaker else 'KI'}")
    backup_enabled = not args.backup_disable
    print(f"  - Biztonsági mentés: {'BE' if backup_enabled else 'KI'}")
    energy_enabled = not args.skip_energy_refine
    print(f"  - Energia alapú korrekció: {'BE' if energy_enabled else 'KI'}")
    print(f"  - Egyszavas szegmensek: {'BE' if args.word_by_word_segments else 'KI'}")
    print(f"  - Speaker change detektálás: {'BE' if args.detect_speaker_changes else 'KI'}")

    speaker_change_detector: Optional[SpeakerChangeDetector] = None
    if args.detect_speaker_changes:
        speaker_change_detector = SpeakerChangeDetector(
            args.hf_token,
            preset_name=args.speaker_change_preset,
            device_name=args.speaker_change_device,
            max_speakers=args.speaker_change_max_speakers,
            refine=not args.speaker_change_no_refine,
            min_region_override=args.speaker_change_min_region,
        )

    process_directory(
        input_dir=input_dir,
        max_pause_s=args.max_pause,
        padding_s=args.timestamp_padding,
        max_segment_s=args.max_segment_duration,
        enforce_single_speaker=args.enforce_single_speaker,
        backup=backup_enabled,
        energy_refine=energy_enabled,
        word_by_word=args.word_by_word_segments,
        speaker_change_detector=speaker_change_detector,
    )


if __name__ == "__main__":
    main()

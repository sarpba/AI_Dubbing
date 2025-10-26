"""
Pitch analyzer utility that compares a narrator reference clip against
translated split outputs within a project.

The script loads the project configuration, finds the narrator reference
audio (single WAV/FLAC/OGG file), estimates its median pitch, then compares
every audio file in `translated_splits` to that baseline. Differences that
exceed the configured tolerance are highlighted in the output.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment dependent
    np = None  # type: ignore[assignment]

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - environment dependent
    sf = None  # type: ignore[assignment]

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

logger = logging.getLogger("pitch_analizer")

# Allowed audio suffixes that soundfile can typically open.
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg"}


@dataclass
class PitchResult:
    """Summary statistics about the detected pitch of an audio clip."""

    path: Path
    sample_rate: int
    mean_hz: float
    median_hz: float
    stdev_hz: float
    voiced_frames: int
    total_frames: int

    @property
    def voiced_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.voiced_frames / self.total_frames


def find_project_root(start: Path | None = None) -> Path:
    """
    Ascend from the given path (or this file) to the repository root
    that contains `config.json`.
    """
    reference = start or Path(__file__).resolve()
    for candidate in reference.parents:
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")


def load_config(project_root: Path) -> dict:
    """Load the root config.json file."""
    config_path = project_root / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Hiba a konfiguráció betöltésekor (%s): %s", config_path, exc)
        sys.exit(1)


def resolve_relative_path(raw: str, project_root: Path) -> Path:
    """
    Resolve a path argument by checking absolute, CWD-relative, then
    project-relative locations.
    """
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (project_root / candidate).resolve()


def list_audio_files(directory: Path) -> List[Path]:
    """Return audio files with known extensions inside a directory."""
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    """
    Load an audio file as mono float32 samples.

    The input is demeaned to remove DC offsets before further analysis.
    """
    data, sample_rate = sf.read(path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32, copy=False)
    if data.size == 0:
        return data, sample_rate
    data = data - np.mean(data)
    max_abs = np.max(np.abs(data))
    if max_abs > 0:
        data = data / max_abs
    return data, sample_rate


def iterate_frames(signal: np.ndarray, frame_length: int, hop_length: int) -> Iterable[np.ndarray]:
    """Yield overlapping frames from the signal."""
    if frame_length <= 0 or hop_length <= 0:
        return
    for offset in range(0, len(signal) - frame_length + 1, hop_length):
        yield signal[offset : offset + frame_length]


def detect_frame_pitch(
    frame: np.ndarray,
    sample_rate: int,
    min_frequency: float,
    max_frequency: float,
    min_autocorr: float,
) -> float:
    """
    Estimate the dominant pitch of a single frame via autocorrelation.
    Returns 0.0 when the frame is considered unvoiced.
    """
    if sample_rate <= 0:
        return 0.0

    window = np.hanning(len(frame))
    windowed = frame * window

    autocorr = np.correlate(windowed, windowed, mode="full")[len(frame) - 1 :]
    if autocorr[0] <= 1e-9:
        return 0.0

    autocorr = autocorr / autocorr[0]

    min_period = max(1, int(round(sample_rate / max_frequency)))
    max_period = max(min_period + 1, int(round(sample_rate / min_frequency)))
    if max_period >= len(autocorr):
        max_period = len(autocorr) - 1
    if max_period <= min_period:
        return 0.0

    segment = autocorr[min_period : max_period + 1]
    peak_index = int(np.argmax(segment))
    peak_value = segment[peak_index]
    if peak_value < min_autocorr:
        return 0.0

    lag = min_period + peak_index
    if lag <= 0:
        return 0.0
    return float(sample_rate / lag)


def summarize_pitch(
    signal: np.ndarray,
    sample_rate: int,
    min_frequency: float,
    max_frequency: float,
    energy_threshold: float,
    min_autocorr: float,
    frame_duration: float = 0.05,
    hop_ratio: float = 0.25,
) -> PitchResult:
    """
    Estimate per-frame pitch and summarize the distribution for a clip.
    """
    frame_length = max(1024, int(sample_rate * frame_duration))
    hop_length = max(1, int(frame_length * hop_ratio))

    total_frames = 0
    voiced_frames = 0
    voiced_pitches: List[float] = []

    for frame in iterate_frames(signal, frame_length, hop_length):
        total_frames += 1
        rms = float(np.sqrt(np.mean(frame**2))) if frame.size else 0.0
        if rms < energy_threshold:
            continue
        pitch = detect_frame_pitch(
            frame,
            sample_rate=sample_rate,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            min_autocorr=min_autocorr,
        )
        if pitch <= 0:
            continue
        voiced_frames += 1
        voiced_pitches.append(pitch)

    if voiced_pitches:
        pitches = np.array(voiced_pitches, dtype=np.float64)
        mean_hz = float(np.mean(pitches))
        median_hz = float(np.median(pitches))
        stdev_hz = float(np.std(pitches, ddof=1)) if pitches.size > 1 else 0.0
    else:
        mean_hz = median_hz = stdev_hz = 0.0

    return PitchResult(
        path=Path(),
        sample_rate=sample_rate,
        mean_hz=mean_hz,
        median_hz=median_hz,
        stdev_hz=stdev_hz,
        voiced_frames=voiced_frames,
        total_frames=total_frames,
    )


def analyze_clip(path: Path, min_frequency: float, max_frequency: float) -> PitchResult:
    """
    Load a clip and compute its pitch summary.
    """
    signal, sample_rate = load_audio_mono(path)
    summary = summarize_pitch(
        signal=signal,
        sample_rate=sample_rate,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        energy_threshold=0.01,
        min_autocorr=0.1,
    )
    summary.path = path
    return summary


def analyze_clip_worker(args: Tuple[str, float, float]) -> PitchResult:
    """
    Helper wrapper so ProcessPoolExecutor can call analyze_clip with serialized inputs.
    """
    path_str, min_frequency, max_frequency = args
    return analyze_clip(Path(path_str), min_frequency=min_frequency, max_frequency=max_frequency)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Narrátor hangmagasság ellenőrzése: összeveti a referencia mintát minden "
            "translated_splits audióval, és jelzi az eltérést."
        )
    )
    parser.add_argument(
        "-p",
        "--project-name",
        required=True,
        help="Projekt neve, amelynek translated_splits könyvtárát ellenőrizzük.",
    )
    parser.add_argument(
        "--narrator",
        required=True,
        help="A narrátor referencia audiót tartalmazó könyvtár.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=20.0,
        help="Megengedett hangmagasság eltérés (Hz). Alapértelmezés: 20 Hz.",
    )
    parser.add_argument(
        "--min-frequency",
        type=float,
        default=60.0,
        help="A várható minimális frekvencia (Hz) a pitch kereséshez.",
    )
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=400.0,
        help="A várható maximális frekvencia (Hz) a pitch kereséshez.",
    )
    parser.add_argument(
        "--delete-outside",
        action="store_true",
        help="Ha engedélyezed, a tolerancián kívüli fájlok törlésre kerülnek.",
    )
    add_debug_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    missing_dependencies: List[str] = []
    if np is None:
        missing_dependencies.append("numpy")
    if sf is None:
        missing_dependencies.append("soundfile")
    if missing_dependencies:
        hint = "pip install " + " ".join(missing_dependencies)
        parser.exit(
            1,
            f"Hiba: hiányzó Python csomag(ok): {', '.join(missing_dependencies)}. "
            f"Telepítés például: {hint}\n",
        )

    project_root = find_project_root()
    config = load_config(project_root)

    log_level = configure_debug_mode(args.debug)
    logging.basicConfig(level=log_level, format="%(levelname)s | %(message)s")
    logger.setLevel(log_level)

    directories = config.get("DIRECTORIES", {})
    subdirs = config.get("PROJECT_SUBDIRS", {})
    workdir_rel = directories.get("workdir")
    translated_rel = subdirs.get("translated_splits")
    if not workdir_rel or not translated_rel:
        logger.error(
            "Hiányzó config kulcs: DIRECTORIES.workdir vagy PROJECT_SUBDIRS.translated_splits."
        )
        return 1

    project_dir = project_root / workdir_rel / args.project_name
    translated_dir = project_dir / translated_rel
    if not translated_dir.is_dir():
        logger.error("A translated_splits könyvtár nem található: %s", translated_dir)
        return 1

    narrator_dir = resolve_relative_path(args.narrator, project_root)
    if not narrator_dir.is_dir():
        logger.error("A narrátor könyvtár nem található: %s", narrator_dir)
        return 1

    narrator_files = list_audio_files(narrator_dir)
    if not narrator_files:
        logger.error(
            "Nem található használható audió fájl a narrátor könyvtárban (%s).", narrator_dir
        )
        return 1
    if len(narrator_files) > 1:
        logger.warning(
            "Több referencia fájl található a narrátor könyvtárban, az elsőt használjuk: %s",
            narrator_files[0].name,
        )
    narrator_path = narrator_files[0]

    logger.info("Narrátor referencia: %s", narrator_path)
    narrator_result = analyze_clip(
        narrator_path, min_frequency=args.min_frequency, max_frequency=args.max_frequency
    )
    if narrator_result.median_hz <= 0:
        logger.error("Nem sikerült megbízható pitch-et meghatározni a narrátor felvételből.")
        return 1

    translated_files = list_audio_files(translated_dir)
    if not translated_files:
        logger.error(
            "Nem található audió fájl a translated_splits könyvtárban: %s", translated_dir
        )
        return 1

    cpu_total = max(1, multiprocessing.cpu_count())
    worker_count = max(1, cpu_total // 2)
    worker_count = min(worker_count, len(translated_files))
    logger.info(
        "Processzor magok: %d | használt munkafolyamatok: %d",
        cpu_total,
        worker_count,
    )

    payloads = [(str(path), args.min_frequency, args.max_frequency) for path in translated_files]
    if worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(analyze_clip_worker, payloads))
    else:
        results = [analyze_clip_worker(payload) for payload in payloads]

    logger.info(
        "Narrátor medián hangmagasság: %.2f Hz (voiced arány: %.0f%%)",
        narrator_result.median_hz,
        narrator_result.voiced_ratio * 100,
    )
    logger.info(
        "Ellenőrzés alatt álló translated_splits fájlok száma: %d (tolerancia: %.2f Hz)",
        len(translated_files),
        args.tolerance,
    )

    outside: List[tuple[Path, float, PitchResult]] = []
    inside = 0

    for audio_path, result in zip(translated_files, results):
        diff = abs(result.median_hz - narrator_result.median_hz)
        status = "OK" if diff <= args.tolerance else "KINT"

        if result.voiced_frames == 0:
            logger.warning(
                "[%s] %s | Nincs hangmintázat (némának tűnik).",
                status,
                audio_path.name,
            )
            outside.append((audio_path, diff, result))
            continue

        logger.info(
            "[%s] %s | medián: %.2f Hz | eltérés: %.2f Hz | voiced: %.0f%%",
            status,
            audio_path.name,
            result.median_hz,
            diff,
            result.voiced_ratio * 100,
        )

        if diff <= args.tolerance:
            inside += 1
        else:
            outside.append((audio_path, diff, result))

    if outside:
        logger.warning("Eltérésen kívüli fájlok száma: %d", len(outside))
        for path, diff, result in sorted(outside, key=lambda item: item[1], reverse=True):
            logger.warning(
                " - %s | medián: %.2f Hz | eltérés: %.2f Hz | voiced: %.0f%%",
                path.name,
                result.median_hz,
                diff,
                result.voiced_ratio * 100,
            )
        if args.delete_outside:
            deleted = 0
            for path, diff, _ in outside:
                try:
                    path.unlink()
                    deleted += 1
                    logger.warning("   Törölve: %s (eltérés: %.2f Hz)", path.name, diff)
                except OSError as exc:
                    logger.error("   Nem sikerült törölni %s: %s", path, exc)
            logger.warning("Összesen törölt fájlok: %d", deleted)
        return 2

    logger.info("Minden fájl a tolerancián belül van (%d/%d).", inside, len(translated_files))
    return 0


if __name__ == "__main__":
    sys.exit(main())

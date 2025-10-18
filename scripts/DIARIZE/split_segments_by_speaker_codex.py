#!/usr/bin/env python3
"""
Improved speaker-based segment splitting using pyannote/speaker-diarization-3.1.

Key changes compared to split_segments_by_speaker.py:
  * Avoids temporary audio extraction by letting pyannote operate on precise time windows.
  * Optional dry-run mode and better configurability for backups/output handling.
  * Smarter word redistribution that keeps individual words unique per speaker run.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rich.console import Console
from rich.progress import Progress

from pyannote.core import Segment

for candidate in Path(__file__).resolve().parents:
    if (candidate / "tools").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from tools.debug_utils import add_debug_argument, configure_debug_mode

console = Console()


def get_project_root() -> Path:
    """
    Felkeresi a projekt gyökerét a config.json alapján.
    """
    for candidate in Path(__file__).resolve().parents:
        config_candidate = candidate / "config.json"
        if config_candidate.is_file():
            return candidate
    raise FileNotFoundError("Nem található config.json a szkript szülő könyvtáraiban.")


@dataclass
class SegmentStats:
    """Aggregate counters for reporting."""

    scanned: int = 0
    split: int = 0
    created: int = 0
    skipped_short: int = 0

    def add(self, other: SegmentStats) -> None:
        self.scanned += other.scanned
        self.split += other.split
        self.created += other.created
        self.skipped_short += other.skipped_short


class SpeakerSegmentSplitterCodex:
    """Improved splitter that relies on pyannote diarization without temp files."""

    def __init__(
        self,
        project: str,
        hf_token: Optional[str] = None,
        audio_exts: str = "wav,flac,mp3,m4a",
        inplace: bool = True,
        min_chunk: float = 0.20,
        round_decimals: int = 2,
        add_speaker_field: bool = True,
        backup: bool = True,
        dry_run: bool = False,
        min_word_overlap: float = 0.5,
        output_suffix: str = "_split",
        log_level: int = logging.INFO,
    ):
        self.project = project
        self.repo_root = get_project_root()

        self.hf_token = self._resolve_hf_token(hf_token)
        self.audio_exts = [ext.strip() for ext in audio_exts.split(",") if ext.strip()]
        self.inplace = inplace
        self.min_chunk = float(min_chunk)
        self.round_decimals = int(round_decimals)
        self.add_speaker_field = add_speaker_field
        self.backup = backup and inplace
        self.dry_run = dry_run
        self.min_word_overlap = float(min_word_overlap)
        self.output_suffix = output_suffix.strip()
        self.log_level = log_level
        self.config = self._load_config()
        self.base_path = self._get_base_path()
        self.log_path = self._setup_logging()

        self.pipeline = None
        self._annotation_cache: Dict[Path, Any] = {}

    # --------------------------------------------------------------------- #
    # Configuration helpers
    # --------------------------------------------------------------------- #
    def _resolve_hf_token(self, cli_token: Optional[str]) -> Optional[str]:
        """Load/remember the Hugging Face token, mirroring the original script."""
        keyholder_path = self.repo_root / "keyholder.json"
        keyholder_data: Dict[str, Any] = {}

        if keyholder_path.exists():
            try:
                keyholder_data = json.loads(keyholder_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                console.print(
                    "[yellow]⚠️  Warning: Could not decode keyholder.json, it might be corrupted."
                )
                logging.warning("Could not decode keyholder.json.")

        env_token = os.getenv("HUGGINGFACE_TOKEN")
        token = cli_token or env_token

        if not token:
            encoded_token = keyholder_data.get("hf_token")
            if encoded_token:
                try:
                    token = base64.b64decode(encoded_token).decode("utf-8")
                    console.print("[blue]ℹ️  Loaded Hugging Face token from keyholder.json")
                except Exception:
                    console.print(
                        "[red]❌ Error decoding token from keyholder.json. "
                        "Please check the file or provide a valid token."
                    )

        # Persist token if it came from CLI or env.
        if token and (cli_token or env_token):
            encoded = base64.b64encode(token.encode("utf-8")).decode("utf-8")
            if encoded != keyholder_data.get("hf_token"):
                keyholder_data["hf_token"] = encoded
                console.print("[yellow]i  Saving Hugging Face token to keyholder.json (base64 encoded)...")
                try:
                    keyholder_path.write_text(json.dumps(keyholder_data, indent=2), encoding="utf-8")
                    logging.info("Saved new hf_token to keyholder.json.")
                except Exception as exc:
                    console.print(f"[red]❌ Could not save token to keyholder.json: {exc}")
                    logging.error("Could not save token to keyholder.json", exc_info=True)

        return token

    def _load_config(self) -> Dict[str, Any]:
        config_path = self.repo_root / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json nem található: {config_path}")
        return json.loads(config_path.read_text(encoding="utf-8"))

    def _get_base_path(self) -> Path:
        workdir = self.repo_root / self.config["DIRECTORIES"]["workdir"]
        separated_audio_speech = self.config["PROJECT_SUBDIRS"]["separated_audio_speech"]
        return workdir / self.project / separated_audio_speech

    def _setup_logging(self) -> Path:
        workdir = self.repo_root / self.config["DIRECTORIES"]["workdir"]
        logs_dir = self.config["PROJECT_SUBDIRS"]["logs"]
        log_dir = workdir / self.project / logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"split_segments_by_speaker_codex_{timestamp}.log"

        logging.basicConfig(
            level=self.log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(),
            ],
            force=True,
        )
        logging.getLogger().setLevel(self.log_level)

        return log_path

    # --------------------------------------------------------------------- #
    # Pyannote pipeline handling
    # --------------------------------------------------------------------- #
    def _initialize_pipeline(self) -> None:
        if self.pipeline is not None:
            return

        if not self.hf_token:
            raise ValueError(
                "HuggingFace token is required. Provide via --hf-token, "
                "HUGGINGFACE_TOKEN env var, or store it in keyholder.json."
            )

        console.print("[cyan]🔄 Initializing pyannote speaker diarization pipeline...")
        try:
            import torch
            from pyannote.audio import Pipeline

            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pipeline.to(device)
            console.print(f"[green]✅ Pipeline initialized on {device}")
            logging.info("Pipeline initialized on %s", device)
        except Exception as exc:
            console.print(f"[red]❌ Failed to initialize pipeline: {exc}")
            logging.error("Failed to initialize pipeline", exc_info=True)
            raise

    # --------------------------------------------------------------------- #
    # Helpers for diarization results
    # --------------------------------------------------------------------- #
    def _get_audio_annotation(self, audio_path: Path):
        """Run diarization once per audio file and cache the annotation."""
        resolved = audio_path.resolve()
        if resolved in self._annotation_cache:
            return self._annotation_cache[resolved]

        console.print(f"[magenta]🔍 Running diarization on entire file: {audio_path.name}")
        logging.info("Running diarization on entire file: %s", audio_path.name)

        try:
            import torch

            with torch.no_grad():
                annotation = self.pipeline(str(audio_path))
        except Exception as exc:
            logging.error("Full-file diarization failed for %s: %s", audio_path.name, exc)
            raise

        self._annotation_cache[resolved] = annotation
        return annotation

    @staticmethod
    def _round_times(obj: Any, decimals: int) -> Any:
        if isinstance(obj, dict):
            return {
                key: SpeakerSegmentSplitterCodex._round_times(value, decimals)
                if key not in {"start", "end"}
                else round(float(value), decimals)
                if isinstance(value, (float, int))
                else value
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [SpeakerSegmentSplitterCodex._round_times(item, decimals) for item in obj]
        return obj

    def _get_speaker_runs(
        self,
        annotation,
        original_start: float,
        original_end: float,
    ) -> List[Tuple[float, float, str]]:
        if annotation is None:
            return [(original_start, original_end, "UNKNOWN")]

        try:
            cropped = annotation.crop(Segment(original_start, original_end), mode="intersection")
        except Exception as exc:
            logging.warning(
                "Failed to crop annotation for window %0.2f-%0.2f: %s",
                original_start,
                original_end,
                exc,
            )
            return [(original_start, original_end, "UNKNOWN")]

        runs: List[Tuple[float, float, str]] = []
        for seg, _, label in cropped.itertracks(yield_label=True):
            runs.append((seg.start, seg.end, label))

        if not runs:
            return [(original_start, original_end, "UNKNOWN")]

        runs.sort(key=lambda item: item[0])

        # Merge consecutive runs for the same speaker with minimal gaps.
        merged: List[Tuple[float, float, str]] = []
        for run_start, run_end, speaker in runs:
            if merged and merged[-1][2] == speaker and abs(merged[-1][1] - run_start) < 0.01:
                merged[-1] = (merged[-1][0], run_end, speaker)
            else:
                merged.append((run_start, run_end, speaker))

        if not merged:
            return [(original_start, original_end, "UNKNOWN")]

        cleaned: List[Tuple[float, float, str]] = []
        for idx, (run_start, run_end, speaker) in enumerate(merged):
            duration = run_end - run_start
            if duration < self.min_chunk and len(merged) > 1:
                # Merge short runs into neighbours when possible.
                if cleaned and cleaned[-1][2] == speaker:
                    cleaned[-1] = (cleaned[-1][0], run_end, speaker)
                    continue
                if idx + 1 < len(merged) and merged[idx + 1][2] == speaker:
                    merged[idx + 1] = (run_start, merged[idx + 1][1], speaker)
                    continue
            cleaned.append((run_start, run_end, speaker))

        if not cleaned:
            return [(original_start, original_end, "UNKNOWN")]

        compact: List[Tuple[float, float, str]] = []
        for run_start, run_end, speaker in cleaned:
            if compact and compact[-1][2] == speaker:
                compact[-1] = (compact[-1][0], run_end, speaker)
            else:
                compact.append((run_start, run_end, speaker))

        return compact or [(original_start, original_end, "UNKNOWN")]

    def _assign_words_to_runs(
        self,
        words: Iterable[Dict[str, Any]],
        runs: List[Tuple[float, float, str]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        assignments: Dict[int, List[Dict[str, Any]]] = {idx: [] for idx in range(len(runs))}
        if not words:
            return assignments

        for word in words:
            try:
                word_start = float(word.get("start", 0.0))
                word_end = float(word.get("end", word_start))
            except (TypeError, ValueError):
                continue

            duration = max(0.0, word_end - word_start)
            word_mid = word_start + duration / 2 if duration > 0 else word_start

            chosen_idx: Optional[int] = None

            for idx, (run_start, run_end, _) in enumerate(runs):
                if run_start <= word_mid <= run_end:
                    chosen_idx = idx
                    break

            if chosen_idx is None and duration > 0:
                best_overlap = 0.0
                for idx, (run_start, run_end, _) in enumerate(runs):
                    overlap = max(0.0, min(run_end, word_end) - max(run_start, word_start))
                    if duration > 0:
                        ratio = overlap / duration
                        if ratio >= self.min_word_overlap and ratio > best_overlap:
                            best_overlap = ratio
                            chosen_idx = idx

            if chosen_idx is None:
                continue

            word_copy = word.copy()
            if self.add_speaker_field:
                word_copy["speaker"] = runs[chosen_idx][2]
            assignments[chosen_idx].append(word_copy)

        return assignments

    # --------------------------------------------------------------------- #
    # JSON processing
    # --------------------------------------------------------------------- #
    def _find_audio_json_pairs(self) -> List[Tuple[Path, Path]]:
        if not self.base_path.exists():
            console.print(f"[red]❌ Base path does not exist: {self.base_path}")
            logging.error("Base path does not exist: %s", self.base_path)
            return []

        audio_files: List[Path] = []
        for ext in self.audio_exts:
            audio_files.extend(self.base_path.glob(f"*.{ext}"))

        pairs: List[Tuple[Path, Path]] = []
        for audio_path in audio_files:
            json_path = audio_path.with_suffix(".json")
            if not json_path.exists():
                console.print(f"[yellow]⚠️  JSON not found for {audio_path.name}, skipping")
                logging.warning("JSON not found for %s", audio_path.name)
                continue
            pairs.append((audio_path, json_path))

        console.print(f"[blue]📁 Found {len(pairs)} audio-JSON pairs")
        logging.info("Found %d audio-JSON pairs", len(pairs))
        return pairs

    def _split_segment(
        self,
        segment: Dict[str, Any],
        audio_path: Path,
        annotation,
    ) -> Tuple[List[Dict[str, Any]], SegmentStats]:
        stats = SegmentStats(scanned=1)

        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            return [segment], stats

        duration = max(0.0, end - start)
        if duration < 0.5:  # 500 ms threshold
            stats.skipped_short += 1
            new_segment = segment.copy()
            if self.add_speaker_field and "speaker" not in new_segment:
                new_segment["speaker"] = "UNKNOWN"
            return [self._round_times(new_segment, self.round_decimals)], stats

        words = segment.get("words", [])

        speaker_runs = self._get_speaker_runs(annotation, start, end)
        assignments = self._assign_words_to_runs(words, speaker_runs)
        if len(speaker_runs) <= 1:
            new_segment = segment.copy()
            if self.add_speaker_field:
                new_segment["speaker"] = speaker_runs[0][2] if speaker_runs else "UNKNOWN"
            if speaker_runs:
                assigned = assignments.get(0, [])
                new_segment["words"] = assigned if assigned else words
            return [self._round_times(new_segment, self.round_decimals)], stats

        new_segments: List[Dict[str, Any]] = []

        for idx, (run_start, run_end, speaker) in enumerate(speaker_runs):
            assigned_words = assignments.get(idx, [])
            if not assigned_words and (run_end - run_start) < self.min_chunk:
                # Drop micro-runs with no lexical support
                continue

            text = " ".join(word.get("word", "") for word in assigned_words).strip()
            if not text:
                # If diarization splits but no words fall inside, keep original text portion.
                text = segment.get("text", "")

            new_segment = {
                "start": run_start,
                "end": run_end,
                "text": text,
                "words": assigned_words,
            }
            if self.add_speaker_field:
                new_segment["speaker"] = speaker
            new_segments.append(self._round_times(new_segment, self.round_decimals))

        if not new_segments:
            new_segment = segment.copy()
            if self.add_speaker_field:
                new_segment["speaker"] = "UNKNOWN"
            return [self._round_times(new_segment, self.round_decimals)], stats

        stats.split += 1
        stats.created += max(0, len(new_segments) - 1)
        return new_segments, stats

    def _process_json_file(self, json_path: Path, audio_path: Path) -> SegmentStats:
        console.print(f"[cyan]📝 Processing {json_path.name}...")

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            console.print(f"[red]❌ Failed to load {json_path.name}: {exc}")
            logging.error("Failed to load %s", json_path.name, exc_info=True)
            return SegmentStats()

        segments = data.get("segments", [])
        if not isinstance(segments, list) or not segments:
            console.print(f"[yellow]⚠️  No segments found in {json_path.name}")
            logging.warning("No segments in %s", json_path.name)
            return SegmentStats()

        if self.backup and not self.dry_run:
            backup_path = json_path.with_suffix(".json.bak")
            backup_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            console.print(f"[blue]💾 Backup created: {backup_path.name}")

        try:
            annotation = self._get_audio_annotation(audio_path)
        except Exception as exc:
            console.print(f"[red]❌ Failed to diarize {audio_path.name}: {exc}")
            logging.error("Failed to diarize %s", audio_path.name, exc_info=True)
            return SegmentStats()

        aggregated = SegmentStats()
        new_segments: List[Dict[str, Any]] = []

        with Progress(transient=True) as progress:
            task_id = progress.add_task("Segments", total=len(segments))
            for segment in segments:
                split_segments, stats = self._split_segment(segment, audio_path, annotation)
                aggregated.add(stats)
                new_segments.extend(split_segments)
                progress.advance(task_id)

        data["segments"] = new_segments

        if not self.dry_run:
            target_path = json_path
            if not self.inplace:
                target_path = json_path.with_name(
                    f"{json_path.stem}{self.output_suffix}{json_path.suffix}"
                )

            target_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            console.print(
                f"[green]✅ Updated {target_path.name}: {aggregated.scanned} → "
                f"{len(new_segments)} segments"
            )
            logging.info(
                "Updated %s (scanned=%d, split=%d, created=%d)",
                target_path.name,
                aggregated.scanned,
                aggregated.split,
                aggregated.created,
            )
        else:
            console.print(
                f"[green]✅ Dry-run finished for {json_path.name}: "
                f"{aggregated.scanned} → {len(new_segments)} (not written)"
            )

        return aggregated

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def process_all(self) -> None:
        console.print(f"[bold blue]🚀 Starting speaker segment splitting for project: {self.project}")
        logging.info("Starting speaker segment splitting for project: %s", self.project)

        self._initialize_pipeline()
        pairs = self._find_audio_json_pairs()
        if not pairs:
            console.print("[yellow]⚠️  No audio-JSON pairs found")
            logging.warning("No audio-JSON pairs found")
            return

        overall = SegmentStats()
        files_processed = 0

        for audio_path, json_path in pairs:
            console.print(f"\n[bold cyan]Processing: {audio_path.name}")
            try:
                stats = self._process_json_file(json_path, audio_path)
            except Exception as exc:
                console.print(f"[red]❌ Failed to process {audio_path.name}: {exc}")
                logging.error("Failed to process %s", audio_path.name, exc_info=True)
                continue

            overall.add(stats)
            files_processed += 1

        console.print("\n[bold green]🎉 Processing complete!")
        console.print(f"[green]Files processed: {files_processed}")
        console.print(f"[green]Segments scanned: {overall.scanned}")
        console.print(f"[green]Segments split: {overall.split}")
        console.print(f"[green]Total new segments created: {overall.created}")
        console.print(f"[blue]Log file: {self.log_path}")
        logging.info(
            "Processing complete (files=%d, scanned=%d, split=%d, created=%d)",
            files_processed,
            overall.scanned,
            overall.split,
            overall.created,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split transcript segments by speaker using pyannote/speaker-diarization-3.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/split_segments_by_speaker_codex.py -p MyProject_001 --hf-token YOUR_HF_TOKEN
  python scripts/split_segments_by_speaker_codex.py -p MyProject_001 --dry-run
  python scripts/split_segments_by_speaker_codex.py -p MyProject_001 --no-inplace --output-suffix _diarized
        """,
    )

    parser.add_argument("-p", "--project", required=True, help="Project leaf folder name under workdir/")
    parser.add_argument("--hf-token", help="HuggingFace access token (overrides env var and keyholder.json)")
    parser.add_argument(
        "--audio-exts",
        default="wav,flac,mp3,m4a",
        help="Comma-separated audio file extensions (default: wav,flac,mp3,m4a)",
    )
    parser.add_argument(
        "--min-chunk",
        type=float,
        default=0.20,
        help="Minimum chunk duration in seconds to keep diarization runs (default: 0.20)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=2,
        help="Number of decimal places for time rounding (default: 2)",
    )
    parser.add_argument(
        "--no-speaker-field",
        dest="add_speaker_field",
        action="store_false",
        help="Do not add a speaker field to resulting segments.",
    )
    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="Disable .bak backup creation (ignored when --no-inplace).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process files but do not write changes (useful for inspection).",
    )
    parser.add_argument(
        "--min-word-overlap",
        type=float,
        default=0.5,
        help="Minimum ratio of word overlap for assignment when the midpoint rule fails (default: 0.5)",
    )
    parser.add_argument(
        "--no-inplace",
        dest="inplace",
        action="store_false",
        help="Write results to <name><suffix>.json instead of overwriting input file.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_split",
        help="Suffix appended to JSON filename when --no-inplace is used (default: _split).",
    )

    add_debug_argument(parser)
    parser.set_defaults(add_speaker_field=True, backup=True, inplace=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    log_level = configure_debug_mode(args.debug)

    try:
        splitter = SpeakerSegmentSplitterCodex(
            project=args.project,
            hf_token=args.hf_token,
            audio_exts=args.audio_exts,
            inplace=args.inplace,
            min_chunk=args.min_chunk,
            round_decimals=args.round,
            add_speaker_field=args.add_speaker_field,
            backup=args.backup,
            dry_run=args.dry_run,
            min_word_overlap=args.min_word_overlap,
            output_suffix=args.output_suffix,
            log_level=log_level,
        )
        splitter.process_all()
    except Exception as exc:
        console.print(f"[red]❌ Fatal error: {exc}")
        logging.error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

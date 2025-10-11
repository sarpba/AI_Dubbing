#!/usr/bin/env python3
"""
Split transcript segments by speaker using pyannote/speaker-diarization-3.1.

This script processes audio files and their corresponding JSON transcripts,
using speaker diarization to split segments that contain multiple speakers
into separate segments for each speaker.

This modified version uses batch processing for efficiency. It processes
chunks of segments together to reduce I/O and model inference calls.
A segment is only split if a speaker change is detected within it.

Usage:
    python scripts/split_segments_by_speaker_batch.py -p MyProject_001 --hf-token $HUGGINGFACE_TOKEN
    python scripts/split_segments_by_speaker_batch.py -p MyProject_001 --min-chunk 0.25 --round 3
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import shutil
import sys
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from rich.console import Console

console = Console()


class SpeakerSegmentSplitter:
    """Main class for splitting transcript segments by speaker using batch processing."""

    # Segments will be processed in batches of this size for efficiency.
    BATCH_SIZE = 15

    def __init__(
        self,
        project: str,
        hf_token: Optional[str] = None,
        audio_exts: str = "wav,flac,mp3,m4a",
        min_chunk: float = 0.20,
        round_decimals: int = 2,
        add_speaker_field: bool = True,
        backup: bool = True,
    ):
        """Initialize the splitter with configuration parameters."""
        self.project = project

        keyholder_path = Path("keyholder.json")
        keyholder_data = {}

        if keyholder_path.exists():
            try:
                with open(keyholder_path, 'r', encoding='utf-8') as f:
                    keyholder_data = json.load(f)
            except json.JSONDecodeError:
                console.print(f"[yellow]⚠️  Warning: Could not decode {keyholder_path}, it might be corrupted.")
                logging.warning(f"Could not decode {keyholder_path}.")

        current_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")

        if not current_token:
            encoded_token = keyholder_data.get("hf_token")
            if encoded_token:
                try:
                    current_token = base64.b64decode(encoded_token).decode('utf-8')
                    console.print("[blue]ℹ️  Loaded Hugging Face token from keyholder.json")
                except Exception:
                    console.print(f"[red]❌ Error decoding token from {keyholder_path}. Please check the file or provide a valid token.")
                    current_token = None

        self.hf_token = current_token

        if self.hf_token and (hf_token or os.getenv("HUGGINGFACE_TOKEN")):
            new_encoded_token = base64.b64encode(self.hf_token.encode('utf-8')).decode('utf-8')

            if new_encoded_token != keyholder_data.get("hf_token"):
                console.print(f"[yellow]i  Saving Hugging Face token to {keyholder_path} (base64 encoded)...")
                keyholder_data["hf_token"] = new_encoded_token
                try:
                    with open(keyholder_path, 'w', encoding='utf-8') as f:
                        json.dump(keyholder_data, f, indent=2)
                    logging.info(f"Saved new hf_token to {keyholder_path}.")
                except Exception as e:
                    console.print(f"[red]❌ Could not save token to {keyholder_path}: {e}")
                    logging.error(f"Could not save token to {keyholder_path}: {e}")

        self.audio_exts = [ext.strip() for ext in audio_exts.split(",")]
        self.min_chunk = min_chunk
        self.round_decimals = round_decimals
        self.add_speaker_field = add_speaker_field
        self.backup = backup

        self.config = self._load_config()
        self.base_path = self._get_base_path()
        self.log_path = self._setup_logging()
        self.pipeline = None

    def _load_config(self) -> Dict[str, Any]:
        config_path = Path("config.json")
        if not config_path.exists():
            raise FileNotFoundError("config.json not found in project root")
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_base_path(self) -> Path:
        workdir = Path(self.config["DIRECTORIES"]["workdir"])
        separated_audio_speech = self.config["PROJECT_SUBDIRS"]["separated_audio_speech"]
        return workdir / self.project / separated_audio_speech

    def _setup_logging(self) -> Path:
        workdir = Path(self.config["DIRECTORIES"]["workdir"])
        logs_dir = self.config["PROJECT_SUBDIRS"]["logs"]
        log_dir = workdir / self.project / logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"split_segments_by_speaker_{timestamp}.log"
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_path, encoding='utf-8'), logging.StreamHandler()])
        return log_path

    def _initialize_pipeline(self) -> None:
        if self.pipeline is not None:
            return
        if not self.hf_token:
            raise ValueError("HuggingFace token is required. Provide via --hf-token, HUGGINGFACE_TOKEN env var, or store it in keyholder.json")
        console.print("[cyan]🔄 Initializing pyannote speaker diarization pipeline...")
        try:
            import torch
            from pyannote.audio import Pipeline
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=self.hf_token)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pipeline.to(device)
            console.print(f"[green]✅ Pipeline initialized on {device}")
            logging.info(f"Pipeline initialized on {device}")
        except Exception as e:
            console.print(f"[red]❌ Failed to initialize pipeline: {e}")
            logging.error(f"Failed to initialize pipeline: {e}")
            raise

    def _find_audio_json_pairs(self) -> List[Tuple[Path, Path]]:
        pairs = []
        if not self.base_path.exists():
            console.print(f"[red]❌ Base path does not exist: {self.base_path}")
            return pairs
        audio_files = []
        for ext in self.audio_exts:
            audio_files.extend(self.base_path.glob(f"*.{ext}"))
        for audio_path in audio_files:
            json_path = audio_path.with_suffix(".json")
            if not json_path.exists():
                logging.warning(f"JSON not found for {audio_path.name}")
                continue
            pairs.append((audio_path, json_path))
        console.print(f"[blue]📁 Found {len(pairs)} audio-JSON pairs")
        return pairs

    def _extract_audio_segment(self, audio_path: Path, start_time: float, end_time: float, segment_index: int) -> Path:
        temp_dir = Path(tempfile.gettempdir()) / "speaker_segments"
        temp_dir.mkdir(exist_ok=True)
        segment_filename = f"{audio_path.stem}_batch_{segment_index:04d}_{start_time:.2f}-{end_time:.2f}.wav"
        segment_path = temp_dir / segment_filename
        cmd = ["ffmpeg", "-y", "-i", str(audio_path), "-ss", str(start_time), "-t", str(end_time - start_time), "-c:a", "pcm_s16le", str(segment_path)]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            logging.debug(f"Extracted batch audio {segment_index}: {segment_path}")
            return segment_path
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to extract segment {start_time}-{end_time}: {e.stderr}")
            raise

    def _run_diarization_on_segment(self, segment_audio_path: Path):
        try:
            import torch
            with torch.no_grad():
                diarization = self.pipeline(str(segment_audio_path))
            return diarization
        except Exception as e:
            logging.error(f"Diarization failed for segment audio {segment_audio_path.name}: {e}")
            raise

    def _cleanup_segment_file(self, segment_path: Path):
        try:
            if segment_path and segment_path.exists():
                segment_path.unlink()
        except Exception as e:
            logging.warning(f"Failed to cleanup segment file {segment_path}: {e}")

    def _get_speaker_runs_from_extracted_segment(self, annotation, original_start: float, original_end: float) -> List[Tuple[float, float, str]]:
        if not annotation:
            return [(original_start, original_end, "UNKNOWN")]
        runs = []
        for seg, _, label in annotation.itertracks(yield_label=True):
            runs.append([original_start + seg.start, original_start + seg.end, label])
        if not runs:
            return [(original_start, original_end, "UNKNOWN")]
        runs.sort(key=lambda x: x[0])
        filled_runs = []
        if runs:
            filled_runs.append(runs[0])
            for i in range(1, len(runs)):
                prev_end = filled_runs[-1][1]
                curr_start = runs[i][0]
                if curr_start > prev_end:
                    filled_runs[-1][1] = curr_start
                filled_runs.append(runs[i])
        merged_runs = []
        if filled_runs:
            merged_runs.append(filled_runs[0])
            for i in range(1, len(filled_runs)):
                if filled_runs[i][2] == merged_runs[-1][2]:
                    merged_runs[-1][1] = filled_runs[i][1]
                else:
                    merged_runs.append(filled_runs[i])
        final_runs = []
        i = 0
        while i < len(merged_runs):
            current_run = merged_runs[i]
            if (current_run[1] - current_run[0]) < self.min_chunk:
                prev_duration = (merged_runs[i-1][1] - merged_runs[i-1][0]) if i > 0 else -1
                next_duration = (merged_runs[i+1][1] - merged_runs[i+1][0]) if i < len(merged_runs) - 1 else -1
                if prev_duration == -1 and next_duration == -1:
                    final_runs.append(current_run)
                elif next_duration > prev_duration:
                    if i < len(merged_runs) - 1: merged_runs[i+1][0] = current_run[0]
                else:
                    if final_runs: final_runs[-1][1] = current_run[1]
            else:
                final_runs.append(current_run)
            i += 1
        if not final_runs:
            return [(original_start, original_end, "UNKNOWN")]
        final_runs[0][0] = original_start
        final_runs[-1][1] = original_end
        return [tuple(run) for run in final_runs]

    def _reassign_words(self, words: List[Dict], start: float, end: float) -> List[Dict]:
        if not words: return []
        reassigned = []
        for word in words:
            word_start, word_end = word.get("start"), word.get("end")
            if word_start is None or word_end is None: continue
            word_mid = (word_start + word_end) / 2
            if start <= word_mid < end:
                reassigned.append(word.copy())
            elif word_start < end and word_end > start:
                overlap = max(0, min(end, word_end) - max(start, word_start))
                duration = word_end - word_start
                if duration > 0 and (overlap / duration) >= 0.5:
                    reassigned.append(word.copy())
        return reassigned

    def _round_times(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: round(float(v), self.round_decimals) if k in ["start", "end"] and isinstance(v, (int, float)) else self._round_times(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._round_times(item) for item in obj]
        return obj

    def _split_single_segment_using_timeline(self, segment: Dict, speaker_timeline: List[Tuple[float, float, str]]) -> List[Dict]:
        seg_start, seg_end = float(segment["start"]), float(segment["end"])
        words = segment.get("words", [])

        relevant_runs = []
        for run_start, run_end, speaker in speaker_timeline:
            if run_start < seg_end and run_end > seg_start:
                overlap_start = max(seg_start, run_start)
                overlap_end = min(seg_end, run_end)
                if overlap_end > overlap_start:
                    relevant_runs.append((overlap_start, overlap_end, speaker))

        if len(relevant_runs) <= 1:
            new_segment = segment.copy()
            if self.add_speaker_field:
                new_segment["speaker"] = relevant_runs[0][2] if relevant_runs else "UNKNOWN"
            return [self._round_times(new_segment)]

        new_segments = []
        for run_start, run_end, speaker in relevant_runs:
            reassigned_words = self._reassign_words(words, run_start, run_end)
            if not reassigned_words and (run_end - run_start) < self.min_chunk:
                continue

            new_text = " ".join(word["word"] for word in reassigned_words).strip()
            final_start = reassigned_words[0]['start'] if reassigned_words else run_start
            final_end = reassigned_words[-1]['end'] if reassigned_words else run_end

            new_seg_dict = {"start": final_start, "end": final_end, "text": new_text}
            if self.add_speaker_field:
                new_seg_dict["speaker"] = speaker
            if "words" in segment:
                new_seg_dict["words"] = reassigned_words
            new_segments.append(self._round_times(new_seg_dict))
        
        if not new_segments:
            new_segment = segment.copy()
            if self.add_speaker_field and "speaker" not in new_segment:
                new_segment["speaker"] = "UNKNOWN"
            return [self._round_times(new_segment)]
            
        return new_segments

    def _process_segment_batch(self, batch: List[Dict], audio_path: Path, batch_index: int) -> Tuple[List[Dict], int]:
        if not batch:
            return [], 0

        batch_start = batch[0]['start']
        batch_end = batch[-1]['end']

        if (batch_end - batch_start) < 0.5:
            return [self._round_times(seg) for seg in batch], 0

        batch_audio_path = None
        try:
            console.print(f"[cyan]    Extracting audio for batch {batch_index+1} ({batch_start:.2f}s - {batch_end:.2f}s)")
            batch_audio_path = self._extract_audio_segment(audio_path, batch_start, batch_end, batch_index)
            diarization = self._run_diarization_on_segment(batch_audio_path)
            speaker_timeline = self._get_speaker_runs_from_extracted_segment(diarization, batch_start, batch_end)

            all_new_segments = []
            split_count = 0
            for original_segment in batch:
                split_segments = self._split_single_segment_using_timeline(original_segment, speaker_timeline)
                if len(split_segments) > 1:
                    split_count += 1
                all_new_segments.extend(split_segments)
            
            console.print(f"[green]    Batch {batch_index+1} processed. {split_count} segment(s) were split.")
            return all_new_segments, split_count

        except Exception as e:
            logging.warning(f"Failed to process batch {batch_index}, returning originals: {e}", exc_info=True)
            console.print(f"[yellow]⚠️  Failed to process batch {batch_index+1}, keeping original segments.")
            return [self._round_times(seg) for seg in batch], 0
        finally:
            self._cleanup_segment_file(batch_audio_path)

    def _process_json_file(self, json_path: Path, audio_path: Path) -> Dict[str, int]:
        console.print(f"[cyan]📝 Processing {json_path.name}...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            console.print(f"[red]❌ Failed to load {json_path.name}: {e}")
            return {"scanned": 0, "split": 0, "created": 0}

        segments = data.get("segments", [])
        if not segments:
            console.print(f"[yellow]⚠️  No segments found in {json_path.name}")
            return {"scanned": 0, "split": 0, "created": 0}

        if self.backup:
            backup_path = json_path.with_suffix(".json.bak.txt")
            if not backup_path.exists():
                shutil.copy2(json_path, backup_path)
                console.print(f"[blue]💾 Backup created: {backup_path.name}")
            else:
                console.print(f"[blue]💾 Backup already exists: {backup_path.name}")

        segment_batches = [segments[i:i + self.BATCH_SIZE] for i in range(0, len(segments), self.BATCH_SIZE)]
        new_segments = []
        stats = {"scanned": len(segments), "split": 0, "created": 0}
        
        console.print(f"[blue]  Split into {len(segment_batches)} batches of up to {self.BATCH_SIZE} segments.")

        for i, batch in enumerate(segment_batches):
            processed_segments, splits_in_batch = self._process_segment_batch(batch, audio_path, i)
            new_segments.extend(processed_segments)
            stats["split"] += splits_in_batch
        
        stats["created"] = len(new_segments)
        data["segments"] = new_segments

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            console.print(f"[green]✅ Updated {json_path.name}: {stats['scanned']} → {stats['created']} segments ({stats['split']} original segments were split)")
            logging.info(f"Updated {json_path.name}: {stats}")
        except Exception as e:
            console.print(f"[red]❌ Failed to write {json_path.name}: {e}")
        
        return stats

    def process_all(self) -> None:
        console.print(f"[bold blue]🚀 Starting speaker segment splitting for project: {self.project}")
        self._initialize_pipeline()
        pairs = self._find_audio_json_pairs()
        if not pairs:
            console.print("[yellow]⚠️  No audio-JSON pairs found")
            return

        total_stats = {"scanned": 0, "split": 0, "created": 0, "files_processed": 0}
        for audio_path, json_path in pairs:
            try:
                console.print(f"\n[bold cyan]Processing: {audio_path.name}")
                file_stats = self._process_json_file(json_path, audio_path)
                for key in ["scanned", "split", "created"]:
                    total_stats[key] += file_stats[key]
                total_stats["files_processed"] += 1
            except Exception as e:
                console.print(f"[red]❌ Failed to process {audio_path.name}: {e}")
                logging.error(f"Failed to process {audio_path.name}: {e}", exc_info=True)
                continue

        console.print(f"\n[bold green]🎉 Processing complete!")
        console.print(f"[green]Files processed: {total_stats['files_processed']}")
        console.print(f"[green]Total segments scanned: {total_stats['scanned']}")
        console.print(f"[green]Original segments split: {total_stats['split']}")
        console.print(f"[green]Total segments created: {total_stats['created']}")
        console.print(f"[blue]Log file: {self.log_path}")
        logging.info(f"Processing complete. Final stats: {total_stats}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Split transcript segments by speaker using pyannote/speaker-diarization-3.1 (Batch Mode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/split_segments_by_speaker_batch.py -p MyProject_001 --hf-token YOUR_HF_TOKEN
  python scripts/split_segments_by_speaker_batch.py -p MyProject_001 --min-chunk 0.25 --round 3
  (If token is in keyholder.json or env var, you can omit --hf-token)
  python scripts/split_segments_by_speaker_batch.py -p MyProject_001
        """
    )
    
    parser.add_argument("-p", "--project", required=True, help="Project leaf folder name under workdir/")
    parser.add_argument("--hf-token", help="HuggingFace access token (overrides env var and keyholder.json)")
    parser.add_argument("--audio-exts", default="wav,flac,mp3,m4a", help="Comma-separated audio file extensions (default: wav,flac,mp3,m4a)")
    parser.add_argument("--min-chunk", type=float, default=0.20, help="Minimum chunk duration in seconds (default: 0.20)")
    parser.add_argument("--round", type=int, default=2, help="Number of decimal places for time rounding (default: 2)")
    parser.add_argument("--no-add-speaker-field", dest='add_speaker_field', action="store_false", help="Do not add speaker field to segments")
    parser.add_argument("--no-backup", dest='backup', action="store_false", help="Do not create backup files before modification")
    
    # A parancsikonok alapértelmezetten igazak maradnak, a --no- előtaggal lehet őket letiltani
    parser.set_defaults(add_speaker_field=True, backup=True)
    
    args = parser.parse_args()
    
    try:
        splitter = SpeakerSegmentSplitter(
            project=args.project,
            hf_token=args.hf_token,
            audio_exts=args.audio_exts,
            min_chunk=args.min_chunk,
            round_decimals=args.round,
            add_speaker_field=args.add_speaker_field,
            backup=args.backup,
        )
        splitter.process_all()
    except Exception as e:
        console.print(f"[red]❌ Fatal error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
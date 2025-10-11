#!/usr/bin/env python3
"""
Split transcript segments by speaker using pyannote/speaker-diarization-3.1.

This script processes audio files and their corresponding JSON transcripts,
using speaker diarization to split segments that contain multiple speakers
into separate segments for each speaker.

Usage:
    python scripts/split_segments_by_speaker.py -p MyProject_001 --hf-token $HUGGINGFACE_TOKEN
    python scripts/split_segments_by_speaker.py -p MyProject_001 --min-chunk 0.25 --round 3
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from rich.console import Console

console = Console()


class SpeakerSegmentSplitter:
    """Main class for splitting transcript segments by speaker."""
    
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
    ):
        """Initialize the splitter with configuration parameters."""
        self.project = project
        
        # --- START OF MODIFIED HUGGING FACE TOKEN HANDLING ---
        
        keyholder_path = Path("keyholder.json")
        keyholder_data = {}

        # 1. Load keyholder.json if it exists
        if keyholder_path.exists():
            try:
                with open(keyholder_path, 'r', encoding='utf-8') as f:
                    keyholder_data = json.load(f)
            except json.JSONDecodeError:
                console.print(f"[yellow]⚠️  Warning: Could not decode {keyholder_path}, it might be corrupted.")
                logging.warning(f"Could not decode {keyholder_path}.")

        # 2. Determine the token source with priority: CLI > ENV > keyholder.json
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

        # 3. Save token to keyholder.json if it came from CLI/env and is different from stored one
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

        # --- END OF MODIFIED HUGGING FACE TOKEN HANDLING ---
        
        self.audio_exts = [ext.strip() for ext in audio_exts.split(",")]
        self.inplace = inplace
        self.min_chunk = min_chunk
        self.round_decimals = round_decimals
        self.add_speaker_field = add_speaker_field
        self.backup = backup
        
        # Load configuration
        self.config = self._load_config()
        self.base_path = self._get_base_path()
        self.log_path = self._setup_logging()
        
        # Initialize diarization pipeline
        self.pipeline = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json."""
        config_path = Path("config.json")
        if not config_path.exists():
            raise FileNotFoundError("config.json not found in project root")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_base_path(self) -> Path:
        """Get the base path for audio and JSON files."""
        workdir = Path(self.config["DIRECTORIES"]["workdir"])
        separated_audio_speech = self.config["PROJECT_SUBDIRS"]["separated_audio_speech"]
        return workdir / self.project / separated_audio_speech
    
    def _setup_logging(self) -> Path:
        """Setup logging configuration and return log file path."""
        workdir = Path(self.config["DIRECTORIES"]["workdir"])
        logs_dir = self.config["PROJECT_SUBDIRS"]["logs"]
        log_dir = workdir / self.project / logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"split_segments_by_speaker_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        return log_path
    
    def _initialize_pipeline(self) -> None:
        """Initialize the pyannote diarization pipeline."""
        if self.pipeline is not None:
            return
            
        if not self.hf_token:
            raise ValueError("HuggingFace token is required. Provide via --hf-token, HUGGINGFACE_TOKEN env var, or store it in keyholder.json")
        
        console.print("[cyan]🔄 Initializing pyannote speaker diarization pipeline...")
        
        try:
            # Import heavy dependencies only when needed
            import torch
            from pyannote.audio import Pipeline
            
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pipeline.to(device)
            
            console.print(f"[green]✅ Pipeline initialized on {device}")
            logging.info(f"Pipeline initialized on {device}")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to initialize pipeline: {e}")
            logging.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def _find_audio_json_pairs(self) -> List[Tuple[Path, Path]]:
        """Find pairs of audio files and their corresponding JSON files."""
        pairs = []
        
        if not self.base_path.exists():
            console.print(f"[red]❌ Base path does not exist: {self.base_path}")
            logging.error(f"Base path does not exist: {self.base_path}")
            return pairs
        
        # Find all audio files
        audio_files = []
        for ext in self.audio_exts:
            audio_files.extend(self.base_path.glob(f"*.{ext}"))
        
        for audio_path in audio_files:
            json_path = audio_path.with_suffix(".json")
            
            if not json_path.exists():
                console.print(f"[yellow]⚠️  JSON not found for {audio_path.name}, skipping")
                logging.warning(f"JSON not found for {audio_path.name}")
                continue
            
            if not audio_path.exists():
                console.print(f"[yellow]⚠️  Audio not found for {json_path.name}, skipping")
                logging.warning(f"Audio not found for {json_path.name}")
                continue
            
            pairs.append((audio_path, json_path))
        
        console.print(f"[blue]📁 Found {len(pairs)} audio-JSON pairs")
        logging.info(f"Found {len(pairs)} audio-JSON pairs")
        
        return pairs
    
    def _extract_audio_segment(self, audio_path: Path, start_time: float, end_time: float, segment_index: int) -> Path:
        """Extract a small audio file for the given time segment."""
        import subprocess
        import tempfile
        
        # Create temporary file for the segment
        temp_dir = Path(tempfile.gettempdir()) / "speaker_segments"
        temp_dir.mkdir(exist_ok=True)
        
        segment_filename = f"{audio_path.stem}_segment_{segment_index:04d}_{start_time:.2f}-{end_time:.2f}.wav"
        segment_path = temp_dir / segment_filename
        
        # Use ffmpeg to extract the segment
        cmd = [
            "ffmpeg",
            "-y",  # overwrite output file
            "-i", str(audio_path),
            "-ss", str(start_time),  # start time
            "-t", str(end_time - start_time),  # duration
            "-c:a", "pcm_s16le", # Use a standard codec like pcm_s16le for WAV
            str(segment_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            logging.debug(f"Extracted segment {segment_index}: {segment_path}")
            return segment_path
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to extract segment {start_time}-{end_time}: {e.stderr}")
            raise
    
    def _run_diarization_on_segment(self, segment_audio_path: Path):
        """Run speaker diarization on a small extracted audio segment."""
        try:
            import torch
            
            with torch.no_grad():
                # Run diarization on the extracted segment file
                diarization = self.pipeline(str(segment_audio_path))
            
            return diarization
            
        except Exception as e:
            logging.error(f"Diarization failed for segment audio {segment_audio_path.name}: {e}")
            raise
    
    def _cleanup_segment_file(self, segment_path: Path):
        """Clean up temporary segment audio file."""
        try:
            if segment_path.exists():
                segment_path.unlink()
                logging.debug(f"Cleaned up segment file: {segment_path}")
        except Exception as e:
            logging.warning(f"Failed to cleanup segment file {segment_path}: {e}")
    
    def _get_speaker_runs_from_extracted_segment(self, annotation, original_start: float, original_end: float) -> List[Tuple[float, float, str]]:
        """Get speaker runs from diarization of an extracted segment, converting back to original timestamps."""
        if not annotation:
            return [(original_start, original_end, "UNKNOWN")]
        
        # Convert to list of (start, end, speaker) tuples
        # Note: times from extracted segment start at 0, need to add original_start
        runs = []
        for seg, _, label in annotation.itertracks(yield_label=True):
            # Convert relative times back to original timestamps
            abs_start = original_start + seg.start
            abs_end = original_start + seg.end
            runs.append((abs_start, abs_end, label))
        
        runs.sort(key=lambda x: x[0])
        
        if not runs:
             return [(original_start, original_end, "UNKNOWN")]

        # Merge adjacent runs of the same speaker
        merged_runs = []
        for run_start, run_end, speaker in runs:
            if merged_runs and merged_runs[-1][2] == speaker and abs(merged_runs[-1][1] - run_start) < 0.01:
                merged_runs[-1] = (merged_runs[-1][0], run_end, speaker)
            else:
                merged_runs.append((run_start, run_end, speaker))
        
        if not merged_runs:
             return [(original_start, original_end, "UNKNOWN")]

        # Filter out very short runs
        final_runs = []
        for i, (run_start, run_end, speaker) in enumerate(merged_runs):
            duration = run_end - run_start
            
            if duration < self.min_chunk and len(merged_runs) > 1:
                if final_runs and final_runs[-1][2] == speaker:
                     final_runs[-1] = (final_runs[-1][0], run_end, speaker)
                elif not final_runs and i < len(merged_runs) -1:
                     final_runs.append((run_start, run_end, speaker)) # Keep if it's the first and short
                else:
                     final_runs.append((run_start, run_end, speaker)) # Keep if cannot be merged
            else:
                final_runs.append((run_start, run_end, speaker))
        
        if not final_runs:
            return [(original_start, original_end, "UNKNOWN")]

        # Post-process to merge consecutive same-speaker segments again
        merged_final_runs = []
        for run_start, run_end, speaker in final_runs:
            if merged_final_runs and merged_final_runs[-1][2] == speaker:
                merged_final_runs[-1] = (merged_final_runs[-1][0], run_end, speaker)
            else:
                merged_final_runs.append((run_start, run_end, speaker))

        return merged_final_runs if merged_final_runs else [(original_start, original_end, "UNKNOWN")]
    
    def _reassign_words(self, words: List[Dict], start: float, end: float) -> List[Dict]:
        """Reassign words that fall within the given time range."""
        if not words:
            return []
        
        reassigned = []
        for word in words:
            word_start = word.get("start", 0)
            word_end = word.get("end", 0)
            word_mid = (word_start + word_end) / 2
            
            if start <= word_mid <= end:
                reassigned.append(word.copy())
            else:
                overlap_start = max(start, word_start)
                overlap_end = min(end, word_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                word_duration = word_end - word_start
                
                if word_duration > 0 and (overlap_duration / word_duration) >= 0.5:
                    reassigned.append(word.copy())
        
        return reassigned
    
    def _round_times(self, obj: Any) -> Any:
        """Recursively round time values in an object."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key in ["start", "end"] and isinstance(value, (int, float)):
                    result[key] = round(float(value), self.round_decimals)
                else:
                    result[key] = self._round_times(value)
            return result
        elif isinstance(obj, list):
            return [self._round_times(item) for item in obj]
        else:
            return obj
    
    def _split_segment(self, segment: Dict, audio_path: Path, segment_index: int) -> List[Dict]:
        """Split a segment by speaker if it contains multiple speakers."""
        start = float(segment["start"])
        end = float(segment["end"])
        words = segment.get("words", [])

        if (end - start) < 0.5: # 500ms threshold
            console.print(f"[yellow]    Segment too short ({end-start:.2f}s), skipping diarization.")
            new_segment = segment.copy()
            if self.add_speaker_field:
                new_segment["speaker"] = "UNKNOWN"
            return [self._round_times(new_segment)]
        
        segment_audio_path = None
        try:
            segment_audio_path = self._extract_audio_segment(audio_path, start, end, segment_index)
            console.print(f"[cyan]    Extracted segment audio: {segment_audio_path.name}")
            diarization = self._run_diarization_on_segment(segment_audio_path)
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to process segment {start}-{end}, keeping original")
            logging.warning(f"Failed to process segment {start}-{end}, keeping original: {e}")
            if segment_audio_path: self._cleanup_segment_file(segment_audio_path)
            new_segment = segment.copy()
            if self.add_speaker_field:
                new_segment["speaker"] = "UNKNOWN"
            return [self._round_times(new_segment)]
        finally:
            if segment_audio_path: self._cleanup_segment_file(segment_audio_path)

        speaker_runs = self._get_speaker_runs_from_extracted_segment(diarization, start, end)
        
        if len(speaker_runs) <= 1:
            new_segment = segment.copy()
            if self.add_speaker_field:
                speaker = speaker_runs[0][2] if speaker_runs else "UNKNOWN"
                new_segment["speaker"] = speaker
            return [self._round_times(new_segment)]
        
        new_segments = []
        for run_start, run_end, speaker in speaker_runs:
            reassigned_words = self._reassign_words(words, run_start, run_end)
            
            if not reassigned_words and (run_end - run_start) < self.min_chunk:
                continue

            new_text = " ".join(word["word"] for word in reassigned_words) if reassigned_words else ""
            if not new_text:
                 logging.warning(f"No words could be reassigned to speaker run {speaker} from {run_start:.2f} to {run_end:.2f}. Segment will have empty text.")

            new_segment = { "start": run_start, "end": run_end, "text": new_text, "words": reassigned_words }
            if self.add_speaker_field:
                new_segment["speaker"] = speaker
            
            new_segments.append(self._round_times(new_segment))
        
        if not new_segments:
            new_segment = segment.copy()
            if self.add_speaker_field:
                new_segment["speaker"] = "UNKNOWN"
            return [self._round_times(new_segment)]

        return new_segments
    
    def _process_json_file(self, json_path: Path, audio_path: Path) -> Dict[str, int]:
        """Process a JSON file and split segments by speaker."""
        console.print(f"[cyan]📝 Processing {json_path.name}...")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            console.print(f"[red]❌ Failed to load {json_path.name}: {e}")
            logging.error(f"Failed to load {json_path.name}: {e}")
            return {"scanned": 0, "split": 0, "created": 0}
        
        segments = data.get("segments", [])
        if not segments:
            console.print(f"[yellow]⚠️  No segments found in {json_path.name}")
            logging.warning(f"No segments found in {json_path.name}")
            return {"scanned": 0, "split": 0, "created": 0}
        
        if self.backup:
            backup_path = json_path.with_suffix(".json.bak")
            shutil.copy2(json_path, backup_path)
            console.print(f"[blue]💾 Backup created: {backup_path.name}")
        
        new_segments = []
        stats = {"scanned": len(segments), "split": 0, "created": 0}
        
        for i, segment in enumerate(segments):
            console.print(f"[cyan]  Processing segment {i+1}/{len(segments)}: {segment.get('start', 0):.2f}s - {segment.get('end', 0):.2f}s")
            split_segments = self._split_segment(segment, audio_path, i)
            if len(split_segments) > 1:
                stats["split"] += 1
                console.print(f"[green]    Split into {len(split_segments)} speaker segments")
            stats["created"] += len(split_segments)
            new_segments.extend(split_segments)
        
        data["segments"] = new_segments
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            console.print(f"[green]✅ Updated {json_path.name}: {stats['scanned']} → {stats['created']} segments")
            logging.info(f"Updated {json_path.name}: {stats}")
        except Exception as e:
            console.print(f"[red]❌ Failed to write {json_path.name}: {e}")
            logging.error(f"Failed to write {json_path.name}: {e}")
        
        return stats
    
    def process_all(self) -> None:
        """Process all audio-JSON pairs in the project."""
        console.print(f"[bold blue]🚀 Starting speaker segment splitting for project: {self.project}")
        logging.info(f"Starting speaker segment splitting for project: {self.project}")
        
        self._initialize_pipeline()
        
        pairs = self._find_audio_json_pairs()
        if not pairs:
            console.print("[yellow]⚠️  No audio-JSON pairs found")
            logging.warning("No audio-JSON pairs found")
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
        console.print(f"[green]Segments scanned: {total_stats['scanned']}")
        console.print(f"[green]Segments split: {total_stats['split']}")
        console.print(f"[green]Total segments created: {total_stats['created']}")
        console.print(f"[blue]Log file: {self.log_path}")
        logging.info(f"Processing complete. Final stats: {total_stats}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Split transcript segments by speaker using pyannote/speaker-diarization-3.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/split_segments_by_speaker.py -p MyProject_001 --hf-token YOUR_HF_TOKEN
  python scripts/split_segments_by_speaker.py -p MyProject_001 --min-chunk 0.25 --round 3
  (If token is in keyholder.json or env var, you can omit --hf-token)
  python scripts/split_segments_by_speaker.py -p MyProject_001
        """
    )
    
    parser.add_argument("-p", "--project", required=True, help="Project leaf folder name under workdir/")
    parser.add_argument("--hf-token", help="HuggingFace access token (overrides env var and keyholder.json)")
    parser.add_argument("--audio-exts", default="wav,flac,mp3,m4a", help="Comma-separated audio file extensions (default: wav,flac,mp3,m4a)")
    parser.add_argument("--inplace", action="store_true", default=True, help="Overwrite files in place (default: true)")
    parser.add_argument("--min-chunk", type=float, default=0.20, help="Minimum chunk duration in seconds (default: 0.20)")
    parser.add_argument("--round", type=int, default=2, help="Number of decimal places for time rounding (default: 2)")
    parser.add_argument("--add-speaker-field", action="store_true", default=True, help="Add speaker field to segments (default: true)")
    parser.add_argument("--backup", action="store_true", default=True, help="Create backup files before modification (default: true)")
    
    args = parser.parse_args()
    
    try:
        splitter = SpeakerSegmentSplitter(
            project=args.project,
            hf_token=args.hf_token,
            audio_exts=args.audio_exts,
            inplace=args.inplace,
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
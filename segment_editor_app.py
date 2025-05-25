import os
import json
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from datetime import timedelta
import uvicorn
import re # For sanitizing filenames if needed, and parsing
from typing import Optional
from pydantic import BaseModel

# --- Pydantic Models ---
class SegmentUpdatePayload(BaseModel):
    text: Optional[str] = None

# --- Configuration Loading ---
try:
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    print("FATAL ERROR: config.json not found. Please ensure it exists in the project root.")
    exit(1)
except json.JSONDecodeError:
    print("FATAL ERROR: config.json is not valid JSON.")
    exit(1)

WORKDIR_CONFIG_VALUE = config.get("DIRECTORIES", {}).get("workdir", "workdir") # Default to "workdir"
if not os.path.isabs(WORKDIR_CONFIG_VALUE):
    WORKDIR = os.path.abspath(os.path.join(os.getcwd(), WORKDIR_CONFIG_VALUE))
else:
    WORKDIR = os.path.abspath(WORKDIR_CONFIG_VALUE)
WORKDIR = Path(os.path.normpath(WORKDIR))

PROJECT_SUBDIRS = config.get("PROJECT_SUBDIRS", {})
ORIGINAL_AUDIO_SUBDIR_NAME = PROJECT_SUBDIRS.get("separated_audio", "2_separated_audio")
SPLITS_SUBDIR_NAME = PROJECT_SUBDIRS.get("splits", "4_splits")

app = FastAPI(title="Segment Editor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
def get_projects():
    if WORKDIR.exists() and WORKDIR.is_dir():
        return sorted([d.name for d in WORKDIR.iterdir() if d.is_dir()])
    return []

def srt_time_to_seconds(srt_time_str: str) -> float:
    """Converts SRT-like time string (HH:MM:SS,mmm or HH-MM-SS-mmm) to seconds."""
    # Normalize separators: replace last '-' (before ms) with '.' and others with ':'
    # Handle cases like 00-00-00-031 or 00:00:00,031
    
    # Check if the last separator for ms is '-' or ','
    if srt_time_str[-4] == '-': # Format like HH-MM-SS-mmm
        time_part = srt_time_str[:-4]
        ms_part = srt_time_str[-3:]
        normalized_time_str = time_part.replace('-', ':') + '.' + ms_part
    elif srt_time_str[-4] == ',': # Format like HH-MM-SS,mmm
        time_part = srt_time_str[:-4]
        ms_part = srt_time_str[-3:]
        normalized_time_str = time_part.replace('-', ':') + '.' + ms_part
    else: # Fallback or unexpected format
        # This case should ideally not be hit if filenames are consistent
        # For safety, try replacing all '-' with ':' and last ',' with '.'
        temp_str = srt_time_str.replace(',', '.')
        parts = temp_str.split(':')
        if len(parts) == 3 and '.' in parts[2]:
             normalized_time_str = temp_str
        else: # If still not parsable, raise error
            raise ValueError(f"Invalid or unsupported SRT time format: {srt_time_str}")

    # Now normalized_time_str should be like HH:MM:SS.mmm
    hms_parts = normalized_time_str.split(':')
    if len(hms_parts) != 3:
        raise ValueError(f"Invalid time format after normalization: {normalized_time_str} (from {srt_time_str})")

    h = int(hms_parts[0])
    m = int(hms_parts[1])
    
    s_ms_part = hms_parts[2].split('.')
    s = int(s_ms_part[0])
    ms = int(s_ms_part[1]) if len(s_ms_part) > 1 else 0
    
    return h * 3600 + m * 60 + s + (ms / 1000.0)

def seconds_to_srt_time_for_filename(seconds: float) -> str:
    """Formats seconds into HH-MM-SS-mmm for filenames."""
    if seconds < 0:
        seconds = 0
    td = timedelta(seconds=abs(seconds))
    total_seconds_int = int(td.total_seconds())
    hours, remainder = divmod(total_seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = int((abs(seconds) - total_seconds_int) * 1000)
    return f"{hours:02}-{minutes:02}-{secs:02}-{milliseconds:03}"

def sanitize_filename_segment(text: str) -> str:
    """Basic sanitization for filename segments, especially speaker names."""
    return re.sub(r'[\\/*?:"<>|]', '', text).replace(" ", "_")

# --- API Endpoints ---
@app.get("/api/editor/projects")
async def list_projects_endpoint():
    return {"projects": get_projects()}

@app.get("/api/editor/projects/{project_name}/segments")
async def list_segments_endpoint(project_name: str):
    project_path = WORKDIR / project_name
    if not project_path.is_dir():
        raise HTTPException(status_code=404, detail="Project not found")

    splits_dir = project_path / SPLITS_SUBDIR_NAME
    if not splits_dir.is_dir():
        return {"segments": [], "error": f"Splits directory '{SPLITS_SUBDIR_NAME}' not found in project."}

    segments = []
    audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

    for item in sorted(splits_dir.iterdir()):
        if item.is_file() and item.suffix.lower() in audio_extensions:
            base_name = item.stem
            txt_file = splits_dir / f"{base_name}.txt"
            
            text_content = ""
            if txt_file.exists():
                try:
                    text_content = txt_file.read_text(encoding="utf-8")
                except Exception as e:
                    print(f"Warning: Could not read text file {txt_file}: {e}")
                    text_content = f"[Error reading text: {e}]"
            
            # Parse start_time, end_time, speaker from filename base_name
            # Filename format: 00-00-00,000_00-00-00,000_SPEAKER_ID
            parts = base_name.split('_')
            if len(parts) < 3:
                print(f"Warning: Could not parse filename {base_name}. Skipping.")
                continue
                
            try:
                start_time_str = parts[0]
                end_time_str = parts[1]
                speaker = "_".join(parts[2:]) # Speaker name might contain underscores

                start_sec = srt_time_to_seconds(start_time_str)
                end_sec = srt_time_to_seconds(end_time_str)
                
                # Construct URL relative to WORKDIR mount point
                relative_audio_path = (Path(project_name) / SPLITS_SUBDIR_NAME / item.name).as_posix()

                segments.append({
                    "id": base_name, # Original base_name used as ID
                    "audio_file_name": item.name,
                    "audio_url": f"/static_workdir_editor/{relative_audio_path}",
                    "text": text_content,
                    "original_start_time_sec": start_sec,
                    "original_end_time_sec": end_sec,
                    "speaker": speaker,
                })
            except ValueError as e:
                print(f"Warning: Error parsing time from filename {base_name}: {e}. Skipping.")
            except Exception as e:
                print(f"Warning: Unexpected error processing file {item.name}: {e}. Skipping.")
                
    # Filter out segments from a potential "deleted_segments" subdirectory
    # This assumes deleted segments are moved into a subdir named as such within SPLITS_SUBDIR_NAME
    # A more robust way might be a manifest file, but this is simpler for now.
    # For this to work, the "deleted_segments" check should be more integrated.
    # Let's assume for now that list_segments_endpoint should NOT show deleted items.
    # The deletion endpoint will handle moving them.
    # So, we need to ensure that when iterating `splits_dir`, we skip `deleted_segments` dir.
    
    final_segments = []
    deleted_subdir_name = "deleted_segments" # Define the name of the deleted segments directory

    for item in sorted(splits_dir.iterdir()):
        if item.is_dir() and item.name == deleted_subdir_name:
            continue # Skip the deleted_segments directory itself

        if item.is_file() and item.suffix.lower() in audio_extensions:
            base_name = item.stem
            txt_file = splits_dir / f"{base_name}.txt"
            
            text_content = ""
            if txt_file.exists():
                try:
                    text_content = txt_file.read_text(encoding="utf-8")
                except Exception as e:
                    print(f"Warning: Could not read text file {txt_file}: {e}")
                    text_content = f"[Error reading text: {e}]"
            
            parts = base_name.split('_')
            if len(parts) < 3:
                print(f"Warning: Could not parse filename {base_name} in {splits_dir}. Skipping.")
                continue
                
            try:
                start_time_str = parts[0]
                end_time_str = parts[1]
                speaker = "_".join(parts[2:])

                start_sec = srt_time_to_seconds(start_time_str)
                end_sec = srt_time_to_seconds(end_time_str)
                
                relative_audio_path = (Path(project_name) / SPLITS_SUBDIR_NAME / item.name).as_posix()

                final_segments.append({
                    "id": base_name,
                    "audio_file_name": item.name,
                    "audio_url": f"/static_workdir_editor/{relative_audio_path}",
                    "text": text_content,
                    "original_start_time_sec": start_sec,
                    "original_end_time_sec": end_sec,
                    "speaker": speaker,
                })
            except ValueError as e:
                print(f"Warning: Error parsing time from filename {base_name}: {e}. Skipping.")
            except Exception as e:
                print(f"Warning: Unexpected error processing file {item.name}: {e}. Skipping.")
                
    return {"segments": final_segments}

@app.post("/api/editor/projects/{project_name}/segments/{segment_id}/update")
async def update_segment_endpoint(
    project_name: str, 
    segment_id: str, 
    start_offset_sec: float = Query(0.0), 
    end_offset_sec: float = Query(0.0),
    payload: Optional[SegmentUpdatePayload] = Body(None) # Changed to accept JSON body
):
    project_path = WORKDIR / project_name
    if not project_path.is_dir():
        raise HTTPException(status_code=404, detail="Project not found")

    original_audio_dir = project_path / ORIGINAL_AUDIO_SUBDIR_NAME
    splits_dir = project_path / SPLITS_SUBDIR_NAME

    if not original_audio_dir.is_dir() or not splits_dir.is_dir():
        raise HTTPException(status_code=404, detail="Required subdirectories not found.")

    # Find the original full audio file, prioritizing "*_speech.wav" as per user feedback
    audio_extensions_order = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    original_full_audio_path = None

    # Try to find "*_speech.wav" first
    speech_wav_files = list(original_audio_dir.glob("*_speech.wav"))
    if speech_wav_files:
        original_full_audio_path = speech_wav_files[0]
        print(f"Found primary speech audio: {original_full_audio_path}")
    else:
        # Fallback: try other extensions with _speech suffix
        for ext in audio_extensions_order:
            if ext == '.wav': continue # Already checked
            speech_files_other_ext = list(original_audio_dir.glob(f"*_speech{ext}"))
            if speech_files_other_ext:
                original_full_audio_path = speech_files_other_ext[0]
                print(f"Found primary speech audio (other ext): {original_full_audio_path}")
                break
    
    if not original_full_audio_path or not original_full_audio_path.is_file():
        error_msg = (f"Original speech audio file (e.g., *_speech.wav or *_speech.mp3) "
                     f"not found in {original_audio_dir}. Please ensure it exists and "
                     f"follows the naming convention, and is not a non_speech.wav file.")
        print(f"ERROR: {error_msg}")
        # Also check for non_speech.wav to provide a more specific message if that's all that's found
        non_speech_files = list(original_audio_dir.glob("*non_speech.wav"))
        if non_speech_files and not any(f.name.endswith(f"_speech{ext}") for ext in audio_extensions_order for f in original_audio_dir.iterdir()):
             error_msg += " Found non_speech.wav, but a speech-specific file is required."
        raise HTTPException(status_code=404, detail=error_msg)

    # Parse original segment details from segment_id (which is the old base_name)
    parts = segment_id.split('_')
    if len(parts) < 3:
        raise HTTPException(status_code=400, detail=f"Invalid segment_id format: {segment_id}")
    
    try:
        original_start_str = parts[0]
        original_end_str = parts[1]
        original_speaker = "_".join(parts[2:])
        
        current_start_sec = srt_time_to_seconds(original_start_str)
        current_end_sec = srt_time_to_seconds(original_end_str)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing time from segment_id {segment_id}: {e}")

    new_start_sec = current_start_sec + start_offset_sec
    new_end_sec = current_end_sec + end_offset_sec

    # Ensure times are valid
    if new_start_sec < 0: new_start_sec = 0
    if new_end_sec < new_start_sec: new_end_sec = new_start_sec + 0.01 # Ensure end is after start, min duration
    
    try:
        print(f"Loading original audio: {original_full_audio_path}")
        audio = AudioSegment.from_file(original_full_audio_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading original audio file: {e}")

    # Calculate new slice times in milliseconds for Pydub
    new_start_ms = int(new_start_sec * 1000)
    new_end_ms = int(new_end_sec * 1000)

    # Ensure slice indices are within bounds
    new_start_ms = max(0, new_start_ms)
    new_end_ms = min(len(audio), new_end_ms)

    if new_start_ms >= new_end_ms:
         # This can happen if offsets make the segment very short or invalid
        print(f"Warning: Adjusted slice times are invalid or result in zero/negative duration. Start: {new_start_ms}ms, End: {new_end_ms}ms. Skipping export of audio for {segment_id}.")
        # We might still want to rename the .txt file if only times change, but for now, let's report an error.
        # Or, we could decide to delete the segment if it becomes invalid.
        # For now, let's prevent creating a bad audio file.
        # The client side should ideally prevent such extreme offsets.
        raise HTTPException(status_code=400, detail=f"Calculated new segment duration is zero or negative. Original: {segment_id}, New times: {new_start_sec:.3f}s - {new_end_sec:.3f}s")


    try:
        new_audio_chunk = audio[new_start_ms:new_end_ms]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error slicing audio: {e}")

    # Determine original segment file extension
    original_segment_audio_file = None
    for ext in audio_extensions_order: # Use the ordered list
        potential_old_file = splits_dir / f"{segment_id}{ext}"
        if potential_old_file.exists():
            original_segment_audio_file = potential_old_file
            break
    
    if not original_segment_audio_file:
        raise HTTPException(status_code=404, detail=f"Original segment audio file for ID {segment_id} not found in {splits_dir}")
    
    original_segment_extension = original_segment_audio_file.suffix # e.g. ".wav"
    export_format = original_segment_extension.lstrip('.') # e.g. "wav"

    # New filename based on new times
    new_start_filename_str = seconds_to_srt_time_for_filename(new_start_sec)
    new_end_filename_str = seconds_to_srt_time_for_filename(new_end_sec)
    new_segment_base_name = f"{new_start_filename_str}_{new_end_filename_str}_{sanitize_filename_segment(original_speaker)}"
    
    new_segment_audio_path = splits_dir / f"{new_segment_base_name}{original_segment_extension}"
    new_segment_text_path = splits_dir / f"{new_segment_base_name}.txt"

    # Define old paths
    old_audio_path = original_segment_audio_file # This is already a Path object
    old_text_path = splits_dir / f"{segment_id}.txt"

    # Check if there's any actual change in filename needed
    if new_segment_base_name == segment_id:
        # This means start_offset_sec and end_offset_sec were effectively zero,
        # and speaker name (if it were part of ID) also didn't change.
        # No file operations needed if filenames are identical.
        print(f"No change in segment timing or speaker for {segment_id}. No file operations performed.")
        return {
            "message": "No change in segment timing. Files remain as is.",
            "new_segment_id": segment_id, # Return original ID
            "new_audio_path": str(old_audio_path.relative_to(WORKDIR)),
            "new_text_path": str(old_text_path.relative_to(WORKDIR)),
        }

    # Proceed with file operations only if filenames will change
    try:
        new_audio_chunk.export(new_segment_audio_path, format=export_format)
        print(f"Exported new audio: {new_segment_audio_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting new audio chunk: {e}")

    # If audio export was successful, now handle the old files
    if old_audio_path.exists() and old_audio_path != new_segment_audio_path: # Ensure we don't delete if overwriting same file (though previous check should prevent this)
        try:
            old_audio_path.unlink()
            print(f"Deleted old audio: {old_audio_path}")
        except Exception as e:
            print(f"Warning: Could not delete old audio file {old_audio_path}: {e}")

    if old_text_path.exists():
        if old_text_path != new_segment_text_path: 
            try:
                # If text is provided in payload, write it to the new file after renaming/moving
                if payload and payload.text is not None:
                    # To be safe, read old content first if rename fails and we need to write new file
                    # However, with rename, the content moves with the file.
                    # The main concern is if the new file needs *new* text.
                    old_text_path.rename(new_segment_text_path)
                    print(f"Renamed old text file {old_text_path} to {new_segment_text_path}")
                    # Now, if payload has text, overwrite the (just renamed) new_segment_text_path
                    if payload and payload.text is not None:
                        new_segment_text_path.write_text(payload.text, encoding="utf-8")
                        print(f"Overwrote {new_segment_text_path} with new text from payload.")
                else: # No new text in payload, just rename
                    old_text_path.rename(new_segment_text_path)
                    print(f"Renamed old text file {old_text_path} to {new_segment_text_path} (no text change in payload).")
            except Exception as e:
                print(f"Warning: Could not rename/write text file {old_text_path} to {new_segment_text_path}: {e}")
        else: # Filenames are the same (e.g. only text changed, no time change)
            if payload and payload.text is not None:
                try:
                    old_text_path.write_text(payload.text, encoding="utf-8")
                    print(f"Overwrote {old_text_path} with new text from payload (filename unchanged).")
                except Exception as e:
                    print(f"Warning: Could not overwrite text file {old_text_path} with new text: {e}")
            else:
                print(f"Text file name {old_text_path} remains unchanged, and no new text provided. No text file operation.")
    else: # Original text file did not exist
        print(f"Warning: Original text file {old_text_path} not found.")
        default_text_to_write = payload.text if (payload and payload.text is not None) else ""
        try:
            new_segment_text_path.write_text(default_text_to_write, encoding="utf-8")
            if default_text_to_write:
                print(f"Created new text file {new_segment_text_path} with text from payload.")
            else:
                print(f"Created empty new text file {new_segment_text_path} as original was missing and no text in payload.")
        except Exception as e:
            print(f"Warning: Could not create new text file {new_segment_text_path}: {e}")

    return {
        "message": "Segment updated successfully",
        "new_segment_id": new_segment_base_name,
        "new_audio_path": str(new_segment_audio_path.relative_to(WORKDIR)),
        "new_text_path": str(new_segment_text_path.relative_to(WORKDIR)),
    }

@app.post("/api/editor/projects/{project_name}/segments/{segment_id}/delete")
async def delete_segment_soft_endpoint(project_name: str, segment_id: str):
    project_path = WORKDIR / project_name
    if not project_path.is_dir():
        raise HTTPException(status_code=404, detail="Project not found")

    splits_dir = project_path / SPLITS_SUBDIR_NAME
    deleted_segments_dir = splits_dir / "deleted_segments"
    
    if not splits_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Splits directory '{SPLITS_SUBDIR_NAME}' not found.")

    # Ensure the deleted_segments directory exists
    deleted_segments_dir.mkdir(parents=True, exist_ok=True)

    # Find the audio file for the segment
    original_segment_audio_file = None
    moved_files_count = 0
    audio_extensions_order = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'] # Consistent with update endpoint

    for ext in audio_extensions_order:
        potential_old_file = splits_dir / f"{segment_id}{ext}"
        if potential_old_file.exists():
            original_segment_audio_file = potential_old_file
            break
    
    if not original_segment_audio_file:
        # Check if it was already deleted (idempotency)
        for ext in audio_extensions_order:
            if (deleted_segments_dir / f"{segment_id}{ext}").exists():
                return {"message": f"Segment {segment_id} already in deleted_segments."}
        raise HTTPException(status_code=404, detail=f"Audio file for segment ID {segment_id} not found in {splits_dir}")

    old_text_path = splits_dir / f"{segment_id}.txt"
    
    new_audio_path_in_deleted = deleted_segments_dir / original_segment_audio_file.name
    new_text_path_in_deleted = deleted_segments_dir / old_text_path.name

    try:
        if original_segment_audio_file.exists():
            shutil.move(str(original_segment_audio_file), str(new_audio_path_in_deleted))
            print(f"Moved audio {original_segment_audio_file} to {new_audio_path_in_deleted}")
            moved_files_count +=1
        
        if old_text_path.exists():
            shutil.move(str(old_text_path), str(new_text_path_in_deleted))
            print(f"Moved text {old_text_path} to {new_text_path_in_deleted}")
            moved_files_count +=1
        
        if moved_files_count == 0:
             return {"message": f"No files found for segment {segment_id} to move to deleted_segments."}

        return {"message": f"Segment {segment_id} moved to deleted_segments successfully."}
    except Exception as e:
        print(f"Error moving segment {segment_id} to deleted_segments: {e}")
        # Attempt to move back if one succeeded and other failed? For now, just raise.
        raise HTTPException(status_code=500, detail=f"Error moving segment to deleted_segments: {e}")


# --- Static File Serving ---
# Serve the frontend (HTML, JS, CSS)
frontend_dir = Path("frontend")
if frontend_dir.exists() and frontend_dir.is_dir():
    app.mount("/editor", StaticFiles(directory=frontend_dir, html=True), name="frontend_editor")
    print(f"Frontend editor served from: {frontend_dir.resolve()} at /editor")
else:
    print(f"Warning: Frontend directory '{frontend_dir}' not found. UI will not be available.")

# Serve WORKDIR for audio files, similar to main_app_new.py but with a different mount point
# to avoid conflicts if both apps run simultaneously.
if WORKDIR.exists() and WORKDIR.is_dir():
    app.mount("/static_workdir_editor", StaticFiles(directory=WORKDIR, html=False, check_dir=True), name="static_workdir_editor_mount")
    print(f"Static files from WORKDIR '{WORKDIR}' mounted at '/static_workdir_editor'")
else:
    print(f"Warning: WORKDIR '{WORKDIR}' not found. Audio files for editor will not be served.")


if __name__ == "__main__":
    # Determine a port different from main_app_new.py (which uses 8001)
    # Let's use 8002 for the segment editor.
    # Using import string for app to allow reloaders to work correctly.
    uvicorn.run("segment_editor_app:app", host="0.0.0.0", port=8002, reload=True)

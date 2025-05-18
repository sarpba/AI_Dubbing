from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import subprocess
import asyncio
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import uvicorn
import shutil
import base64
from datetime import datetime
from mutagen import File as MutagenFile # For audio duration

# Load existing config from original app
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

WORKDIR_CONFIG_VALUE = config["DIRECTORIES"]["workdir"]
if not os.path.isabs(WORKDIR_CONFIG_VALUE):
    # If workdir is relative in config, assume it's relative to the project root (CWD)
    WORKDIR = os.path.abspath(os.path.join(os.getcwd(), WORKDIR_CONFIG_VALUE))
else:
    WORKDIR = os.path.abspath(WORKDIR_CONFIG_VALUE) # Normalize even if absolute
WORKDIR = os.path.normpath(WORKDIR)
print(f"Resolved WORKDIR to: {WORKDIR}")


KEYHOLDER_FILE = "keyholder.json"

app = FastAPI(title="AI Dubbing API",
             description="Modern interface for dubbing workflow")

# CORS setup for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ProjectCreateBody(BaseModel): # For request body when not using Form
    project_name: str

class CheckerTextSave(BaseModel):
    text: str

class ApiKeys(BaseModel):
    hf_token: Optional[str] = None
    deepL_api_key: Optional[str] = None

class ScriptRunParams(BaseModel): # This might be less used if params are passed directly in dicts
    # Base parameters for all scripts
    hf_token: Optional[str] = None
    deepL_api_key: Optional[str] = None
    # Will be extended per script

# WebSocket manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, project_name: str, websocket: WebSocket):
        await websocket.accept()
        if project_name not in self.active_connections:
            self.active_connections[project_name] = []
        self.active_connections[project_name].append(websocket)

    def disconnect(self, project_name: str, websocket: WebSocket):
        if project_name in self.active_connections:
            self.active_connections[project_name].remove(websocket)

    async def broadcast(self, project_name: str, message: str):
        if project_name in self.active_connections:
            for connection in self.active_connections[project_name]:
                try:
                    await connection.send_text(message)
                except:
                    self.disconnect(project_name, connection)

manager = ConnectionManager()

# Helper functions from original app
def encode_key(key):
    return base64.b64encode(key.encode('utf-8')).decode('utf-8') if key else ""

def decode_key(encoded_key):
    return base64.b64decode(encoded_key.encode('utf-8')).decode('utf-8') if encoded_key else ""

def load_keys():
    if os.path.exists(KEYHOLDER_FILE):
        try:
            with open(KEYHOLDER_FILE, "r", encoding="utf-8") as f:
                encoded_keys = json.load(f)
            return {
                "hf_token": decode_key(encoded_keys.get("hf_token", "")),
                "deepL_api_key": decode_key(encoded_keys.get("deepL_api_key", ""))
            }
        except Exception as e:
            log_action(f"Error reading keyholder file: {e}", "")
            return {"hf_token": "", "deepL_api_key": ""}
    return {"hf_token": "", "deepL_api_key": ""}

def save_keys(hf_token, deepL_api_key):
    keys = {
        "hf_token": encode_key(hf_token),
        "deepL_api_key": encode_key(deepL_api_key)
    }
    try:
        with open(KEYHOLDER_FILE, "w", encoding="utf-8") as f:
            json.dump(keys, f, indent=2)
    except Exception as e:
        log_action(f"Error writing keyholder file: {e}", "")

def get_projects():
    if os.path.exists(WORKDIR):
        return [d for d in os.listdir(WORKDIR) if os.path.isdir(os.path.join(WORKDIR, d))]
    return []

# Functions to get dynamic choices for frontend
def get_tts_subdirs():
    """Return a list of subdirectories in the TTS folder."""
    tts_dir = config["DIRECTORIES"]["TTS"]
    if os.path.exists(tts_dir):
        return [d for d in os.listdir(tts_dir) if os.path.isdir(os.path.join(tts_dir, d))]
    return []

def get_normaliser_subdirs():
    """Return a list of subdirectories in the normalisers folder."""
    norm_dir = config["DIRECTORIES"]["normalisers"]
    if os.path.exists(norm_dir):
        return [d for d in os.listdir(norm_dir) if os.path.isdir(os.path.join(norm_dir, d))]
    return []

def get_demucs_models():
    """Return a list of available Demucs models."""
    # These are typically hardcoded or based on what the script supports
    return ["htdemucs", "htdemucs_ft", "hdemucs_mmi", "mdx", "mdx_extra", "htdemucs_6s", "mdx_q", "mdx_extra_q"]

# --- Helper functions for Checker App ---
def parse_timestamp_from_basename(basename: str) -> float:
    ts_part = basename.split('_')[0] # Assuming format like "0-00-01.234_chunk_0"
    parts = ts_part.split('-')
    if len(parts) < 3:
        return 0.0
    try:
        h = int(parts[0])
        m = int(parts[1])
        s_ms = parts[2].split('.')
        s = int(s_ms[0])
        ms = int(s_ms[1]) if len(s_ms) > 1 else 0
        return h * 3600 + m * 60 + s + (ms / 1000.0)
    except Exception as e:
        # Log this error appropriately if integrating with a logger
        print(f"Error parsing timestamp from {ts_part}: {e}")
        return 0.0

def load_checker_project_data(project_name: str) -> List[Dict[str, Any]]:
    project_dir = os.path.join(WORKDIR, project_name)
    splits_dir = os.path.join(project_dir, config["PROJECT_SUBDIRS"]["splits"])
    trans_dir = os.path.join(project_dir, config["PROJECT_SUBDIRS"]["translated_splits"])
    
    print(f"Checker: Loading data for project '{project_name}' from WORKDIR: {WORKDIR}")
    print(f"Checker: Splits directory: {splits_dir}")
    print(f"Checker: Translated splits directory: {trans_dir}")

    data = []
    if not os.path.isdir(splits_dir):
        print(f"Checker Warning: Splits directory not found: {splits_dir}")
        return data

    raw_files = {} 

    for file_type, directory in [("splits", splits_dir), ("translated", trans_dir)]:
        if not os.path.isdir(directory):
            print(f"Checker Warning: {file_type.capitalize()} directory not found: {directory}")
            continue
        for file in os.listdir(directory):
            basename, ext = os.path.splitext(file)
            if basename not in raw_files:
                raw_files[basename] = {"basename": basename}
            
            if ext == ".wav":
                raw_files[basename][f"{file_type}_wav_path"] = os.path.join(directory, file)
            elif ext == ".txt":
                try:
                    with open(os.path.join(directory, file), "r", encoding="utf-8") as tf:
                        raw_files[basename][f"{file_type}_txt"] = tf.read()
                except Exception as e:
                    print(f"Checker Warning: Could not read text file {os.path.join(directory, file)}: {e}")
                    raw_files[basename][f"{file_type}_txt"] = ""

    first_url_logged = False
    for bn, item_data in raw_files.items():
        item_data.setdefault("splits_txt", "")
        item_data.setdefault("translated_txt", "") # Note: key changed from trans_txt
        item_data["start_time"] = parse_timestamp_from_basename(bn)
        data.append(item_data)
        
    data.sort(key=lambda x: x["start_time"])

    for i in range(len(data)):
        item = data[i]
        item["allowed_interval"] = (data[i+1]["start_time"] - item["start_time"]) if i + 1 < len(data) else None
        item["overlong"] = False
        item["diff_seconds"] = 0.0
        item["trans_duration"] = 0.0

        # Process translated audio file (key is now 'translated_wav_path')
        current_trans_wav_path = item.get("translated_wav_path")
        if current_trans_wav_path and os.path.exists(current_trans_wav_path):
            try:
                audio = MutagenFile(current_trans_wav_path)
                if audio and audio.info:
                    duration = audio.info.length
                    item["trans_duration"] = duration
                    if item["allowed_interval"] is not None and duration > item["allowed_interval"]:
                        item["overlong"] = True
                        item["diff_seconds"] = duration - item["allowed_interval"]
            except Exception as e:
                print(f"Checker Warning: Could not read audio info for {current_trans_wav_path}: {e}")
            
            relative_path_trans = os.path.relpath(current_trans_wav_path, WORKDIR)
            item["trans_wav_url"] = f"/static_workdir/{relative_path_trans.replace(os.sep, '/')}"
            if not first_url_logged: 
                print(f"Checker Example URL (translated): {item['trans_wav_url']} from {current_trans_wav_path}")
        else:
            if current_trans_wav_path: # Path was expected but file not found
                 print(f"Checker Warning: Translated audio file not found: {current_trans_wav_path}")
            item["translated_wav_path"] = None
            item["trans_wav_url"] = None
            item["trans_duration"] = 0.0
            if not first_url_logged and current_trans_wav_path:
                 print(f"Checker: No URL for translated audio (file not found): {current_trans_wav_path}")


        # Process splits audio file
        current_splits_wav_path = item.get("splits_wav_path")
        if current_splits_wav_path and os.path.exists(current_splits_wav_path):
            relative_path_splits = os.path.relpath(current_splits_wav_path, WORKDIR)
            item["splits_wav_url"] = f"/static_workdir/{relative_path_splits.replace(os.sep, '/')}"
            if not first_url_logged:
                print(f"Checker Example URL (splits): {item['splits_wav_url']} from {current_splits_wav_path}")
                first_url_logged = True # Log only one set of examples
        else:
            if current_splits_wav_path: # Path was expected but file not found
                print(f"Checker Warning: Splits audio file not found: {current_splits_wav_path}")
            item["splits_wav_path"] = None
            item["splits_wav_url"] = None
            if not first_url_logged and current_splits_wav_path:
                 print(f"Checker: No URL for splits audio (file not found): {current_splits_wav_path}")
                 first_url_logged = True


    print(f"Checker: Loaded {len(data)} items for project {project_name}.")
    return data
# --- End Helper functions for Checker App ---


def log_action(message, project):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp}: {message}\n"
    if project:
        log_dir = os.path.join(WORKDIR, project, config["PROJECT_SUBDIRS"]["logs"])
    else:
        log_dir = os.path.join(WORKDIR, config["PROJECT_SUBDIRS"]["logs"])
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_message)

async def run_subprocess_command(command: List[str], project_name: str, description: str = ""):
    """Run a subprocess command and stream output via WebSocket"""
    desc_text = description if description else "script"
    start_message = f"[INFO] Starting: {desc_text}..."
    await manager.broadcast(project_name, start_message)
    log_action(start_message, project_name)
    log_action(f"Executing command: {' '.join(command)}", project_name)
    
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    async def read_stream(stream, stream_type):
        while True:
            line = await stream.readline()
            if line:
                output = line.decode(errors='replace').strip()
                await manager.broadcast(project_name, f"[{stream_type}] {output}")
                log_action(f"({desc_text}) [{stream_type}] {output}", project_name)
            else:
                # EOF
                break

    # Wait for both stdout and stderr to be fully processed
    await asyncio.gather(
        read_stream(process.stdout, "STDOUT"),
        read_stream(process.stderr, "STDERR")
    )
    
    # Wait for the process to terminate and get the return code
    return_code = await process.wait()
    
    status_message = "completed successfully" if return_code == 0 else f"failed with exit code {return_code}"
    finish_message = f"[INFO] Finished: {desc_text} - {status_message}."
    await manager.broadcast(project_name, finish_message)
    log_action(finish_message, project_name)
    
    return return_code

# API Endpoints
@app.get("/api/projects")
async def list_projects():
    return {"projects": get_projects()}

@app.post("/api/projects")
async def create_project(project_name: str = Form(...), mkv_file: UploadFile = File(None)):
    project_path = os.path.join(WORKDIR, project_name)
    if os.path.exists(project_path):
        raise HTTPException(status_code=400, detail="Project already exists")

    os.makedirs(project_path, exist_ok=True)
    logs_subdir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["logs"])
    os.makedirs(logs_subdir, exist_ok=True) # Ensure logs directory is created

    upload_subdir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["upload"])
    os.makedirs(upload_subdir, exist_ok=True)

    if mkv_file:
        file_location = os.path.join(upload_subdir, mkv_file.filename)
        try:
            with open(file_location, "wb+") as file_object:
                shutil.copyfileobj(mkv_file.file, file_object)
            log_action(f"Uploaded MKV file '{mkv_file.filename}' to project '{project_name}'", project_name)
        except Exception as e:
            log_action(f"Error uploading MKV file for project '{project_name}': {e}", project_name)
            # Optionally, clean up created project directory if upload fails critically
            # shutil.rmtree(project_path)
            raise HTTPException(status_code=500, detail=f"Could not upload MKV file: {e}")
        finally:
            mkv_file.file.close() # Ensure the file is closed

    log_action(f"Project '{project_name}' created", project_name)
    return {"message": "Project created", "project_name": project_name, "mkv_uploaded": mkv_file.filename if mkv_file else None}


@app.websocket("/ws/projects/{project_name}/status")
async def websocket_endpoint(websocket: WebSocket, project_name: str):
    await manager.connect(project_name, websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(project_name, websocket)

# Script execution endpoints
@app.post("/api/projects/{project_name}/run/separate_audio")
async def run_separate_audio(project_name: str, params: dict):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")
    
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "separate_audio.py")
    command = [
        "python", script_path,
        "-i", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["upload"]),
        "-o", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["separated_audio"]),
        "--device", params.get("device", "cuda")
    ]
    
    if params.get("keep_full_audio", False):
        command.append("--keep_full_audio")
    if params.get("non_speech_silence", False):
        command.append("--non_speech_silence")
    
    command.extend(["--chunk_size", str(params.get("chunk_size", 5))])
    command.extend(["--model", params.get("model", "htdemucs")])
    
    return_code = await run_subprocess_command(command, project_name, "separate_audio.py")
    return {"status": "completed" if return_code == 0 else "failed"}

@app.post("/api/projects/{project_name}/run/transcribe_align")
async def run_transcribe_align(project_name: str, params: dict):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")
    
    hf_token_to_use = params.get("hf_token")
    if not hf_token_to_use: # If token not provided in params, try to load saved one
        saved_keys = load_keys()
        hf_token_to_use = saved_keys.get("hf_token")
    else: # If token IS provided in params, save it for future use
        keys_to_save = load_keys()
        keys_to_save["hf_token"] = hf_token_to_use
        save_keys(keys_to_save["hf_token"], keys_to_save.get("deepL_api_key", ""))

    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "whisx.py")
    command = ["python", script_path]
    
    if hf_token_to_use: # Use the determined token
        command.extend(["--hf_token", hf_token_to_use])
    if params.get("language"):
        command.extend(["--language", params["language"]])
    
    command.append(os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["separated_audio"]))
    
    return_code = await run_subprocess_command(command, project_name, "whisx.py")
    return {"status": "completed" if return_code == 0 else "failed"}

@app.post("/api/projects/{project_name}/run/audio_split")
async def run_audio_split(project_name: str):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")
    
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "splitter.py")
    command = [
        "python", script_path,
        "--input_dir", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["separated_audio"]),
        "--output_dir", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["splits"])
    ]
    
    return_code = await run_subprocess_command(command, project_name, "splitter.py")
    return {"status": "completed" if return_code == 0 else "failed"}

@app.post("/api/projects/{project_name}/run/translate")
async def run_translate(project_name: str, params: dict):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")

    auth_key_to_use = params.get("auth_key")
    if not auth_key_to_use: # If key not provided in params, try to load saved one
        saved_keys = load_keys()
        auth_key_to_use = saved_keys.get("deepL_api_key")
    else: # If key IS provided in params, save it for future use
        keys_to_save = load_keys()
        keys_to_save["deepL_api_key"] = auth_key_to_use
        save_keys(keys_to_save.get("hf_token", ""), keys_to_save["deepL_api_key"])

    if not auth_key_to_use: # If still no key after trying to load, raise error
        raise HTTPException(status_code=400, detail="DeepL API key is required and not found. Please save it in API Key Settings or provide it for the script.")

    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "translate.py")
    command = [
        "python", script_path,
        "-input_dir", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["splits"]),
        "-output_dir", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["translated_splits"]),
        "-input_language", params.get("input_language", "EN"),
        "-output_language", params.get("output_language", "HU"),
        "-auth_key", auth_key_to_use # Use the determined key
    ]
    
    return_code = await run_subprocess_command(command, project_name, "translate.py")
    return {"status": "completed" if return_code == 0 else "failed"}

@app.post("/api/projects/{project_name}/run/generate_tts")
async def run_generate_tts(project_name: str, params: dict):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")
    
    tts_base = config["DIRECTORIES"]["TTS"]
    tts_path = os.path.join(tts_base, params.get("tts_subdir", ""))
    
    # Find checkpoint and vocab files
    ckpt_file = None
    vocab_file = None
    for f in os.listdir(tts_path):
        if f.endswith(".pt"):
            ckpt_file = os.path.join(tts_path, f)
        if f.endswith(".txt"):
            vocab_file = os.path.join(tts_path, f)
    
    if not ckpt_file or not vocab_file:
        raise HTTPException(status_code=400, detail="Missing .pt or .txt files in TTS directory")
    
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "f5_tts_infer_API.py")
    command = [
        "conda", "run", "-n", "f5-tts", "python", script_path,
        "-i", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["splits"]),
        "-ig", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["translated_splits"]),
        "-o", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["translated_splits"]),
        "--vocab_file", vocab_file,
        "--ckpt_file", ckpt_file,
        "--speed", str(params.get("speed", 1.0)),
        "--nfe_step", str(params.get("nfe_step", 32)),
        "--norm", params.get("norm_selection", "hun"),
        "--seed", str(params.get("seed", -1))
    ]
    
    if params.get("remove_silence", False):
        command.append("--remove_silence")
    
    return_code = await run_subprocess_command(command, project_name, "f5_tts_infer_API.py")
    return {"status": "completed" if return_code == 0 else "failed"}

@app.post("/api/projects/{project_name}/run/transcribe_align_chunks")
async def run_transcribe_align_chunks(project_name: str, params: dict):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not params.get("splits_lang") or not params.get("translated_splits_lang"):
        raise HTTPException(status_code=400, detail="Both language selections are required")
    
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "VAD.py")
    
    # Run for splits
    splits_dir = os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["splits"])
    command1 = ["python", script_path, splits_dir, "--lang", params["splits_lang"]]
    return_code1 = await run_subprocess_command(command1, project_name, "VAD.py on splits")
    
    # Run for translated splits
    translated_dir = os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["translated_splits"])
    command2 = ["python", script_path, translated_dir, "--lang", params["translated_splits_lang"]]
    return_code2 = await run_subprocess_command(command2, project_name, "VAD.py on translated splits")
    
    return {
        "status": "completed" if return_code1 == 0 and return_code2 == 0 else "failed",
        "splits_status": return_code1,
        "translated_status": return_code2
    }

@app.post("/api/projects/{project_name}/run/normalize_cut")
async def run_normalize_cut(project_name: str, params: dict):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")
    
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "normalise_and_cut.py")
    command = [
        "python", script_path,
        "-i", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["translated_splits"]),
        "-rj", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["splits"]),
        "--ira", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["splits"]),
        "-db", str(params.get("min_db", -40.0))
    ]
    
    if params.get("delete_empty", False):
        command.append("--delete_empty")
    
    return_code = await run_subprocess_command(command, project_name, "normalise_and_cut.py")
    return {"status": "completed" if return_code == 0 else "failed"}

@app.post("/api/projects/{project_name}/run/inspect_repair")
async def run_inspect_repair(project_name: str):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")
    
    command = ["python", "check_app.py", "--project", project_name]
    try:
        subprocess.Popen(command)
        return {"status": "started", "message": "Check app launched in new window"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to launch check_app: {str(e)}")

@app.post("/api/projects/{project_name}/run/merge_chunks_bg")
async def run_merge_chunks_bg(project_name: str):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Find background audio file
    separated_dir = os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["separated_audio"])
    bg_file = None
    for f in os.listdir(separated_dir):
        if f.endswith("non_speech.wav"):
            bg_file = os.path.join(separated_dir, f)
            break
    
    if not bg_file:
        raise HTTPException(status_code=400, detail="non_speech.wav not found")
    
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "merge_chunks_with_background.py")
    command = [
        "python", script_path,
        "-i", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["translated_splits"]),
        "-o", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["film_dubbing"]),
        "-bg", bg_file
    ]
    
    return_code = await run_subprocess_command(command, project_name, "merge_chunks_with_background.py")
    return {"status": "completed" if return_code == 0 else "failed"}

@app.post("/api/projects/{project_name}/run/merge_video")
async def run_merge_video(project_name: str, params: dict):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Find input video
    upload_dir = os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["upload"])
    input_video = None
    for f in os.listdir(upload_dir):
        if f.lower().endswith(".mkv"):
            input_video = os.path.join(upload_dir, f)
            break
    
    if not input_video:
        raise HTTPException(status_code=400, detail="No MKV file found")
    
    # Find most recent audio file
    dubbing_dir = os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["film_dubbing"])
    if not os.path.exists(dubbing_dir) or not os.listdir(dubbing_dir):
        raise HTTPException(status_code=400, detail="No dubbed audio files found")
    
    audio_files = [os.path.join(dubbing_dir, f) for f in os.listdir(dubbing_dir) 
                 if os.path.isfile(os.path.join(dubbing_dir, f))]
    input_audio = max(audio_files, key=os.path.getmtime)
    
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "merge_to_video.py")
    command = [
        "python", script_path,
        "-i", input_video,
        "-ia", input_audio,
        "-lang", params.get("language", "HUN"),
        "-o", os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["download"])
    ]
    
    return_code = await run_subprocess_command(command, project_name, "merge_to_video.py")
    return {"status": "completed" if return_code == 0 else "failed"}

@app.get("/api/projects/{project_name}/files/downloadable")
async def list_downloadable_files(project_name: str):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")
    
    download_dir = os.path.join(WORKDIR, project_name, config["PROJECT_SUBDIRS"]["download"])
    if not os.path.exists(download_dir):
        return {"files": []}
    
    files = [f for f in os.listdir(download_dir) 
            if os.path.isfile(os.path.join(download_dir, f))]
    return {"files": files}

@app.get("/api/projects/{project_name}/files/download/{filename}")
async def download_file(project_name: str, filename: str):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")
    
    file_path = os.path.join(WORKDIR, project_name, 
                           config["PROJECT_SUBDIRS"]["download"], filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

# --- Endpoints for dynamic choices ---
@app.get("/api/tts-models")
async def list_tts_models():
    return {"tts_models": get_tts_subdirs()}

@app.get("/api/normalizers")
async def list_normalizers():
    return {"normalizers": get_normaliser_subdirs()}

@app.get("/api/demucs-models")
async def list_demucs_models():
    return {"demucs_models": get_demucs_models()}

# --- Endpoints for API Keys ---
@app.get("/api/keys")
async def get_api_keys():
    return load_keys()

@app.post("/api/keys")
async def set_api_keys(keys: ApiKeys):
    current_keys = load_keys()
    hf_token_to_save = keys.hf_token if keys.hf_token is not None else current_keys.get("hf_token", "")
    deepL_api_key_to_save = keys.deepL_api_key if keys.deepL_api_key is not None else current_keys.get("deepL_api_key", "")
    save_keys(hf_token_to_save, deepL_api_key_to_save)
    return {"message": "API keys updated successfully."}


if __name__ == "__main__":
    uvicorn.run("main_app_new:app", host="0.0.0.0", port=8001, reload=True)

# Mount WORKDIR for serving audio files if needed by checker.html
print(f"Static files: Attempting to mount WORKDIR '{WORKDIR}' at '/static_workdir'")
if os.path.exists(WORKDIR) and os.path.isdir(WORKDIR):
    app.mount("/static_workdir", StaticFiles(directory=WORKDIR, html=False, check_dir=True), name="static_workdir_mount")
    print(f"Static files: Successfully mounted WORKDIR '{WORKDIR}' at '/static_workdir'")
else:
    print(f"Static files Warning: WORKDIR '{WORKDIR}' not found or not a directory. Static file serving for checker audio will not work.")


# --- Checker App API Endpoints ---
@app.get("/api/check/{project_name}/data")
async def get_checker_data(project_name: str):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        data = load_checker_project_data(project_name)
        return JSONResponse(content=data)
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Error loading checker data: {str(e)}")

@app.post("/api/check/{project_name}/item/{basename}/save_text")
async def save_checker_text(project_name: str, basename: str, payload: CheckerTextSave):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = os.path.join(WORKDIR, project_name)
    trans_dir = os.path.join(project_dir, config["PROJECT_SUBDIRS"]["translated_splits"])
    
    txt_path = os.path.join(trans_dir, basename + ".txt")
    wav_path = os.path.join(trans_dir, basename + ".wav")
    json_path = os.path.join(trans_dir, basename + ".json") # Assuming .json might exist

    try:
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(payload.text)
        log_action(f"Checker: Saved text for {basename} in project {project_name}", project_name)

        if os.path.exists(wav_path):
            os.remove(wav_path)
            log_action(f"Checker: Deleted WAV {wav_path} after text save", project_name)
        if os.path.exists(json_path): # Also remove associated JSON if it exists
            os.remove(json_path)
            log_action(f"Checker: Deleted JSON {json_path} after text save", project_name)
            
        return {"message": f"Text for {basename} saved. Associated audio/JSON removed."}
    except Exception as e:
        log_action(f"Checker: Error saving text for {basename} in {project_name}: {e}", project_name)
        raise HTTPException(status_code=500, detail=f"Error saving text: {str(e)}")

@app.delete("/api/check/{project_name}/item/{basename}/delete_audio")
async def delete_checker_audio(project_name: str, basename: str):
    if project_name not in get_projects():
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = os.path.join(WORKDIR, project_name)
    trans_dir = os.path.join(project_dir, config["PROJECT_SUBDIRS"]["translated_splits"])

    wav_path = os.path.join(trans_dir, basename + ".wav")
    json_path = os.path.join(trans_dir, basename + ".json") # Assuming .json might exist

    deleted_files = []
    try:
        if os.path.exists(wav_path):
            os.remove(wav_path)
            deleted_files.append(wav_path)
            log_action(f"Checker: Deleted WAV {wav_path}", project_name)
        if os.path.exists(json_path):
            os.remove(json_path)
            deleted_files.append(json_path)
            log_action(f"Checker: Deleted JSON {json_path}", project_name)
        
        if not deleted_files:
            return {"message": f"No audio/JSON files found for {basename} to delete."}
        return {"message": f"Audio/JSON for {basename} deleted: {', '.join(deleted_files)}"}
    except Exception as e:
        log_action(f"Checker: Error deleting audio/JSON for {basename} in {project_name}: {e}", project_name)
        raise HTTPException(status_code=500, detail=f"Error deleting files: {str(e)}")

import os
import json
import shutil
import subprocess
import datetime
import base64
import gradio as gr
import sys
import argparse

# Load configuration from config.json
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# WORKDIR path based on config
WORKDIR = config["DIRECTORIES"]["workdir"]

# File to hold API keys (encoded)
KEYHOLDER_FILE = "keyholder.json"

def encode_key(key):
    """Encode a key using Base64."""
    return base64.b64encode(key.encode('utf-8')).decode('utf-8') if key else ""

def decode_key(encoded_key):
    """Decode a Base64 encoded key."""
    return base64.b64decode(encoded_key.encode('utf-8')).decode('utf-8') if encoded_key else ""

def load_keys():
    """Load saved keys from the keyholder file and decode them."""
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
    else:
        return {"hf_token": "", "deepL_api_key": ""}

def save_keys(hf_token, deepL_api_key):
    """Encode and save the provided keys to the keyholder file."""
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
    """Return a list of project directories in the WORKDIR folder."""
    if os.path.exists(WORKDIR):
        return [d for d in os.listdir(WORKDIR) if os.path.isdir(os.path.join(WORKDIR, d))]
    return []

def log_action(message, project):
    """Write the given log message to the appropriate log directory."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp}: {message}\n"
    if project:
        log_dir = os.path.join(WORKDIR, project, config["PROJECT_SUBDIRS"]["logs"])
    else:
        log_dir = os.path.join(WORKDIR, config["PROJECT_SUBDIRS"]["logs"])
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_message)

def run_subprocess_command(command, current_project, description=""):
    """
    Helper function to run a subprocess command and log its execution.
    
    Args:
        command (list): The command to run.
        current_project (str): The current project name.
        description (str): A short description of the command (e.g. script name).
    
    Returns:
        str: The standard output if successful, otherwise an error message.
    """
    desc_text = f"{description} " if description else ""
    log_action(f"Running {desc_text}with command: {' '.join(command)}", current_project)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error: {e.stderr}"
    log_action(f"{description} finished with output: {output}", current_project)
    return output

def toggle_project_fields(mode):
    """Toggle the visibility of the project panels based on the radio selection."""
    if mode == "Select Existing Project":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

def confirm_project(project_mode, existing_project, new_project_name, uploaded_file):
    """
    Choose or create a project based on user selection.

    Returns:
      1. Updated header (Markdown text),
      2. Message for the output window,
      3. Updated project panel visibility,
      4. Updated dropdown choices,
      5. The current project state.
    """
    if project_mode == "Select Existing Project":
        selected_project = existing_project
        msg = f"Project '{selected_project}' selected!"
        updated_choices = get_projects()
    else:
        if not new_project_name:
            return gr.update(value=""), "New project name is required!", gr.update(visible=True), gr.update(choices=get_projects()), ""
        project_path = os.path.join(WORKDIR, new_project_name)
        if os.path.exists(project_path):
            return gr.update(value=""), f"Project '{new_project_name}' already exists!", gr.update(visible=True), gr.update(choices=get_projects()), ""
        os.makedirs(project_path, exist_ok=True)
        upload_subdir = config["PROJECT_SUBDIRS"]["upload"]
        upload_path = os.path.join(project_path, upload_subdir)
        os.makedirs(upload_path, exist_ok=True)
        logs_subdir = config["PROJECT_SUBDIRS"]["logs"]
        logs_path = os.path.join(project_path, logs_subdir)
        os.makedirs(logs_path, exist_ok=True)
        if uploaded_file:
            file_path = uploaded_file[0] if isinstance(uploaded_file, list) else uploaded_file
            target_path = os.path.join(upload_path, os.path.basename(file_path))
            shutil.move(file_path, target_path)
        selected_project = new_project_name
        msg = f"New project '{new_project_name}' created and selected!"
        updated_choices = get_projects()
    log_action(msg, selected_project)
    header_text = f"**Current Project:** {selected_project}"
    return header_text, msg, gr.update(visible=False), gr.update(choices=updated_choices, value=selected_project), selected_project

def run_separate_audio(device, keep_full_audio, non_speech_silence, chunk_size, model, current_project):
    """
    Futtatja a separate_audio.py scriptet az alábbi paraméterekkel:
      - INPUT_DIR: a projekt "upload" alkönyvtára.
      - OUTPUT_DIR: a projekt "separated_audio" alkönyvtára.
      - --device: "cuda" vagy "cpu".
      - --keep_full_audio és --non_speech_silence: zászlók.
      - --chunk_size: darabolás hossza percben.
      - --model: a használt modell.
    """
    if not current_project:
        return "No active project selected!"
    project_path = os.path.join(WORKDIR, current_project)
    input_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["upload"])
    output_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["separated_audio"])
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "separate_audio.py")
    command = ["python", script_path, "-i", input_dir, "-o", output_dir, "--device", device]
    if keep_full_audio:
        command.append("--keep_full_audio")
    if non_speech_silence:
        command.append("--non_speech_silence")
    command += ["--chunk_size", str(chunk_size)]
    command += ["--model", model]
    return run_subprocess_command(command, current_project, "separate_audio.py")


def run_transcribe_align(hf_token, language, current_project):
    """
    Run the whisx.py script. Temporarily moves the "non_speech.wav" file, then restores it.
    Also, if a Hugging Face token is provided, it is saved (encoded) for subsequent runs.
    """
    if not current_project:
        return "No active project selected!"
    # Update key file if hf_token is provided
    if hf_token:
        keys = load_keys()
        keys["hf_token"] = hf_token
        save_keys(keys["hf_token"], keys.get("deepL_api_key", ""))
    project_path = os.path.join(WORKDIR, current_project)
    separated_audio_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["separated_audio"])
    non_speech_file = None
    for f in os.listdir(separated_audio_dir):
        if f.endswith("non_speech.wav"):
            non_speech_file = f
            break
    moved = False
    if non_speech_file:
        original_path = os.path.join(separated_audio_dir, non_speech_file)
        temp_path = os.path.join(project_path, non_speech_file)
        try:
            shutil.move(original_path, temp_path)
            moved = True
            log_action(f"Temporarily moved {non_speech_file} to project root.", current_project)
        except Exception as e:
            return f"Error moving file: {e}"
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "whisx.py")
    command = ["python", script_path]
    if hf_token:
        command += ["--hf_token", hf_token]
    if language:
        command += ["--language", language]
    command.append(separated_audio_dir)
    output = run_subprocess_command(command, current_project, "whisx.py")
    if moved:
        try:
            shutil.move(temp_path, original_path)
            log_action(f"Restored {non_speech_file} to its original location.", current_project)
        except Exception as e:
            log_action(f"Error restoring {non_speech_file}: {e}", current_project)
    return output

def run_audio_split(current_project):
    """
    Run the splitter.py script which processes JSON and audio files from the project's "separated_audio" directory.
    """
    if not current_project:
        return "No active project selected!"
    project_path = os.path.join(WORKDIR, current_project)
    input_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["separated_audio"])
    output_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["splits"])
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "splitter.py")
    command = ["python", script_path, "--input_dir", input_dir, "--output_dir", output_dir]
    return run_subprocess_command(command, current_project, "splitter.py")

def run_translate(auth_key, input_language, output_language, current_project):
    """
    Run the translate.py script using DeepL.
      - INPUT_DIR: the project's "splits" subdirectory.
      - OUTPUT_DIR: the project's "translated_splits" subdirectory.
    Also, if a DeepL API key is provided, it is saved (encoded) for subsequent runs.
    """
    if not current_project:
        return "No active project selected!"
    # Update key file if auth_key is provided
    if auth_key:
        keys = load_keys()
        keys["deepL_api_key"] = auth_key
        save_keys(keys.get("hf_token", ""), keys["deepL_api_key"])
    project_path = os.path.join(WORKDIR, current_project)
    input_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["splits"])
    output_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["translated_splits"])
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "translate.py")
    command = [
        "python", script_path,
        "-input_dir", input_dir,
        "-output_dir", output_dir,
        "-input_language", input_language,
        "-output_language", output_language,
        "-auth_key", auth_key
    ]
    return run_subprocess_command(command, current_project, "translate.py")

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

def run_generate_tts_dubbing(remove_silence, tts_subdir, speed, nfe_step, norm_selection, seed, current_project):
    """
    Run the f5_tts_infer_API.py script from the f5-tts Anaconda environment.
      - INPUT_DIR: the project's "splits" subdirectory.
      - INPUT_GEN_DIR: the project's "translated_splits" subdirectory.
      - OUTPUT_DIR: the same as INPUT_GEN_DIR.
      - --vocab_file and --ckpt_file: from the selected TTS subdirectory (*.pt and *.txt).
      - Additional parameters: --speed, --nfe_step, --norm, --seed, and optionally --remove_silence.
    """
    if not current_project:
        return "No active project selected!"
    project_path = os.path.join(WORKDIR, current_project)
    input_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["splits"])
    input_gen_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["translated_splits"])
    output_dir = input_gen_dir  # Output goes to the same directory
    tts_base = config["DIRECTORIES"]["TTS"]
    tts_path = os.path.join(tts_base, tts_subdir)
    ckpt_file = None
    vocab_file = None
    for f in os.listdir(tts_path):
        if f.endswith(".pt"):
            ckpt_file = os.path.join(tts_path, f)
        if f.endswith(".txt"):
            vocab_file = os.path.join(tts_path, f)
    if not ckpt_file or not vocab_file:
        return "Error: Appropriate .pt or .txt file not found in the selected TTS directory."
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "f5_tts_infer_API.py")
    command = [
        "conda", "run", "-n", "f5-tts", "python", script_path,
        "-i", input_dir,
        "-ig", input_gen_dir,
        "-o", output_dir,
        "--vocab_file", vocab_file,
        "--ckpt_file", ckpt_file,
        "--speed", str(speed),
        "--nfe_step", str(nfe_step),
        "--norm", norm_selection,
        "--seed", str(seed)
    ]
    if remove_silence:
        command.append("--remove_silence")
    return run_subprocess_command(command, current_project, "f5_tts_infer_API.py")

def run_transcribe_align_chunks(current_project, splits_lang, translated_splits_lang):
    """
    Run the whisx_turbo.py script on two directories sequentially:
      1. The project's "splits" subdirectory.
      2. The project's "translated_splits" subdirectory.
    The language for each directory is passed via:
      - splits_lang: language for the splits directory.
      - translated_splits_lang: language for the translated_splits directory.
    Combines the results and returns them.
    """
    if not current_project:
        return "No active project selected!"
    if not splits_lang or not translated_splits_lang:
        return "Both language selections are required!"
    project_path = os.path.join(WORKDIR, current_project)
    splits_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["splits"])
    translated_splits_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["translated_splits"])
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "whisx_turbo.py")
    
    # Futtatás a splits könyvtárra
    command1 = [sys.executable, script_path, splits_dir, "--lang", splits_lang]
    output1 = run_subprocess_command(command1, current_project, f"whisx_turbo.py on {splits_dir}")
    
    # Futtatás a translated_splits könyvtárra
    command2 = [sys.executable, script_path, translated_splits_dir, "--lang", translated_splits_lang]
    output2 = run_subprocess_command(command2, current_project, f"whisx_turbo.py on {translated_splits_dir}")
    
    combined_output = f"Splits output:\n{output1}\n\nTranslated Splits output:\n{output2}"
    return combined_output


def run_normalise_and_cut(current_project, delete_empty, min_db):
    """
    Run the normalise_and_cut.py script.
      - -i INPUT_DIR: the project's "translated_splits" subdirectory.
      - -rj REFERENCE_JSON_DIR: the project's "splits" subdirectory.
      - --ira IRA: also the project's "splits" subdirectory.
      - -db MIN_DB: the specified value (default: -40.0 dB).
    After running the script, if the checkbox is active, deletes all .json files
    from both the "translated_splits" and "splits" directories.
    """
    if not current_project:
        return "No active project selected!"
    project_path = os.path.join(WORKDIR, current_project)
    input_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["translated_splits"])
    reference_json_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["splits"])
    ira = reference_json_dir  # Using splits directory as IRA parameter
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "normalise_and_cut.py")
    command = [
        "python", script_path,
        "-i", input_dir,
        "-rj", reference_json_dir,
        "--ira", ira,
        "-db", str(min_db)
    ]
    if delete_empty:
        command.append("--delete_empty")
    output = run_subprocess_command(command, current_project, "normalise_and_cut.py")
    
    # Delete all .json files from both translated_splits and splits directories
    deleted_files = []
    for dir_path in [input_dir, reference_json_dir]:
        for f in os.listdir(dir_path):
            if f.lower().endswith(".json"):
                file_path = os.path.join(dir_path, f)
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                except Exception as e:
                    log_action(f"Error deleting {file_path}: {e}", current_project)
    log_action(f"Deleted JSON files: {deleted_files}", current_project)
    output += f"\nDeleted {len(deleted_files)} JSON files."
    return output

def on_inspect_repair(current_project):
    """
    Launch the check_app.py in a new process (new window) passing the current project
    via the --project parameter.
    """
    if not current_project:
        return "No active project selected!"
    
    command = [sys.executable, "check_app.py", "--project", current_project]
    try:
        subprocess.Popen(command)
        return (
            f"check_app.py launched for project '{current_project}' in a new window.\n"
            "Check the terminal for port number!\n"
            "\n"
            "After done the chunk repair, close the window and return here.\n"
            "Run the Generate TTS Dubbing button again.\n"
            "Run the Normalize & Cut Chunks button again.\n"
            "Then continue with the Merge Chunks with Background button."
        )
    except Exception as e:
        return f"Failed to launch check_app.py: {e}"

def on_merge_chunks_bg(current_project):
    """
    Run the merge_chunks_with_background.py script with the following parameters:
      -i INPUT: the project's "translated_splits" directory,
      -o OUTPUT: the project's "film_dubbing" directory,
      -bg BACKGROUND: the file ending with "non_speech.wav" in the project's "separated_audio" directory.
    """
    if not current_project:
        return "No active project selected!"
    
    project_path = os.path.join(WORKDIR, current_project)
    input_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["translated_splits"])
    output_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["film_dubbing"])
    separated_audio_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["separated_audio"])
    
    # Find the non_speech.wav file in the separated_audio directory
    bg_file = None
    for f in os.listdir(separated_audio_dir):
        if f.endswith("non_speech.wav"):
            bg_file = os.path.join(separated_audio_dir, f)
            break
    if not bg_file:
        return "non_speech.wav not found in the separated_audio directory!"
    
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "merge_chunks_with_background.py")
    command = [
        "python", script_path,
        "-i", input_dir,
        "-o", output_dir,
        "-bg", bg_file
    ]
    return run_subprocess_command(command, current_project, "merge_chunks_with_background.py")

def on_merge_video(current_project, language):
    """
    Run the merge_to_video.py script with the following parameters:
      -i INPUT_VIDEO: the mkv file in the project's "upload" folder,
      -ia INPUT_AUDIO: the most recently saved file in the project's "film_dubbing" directory,
      -lang LANGUAGE: the 3-letter language code provided by the user,
      -o OUTPUT_DIR: the project's "download" directory.
    """
    if not current_project:
        return "No active project selected!"
    
    project_path = os.path.join(WORKDIR, current_project)
    # Find the mkv file in the upload directory
    upload_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["upload"])
    input_video = None
    for f in os.listdir(upload_dir):
        if f.lower().endswith(".mkv"):
            input_video = os.path.join(upload_dir, f)
            break
    if not input_video:
        return "No mkv file found in the upload directory!"
    
    # Get the most recent audio file from the film_dubbing directory
    film_dubbing_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["film_dubbing"])
    if not os.path.exists(film_dubbing_dir) or not os.listdir(film_dubbing_dir):
        return "No file in the film_dubbing directory!"
    audio_files = [os.path.join(film_dubbing_dir, f) for f in os.listdir(film_dubbing_dir) if os.path.isfile(os.path.join(film_dubbing_dir, f))]
    if not audio_files:
        return "No file in the film_dubbing directory!"
    input_audio = max(audio_files, key=os.path.getmtime)
    
    # Ensure the download directory exists
    output_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["download"])
    os.makedirs(output_dir, exist_ok=True)
    
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "merge_to_video.py")
    command = [
        "python", script_path,
        "-i", input_video,
        "-ia", input_audio,
        "-lang", language,
        "-o", output_dir
    ]
    return run_subprocess_command(command, current_project, "merge_to_video.py")

def update_download_dropdown(current_project):
    """
    Refresh the contents of the project's download directory and return file names for a Dropdown.
    """
    if not current_project:
        return "No active project selected!", gr.update(choices=[])
    project_path = os.path.join(WORKDIR, current_project)
    download_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["download"])
    if not os.path.exists(download_dir):
        return "Download folder does not exist!", gr.update(choices=[])
    files = [f for f in os.listdir(download_dir) if os.path.isfile(os.path.join(download_dir, f))]
    if not files:
        return "No file in the download folder!", gr.update(choices=[])
    return "Download folder updated.", gr.update(choices=files, value=files[0])

def download_file(current_project, selected_file):
    """
    Return the selected file's path for download.
    """
    if not current_project or not selected_file:
        return None
    project_path = os.path.join(WORKDIR, current_project)
    download_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["download"])
    file_path = os.path.join(download_dir, selected_file)
    return file_path if os.path.exists(file_path) else None

def on_download(current_project):
    """
    Simple feedback for the Download button.
    (In this example, the actual update is handled by the update_download_dropdown function.)
    """
    return f"Download button pressed! (Project: {current_project})"

def main(share, host):
    # Load saved keys to pre-populate the token fields (decoded)
    saved_keys = load_keys()
    hf_token_default = saved_keys.get("hf_token", "")
    deepL_api_default = saved_keys.get("deepL_api_key", "")
    
    initial_project = get_projects()[0] if get_projects() else ""
    # A custom CSS-sel állítjuk be, hogy a gomb klikkeléskor narancssárgára váltson
    css = """
    button:active {
        background-color: orange !important;
    }
    """
    with gr.Blocks(css=css) as demo:
        current_project_state = gr.State(initial_project)
        header = gr.Markdown(f"**Current Project:** {initial_project}")
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                # Project panel
                btn_project_panel = gr.Button("Create or Select Project")
                with gr.Column(visible=False, elem_id="project_panel") as project_panel:
                    project_mode = gr.Radio(
                        choices=["Select Existing Project", "Create New Project"],
                        label="Project Selection",
                        value="Select Existing Project"
                    )
                    with gr.Column(visible=True) as existing_project_container:
                        existing_project_dropdown = gr.Dropdown(
                            choices=get_projects(),
                            label="Choose Project",
                            value=get_projects()[0] if get_projects() else None
                        )
                    with gr.Column(visible=False) as new_project_container:
                        new_project_name = gr.Textbox(label="New Project Name", placeholder="Enter new project name")
                        mkv_upload = gr.File(label="Upload MKV File", file_types=[".mkv"], file_count="single")
                    btn_confirm_project = gr.Button("Confirm")
                
                # Function buttons
                btn_separate = gr.Button("Separate Speech")
                with gr.Column(visible=False, elem_id="separate_audio_panel") as separate_audio_panel:
                    device_dropdown = gr.Dropdown(choices=["cuda", "cpu"], label="Device", value="cuda")
                    keep_full_audio_checkbox = gr.Checkbox(label="Keep full audio", value=False, info="Use for nothing at the moment")
                    non_speech_silence_checkbox = gr.Checkbox(label="Background silence", value=False, info="Check if the video have speak only")
                    chunk_size_input = gr.Number(label="Chunk Size (minutes)", value=5, precision=0)
                    model_dropdown = gr.Dropdown(choices=["htdemucs", "htdemucs_ft", "hdemucs_mmi", "mdx", "mdx_extra"], label="Model", value="htdemucs")
                    btn_run_separate = gr.Button("Run")

                
                btn_transcribe_align = gr.Button("Transcribe & Align")
                with gr.Column(visible=False, elem_id="transcribe_align_panel") as transcribe_align_panel:
                    hf_token_input = gr.Textbox(label="Hugging Face Access Token (optional)", placeholder="Enter HF token", value=hf_token_default)
                    language_dropdown = gr.Dropdown(choices=["", "en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt", "ar", "cs", "ru", "pl", "hu", "fi", "fa", "el", "tr", "da", "he", "vi", "ko", "ur", "te", "hi", "ca", "ml", "no", "nn", "sk", "sl", "hr", "ro", "eu", "gl", "ka"], label="Language", value="")
                    btn_run_transcribe_align = gr.Button("Run")
                
                btn_audio_split = gr.Button("Audio Splitting")
                
                btn_translate = gr.Button("Translate")
                with gr.Column(visible=False, elem_id="translate_panel") as translate_panel:
                    input_language_dropdown = gr.Dropdown(
                        choices=["EN", "HU", "DE", "FR", "ES", "IT", "NL", "PL", "RU", "ZH", "JA"],
                        label="Input Language",
                        value="EN"
                    )
                    output_language_dropdown = gr.Dropdown(
                        choices=["EN-US", "EN-UK", "HU", "DE", "FR", "ES", "IT", "NL", "PL", "RU", "ZH"],
                        label="Output Language",
                        value="HU"
                    )
                    auth_key_input = gr.Textbox(label="DeepL API Authentication Key", placeholder="Enter DeepL API key", value=deepL_api_default)
                    btn_run_translate = gr.Button("Run")
                
                btn_generate_tts = gr.Button("Generate TTS Dubbing")
                with gr.Column(visible=False, elem_id="tts_panel") as tts_panel:
                    remove_silence_checkbox = gr.Checkbox(label="Remove silence", value=False)
                    tts_subdir_dropdown = gr.Dropdown(
                        choices=get_tts_subdirs(),
                        label="Select TTS Directory",
                        value=get_tts_subdirs()[0] if get_tts_subdirs() else ""
                    )
                    speed_slider = gr.Slider(minimum=0.3, maximum=2.0, step=0.1, label="Speed", value=1.0)
                    nfe_slider = gr.Slider(minimum=16, maximum=64, step=1, label="NFE Step", value=32)
                    norm_dropdown = gr.Dropdown(
                        choices=get_normaliser_subdirs(),
                        label="Normalizer",
                        value=get_normaliser_subdirs()[0] if get_normaliser_subdirs() else ""
                    )
                    seed_input = gr.Number(label="Seed", value=-1)
                    btn_run_tts = gr.Button("Run")
                
                # Transcribe & Align Chunks:
                # Alapértelmezetten a nyelvválasztó panel rejtve van.
                btn_transcribe_align_chunks = gr.Button("Transcribe & Align Chunks")
                with gr.Column(visible=False, elem_id="transcribe_align_chunks_panel") as transcribe_align_chunks_panel:
                    splits_lang_dropdown = gr.Dropdown(
                        choices=["", "en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt", "ar", "cs", "ru", "pl", "hu", "fi", "fa", "el", "tr", "da", "he", "vi", "ko", "ur", "te", "hi", "ca", "ml", "no", "nn", "sk", "sl", "hr", "ro", "eu", "gl", "ka"
],
                        label="Language for Splits",
                        value=""
                    )
                    translated_splits_lang_dropdown = gr.Dropdown(
                        choices=["", "en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt", "ar", "cs", "ru", "pl", "hu", "fi", "fa", "el", "tr", "da", "he", "vi", "ko", "ur", "te", "hi", "ca", "ml", "no", "nn", "sk", "sl", "hr", "ro", "eu", "gl", "ka"
],
                        label="Language for Translated Splits",
                        value=""
                    )
                    btn_run_transcribe_align_chunks = gr.Button("Run")
                
                btn_normalise_cut = gr.Button("Normalize & Cut Chunks")
                with gr.Column(visible=False, elem_id="normalise_cut_panel") as normalise_cut_panel:
                    delete_empty_checkbox = gr.Checkbox(label="Delete empty JSON files after processing", value=False)
                    min_db_input = gr.Number(label="Min dB", value=-40.0)
                    btn_run_normalise_cut = gr.Button("Run")
                
                btn_inspect_repair = gr.Button("Inspect & Repair Chunks")
                btn_merge_chunks_bg = gr.Button("Merge Chunks with Background")
                
                merge_video_lang_input = gr.Textbox(label="Language (3-letter code)", placeholder="e.g. ENG, HUN", value="HUN")
                btn_merge_video = gr.Button("Merge to Video")
                
                btn_download = gr.Button("Refresh Download Dir")
                download_dropdown = gr.Dropdown(label="Select file to download", choices=[])
                btn_download_file = gr.Button("Download")
                download_file_output = gr.File(label="Download File")
            with gr.Column(scale=3):
                output_text = gr.Textbox(label="Output", placeholder="Output will be displayed here", lines=10)
        
        # Event handlers
        btn_project_panel.click(lambda: gr.update(visible=True), None, project_panel)
        project_mode.change(
            toggle_project_fields,
            inputs=project_mode,
            outputs=[existing_project_container, new_project_container]
        )
        btn_confirm_project.click(
            confirm_project,
            inputs=[project_mode, existing_project_dropdown, new_project_name, mkv_upload],
            outputs=[header, output_text, project_panel, existing_project_dropdown, current_project_state]
        )
        btn_separate.click(lambda: gr.update(visible=True), None, separate_audio_panel)
        btn_run_separate.click(
            run_separate_audio,
            inputs=[device_dropdown, keep_full_audio_checkbox, non_speech_silence_checkbox, chunk_size_input, model_dropdown, current_project_state],
            outputs=output_text
        )
        btn_transcribe_align.click(lambda: gr.update(visible=True), None, transcribe_align_panel)
        btn_run_transcribe_align.click(
            run_transcribe_align,
            inputs=[hf_token_input, language_dropdown, current_project_state],
            outputs=output_text
        )
        btn_audio_split.click(
            run_audio_split,
            inputs=[current_project_state],
            outputs=output_text
        )
        btn_translate.click(lambda: gr.update(visible=True), None, translate_panel)
        btn_run_translate.click(
            run_translate,
            inputs=[auth_key_input, input_language_dropdown, output_language_dropdown, current_project_state],
            outputs=output_text
        )
        btn_generate_tts.click(lambda: gr.update(visible=True), None, tts_panel)
        btn_run_tts.click(
            run_generate_tts_dubbing,
            inputs=[remove_silence_checkbox, tts_subdir_dropdown, speed_slider, nfe_slider, norm_dropdown, seed_input, current_project_state],
            outputs=output_text
        )
        # A "Transcribe & Align Chunks" gomb megnyomása megjeleníti a nyelvválasztó panelt
        btn_transcribe_align_chunks.click(lambda: gr.update(visible=True), None, transcribe_align_chunks_panel)
        # A panelban lévő "Run" gomb elindítja a feldolgozást
        btn_run_transcribe_align_chunks.click(
            run_transcribe_align_chunks,
            inputs=[current_project_state, splits_lang_dropdown, translated_splits_lang_dropdown],
            outputs=output_text
        )
        btn_normalise_cut.click(lambda: gr.update(visible=True), None, normalise_cut_panel)
        btn_run_normalise_cut.click(
            run_normalise_and_cut,
            inputs=[current_project_state, delete_empty_checkbox, min_db_input],
            outputs=output_text
        )
        btn_inspect_repair.click(on_inspect_repair, inputs=[current_project_state], outputs=output_text)
        btn_merge_chunks_bg.click(on_merge_chunks_bg, inputs=[current_project_state], outputs=output_text)
        btn_merge_video.click(
            on_merge_video,
            inputs=[current_project_state, merge_video_lang_input],
            outputs=output_text
        )
        btn_download.click(
            update_download_dropdown,
            inputs=[current_project_state],
            outputs=[output_text, download_dropdown]
        )
        btn_download_file.click(
            download_file,
            inputs=[current_project_state, download_dropdown],
            outputs=download_file_output
        )
    
    demo.launch(share=share, server_name=host)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Gradio app with optional sharing and host parameters.")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app on a public link.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to serve the app (default: 0.0.0.0)")
    args = parser.parse_args()
    main(share=args.share, host=args.host)

import os
import json
import shutil
import subprocess
import datetime
import gradio as gr

# Config betöltése
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# A munkakönyvtár elérési útja a config alapján
WORKDIR = config["DIRECTORIES"]["workdir"]

def get_projects():
    """Visszaadja a WORKDIR mappában található projekt könyvtárak listáját."""
    if os.path.exists(WORKDIR):
        return [d for d in os.listdir(WORKDIR) if os.path.isdir(os.path.join(WORKDIR, d))]
    else:
        return []

def log_action(message, project):
    """Az adott lépés üzenetét a megfelelő log mappába írja."""
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

def toggle_project_fields(mode):
    """A rádió gomb alapján beállítja, melyik panel legyen látható."""
    if mode == "Létező projekt kiválasztása":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

def confirm_project(project_mode, existing_project, new_project_name, uploaded_file):
    """
    A felhasználó választása alapján kiválasztja vagy létrehozza a projektet.
    Visszatér:
      1. Frissített fejléc (Markdown szöveg),
      2. Üzenet a kimeneti ablakba,
      3. A projektpanel elrejtése,
      4. Frissített dropdown lista,
      5. Az aktuális projekt gr.State értéke.
    """
    if project_mode == "Létező projekt kiválasztása":
        selected_project = existing_project
        msg = f"Projekt '{selected_project}' kiválasztva!"
        updated_choices = get_projects()
    else:
        if not new_project_name:
            return gr.update(value=""), "Új projekt neve szükséges!", gr.update(visible=True), gr.update(choices=get_projects()), ""
        project_path = os.path.join(WORKDIR, new_project_name)
        if os.path.exists(project_path):
            return gr.update(value=""), f"A '{new_project_name}' projekt már létezik!", gr.update(visible=True), gr.update(choices=get_projects()), ""
        os.makedirs(project_path, exist_ok=True)
        upload_subdir = config["PROJECT_SUBDIRS"]["upload"]
        upload_path = os.path.join(project_path, upload_subdir)
        os.makedirs(upload_path, exist_ok=True)
        logs_subdir = config["PROJECT_SUBDIRS"]["logs"]
        logs_path = os.path.join(project_path, logs_subdir)
        os.makedirs(logs_path, exist_ok=True)
        if uploaded_file:
            if isinstance(uploaded_file, list):
                file_path = uploaded_file[0]
            else:
                file_path = uploaded_file
            target_path = os.path.join(upload_path, os.path.basename(file_path))
            shutil.move(file_path, target_path)
        selected_project = new_project_name
        msg = f"Új projekt '{new_project_name}' létrehozva és kiválasztva!"
        updated_choices = get_projects()
    log_action(msg, selected_project)
    header_text = f"**Aktuális projekt:** {selected_project}"
    return header_text, msg, gr.update(visible=False), gr.update(choices=updated_choices, value=selected_project), selected_project

def run_separate_audio(device, keep_full_audio, non_speech_silence, current_project):
    """
    Futtatja a separate_audio.py scriptet.
      - INPUT_DIR: a projekt "upload" almappája.
      - OUTPUT_DIR: a projekt "separated_audio" almappája.
      - --device: "cuda" vagy "cpu".
      - --keep_full_audio és --non_speech_silence: flag-ek.
    """
    if not current_project:
        return "Nincs aktív projekt kiválasztva!"
    project_path = os.path.join(WORKDIR, current_project)
    input_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["upload"])
    output_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["separated_audio"])
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "separate_audio.py")
    command = ["python", script_path, "-i", input_dir, "-o", output_dir, "--device", device]
    if keep_full_audio:
        command.append("--keep_full_audio")
    if non_speech_silence:
        command.append("--non_speech_silence")
    log_action(f"Running separate_audio.py with command: {' '.join(command)}", current_project)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error: {e.stderr}"
    log_action(f"separate_audio.py finished with output: {output}", current_project)
    return output

def run_transcribe_align(hf_token, language, current_project):
    """
    Futtatja a whisx.py scriptet. Átmenetileg áthelyezi a "non_speech.wav" fájlt, majd visszahelyezi.
    """
    if not current_project:
        return "Nincs aktív projekt kiválasztva!"
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
            log_action(f"Áthelyezve ideiglenesen a {non_speech_file} fájl a projekt gyökérbe.", current_project)
        except Exception as e:
            return f"Hiba a fájl áthelyezésekor: {e}"
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "whisx.py")
    command = ["python", script_path]
    if hf_token:
        command += ["--hf_token", hf_token]
    if language:
        command += ["--language", language]
    command.append(separated_audio_dir)
    log_action(f"Running whisx.py with command: {' '.join(command)}", current_project)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error: {e.stderr}"
    log_action(f"whisx.py finished with output: {output}", current_project)
    if moved:
        try:
            shutil.move(temp_path, original_path)
            log_action(f"A {non_speech_file} fájl visszahelyezve az eredeti mappába.", current_project)
        except Exception as e:
            log_action(f"Hiba a {non_speech_file} visszahelyezésekor: {e}", current_project)
    return output

def run_audio_split(current_project):
    """
    Futtatja a splitter.py scriptet, amely a projekt "separated_audio" könyvtárából dolgozza fel a JSON és audio fájlokat.
    """
    if not current_project:
        return "Nincs aktív projekt kiválasztva!"
    project_path = os.path.join(WORKDIR, current_project)
    input_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["separated_audio"])
    output_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["splits"])
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "splitter.py")
    command = ["python", script_path, "--input_dir", input_dir, "--output_dir", output_dir]
    log_action(f"Running splitter.py with command: {' '.join(command)}", current_project)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error: {e.stderr}"
    log_action(f"splitter.py finished with output: {output}", current_project)
    return output

def run_translate(auth_key, input_language, output_language, current_project):
    """
    Futtatja a translate.py scriptet DeepL segítségével.
      - INPUT_DIR: a projekt "splits" almappája.
      - OUTPUT_DIR: a projekt "translated_splits" almappája.
    """
    if not current_project:
        return "Nincs aktív projekt kiválasztva!"
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
    log_action(f"Running translate.py with command: {' '.join(command)}", current_project)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error: {e.stderr}"
    log_action(f"translate.py finished with output: {output}", current_project)
    return output

def get_tts_subdirs():
    """Visszaadja a TTS könyvtár alkönyvtárainak listáját."""
    tts_dir = config["DIRECTORIES"]["TTS"]
    if os.path.exists(tts_dir):
        return [d for d in os.listdir(tts_dir) if os.path.isdir(os.path.join(tts_dir, d))]
    else:
        return []

def get_normaliser_subdirs():
    """Visszaadja a normalisers könyvtár alkönyvtárainak listáját."""
    norm_dir = config["DIRECTORIES"]["normalisers"]
    if os.path.exists(norm_dir):
        return [d for d in os.listdir(norm_dir) if os.path.isdir(os.path.join(norm_dir, d))]
    else:
        return []

def run_generate_tts_dubbing(remove_silence, tts_subdir, speed, nfe_step, norm_selection, seed, current_project):
    """
    Futtatja az f5_tts_infer_API.py scriptet a f5-tts Anaconda környezetből.
      - INPUT_DIR: a projekt "splits" almappája.
      - INPUT_GEN_DIR: a projekt "translated_splits" almappája.
      - OUTPUT_DIR: a projekt "translated_splits" almappája.
      - --vocab_file és --ckpt_file: a kiválasztott TTS alkönyvtárból (*.pt és *.txt).
      - További paraméterek: --speed, --nfe_step, --norm, --seed, és opcionálisan --remove_silence.
    """
    if not current_project:
        return "Nincs aktív projekt kiválasztva!"
    project_path = os.path.join(WORKDIR, current_project)
    input_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["splits"])
    input_gen_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["translated_splits"])
    output_dir = input_gen_dir  # a kimenet ugyanoda kerül
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
        return "Hiba: nem található megfelelő .pt vagy .txt fájl a kiválasztott TTS könyvtárban."
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
    log_action(f"Running f5_tts_infer_API.py with command: {' '.join(command)}", current_project)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error: {e.stderr}"
    log_action(f"f5_tts_infer_API.py finished with output: {output}", current_project)
    return output

def run_transcribe_align_chunks(current_project):
    """
    Futtatja a whisx_turbo.py scriptet két könyvtáron egymást követően:
      1. Először a projekt "splits" almappáján.
      2. Majd a projekt "translated_splits" almappáján.
    Az eredményeket egyesíti, és visszaadja a kimeneti ablakban.
    """
    if not current_project:
        return "Nincs aktív projekt kiválasztva!"
    project_path = os.path.join(WORKDIR, current_project)
    splits_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["splits"])
    translated_splits_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["translated_splits"])
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "whisx_turbo.py")
    
    # Első futtatás a splits könyvtáron
    command1 = ["python", script_path, splits_dir]
    log_action(f"Running whisx_turbo.py on {splits_dir} with command: {' '.join(command1)}", current_project)
    try:
        result1 = subprocess.run(command1, capture_output=True, text=True, check=True)
        output1 = result1.stdout
    except subprocess.CalledProcessError as e:
        output1 = f"Error on splits: {e.stderr}"
    log_action(f"whisx_turbo.py finished on splits with output: {output1}", current_project)
    
    # Második futtatás a translated_splits könyvtáron
    command2 = ["python", script_path, translated_splits_dir]
    log_action(f"Running whisx_turbo.py on {translated_splits_dir} with command: {' '.join(command2)}", current_project)
    try:
        result2 = subprocess.run(command2, capture_output=True, text=True, check=True)
        output2 = result2.stdout
    except subprocess.CalledProcessError as e:
        output2 = f"Error on translated_splits: {e.stderr}"
    log_action(f"whisx_turbo.py finished on translated_splits with output: {output2}", current_project)
    
    combined_output = f"Splits output:\n{output1}\n\nTranslated Splits output:\n{output2}"
    return combined_output

def run_normalise_and_cut(current_project, delete_empty, min_db):
    """
    Futtatja a normalise_and_cut.py scriptet.
      - -i INPUT_DIR: a projekt "translated_splits" almappája.
      - -rj REFERENCE_JSON_DIR: a projekt "splits" almappája.
      - --ira IRA: szintén a projekt "splits" almappája.
      - -db MIN_DB: a megadott érték (alapértelmezett: -40.0 dB).
    Script futtatása után, ha a checkbox aktiv, törli az összes .json fájlt
    mind a "translated_splits", mind a "splits" könyvtárakból.
    """
    if not current_project:
        return "Nincs aktív projekt kiválasztva!"
    project_path = os.path.join(WORKDIR, current_project)
    input_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["translated_splits"])
    reference_json_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["splits"])
    # IRA paraméterként is a splits könyvtárat használjuk
    ira = reference_json_dir
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
    log_action(f"Running normalise_and_cut.py with command: {' '.join(command)}", current_project)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error: {e.stderr}"
    log_action(f"normalise_and_cut.py finished with output: {output}", current_project)
    
    # Töröljük az összes .json fájlt a translated_splits és splits könyvtárakból
    deleted_files = []
    for dir_path in [input_dir, reference_json_dir]:
        for f in os.listdir(dir_path):
            if f.lower().endswith(".json"):
                file_path = os.path.join(dir_path, f)
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                except Exception as e:
                    log_action(f"Hiba a {file_path} törlésekor: {e}", current_project)
    log_action(f"Törölt JSON fájlok: {deleted_files}", current_project)
    output += f"\nTörölt JSON fájlok: {len(deleted_files)} db."
    return output

def on_inspect_repair(current_project):
    return f"Inspect & Repair Chunks gomb megnyomva! (Projekt: {current_project})"

def on_merge_chunks_bg(current_project):
    """
    Futtatja a merge_chunks_with_background.py scriptet a következő paraméterekkel:
      -i INPUT: a projekt "translated_splits" könyvtára,
      -o OUTPUT: a projekt "film_dubbing" könyvtára,
      -bg BACKGROUND: a projekt "separated_audio" könyvtárában található, "non_speech.wav" végződésű fájl.
    """
    if not current_project:
        return "Nincs aktív projekt kiválasztva!"
    
    # Projekt útvonalak beállítása
    project_path = os.path.join(WORKDIR, current_project)
    input_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["translated_splits"])
    output_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["film_dubbing"])
    separated_audio_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["separated_audio"])
    
    # Keressük a non_speech.wav fájlt a separated_audio könyvtárban
    bg_file = None
    for f in os.listdir(separated_audio_dir):
        if f.endswith("non_speech.wav"):
            bg_file = os.path.join(separated_audio_dir, f)
            break
    if not bg_file:
        return "Nem található non_speech.wav a separated_audio könyvtárban!"
    
    # Script útvonalának meghatározása a config.json alapján
    script_path = os.path.join(config["DIRECTORIES"]["scripts"], "merge_chunks_with_background.py")
    
    # Parancs összeállítása
    command = [
        "python", script_path,
        "-i", input_dir,
        "-o", output_dir,
        "-bg", bg_file
    ]
    log_action(f"Running merge_chunks_with_background.py with command: {' '.join(command)}", current_project)
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error: {e.stderr}"
    
    log_action(f"merge_chunks_with_background.py finished with output: {output}", current_project)
    return output

def on_merge_video(current_project, language):
    """
    Futtatja a merge_to_video.py scriptet a következő paraméterekkel:
      -i INPUT_VIDEO: a projekt "upload" mappában lévő mkv file,
      -ia INPUT_AUDIO: a projekt "film_dubbing" könyvtárában található, a legutoljára lementett fájl,
      -lang LANGUAGE: a felhasználó által megadott 3 betűs nyelvkód,
      -o OUTPUT_DIR: a projekt "download" könyvtára.
    """
    if not current_project:
        return "Nincs aktív projekt kiválasztva!"
    
    project_path = os.path.join(WORKDIR, current_project)
    # INPUT_VIDEO: az upload mappában keresünk egy mkv kiterjesztésű fájlt
    upload_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["upload"])
    input_video = None
    for f in os.listdir(upload_dir):
        if f.lower().endswith(".mkv"):
            input_video = os.path.join(upload_dir, f)
            break
    if not input_video:
        return "Nem található mkv fájl az upload könyvtárban!"
    
    # INPUT_AUDIO: a film_dubbing könyvtárból a legutoljára módosított fájl
    film_dubbing_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["film_dubbing"])
    if not os.path.exists(film_dubbing_dir) or not os.listdir(film_dubbing_dir):
        return "Nincs fájl a film_dubbing könyvtárban!"
    audio_files = [os.path.join(film_dubbing_dir, f) for f in os.listdir(film_dubbing_dir) if os.path.isfile(os.path.join(film_dubbing_dir, f))]
    if not audio_files:
        return "Nincs fájl a film_dubbing könyvtárban!"
    input_audio = max(audio_files, key=os.path.getmtime)
    
    # OUTPUT_DIR: a download könyvtár
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
    log_action(f"Running merge_to_video.py with command: {' '.join(command)}", current_project)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error: {e.stderr}"
    log_action(f"merge_to_video.py finished with output: {output}", current_project)
    return output

def update_download_dropdown(current_project):
    """
    Frissíti a projekt download könyvtárának tartalmát, és visszaadja a fájlneveket egy Dropdown számára.
    """
    if not current_project:
        return "Nincs aktív projekt kiválasztva!", gr.update(choices=[])
    project_path = os.path.join(WORKDIR, current_project)
    download_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["download"])
    if not os.path.exists(download_dir):
        return "A download mappa nem létezik!", gr.update(choices=[])
    files = [f for f in os.listdir(download_dir) if os.path.isfile(os.path.join(download_dir, f))]
    if not files:
        return "Nincs fájl a download mappában!", gr.update(choices=[])
    return "A download mappa tartalma frissítve.", gr.update(choices=files, value=files[0])

def download_file(current_project, selected_file):
    """
    A kiválasztott fájlt visszaadja letöltendőként.
    """
    if not current_project or not selected_file:
        return None
    project_path = os.path.join(WORKDIR, current_project)
    download_dir = os.path.join(project_path, config["PROJECT_SUBDIRS"]["download"])
    file_path = os.path.join(download_dir, selected_file)
    if os.path.exists(file_path):
        return file_path
    else:
        return None

def on_download(current_project):
    """
    Egyszerű visszajelzés a Download gombhoz.
    Ebben a példában a tényleges frissítést az update_download_dropdown függvény végzi.
    """
    return f"Download gomb megnyomva! (Projekt: {current_project})"

def main():
    initial_project = get_projects()[0] if get_projects() else ""
    with gr.Blocks() as demo:
        current_project_state = gr.State(initial_project)
        header = gr.Markdown(f"**Aktuális projekt:** {initial_project}")
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                # Projekt panel
                btn_project_panel = gr.Button("Create or Select Projekt")
                with gr.Column(visible=False, elem_id="project_panel") as project_panel:
                    project_mode = gr.Radio(
                        choices=["Létező projekt kiválasztása", "Új projekt létrehozása"],
                        label="Projekt kiválasztása",
                        value="Létező projekt kiválasztása"
                    )
                    with gr.Column(visible=True) as existing_project_container:
                        existing_project_dropdown = gr.Dropdown(
                            choices=get_projects(),
                            label="Válassz projektet",
                            value=get_projects()[0] if get_projects() else None
                        )
                    with gr.Column(visible=False) as new_project_container:
                        new_project_name = gr.Textbox(label="Új projekt neve", placeholder="Add meg az új projekt nevét")
                        mkv_upload = gr.File(label="MKV fájl feltöltése", file_types=[".mkv"], file_count="single")
                    btn_confirm_project = gr.Button("Megerősít")
                
                # Funkciógombok
                btn_separate = gr.Button("Separate Speach")
                with gr.Column(visible=False, elem_id="separate_audio_panel") as separate_audio_panel:
                    device_dropdown = gr.Dropdown(choices=["cuda", "cpu"], label="Eszköz", value="cuda")
                    keep_full_audio_checkbox = gr.Checkbox(label="Teljes audio megtartása", value=False)
                    non_speech_silence_checkbox = gr.Checkbox(label="Non-speech silence", value=False)
                    btn_run_separate = gr.Button("Futtatás")
                
                btn_transcribe_align = gr.Button("Transcribe & Align")
                with gr.Column(visible=False, elem_id="transcribe_align_panel") as transcribe_align_panel:
                    hf_token_input = gr.Textbox(label="Hugging Face Access Token (opcionális)", placeholder="Add meg a HF token-t", value="")
                    language_dropdown = gr.Dropdown(choices=["", "en", "es", "de", "fr", "hu"], label="Nyelv", value="")
                    btn_run_transcribe_align = gr.Button("Futtatás")
                
                btn_audio_split = gr.Button("Audio Spliting")
                
                btn_translate = gr.Button("Translate")
                with gr.Column(visible=False, elem_id="translate_panel") as translate_panel:
                    input_language_dropdown = gr.Dropdown(
                        choices=["EN", "HU", "DE", "FR", "ES", "IT", "NL", "PL", "RU", "ZH", "JA"],
                        label="Bemeneti nyelv",
                        value="EN"
                    )
                    output_language_dropdown = gr.Dropdown(
                        choices=["EN-US", "EN-UK", "HU", "DE", "FR", "ES", "IT", "NL", "PL", "RU", "ZH"],
                        label="Kimeneti nyelv",
                        value="HU"
                    )
                    auth_key_input = gr.Textbox(label="DeepL API hitelesítési kulcs", placeholder="Add meg a DeepL API kulcsot", value="")
                    btn_run_translate = gr.Button("Futtatás")
                
                btn_generate_tts = gr.Button("Generate TTS dubbing")
                with gr.Column(visible=False, elem_id="tts_panel") as tts_panel:
                    remove_silence_checkbox = gr.Checkbox(label="Remove silence", value=False)
                    tts_subdir_dropdown = gr.Dropdown(
                        choices=get_tts_subdirs(),
                        label="Válassz TTS könyvtárat",
                        value=get_tts_subdirs()[0] if get_tts_subdirs() else ""
                    )
                    speed_slider = gr.Slider(minimum=0.3, maximum=2.0, step=0.1, label="Speed", value=1.0)
                    nfe_slider = gr.Slider(minimum=16, maximum=64, step=1, label="NFE Step", value=32)
                    norm_dropdown = gr.Dropdown(
                        choices=get_normaliser_subdirs(),
                        label="Normaliser",
                        value=get_normaliser_subdirs()[0] if get_normaliser_subdirs() else ""
                    )
                    seed_input = gr.Number(label="Seed", value=-1)
                    btn_run_tts = gr.Button("Futtatás")
                
                # Új: Transcribe & Align Chunks
                btn_transcribe_align_chunks = gr.Button("Transcribe & Align Chunks")
                
                # Új: Normalise & Cut Chunks panel
                btn_normalise_cut = gr.Button("Normalise & Cut Chunks")
                with gr.Column(visible=False, elem_id="normalise_cut_panel") as normalise_cut_panel:
                    delete_empty_checkbox = gr.Checkbox(label="Delete empty JSON files after processing", value=False)
                    min_db_input = gr.Number(label="Min dB", value=-40.0)
                    btn_run_normalise_cut = gr.Button("Futtatás")
                
                btn_inspect_repair = gr.Button("Inspect & Repair Chunks")
                btn_merge_chunks_bg = gr.Button("Merge Chunks with Background")
                
                # Új: Merge to Video panelhoz egy szövegbeviteli mező a language paraméterhez
                merge_video_lang_input = gr.Textbox(label="Language (3 betű)", placeholder="Pl: ENG", value="ENG")
                btn_merge_video = gr.Button("Merge to Video")
                
                # Új: Download panel
                btn_download = gr.Button("Download")
                download_dropdown = gr.Dropdown(label="Válassz letöltendő fájlt", choices=[])
                btn_download_file = gr.Button("Letöltés")
                download_file_output = gr.File(label="Letöltendő fájl")
            with gr.Column(scale=3):
                output_text = gr.Textbox(label="Kimenet", placeholder="Itt jelenik meg a kimenet", lines=10)
        
        # Eseménykezelők
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
            inputs=[device_dropdown, keep_full_audio_checkbox, non_speech_silence_checkbox, current_project_state],
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
        btn_transcribe_align_chunks.click(
            run_transcribe_align_chunks,
            inputs=[current_project_state],
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
        # A Download gombbal frissítjük a letöltendő fájlok listáját
        btn_download.click(
            update_download_dropdown,
            inputs=[current_project_state],
            outputs=[output_text, download_dropdown]
        )
        # A "Letöltés" gomb megnyomására visszaadjuk a kiválasztott fájl elérési útját, amely megjelenik a download_file_output komponensben
        btn_download_file.click(
            download_file,
            inputs=[current_project_state, download_dropdown],
            outputs=download_file_output
        )
    
    demo.launch()

if __name__ == "__main__":
    main()

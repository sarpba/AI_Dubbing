# tabs/transcription.py
import os
import shutil
import torch
from .utils import run_script, ensure_directory, list_projects

def transcribe_audio_whisperx(proj_name, hf_token, selected_device, selected_device_index, workdir="workdir"):
    """
    Transzkripciót készít WhisperX segítségével külső script hívásával.

    Args:
        proj_name (str): A kiválasztott projekt neve.
        hf_token (str): Hugging Face token.
        selected_device (str): "cpu" vagy "cuda".
        selected_device_index (str): GPU index, ha "cuda" van választva.
        workdir (str): A munkakönyvtár alapértelmezett útvonala.

    Yields:
        str: A script kimenete folyamatosan frissülő eredmény ablakban.
    """
    try:
        project_path = os.path.join(workdir, proj_name)
        audio_path = os.path.join(project_path, "audio")
        transcripts_path = os.path.join(project_path, "transcripts")
        ensure_directory(transcripts_path)

        # Ellenőrizzük, hogy van-e audio fájl
        audio_files = [f for f in os.listdir(audio_path) if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.opus'))]
        if not audio_files:
            yield "Nincs audio fájl a projektben."
            return

        audio_file = os.path.join(audio_path, audio_files[0])
        audio_directory = os.path.dirname(audio_file)

        # Eszköz és index beállítása
        gpus = []
        warning = ""
        if selected_device == "cuda":
            device_index = int(selected_device_index)
            if device_index < torch.cuda.device_count():
                gpus = [str(device_index)]
            else:
                warning = f"A kiválasztott GPU (cuda:{selected_device_index}) nem elérhető. CPU-t használunk helyette."

        # Külső script hívása
        whisperx_script = os.path.join("scripts", "whisx.py")  # Ha más helyen van, add meg a teljes elérési utat

        # Parancs összeállítása
        cmd = ["python", "-u", whisperx_script]
        if gpus:
            cmd += ["--gpus", ",".join(gpus)]
        cmd += ["--hf_token", hf_token, audio_directory]

        # Script futtatása és kimenet olvasása
        for output in run_script(cmd):
            yield output

        # JSON fájl áthelyezése a transcripts mappába
        # Feltételezzük, hogy a JSON fájl az audio_directory mappában jön létre
        json_files = [f for f in os.listdir(audio_directory) if f.lower().endswith('.json')]
        if not json_files:
            yield "A transzkripció sikeres volt, de nem található JSON fájl."
            return

        for json_file in json_files:
            src_json_path = os.path.join(audio_directory, json_file)
            dest_json_path = os.path.join(transcripts_path, json_file)
            shutil.move(src_json_path, dest_json_path)

        if warning:
            yield warning + f"\nTranszkripció kész: {dest_json_path}"
        else:
            yield f"Transzkripció kész: {dest_json_path}"

    except Exception as e:
        yield f"Hiba történt a transzkripció során: {str(e)}"


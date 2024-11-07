# tabs/transcription.py
import os
import shutil
import torch
from .utils import run_script, ensure_directory, list_projects

def transcribe_audio_whisperx(proj_name, hf_token, selected_device, selected_device_index, audio_source='speech_removed', language=None):

    workdir = "workdir"
    """
    Transzkripciót készít WhisperX segítségével külső script hívásával.

    Args:
        proj_name (str): A kiválasztott projekt neve.
        hf_token (str): Hugging Face token.
        selected_device (str): "cpu" vagy "cuda".
        selected_device_index (str): GPU index, ha "cuda" van választva.
        audio_source (str): "audio" vagy "speech_removed" mappa.
        language(str, optional): Opcionális nyelvi kód (pl. 'en', 'es'). Alapértelmezett: None.

    Yields:
        str: A script kimenete folyamatosan frissülő eredmény ablakban.
    """
    try:
        project_path = os.path.join(workdir, proj_name)
        transcripts_path = os.path.join(project_path, "transcripts")
        ensure_directory(transcripts_path)

        if audio_source == 'speech_removed':
            # Ideiglenes mappa létrehozása, előtte töröljük, ha létezik
            audio_temp_path = os.path.join(project_path, "audio_temp")
            if os.path.exists(audio_temp_path):
                shutil.rmtree(audio_temp_path)
            ensure_directory(audio_temp_path)

            # Eredeti audio útvonal
            speech_removed_path = os.path.join(project_path, "speech_removed")

            # Csak a '_speech.wav' fájlokat dolgozzuk fel, de kizárjuk a '_non_speech.wav' végűeket
            speech_files = [
                f for f in os.listdir(speech_removed_path)
                if f.lower().endswith('_speech.wav') and not f.lower().endswith('_non_speech.wav')
            ]
            if not speech_files:
                yield "Nem található megfelelő '_speech.wav' fájl a 'speech_removed' mappában."
                return

            # Fájl másolása és átnevezése az 'audio_temp' mappába
            for original_file in speech_files:
                src_file_path = os.path.join(speech_removed_path, original_file)
                new_filename = original_file.replace('_speech', '')  # '_speech' eltávolítása
                dest_file_path = os.path.join(audio_temp_path, new_filename)
                shutil.copy(src_file_path, dest_file_path)

            # Az audio_path az 'audio_temp' mappa lesz
            audio_path = audio_temp_path

        else:
            # Az 'audio' mappa használata
            audio_path = os.path.join(project_path, 'audio')
            if not os.path.exists(audio_path):
                yield "Az 'audio' mappa nem található."
                return

        # Feldolgozható audio fájlok keresése az audio_path mappában
        audio_files = [f for f in os.listdir(audio_path) if f.lower().endswith('.wav')]
        if not audio_files:
            yield "Nincs feldolgozható audio fájl a kiválasztott mappában."
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
        cmd += ["--hf_token", hf_token]
        cmd += [audio_directory]  # Az audio fájl könyvtárát adjuk át

        # Opcionális nyelvi kód hozzáadása
        if language:
            cmd += ["--language", language]

        # Script futtatása és kimenet olvasása
        for output in run_script(cmd):
            yield output

        # JSON fájlok áthelyezése a transcripts mappába
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

        # Ideiglenes mappa törlése
        if audio_source == 'speech_removed':
            shutil.rmtree(audio_temp_path)

    except Exception as e:
        yield f"Hiba történt a transzkripció során: {str(e)}"

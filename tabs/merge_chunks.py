# tabs/merge_chunks.py
import os
from .utils import run_script, ensure_directory

def merge_chunks(proj_name, workdir="workdir"):
    """
    Chunks egyesítése a merge_audio.py script segítségével.

    Args:
        proj_name (str): A kiválasztott projekt neve.
        workdir (str): A munkakönyvtár alapértelmezett útvonala.

    Yields:
        str: A script kimenete folyamatosan frissülő eredmény ablakban.
    """
    try:
        project_path = os.path.join(workdir, proj_name)
        input_dir = os.path.join(project_path, "sync")
        output_dir = os.path.join(project_path, "sync_merged")
        ensure_directory(output_dir)
        output_file = os.path.join(output_dir, "sync.wav")

        # Keresés a háttérzene fájlra
        speech_removed_dir = os.path.join(project_path, "speech_removed")
        background_files = [f for f in os.listdir(speech_removed_dir) if f.endswith("_temp_non_speech.wav")]

        if not background_files:
            yield "Nem található háttérzene fájl a speech_removed könyvtárban."
            return

        # Feltételezzük, hogy csak egy ilyen fájl van
        background_file = os.path.join(speech_removed_dir, background_files[0])

        # Ellenőrizzük, hogy vannak-e WAV fájlok az input könyvtárban
        wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
        if not wav_files:
            yield "Nincs egyesítendő WAV fájl a sync könyvtárban."
            return

        # Külső merge_audio.py script hívása
        merge_script = os.path.join("scripts", "merge_audio.py")  # Ha más helyen van, add meg a teljes elérési utat

        cmd = [
            "python", "-u", merge_script,
            "-i", input_dir,
            "-o", output_file,
            "-bg", background_file
        ]

        # Script futtatása és kimenet olvasása
        for output in run_script(cmd):
            yield output

        yield f"\nChunks egyesítése kész: {output_file}"

    except Exception as e:
        yield f"Hiba történt a chunks egyesítése során: {str(e)}"


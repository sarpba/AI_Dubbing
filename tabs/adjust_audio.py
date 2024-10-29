# tabs/adjust_audio.py
import os
from .utils import run_script, ensure_directory

def adjust_audio(proj_name, use_delete_empty, use_db_input, db_value, workdir="workdir"):
    """
    Audio illesztés és hangerő normalizálás az audio_cuter_normaliser.py script segítségével.

    Args:
        proj_name (str): A kiválasztott projekt neve.
        use_delete_empty (bool): Ha True, akkor használja a --delete_empty kapcsolót.
        use_db_input (bool): Ha True, akkor használja a -db paramétert.
        db_value (str): A megadott dB érték.
        workdir (str): A munkakönyvtár alapértelmezett útvonala.

    Yields:
        str: A script kimenete folyamatosan frissülő eredmény ablakban.
    """
    try:
        project_path = os.path.join(workdir, proj_name)
        input_dir = os.path.join(project_path, "sync")
        reference_json_dir = os.path.join(project_path, "transcripts_split")
        ira_dir = os.path.join(project_path, "split_audio")

        # Ellenőrizzük, hogy a szükséges könyvtárak léteznek-e
        if not os.path.exists(input_dir):
            yield f"A {input_dir} könyvtár nem található."
            return

        if not os.path.exists(reference_json_dir):
            yield f"A {reference_json_dir} könyvtár nem található."
            return

        if not os.path.exists(ira_dir):
            yield f"A {ira_dir} könyvtár nem található."
            return

        # Külső audio_cuter_normaliser.py script hívása
        script_path = os.path.join("scripts", "audio_cuter_normaliser.py")  # Módosítsd, ha szükséges

        cmd = [
            "python", "-u", script_path,
            "-i", input_dir,
            "-rj", reference_json_dir,
            "--ira", ira_dir
        ]

        if use_delete_empty:
            cmd.append("--delete_empty")

        if use_db_input and db_value.strip():
            cmd.extend(["-db", db_value.strip()])

        # Script futtatása és kimenet olvasása
        for output in run_script(cmd):
            yield output

        yield "\nAudio illesztés és hangerő normalizálás befejeződött."

    except Exception as e:
        yield f"Hiba történt az audio illesztése során: {str(e)}"


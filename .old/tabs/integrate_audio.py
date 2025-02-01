# tabs/integrate_audio.py
import os
from .utils import run_script, ensure_directory

def integrate_audio(proj_name, language_code, workdir="workdir"):
    """
    Audio fájl integrálása a videóba a merge_video.py script segítségével.

    Args:
        proj_name (str): A kiválasztott projekt neve.
        language_code (str): A nyelvi címke (pl. eng, hun).
        workdir (str): A munkakönyvtár alapértelmezett útvonala.

    Yields:
        str: A script kimenete folyamatosan frissülő eredmény ablakban.
    """
    try:
        project_path = os.path.join(workdir, proj_name)
        input_video_dir = os.path.join(project_path, "uploads")
        input_audio_dir = os.path.join(project_path, "sync_merged")
        output_dir = os.path.join(project_path, "output")
        ensure_directory(output_dir)

        # Bemeneti videó fájl keresése
        video_files = [f for f in os.listdir(input_video_dir) if f.lower().endswith(('.mp4', '.mkv', '.avi'))]
        if not video_files:
            yield "Nem található videó fájl a uploads könyvtárban."
            return

        # Feltételezzük, hogy csak egy videó fájl van
        input_video_file = os.path.join(input_video_dir, video_files[0])

        # Bemeneti audio fájl keresése
        audio_files = [f for f in os.listdir(input_audio_dir) if f.lower().endswith('.wav')]
        if not audio_files:
            yield "Nem található audio fájl a sync_merged könyvtárban."
            return

        # Feltételezzük, hogy csak egy audio fájl van
        input_audio_file = os.path.join(input_audio_dir, audio_files[0])

        # Külső merge_video.py script hívása
        merge_video_script = os.path.join("scripts", "merge_video.py")  # Ha más helyen van, add meg a teljes elérési utat

        cmd = [
            "python", "-u", merge_video_script,
            "-i", input_video_file,
            "-ia", input_audio_file,
            "-lang", language_code,
            "-o", output_dir
        ]

        # Script futtatása és kimenet olvasása
        for output in run_script(cmd):
            yield output

        yield f"\nAudio integrálása kész: {output_dir}"

    except Exception as e:
        yield f"Hiba történt az audio integrálása során: {str(e)}"


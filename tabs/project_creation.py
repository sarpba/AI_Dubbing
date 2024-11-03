# tabs/project_creation.py
import os
import shutil
from moviepy.editor import VideoFileClip
from .utils import run_script, ensure_directory

def upload_and_extract_audio(proj_name, video_path, workdir="workdir"):
    """
    Feltölti a videófájlt, létrehozza a projekt struktúráját, és kivonja az audiót.

    Args:
        proj_name (str): A projekt neve.
        video_path (str): Az feltöltött videófájl elérési útja.
        workdir (str): A munkakönyvtár alapértelmezett útvonala.

    Returns:
        str: Az eredmény üzenete.
    """
    try:
        # Ellenőrizzük, hogy egy fájl lett-e feltöltve
        if video_path is None:
            return "Nincs feltöltött fájl."

        # Ellenőrizzük a fájl méretét (max 100MB)
        file_size = os.path.getsize(video_path)
        max_size = 2000 * 1024 * 1024  # 1000MB
        if file_size > max_size:
            return "A feltöltött fájl mérete meghaladja a 100MB-ot."

        # Projekt mappa létrehozása
        project_path = os.path.join(workdir, proj_name)
        uploads_path = os.path.join(project_path, "uploads")
        audio_path = os.path.join(project_path, "audio")
        ensure_directory(uploads_path)
        ensure_directory(audio_path)

        # Videó másolása az uploads mappába
        video_filename = os.path.basename(video_path)
        dest_video_path = os.path.join(uploads_path, video_filename)
        shutil.copy(video_path, dest_video_path)

        # Audio kivonása
        video = VideoFileClip(dest_video_path)
        audio = video.audio
        audio_file_path = os.path.join(audio_path, f"{os.path.splitext(video_filename)[0]}.mp3")
        audio.write_audiofile(audio_file_path)
        audio.close()
        video.close()

        return f"Videó és audio sikeresen feltöltve és kivonva.\nAudio fájl elérhető itt: {audio_file_path}"
    except Exception as e:
        return f"Hiba történt a feltöltés vagy az audio kivonása során: {str(e)}"

# tabs/audio_splitting.py
import os
import shutil
from .utils import run_script, ensure_directory, list_projects

def split_audio(proj_name, audio_choice, workdir="workdir"):
    """
    Audio fájlok darabolása splitter.py script segítségével.

    Args:
        proj_name (str): A kiválasztott projekt neve.
        audio_choice (str): "Teljes audio" vagy "Beszéd eltávolított audio".
        workdir (str): A munkakönyvtár alapértelmezett útvonala.

    Yields:
        str: A script kimenete folyamatosan frissülő eredmény ablakban.
    """
    try:
        project_path = os.path.join(workdir, proj_name)

        # Kiválasztott audio fájl elérési útjának meghatározása
        if audio_choice == "Full Audio":
            audio_dir = os.path.join(project_path, "audio")
            audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.mp3'))]
            if not audio_files:
                yield "Nem található teljes audio fájl a projektben."
                return
            selected_audio = os.path.join(audio_dir, audio_files[0])
        elif audio_choice == "Speech Only":
            audio_dir = os.path.join(project_path, "speech_removed")
            audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('_speech.wav') and not f.lower().endswith('_non_speech.wav')]
            if not audio_files:
                yield "Nincs található beszéd eltávolított audio fájl a projektben."
                return
            selected_audio = os.path.join(audio_dir, audio_files[0])
        else:
            yield "Érvénytelen audio választás."
            return

        # JSON fájl megtalálása a transcripts mappában
        transcripts_dir = os.path.join(project_path, "transcripts")
        audio_basename = os.path.splitext(os.path.basename(selected_audio))[0]
        json_files = [f for f in os.listdir(transcripts_dir) if f.lower().endswith('.json') and os.path.splitext(f)[0] == audio_basename]
        if not json_files:
            # Ha nincs azonos nevű JSON, keresünk más JSON fájlt és átnevezzük
            json_files = [f for f in os.listdir(transcripts_dir) if f.lower().endswith('.json')]
            if not json_files:
                yield "Nincs található megfelelő JSON fájl a transzkripciók között."
                return
            # Válasszuk az elsőt és nevezzük át
            src_json = os.path.join(transcripts_dir, json_files[0])
            dest_json = os.path.join(transcripts_dir, f"{audio_basename}.json")
            shutil.copy(src_json, dest_json)
        else:
            dest_json = os.path.join(transcripts_dir, json_files[0])

        # TEMP könyvtár létrehozása
        temp_dir = os.path.join(project_path, "TEMP")
        ensure_directory(temp_dir)

        # Audio fájl és JSON fájl másolása a TEMP könyvtárba, átnevezve, ha szükséges
        shutil.copy(selected_audio, temp_dir)
        shutil.copy(dest_json, temp_dir)

        # Ha a JSON fájl neve nem egyezik az audio fájl nevével, átnevezzük
        temp_json_path = os.path.join(temp_dir, f"{audio_basename}.json")
        if os.path.basename(dest_json) != f"{audio_basename}.json":
            os.rename(os.path.join(temp_dir, os.path.basename(dest_json)), temp_json_path)

        # Kimeneti könyvtár meghatározása
        split_output_dir = os.path.join(project_path, "split_audio")
        ensure_directory(split_output_dir)

        # Külső splitter.py script hívása
        splitter_script = os.path.join("scripts", "splitter.py")  # Ha más helyen van, add meg a teljes elérési utat
        cmd = ["python", "-u", splitter_script, "--input_dir", temp_dir, "--output_dir", split_output_dir]

        # Script futtatása és kimenet olvasása
        for output in run_script(cmd):
            yield output

        # TEMP könyvtár törlése
        shutil.rmtree(temp_dir)

        yield f"\nAudio sikeresen darabolva.\nEredmény itt: {split_output_dir}"

    except Exception as e:
        yield f"Hiba történt az audio darabolása során: {str(e)}"


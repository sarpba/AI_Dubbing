# tabs/verify_chunks.py
import os
import shutil
from .utils import run_script, ensure_directory

def verify_chunks_whisperx(proj_name, workdir="workdir"):
    """
    Ellenőrzi a darabolt audio fájlokat WhisperX segítségével a split_audio és sync könyvtárakból.

    Args:
        proj_name (str): A kiválasztott projekt neve.
        workdir (str): A munkakönyvtár alapértelmezett útvonala.

    Yields:
        str: A script kimenete folyamatosan frissülő eredmény ablakban.
    """
    try:
        project_path = os.path.join(workdir, proj_name)
        split_audio_dir = os.path.join(project_path, "split_audio")
        sync_dir = os.path.join(project_path, "sync")
        transcripts_split_dir = os.path.join(project_path, "transcripts_split")
        
        # Új kód: Ellenőrizzük, hogy létezik-e a transcripts_split mappa
        if os.path.exists(transcripts_split_dir):
            json_files = [f for f in os.listdir(transcripts_split_dir) if f.lower().endswith('.json')]
            for json_file in json_files:
                src_json_path = os.path.join(transcripts_split_dir, json_file)
                dest_json_path = os.path.join(split_audio_dir, json_file)
                shutil.move(src_json_path, dest_json_path)
            # Opció: Tájékoztathatjuk a felhasználót a műveletről
            print(f"A {transcripts_split_dir} mappából vissza lettek helyezve a JSON fájlok a {split_audio_dir} mappába.")

        # Biztosítjuk, hogy a transcripts_split mappa létezik
        ensure_directory(transcripts_split_dir)

        # Ellenőrizzük, hogy van-e darabolt audio fájl a split_audio könyvtárban
        audio_files_split = [f for f in os.listdir(split_audio_dir) if f.lower().endswith(('.wav', '.mp3'))] if os.path.exists(split_audio_dir) else []

        # Ellenőrizzük, hogy van-e audio fájl a sync könyvtárban
        audio_files_sync = [f for f in os.listdir(sync_dir) if f.lower().endswith(('.wav', '.mp3'))] if os.path.exists(sync_dir) else []

        if not audio_files_split and not audio_files_sync:
            yield "Nincs darabolt audio fájl a projektben a split_audio vagy sync könyvtárban."
            return

        # WhisperX script elérési útja
        whisperx_script = os.path.join("scripts", "whisx_simple.py")  # Módosítsd az elérési utat, ha szükséges

        # Függvény a WhisperX futtatásához egy adott könyvtárban
        def run_whisperx_on_directory(audio_dir, move_json_files):
            cmd = ["python", "-u", whisperx_script, audio_dir]
            # Script futtatása és kimenet olvasása
            for output in run_script(cmd):
                yield output

            if move_json_files:
                # JSON fájlok áthelyezése a transcripts_split mappába
                json_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.json')]
                if not json_files:
                    yield f"A chunks ellenőrzése sikeres volt a {audio_dir} könyvtárban, de nem található JSON fájl."
                    return

                for json_file in json_files:
                    src_json_path = os.path.join(audio_dir, json_file)
                    dest_json_path = os.path.join(transcripts_split_dir, json_file)
                    shutil.move(src_json_path, dest_json_path)

                yield f"\nChunks ellenőrzése kész: {transcripts_split_dir}"
            else:
                yield f"\nChunks ellenőrzése kész a {audio_dir} könyvtárban."

        # Futtassuk a WhisperX-et a split_audio könyvtárban, és mozgassuk a JSON fájlokat
        if audio_files_split:
            yield f"\nWhisperX futtatása a split_audio könyvtárban..."
            for output in run_whisperx_on_directory(split_audio_dir, move_json_files=True):
                yield output

        # Ha létezik a sync könyvtár, futtassuk ott is, de ne mozgassuk a JSON fájlokat
        if audio_files_sync:
            yield f"\nWhisperX futtatása a sync könyvtárban..."
            for output in run_whisperx_on_directory(sync_dir, move_json_files=False):
                yield output

    except Exception as e:
        yield f"Hiba történt a chunks ellenőrzése során: {str(e)}"


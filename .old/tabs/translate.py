# tabs/translate.py
import os
from .utils import run_script, ensure_directory

def translate_chunks(proj_name, input_language, output_language, auth_key, workdir="workdir"):
    """
    Chunkok fordítása DeepL API segítségével translate.py script hívásával.

    Args:
        proj_name (str): A kiválasztott projekt neve.
        input_language (str): A bemeneti nyelv kódja (pl. EN, HU).
        output_language (str): A kimeneti nyelv kódja (pl. EN, HU).
        auth_key (str): A DeepL API hitelesítési kulcs.
        workdir (str): A munkakönyvtár alapértelmezett útvonala.

    Yields:
        str: A script kimenete folyamatosan frissülő eredmény ablakban.
    """
    try:
        project_path = os.path.join(workdir, proj_name)
        input_dir = os.path.join(project_path, "split_audio")
        output_dir = os.path.join(project_path, "translations")
        ensure_directory(output_dir)

        # Ellenőrizzük, hogy vannak-e fordítandó .txt fájlok
        txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
        if not txt_files:
            yield "Nincs fordítandó .txt fájl a split_audio könyvtárban."
            return

        # Külső translate.py script hívása
        translate_script = os.path.join("scripts", "translate.py")  # Ha más helyen van, add meg a teljes elérési utat

        # Parancs összeállítása
        cmd = [
            "python", "-u", translate_script,
            "-input_dir", input_dir,
            "-output_dir", output_dir,
            "-input_language", input_language,
            "-output_language", output_language,
            "-auth_key", auth_key
        ]

        # Script futtatása és kimenet olvasása
        for output in run_script(cmd):
            yield output

        yield f"\nChunks fordítása kész: {output_dir}"

    except Exception as e:
        yield f"Hiba történt a fordítás során: {str(e)}"


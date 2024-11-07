# tabs/speech_removal.py
import os
from .utils import run_script, ensure_directory

def separate_audio(proj_name, device, keep_full_audio, selected_model, workdir="workdir"):
    """
    Videófájlokból audio kivonása és szétválasztása Demucs MDX segítségével külső script hívásával.

    Args:
        proj_name (str): A kiválasztott projekt neve.
        device (str): "cpu" vagy "cuda".
        keep_full_audio (bool): Meghatározza, hogy megtartja-e a teljes audio fájlt.
        workdir (str): A munkakönyvtár alapértelmezett útvonala.

    Yields:
        str: A script kimenete folyamatosan frissülő eredmény ablakban.
    """
    try:
        project_path = os.path.join(workdir, proj_name)
        input_dir = os.path.join(project_path, "uploads")
        output_dir = os.path.join(project_path, "speech_removed")
        ensure_directory(output_dir)

        # Külső script hívása
        separate_script = os.path.join("scripts", "separate.py")
        cmd = ["python", "-u", separate_script, "-i", input_dir, "-o", output_dir]

        # Opciók hozzáadása
        if device:
            cmd += ["--device", device]
        if keep_full_audio:
            cmd += ["--keep_full_audio"]
        if selected_model:
            cmd += ["--model", selected_model]

        # Script futtatása és kimenet olvasása
        for output in run_script(cmd):
            yield output

        yield f"\nBeszéd sikeresen eltávolítva.\nEredmény itt: {output_dir}"

    except Exception as e:
        yield f"Hiba történt a beszéd eltávolítása során: {str(e)}"


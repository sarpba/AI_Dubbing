# tabs/utils.py
import os
import subprocess
import torch
import string
import html
import shutil
import subprocess

def list_projects(workdir="workdir"):
    """
    Listázza a meglévő projekteket a munkakönyvtárban.
    """
    try:
        projects = [d for d in os.listdir(workdir) if os.path.isdir(os.path.join(workdir, d))]
        return projects
    except Exception as e:
        print(f"Hiba a projektek listázása során: {e}")
        return []

def get_available_gpus():
    """
    Lekérdezi a PyTorch által elérhető GPU-k indexeit.
    """
    try:
        count = torch.cuda.device_count()
        return list(range(count))
    except Exception as e:
        print(f"Hiba a GPU-k lekérdezése során: {e}")
        return []


def ensure_directory(path):
    """
    Biztosítja, hogy a megadott könyvtár létezik. Ha nem, létrehozza.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def run_script(cmd, shell=False):
    """
    Egy külső script futtatása és a kimenet olvasása soronként.

    Args:
        cmd (list vagy str): A futtatni kívánt parancs. Lehet list vagy str.
        shell (bool): Ha True, a parancsot shell-ben futtatja.

    Yields:
        str: A script aktuális kimenete soronként.
    """
    if shell and isinstance(cmd, list):
        cmd = ' '.join(cmd)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=shell,
        bufsize=1,
        universal_newlines=True
    )
    try:
        for line in process.stdout:
            yield line.rstrip()
    except Exception as e:
        yield f"\nHiba történt a script futtatása közben: {e}"
    finally:
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            yield f"\nScript végrehajtása sikertelen, kilépési kód: {return_code}"
        else:
            yield f"\nScript végrehajtása sikeresen befejeződött."


def normalize_text(text):
    """
    Normalizálja a szöveget: kisbetűssé alakítja és eltávolítja az írásjeleket.

    Args:
        text (str): Eredeti szöveg.

    Returns:
        str: Normalizált szöveg.
    """
    text = text.lower()  # Kisbetűssé alakítás
    translator = str.maketrans('', '', string.punctuation)  # Írásjelek eltávolítása
    return text.translate(translator).strip()

def escape_html_text(text):
    """
    HTML-escape-eli a szöveget.

    Args:
        text (str): Eredeti szöveg.

    Returns:
        str: Escaped szöveg.
    """
    return html.escape(text).replace("\n", "<br>")

def ensure_directory(path):
    """
    Biztosítja, hogy a megadott könyvtár létezzen.

    Args:
        path (str): Könyvtár útvonala.
    """
    os.makedirs(path, exist_ok=True)


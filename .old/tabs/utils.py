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


from typing import Union, List, Generator

def run_script(cmd: Union[List[str], str], shell: bool = False, logfile: str = "script.log") -> Generator[str, None, None]:
    """
    Egy külső script futtatása és a kimenet olvasása soronként, miközben minden sort naplóz egy logfájlba.

    Args:
        cmd (list vagy str): A futtatni kívánt parancs. Lehet list vagy str.
        shell (bool): Ha True, a parancsot shell-ben futtatja.
        logfile (str): A logfájl elérési útja.

    Yields:
        str: A script aktuális kimenete soronként.
    """
    if shell and isinstance(cmd, list):
        cmd = ' '.join(cmd)

    try:
        log_dir = os.path.dirname(logfile)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"Hiba a logfájl könyvtárának létrehozásakor: {e}")
        # Folytatjuk a futást, még ha a logfájl nem is írható

    try:
        with open(logfile, "a", encoding="utf-8") as log_file:
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
                    clean_line = line.rstrip()
                    log_file.write(clean_line + "\n")
                    log_file.flush()  # Azonnali írás a fájlba
                    yield clean_line
            except Exception as e:
                error_msg = f"\nHiba történt a script futtatása közben: {e}"
                log_file.write(error_msg + "\n")
                log_file.flush()
                yield error_msg
            finally:
                process.stdout.close()
                return_code = process.wait()
                if return_code != 0:
                    failure_msg = f"\nScript végrehajtása sikertelen, kilépési kód: {return_code}"
                    log_file.write(failure_msg + "\n")
                    log_file.flush()
                    yield failure_msg
                else:
                    success_msg = f"\nScript végrehajtása sikeresen befejeződött."
                    log_file.write(success_msg + "\n")
                    log_file.flush()
                    yield success_msg
    except Exception as e:
        error_msg = f"Hiba a logfájl írásakor: {e}"
        print(error_msg)
        yield error_msg

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

def get_available_demucs_models():
    try:
        from demucs.pretrained import MODEL_HASHES
        return list(MODEL_HASHES.keys())
    except ImportError:
        # Alapértelmezett lista további modellekkel
        return ["htdemucs", "htdemucs_ft", "hdemucs_mmi", "mdx", "mdx_extra",]


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


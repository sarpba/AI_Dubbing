# tabs/tts_generation.py
import os
import glob
from .utils import run_script, ensure_directory

def tts_generation(proj_name, workdir="workdir"):
    """
    TTS hangfájlok generálása a generate parancs segítségével.

    Args:
        proj_name (str): A kiválasztott projekt neve.
        workdir (str): A munkakönyvtár alapértelmezett útvonala.

    Yields:
        str: A script aktuális kimenete.
    """
    try:
        # Meghatározzuk a fő alkalmazás útvonalát (a main_app.py könyvtára)
        main_app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Definiáljuk a szükséges könyvtárak elérési útját
        TTS_dir = os.path.join(main_app_dir, "TTS")
        split_audio_dir = os.path.join(workdir, proj_name, "split_audio")
        translations_dir = os.path.join(workdir, proj_name, "translations")
        sync_dir = os.path.join(workdir, proj_name, "sync")
        
        # Ellenőrizzük, hogy a szükséges könyvtárak léteznek-e
        if not os.path.exists(split_audio_dir):
            yield f"Hiba: A split_audio könyvtár nem található: {split_audio_dir}"
            return

        if not os.path.exists(translations_dir):
            yield f"Hiba: A translations könyvtár nem található: {translations_dir}"
            return

        if not os.path.exists(sync_dir):
            os.makedirs(sync_dir)
            yield f"A sync könyvtár létre lett hozva: {sync_dir}"
        
        # Ellenőrizzük, hogy a TTS könyvtár létezik-e
        if not os.path.exists(TTS_dir):
            yield f"Hiba: A TTS könyvtár nem található: {TTS_dir}"
            return

        # Definiáljuk a szükséges fájlok elérési útját
        model_files = glob.glob(os.path.join(TTS_dir, "*.pt"))
        if not model_files:
            yield f"Hiba: Nincsenek model (.pt) fájlok a TTS könyvtárban: {TTS_dir}"
            return
        # Ha több model fájl van, feltételezzük, hogy csak egyet használunk. Ha többt is, módosítsd a logikát.
        model_path = model_files[0]

        vocab_path = os.path.join(TTS_dir, "vocab.txt")
        if not os.path.exists(vocab_path):
            yield f"Hiba: Hiányzik a vocab.txt a TTS könyvtárban: {vocab_path}"
            return

        # Ellenőrizzük, hogy vannak-e wav fájlok a split_audio_dir-ben
        wav_files = [f for f in os.listdir(split_audio_dir) if f.lower().endswith('.wav')]
        if not wav_files:
            yield f"Nincsenek wav fájlok a split_audio könyvtárban: {split_audio_dir}"
            return

        # Ellenőrizzük, hogy a split_audio_dir-ben léteznek-e a megfelelő txt fájlok
        missing_split_texts = []
        for wav_file in wav_files:
            wav_basename = os.path.splitext(wav_file)[0]
            split_txt_path = os.path.join(split_audio_dir, f"{wav_basename}.txt")
            if not os.path.exists(split_txt_path):
                missing_split_texts.append(split_txt_path)
        if missing_split_texts:
            yield f"Hiba: Hiányzó split txt fájlok:\n" + "\n".join(missing_split_texts)
            return

        # Ellenőrizzük, hogy a translations_dir-ben léteznek-e a megfelelő txt fájlok
        missing_translation_texts = []
        for wav_file in wav_files:
            wav_basename = os.path.splitext(wav_file)[0]
            translation_txt_path = os.path.join(translations_dir, f"{wav_basename}.txt")
            if not os.path.exists(translation_txt_path):
                missing_translation_texts.append(translation_txt_path)
        if missing_translation_texts:
            yield f"Hiba: Hiányzó translation txt fájlok:\n" + "\n".join(missing_translation_texts)
            return

        # Parancs összeállítása a generate futtatásához az F5-TTS környezetben
        cmd = [
            "conda", "run", "-n", "F5-TTS",
            "python", "./scripts/generate_v3.py",
            "-m", "F5-TTS",
            "-p", model_path,
            "-v", vocab_path,
            "-rd", split_audio_dir,
            "-rt", split_audio_dir,  # Assuming the split text files are in split_audio_dir
            "-rt_gen", translations_dir,
            "-o", sync_dir
        ]

        # Logolás megkezdése
        cmd_str = ' '.join([f'"{c}"' if ' ' in c else c for c in cmd])
        yield f"Futtatás: {cmd_str}"

        # Script futtatása és kimenet olvasása
        for output in run_script(cmd):
            yield output

        yield f"\nTTS hangfájlok generálása befejeződött."
    
    except Exception as e:
        yield f"Hiba történt a TTS hangfájlok generálása során: {e}"

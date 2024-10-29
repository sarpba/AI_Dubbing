# tabs/tts_generation.py
import os
import glob
from .utils import run_script, ensure_directory
from .huntextnormalizer import HungarianTextNormalizer  # Importáljuk a normalizálót

def tts_generation(proj_name, workdir="workdir", logdir="LOG"):
    """
    TTS hangfájlok generálása a f5-tts_infer-cli parancs segítségével, és log fájl írása a /LOG könyvtárba.

    Args:
        proj_name (str): A kiválasztott projekt neve.
        workdir (str): A munkakönyvtár alapértelmezett útvonala.
        logdir (str): A log fájlok könyvtára.

    Yields:
        str: A script aktuális kimenete.
    """
    try:
        # Meghatározzuk a fő alkalmazás útvonalát (a main_app.py könyvtára)
        main_app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Definiáljuk a szükséges könyvtárak elérési útját
        TTS_dir = os.path.join(main_app_dir, "TTS")
        split_audio_dir = os.path.join(workdir, proj_name, "split_audio")
        translations_dir = os.path.join(workdir, proj_name, "translations")  # Javított útvonal
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

        # --load_vocoder_from_local és vocoder_path eltávolítva

        # Log könyvtár biztosítása
        ensure_directory(logdir)
        log_file_path = os.path.join(logdir, f"tts_generation_{proj_name}.log")

        # Keresési minta a wav fájlokra a split_audio könyvtárban
        wav_files = [f for f in os.listdir(split_audio_dir) if f.lower().endswith('.wav')]
        if not wav_files:
            yield f"Nincsenek wav fájlok a split_audio könyvtárban: {split_audio_dir}"
            return

        # Inicializáljuk a HungarianTextNormalizer-t
        normalizer = HungarianTextNormalizer()

        # Futtatjuk a f5-tts_infer-cli parancsot minden wav fájlra
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            for wav_file in wav_files:
                wav_basename = os.path.splitext(wav_file)[0]
                wav_path = os.path.join(split_audio_dir, wav_file)

                # Megfelelő txt fájlok elérési útja
                split_txt_path = os.path.join(split_audio_dir, f"{wav_basename}.txt")
                translate_txt_path = os.path.join(translations_dir, f"{wav_basename}.txt")  # Javított útvonal

                # Ellenőrizzük, hogy a txt fájlok léteznek-e
                if not os.path.exists(split_txt_path):
                    log_message = f"Hiba: Hiányzik a split_audio txt fájl: {split_txt_path}"
                    log_file.write(log_message + '\n')
                    log_file.flush()
                    yield log_message
                    continue

                if not os.path.exists(translate_txt_path):
                    log_message = f"Hiba: Hiányzik a translations txt fájl: {translate_txt_path}"
                    log_file.write(log_message + '\n')
                    log_file.flush()
                    yield log_message
                    continue

                # Beolvassuk a txt fájlok tartalmát
                with open(split_txt_path, 'r', encoding='utf-8') as f_split:
                    split_text = f_split.read().strip()
                with open(translate_txt_path, 'r', encoding='utf-8') as f_translate:
                    translate_text = f_translate.read().strip()

                # Normalizáljuk csak a -t szöveget
                normalized_translate_text = normalizer.normalize(translate_text)

                # Parancs összeállítása a f5-tts_infer-cli futtatásához az F5-TTS környezetben
                cmd = [
                    "conda", "run", "-n", "F5-TTS",
                    "f5-tts_infer-cli",
                    "-m", "F5-TTS",
                    "-p", model_path,
                    "-v", vocab_path,
                    "-r", wav_path,
                    "-s", split_text,  # Eredeti szöveg
                    "-t", normalized_translate_text,  # Normalizált szöveg
                    "-o", sync_dir
                ]

                # Logolás megkezdése
                # Az idézőjelek a szövegek körül, hogy kezeljük a szóközöket
                start_message = f"Futtatás: {' '.join(cmd[:7])} -s \"{split_text}\" -t \"{normalized_translate_text}\" {' '.join(cmd[8:])}"
                log_file.write(start_message + '\n')
                log_file.flush()
                yield start_message

                # Script futtatása és kimenet olvasása
                for output in run_script(cmd):
                    log_file.write(output + '\n')
                    log_file.flush()
                    yield output

                # `infer_cli_out.wav` átnevezése az eredeti wav fájl nevére
                out_wav_path = os.path.join(sync_dir, "infer_cli_out.wav")  # Átnevezés
                renamed_wav_path = os.path.join(sync_dir, f"{wav_basename}.wav")

                if os.path.exists(out_wav_path):
                    os.rename(out_wav_path, renamed_wav_path)
                    rename_message = f"Átnevezve: {out_wav_path} -> {renamed_wav_path}"
                    log_file.write(rename_message + '\n')
                    log_file.flush()
                    yield rename_message
                else:
                    missing_out_message = f"Hiba: Hiányzik az infer_cli_out.wav a sync könyvtárban: {out_wav_path}"
                    log_file.write(missing_out_message + '\n')
                    log_file.flush()
                    yield missing_out_message

        yield f"\nTTS hangfájlok generálása befejeződött. A log fájl itt található: {log_file_path}"
    
    except Exception as e:
        yield f"Hiba történt a TTS hangfájlok generálása során: {e}"


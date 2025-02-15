#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import gc
import time
import datetime
import sys
from multiprocessing import Process, Queue, current_process, Manager
import subprocess
import json
import queue  # a queue.Empty kivételhez

import logging
# Letiltjuk a Pyannote logolását, hogy ne jelenjenek meg üzenetek
logging.getLogger("pyannote").disabled = True
logging.getLogger("pyannote.audio").disabled = True

# Maximális próbálkozások egy fájl feldolgozására
MAX_RETRIES = 3
# Időtúllépés másodpercben (nem implementált a jelenlegi szkriptben)
TIMEOUT = 600  # 10 perc
# Tétlenség-limit a GPU-knál (mp)
INACTIVITY_LIMIT = 10

def get_available_gpus():
    """
    Lekérdezi a rendelkezésre álló GPU indexeket az nvidia-smi segítségével.
    """
    try:
        command = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gpu_indices = result.stdout.decode().strip().split('\n')
        gpu_ids = [int(idx) for idx in gpu_indices if idx.strip().isdigit()]
        return gpu_ids
    except Exception as e:
        print(f"Hiba a GPU-k lekérdezése során: {e}")
        return []

def get_audio_duration(audio_file):
    """
    Visszaadja az audio fájl hosszát másodpercben.
    """
    command = [
        "ffprobe",
        "-i", audio_file,
        "-show_entries", "format=duration",
        "-v", "quiet",
        "-of", "csv=p=0"
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        duration_str = result.stdout.decode().strip()
        duration = float(duration_str)
        return duration
    except Exception as e:
        print(f"Nem sikerült meghatározni az audio hosszát a következő fájlhoz: {audio_file} - {e}")
        return 0

def worker(gpu_id, task_queue, progress_queue, last_activity, lang_override):
    """
    A GPU-khoz rendelt folyamatokat kezelő függvény.
    Nem törli a betöltött modelleket minden fájl után, csak a folyamat legvégén.
    Használja a progress_queue-t, hogy jelezze a sikeres feldolgozás befejezését vagy a sikertelenséget.
    A last_activity[gpu_id] értékét frissíti, ha új feladatot kap és ha befejez egy feladatot.
    
    A lang_override értéke, ha meg van adva, mind az átírás, mind az alignment során
    az adott nyelv használatát eredményezi.
    """
    # Beállítjuk a CUDA_VISIBLE_DEVICES környezeti változót, hogy csak az adott GPU látható legyen
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda"  # 'cuda' mostantól az adott GPU-t jelenti

    model = None  # WhisperX transcribe modell
    alignment_models = {}  # Nyelvenként cache-elt alignment modellek: { 'en': (model_a, metadata), ... }

    try:
        import torch
        import whisperx

        print(f"Folyamat {current_process().name} beállítva a GPU-{gpu_id} eszközre.")
        if lang_override is not None:
            print(f"GPU-{gpu_id}: A megadott nyelv: {lang_override}")

        # WhisperX fő (transcribe) modell betöltése (csak egyszer)
        print(f"GPU-{gpu_id}: WhisperX transcribe modell betöltése...")
        model = whisperx.load_model("large-v3-turbo", device=device, compute_type="float16")
        print(f"GPU-{gpu_id}: Transcribe modell betöltve.")

        no_task_count = 0
        max_no_task_tries = 3

        while True:
            try:
                # Megpróbálunk azonnal feladatot kivenni
                task = task_queue.get_nowait()
                # Ha kaptunk feladatot, frissítjük az utolsó aktivitást
                last_activity[gpu_id] = time.time()
            except queue.Empty:
                # Ha nincs feladat a queue-ban, kicsit várunk
                if no_task_count < max_no_task_tries:
                    no_task_count += 1
                    time.sleep(1)
                    continue
                else:
                    # Ha 3x sem érkezik feladat, kilépünk a workerből
                    print(f"GPU-{gpu_id}: Nincs több feladat, kilépek a workerből.")
                    break
            except Exception as e:
                print(f"GPU-{gpu_id}: Hiba a feladat kérésekor: {e}")
                break

            no_task_count = 0  # Ha sikerült feladatot kapni, nullázzuk a számlálót

            audio_file, retries = task
            json_file = os.path.splitext(audio_file)[0] + ".json"

            if os.path.exists(json_file):
                print(f"Már létezik: {json_file}, kihagyás...")
                progress_queue.put({
                    "status": "done",
                    "file": audio_file,
                    "processing_time": 0
                })
                continue

            try:
                print(f"GPU-{gpu_id} használatával feldolgozás: {audio_file}")
                start_time = time.time()
                start_datetime = datetime.datetime.now()

                # 1) Audio betöltése és átírás
                audio = whisperx.load_audio(audio_file)
                if lang_override is not None:
                    result = model.transcribe(audio, batch_size=16, language=lang_override)
                else:
                    result = model.transcribe(audio, batch_size=16)
                print(f"Átírás befejezve: {audio_file}")

                # 2) Alignálás a transzkripció eredményével
                if lang_override is not None:
                    align_language_code = lang_override
                else:
                    align_language_code = result["language"]

                if align_language_code not in alignment_models:
                    print(f"GPU-{gpu_id}: Alignment modell betöltése a következő nyelvhez: {align_language_code}")
                    model_a, metadata = whisperx.load_align_model(
                        language_code=align_language_code,
                        device=device
                    )
                    alignment_models[align_language_code] = (model_a, metadata)
                else:
                    model_a, metadata = alignment_models[align_language_code]

                result_aligned = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False
                )
                print(f"Alignálás befejezve: {audio_file}")

                # 3) Eredmények mentése JSON-be
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(result_aligned, f, ensure_ascii=False, indent=4)

                end_time = time.time()
                end_datetime = datetime.datetime.now()
                processing_time = end_time - start_time
                audio_duration = get_audio_duration(audio_file)
                ratio = audio_duration / processing_time if processing_time > 0 else 0

                print(f"Sikeresen feldolgozva GPU-{gpu_id} által:")
                print(f"  Feldolgozott fájl: {audio_file}")
                print(f"  Audio hossza: {audio_duration:.2f} s")
                print(f"  Feldolgozási idő: {processing_time:.2f} s")
                print(f"  Arány: {ratio:.2f}")
                print(f"  Kezdés: {start_datetime.strftime('%Y.%m.%d %H:%M')}")
                print(f"  Befejezés: {end_datetime.strftime('%Y.%m.%d %H:%M')}\n")

                progress_queue.put({
                    "status": "done",
                    "file": audio_file,
                    "processing_time": processing_time
                })

                last_activity[gpu_id] = time.time()

            except Exception as e:
                print(f"Hiba a következő fájl feldolgozása során GPU-{gpu_id}-n: {audio_file} - {e}")
                if retries < MAX_RETRIES:
                    print(f"Újrapróbálkozás {retries + 1}/{MAX_RETRIES}...\n")
                    task_queue.put((audio_file, retries + 1))
                else:
                    print(f"Maximális próbálkozások elérve: {audio_file} feldolgozása sikertelen.\n")
                    progress_queue.put({
                        "status": "failed",
                        "file": audio_file,
                        "error": str(e)
                    })

    except Exception as main_e:
        print(f"Fő hiba a GPU-{gpu_id} folyamatban: {main_e}")

    finally:
        if model is not None:
            try:
                print(f"GPU-{gpu_id}: GPU memória felszabadítása...")
                del model
                for lang_code, (model_a, _) in alignment_models.items():
                    del model_a
                alignment_models.clear()
                gc.collect()
                import torch
                torch.cuda.empty_cache()
                print(f"GPU-{gpu_id}: GPU memória felszabadítva.")
            except Exception as cleanup_e:
                print(f"Hiba a GPU-{gpu_id} memória felszabadítása során: {cleanup_e}")
        else:
            print(f"GPU-{gpu_id}: Modell nem volt betöltve, nincs mit felszabadítani.")

def get_audio_files(directory):
    """
    Az adott könyvtárban és almappáiban található összes audio fájl gyűjtése.
    """
    audio_extensions = (".mp3", ".wav", ".flac", ".m4a", ".opus")
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files

def transcribe_directory(directory, gpu_ids, lang):
    """
    Folyamatokat indító és feladatlistát kezelő függvény.
    Elindít minden GPU-ra egy worker folyamatot,
    figyeli a progress_queue-t, és ellenőrzi a GPU-k tétlenségét.
    
    A lang paraméter, ha nem None, a megadott nyelvet használja a transzkripció és alignálás során.
    """
    audio_files = get_audio_files(directory)
    task_queue = Queue()
    tasks_added = 0

    for audio_file in audio_files:
        json_file = os.path.splitext(audio_file)[0] + ".json"
        if not os.path.exists(json_file):
            task_queue.put((audio_file, 0))
            tasks_added += 1
        else:
            print(f"Már létezik: {json_file}, kihagyás...")

    if tasks_added == 0:
        print("Nincs feldolgozandó fájl.")
        return

    print(f"Összesen {tasks_added} fájl vár feldolgozásra.")

    manager = Manager()
    progress_queue = manager.Queue()
    last_activity = manager.dict()

    for gpu_id in gpu_ids:
        last_activity[gpu_id] = time.time()

    processes = []
    for gpu_id in gpu_ids:
        p = Process(
            target=worker,
            args=(gpu_id, task_queue, progress_queue, last_activity, lang),
            name=f"GPU-{gpu_id}-Process"
        )
        processes.append(p)
        p.start()
        print(f"Folyamat indítva: {p.name} a GPU-{gpu_id}-n.")

    tasks_done = 0
    tasks_failed = 0
    failed_files = []
    start_time = time.time()

    while tasks_done + tasks_failed < tasks_added:
        try:
            message = progress_queue.get(timeout=1.0)
            if message["status"] == "done":
                tasks_done += 1
                elapsed_time = time.time() - start_time
                remaining = tasks_added - (tasks_done + tasks_failed)
                avg_time_per_file = elapsed_time / tasks_done if tasks_done > 0 else 0
                est_remaining_time = avg_time_per_file * remaining
                finish_time_est = datetime.datetime.now() + datetime.timedelta(seconds=est_remaining_time)
                progress_percent = ((tasks_done + tasks_failed) / tasks_added) * 100

                print(
                    f"[{tasks_done + tasks_failed}/{tasks_added} - {progress_percent:.1f}%] "
                    f"Kész: {message['file']} | "
                    f"Várható befejezés: {finish_time_est.strftime('%Y-%m-%d %H:%M:%S')}"
                )

            elif message["status"] == "failed":
                tasks_failed += 1
                failed_files.append((message["file"], message.get("error", "Ismeretlen hiba")))
                elapsed_time = time.time() - start_time
                remaining = tasks_added - (tasks_done + tasks_failed)
                avg_time_per_file = elapsed_time / (tasks_done + tasks_failed) if (tasks_done + tasks_failed) > 0 else 0
                est_remaining_time = avg_time_per_file * remaining
                finish_time_est = datetime.datetime.now() + datetime.timedelta(seconds=est_remaining_time)
                progress_percent = ((tasks_done + tasks_failed) / tasks_added) * 100

                print(
                    f"[{tasks_done + tasks_failed}/{tasks_added} - {progress_percent:.1f}%] "
                    f"Sikertelen: {message['file']} | "
                    f"Várható befejezés: {finish_time_est.strftime('%Y-%m-%d %H:%M:%S')}"
                )

        except queue.Empty:
            pass

        if len(gpu_ids) > 1:
            now = time.time()
            for g in gpu_ids:
                if (now - last_activity[g]) > INACTIVITY_LIMIT:
                    print(f"FIGYELEM: GPU-{g} több mint {INACTIVITY_LIMIT} mp-ig tétlen.")
                    print("Minden folyamat leállítása és a script újraindítása...")
                    for proc in processes:
                        if proc.is_alive():
                            proc.terminate()
                    python = sys.executable
                    os.execl(python, python, *sys.argv)

    for p in processes:
        p.join()
        print(f"Folyamat befejezve: {p.name}")

    total_time = time.time() - start_time
    print(f"Minden feladat elkészült. Teljes feldolgozási idő: {total_time:.2f} mp")

    if failed_files:
        print(f"\n{len(failed_files)} fájl feldolgozása sikertelen:")
        for file, error in failed_files:
            print(f"  - {file}: {error}")
        failed_log = os.path.join(directory, "failed_files.log")
        try:
            with open(failed_log, "w", encoding="utf-8") as f:
                for file, error in failed_files:
                    f.write(f"{file}: {error}\n")
            print(f"Sikertelen fájlok listája mentve: {failed_log}")
        except Exception as e:
            print(f"Hiba a sikertelen fájlok mentése során: {e}")
    else:
        print("Minden fájl sikeresen feldolgozva.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audio fájlok átírása és alignálása (WhisperX) több GPU-val, hibajavító funkcióval."
    )
    parser.add_argument("directory", type=str, help="A könyvtár, amely tartalmazza az audio fájlokat.")
    parser.add_argument('--gpus', type=str, default=None, 
                        help="Használni kívánt GPU indexek, vesszővel elválasztva (pl. '0,2,3')")
    parser.add_argument('--lang', type=str, default=None,
                        help="Kényszerített nyelvkód a transzkripcióhoz és alignáláshoz (pl. 'hu', 'en'). Ha nincs megadva, automatikus felismerés történik.")

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Hiba: A megadott könyvtár nem létezik: {args.directory}")
        sys.exit(1)

    if args.gpus:
        try:
            specified_gpus = [int(x.strip()) for x in args.gpus.split(',')]
        except ValueError:
            print("Hiba: A --gpus argumentumnak egész számok vesszővel elválasztott listájának kell lennie.")
            sys.exit(1)
        available_gpus = get_available_gpus()
        if not available_gpus:
            print("Hiba: Nincsenek elérhető GPU-k.")
            sys.exit(1)
        invalid_gpus = [gpu for gpu in specified_gpus if gpu not in available_gpus]
        if invalid_gpus:
            print(f"Hiba: A megadott GPU-k nem érhetők el: {invalid_gpus}")
            sys.exit(1)
        gpu_ids = specified_gpus
    else:
        gpu_ids = get_available_gpus()
        if not gpu_ids:
            print("Hiba: Nincsenek elérhető GPU-k.")
            sys.exit(1)

    print(f"Használt GPU-k: {gpu_ids}")

    transcribe_directory(args.directory, gpu_ids, args.lang)

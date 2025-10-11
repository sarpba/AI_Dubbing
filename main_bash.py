#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import subprocess
import logging
import argparse
import shutil
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Callable

VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.webm')

_CONDA_INFO_CACHE: Optional[dict] = None


def get_conda_info(force_refresh: bool = False) -> Optional[dict]:
    global _CONDA_INFO_CACHE
    if _CONDA_INFO_CACHE is not None and not force_refresh:
        return _CONDA_INFO_CACHE
    try:
        result = subprocess.run(
            ['conda', 'info', '--json'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        info = json.loads(result.stdout)
        _CONDA_INFO_CACHE = info
        return info
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Hiba a Conda információk lekérdezése közben: {e}")
        return None


def _map_env_name_to_path(info: dict) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    envs: List[str] = info.get('envs', []) or []
    root = info.get('root_prefix', '')
    if root:
        mapping['base'] = root
        mapping[os.path.basename(root)] = root
    for p in envs:
        name = os.path.basename(p.rstrip(os.sep))
        mapping[name] = p
    return mapping


def get_conda_python_executable(env_name: str) -> Optional[str]:
    info = get_conda_info()
    if not info:
        return None

    env_map = _map_env_name_to_path(info)
    env_path = env_map.get(env_name)
    if not env_path:
        for envs_dir in info.get('envs_dirs', []):
            potential_path = os.path.join(envs_dir, env_name)
            if os.path.isdir(potential_path):
                env_path = potential_path
                break

    if not env_path:
        logging.error(f"A(z) '{env_name}' nevű Conda környezet nem található.")
        return None

    executable_path = os.path.join(env_path, "python.exe") if sys.platform == "win32" else os.path.join(env_path, "bin", "python")
    if os.path.exists(executable_path):
        logging.debug(f"Python végrehajtható fájl: {executable_path}")
        return executable_path
    else:
        logging.error(f"Nem található a Python végrehajtható fájl itt: {executable_path}")
        return None


def load_config() -> dict:
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Hiba: 'config.json' nem található.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Hiba: 'config.json' hibás formátumú.")
        sys.exit(1)


def save_config(data: dict) -> None:
    try:
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info("Beállítások mentve a 'config.json' fájlba.")
    except IOError as e:
        logging.error(f"Hiba a 'config.json' írása közben: {e}")


def setup_logging(project_path: str, logs_subdir: str, project_name: str) -> Tuple[logging.Handler, str]:
    log_dir = os.path.join(project_path, logs_subdir)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{project_name}_run_{timestamp}.log')

    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

    logging.info(f"Log fájl ehhez a projekthez: {log_file}")
    return file_handler, log_file


def remove_logging_handler(handler: logging.Handler) -> None:
    logger = logging.getLogger()
    try:
        logger.removeHandler(handler)
    except Exception:
        pass
    try:
        handler.close()
    except Exception:
        pass


def create_project_structure(project_path: str, subdirs_config: Dict[str, str]) -> None:
    logging.info(f"Projekt könyvtár létrehozása/ellenőrzése: {project_path}")
    os.makedirs(project_path, exist_ok=True)
    for subdir_name in subdirs_config.values():
        os.makedirs(os.path.join(project_path, subdir_name), exist_ok=True)
    logging.info("Projekt struktúra sikeresen létrehozva/ellenőrizve.")


def run_command(command_list: List[str], log_prefix: str = "", timeout: Optional[float] = None, env: Optional[dict] = None) -> bool:
    command_str = ' '.join(f'"{c}"' if ' ' in c else c for c in command_list)
    logging.info(f"{log_prefix} Parancs futtatása: {command_str}")
    try:
        proc = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            env=env or os.environ.copy(),
        )
        start = time.time()
        if proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                logging.info(f"{log_prefix} {line.rstrip()}")
                if timeout and (time.time() - start) > timeout:
                    proc.kill()
                    logging.error(f"{log_prefix} Timeout ({timeout}s) miatt megszakítva.")
                    return False
        rc = proc.wait()
        if rc != 0:
            logging.error(f"{log_prefix} A parancs hibával fejeződött be! Kód: {rc}")
            return False
        logging.info(f"{log_prefix} Parancs sikeresen lefutott.")
        return True
    except Exception as e:
        logging.error(f"{log_prefix} Váratlan hiba a parancs futtatása közben: {e}", exc_info=True)
        return False


def any_file_with_extensions(path: str, extensions: Tuple[str, ...]) -> bool:
    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith(extensions):
                    return True
    except FileNotFoundError:
        return False
    return False


def has_any_file(path: str) -> bool:
    try:
        with os.scandir(path) as it:
            for _ in it:
                return True
    except FileNotFoundError:
        return False
    return False


def skip_unpack_srt(upload_dir: str) -> bool:
    return any_file_with_extensions(upload_dir, ('.srt',))


def skip_separate_audio(speech_dir: str, background_dir: str) -> bool:
    return has_any_file(speech_dir) and has_any_file(background_dir)


def skip_parakeet_transcribe(speech_dir: str) -> bool:
    return any_file_with_extensions(speech_dir, ('.json',))


def skip_split_by_speaker(speech_dir: str) -> bool:
    return any_file_with_extensions(speech_dir, ('.bak',))


def skip_translate(translated_dir: str) -> bool:
    return any_file_with_extensions(translated_dir, ('.json',))


def gpu_cooldown_wait(seconds: int = 5) -> None:
    logging.info(f">>> Várakozás {seconds} másodpercet a GPU memória felszabadulásáért... <<<")
    time.sleep(seconds)


def run_dubbing_workflow(project_name: str, project_path: str, config: dict, params: dict) -> bool:
    logging.info("A Conda környezetek Python végrehajtható fájljainak keresése...")
    sync_python = get_conda_python_executable("sync")
    parakeet_python = get_conda_python_executable("parakeet-fix")
    tts_python = get_conda_python_executable("f5-tts_hun")

    if not all([sync_python, parakeet_python, tts_python]):
        logging.critical("Egy vagy több Conda környezet Pythonja nem található. A folyamat leáll.")
        return False

    translation_context = params['translation_context']
    tts_path = params['tts_model_path']
    tts_path_run4 = params['tts_model_path_run4']
    lang_code = params['lang_code']

    logging.info(f"--- Dubbing Workflow Indítása: '{project_name}' ---")

    subdirs = config['PROJECT_SUBDIRS']
    upload_dir = os.path.join(project_path, subdirs['upload'])
    speech_dir = os.path.join(project_path, subdirs['separated_audio_speech'])
    background_dir = os.path.join(project_path, subdirs['separated_audio_background'])
    translated_dir = os.path.join(project_path, subdirs['translated'])

    Step = Tuple[List[str], str, bool, Optional[Callable[[], bool]]]
    steps: List[Step] = [
        ([sync_python, 'scripts/extract_audio_easy_channels.py', project_name], "[extract_audio]", True, None),
        ([sync_python, 'scripts/separate_audio_easy_channels.py', '-p', project_name], "[separate_audio]", True, lambda: skip_separate_audio(speech_dir, background_dir)),
        ([sync_python, 'scripts/unpack_srt_from_mkv_easy.py', project_name], "[unpack_srt]", False, lambda: skip_unpack_srt(upload_dir)),
        ([parakeet_python, 'scripts/Nvidia_asr_eng/parakeet_transcribe_wordts_4.0_easy.py', '-p', project_name], "[parakeet_transcribe]", True, lambda: skip_parakeet_transcribe(speech_dir)),
        #([sync_python, 'scripts/split_segments_by_speaker.py', '-p', project_name], "[split_by_speaker]", True, lambda: skip_split_by_speaker(speech_dir)),
        ([sync_python, 'scripts/translate_chatgpt_srt_easy_250910.py', '-project_name', project_name, '-context', translation_context], "[translate_chatgpt]", True, lambda: skip_translate(translated_dir)),
        ([tts_python, 'scripts/f5_tts_easy_gai.py', project_name, '--phonetic-ref', '--norm', 'hun', '--model_dir', tts_path], "[f5_tts_run1]", False, None),
        ([tts_python, 'scripts/f5_tts_easy_gai.py', project_name, '--phonetic-ref', '--norm', 'hun', '--model_dir', tts_path], "[f5_tts_run2]", False, None),
        ([tts_python, 'scripts/f5_tts_easy_gai.py', project_name, '--phonetic-ref', '--norm', 'hun', '--save-failures', '--normalize-ref-audio', '--model_dir', tts_path], "[f5_tts_run3]", True, None),
        ([tts_python, 'scripts/f5_tts_easy_gai.py', project_name, '--phonetic-ref', '--norm', 'hun', '--save-failures', '--normalize-ref-audio', '--model_dir', tts_path_run4], "[f5_tts_run4]", True, None),
        ([sync_python, 'scripts/normalise_and_cut_json_easy2.py', project_name], "[normalise_and_cut]", True, None),
        ([sync_python, 'scripts/merge_chunks_with_background_easy.py', project_name], "[merge_chunks]", True, None),
        ([sync_python, 'scripts/merge_to_video_easy.py', project_name, '-lang', lang_code], "[merge_to_video]", True, None),
    ]

    for cmd, prefix, is_critical, skip_fn in steps:
        if skip_fn:
            try:
                if skip_fn():
                    logging.info(f"{prefix} Kihagyva (kimeneti fájlok jelen vannak).")
                    continue
            except Exception as e:
                logging.warning(f"{prefix} Skip feltétel ellenőrzése közben hiba: {e}")

        if not run_command(cmd, log_prefix=prefix):
            if is_critical:
                logging.critical(f"{prefix} Kritikus lépés sikertelen. '{project_name}' feldolgozása leáll.")
                return False

        if prefix in ("[f5_tts_run1]", "[f5_tts_run2]", "[f5_tts_run3]"):
            gpu_cooldown_wait(2)

    logging.info(f"--- Dubbing Workflow Sikeresen Befejeződött: '{project_name}' ---")
    return True


def copy_results_to_source(project_path: str, source_video_path: str, config: dict) -> None:
    try:
        logging.info("Eredmények visszamásolása a forráskönyvtárba...")
        destination_dir = os.path.join(os.path.dirname(source_video_path), "AI_szinkron")
        os.makedirs(destination_dir, exist_ok=True)

        project_download_dir = os.path.join(project_path, config['PROJECT_SUBDIRS']['download'])
        if os.path.isdir(project_download_dir):
            with os.scandir(project_download_dir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith(VIDEO_EXTENSIONS):
                        shutil.copy2(entry.path, destination_dir)

        project_upload_dir = os.path.join(project_path, config['PROJECT_SUBDIRS']['upload'])
        if os.path.isdir(project_upload_dir):
            with os.scandir(project_upload_dir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith('.srt'):
                        shutil.copy2(entry.path, destination_dir)

        logging.info("Visszamásolás sikeres.")
    except Exception as e:
        logging.error(f"Hiba az eredmények visszamásolása közben: {e}", exc_info=True)


def get_input_with_default(prompt: str, default_value: Optional[str]) -> Optional[str]:
    prompt_text = f"{prompt} [Alapértelmezett: {default_value}] (Enter): " if default_value else f"{prompt}: "
    try:
        user_input = input(prompt_text)
    except KeyboardInterrupt:
        print("\nMegszakítva.")
        sys.exit(1)
    return default_value if user_input == "" and default_value else user_input


def select_from_list(options: List[str], prompt_message: str, default_value: Optional[str] = None) -> Optional[str]:
    if not options:
        logging.error(f"Nincsenek opciók: {prompt_message}")
        return None

    print(prompt_message)
    default_index = -1
    for i, option in enumerate(options, 1):
        if option == default_value:
            print(f"  {i}) {option}  <-- Alapértelmezett")
            default_index = i
        else:
            print(f"  {i}) {option}")

    prompt = f"Válasszon (1-{len(options)}) [Enter = {default_index}]: " if default_index != -1 else f"Válasszon (1-{len(options)}): "
    while True:
        try:
            choice_input = input(prompt)
            if choice_input == "" and default_index != -1:
                return options[default_index - 1]
            choice = int(choice_input)
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print("Érvénytelen szám.")
        except ValueError:
            print("Érvénytelen bemenet.")
        except KeyboardInterrupt:
            print("\nMegszakítva.")
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Videó szinkronizálási vezérlő.", formatter_class=argparse.RawTextHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--project_name', type=str, help="Projekt neve (egyszeri futtatás).")
    group.add_argument('-i', '--input_dir', type=str, help="Bemeneti könyvtár videókkal (kötegelt futtatás).")
    parser.add_argument('--auto', action='store_true', help="Automatikus mód: paraméterek olvasása a config.json-ból.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

    if get_conda_info() is None:
        logging.critical("Nem sikerült lekérdezni a Conda információkat. A program leáll.")
        sys.exit(1)

    config = load_config()
    workdir = config['DIRECTORIES']['workdir']

    if args.auto:
        logging.info("Automatikus mód aktív. Paraméterek betöltése.")
        params = config.get("LAST_USED_PARAMS", {})
        required = ["translation_context", "tts_model_path", "tts_model_path_run4", "lang_code"]
        if not all(k in params for k in required):
            logging.error("Nincsenek elmentett paraméterek az automatikus futatáshoz! Futtasd interaktívan (--auto nélkül).")
            sys.exit(1)
    else:
        params = {}

    if args.input_dir:
        video_files: List[str] = []
        for r, _, fs in os.walk(args.input_dir):
            for f in fs:
                if f.lower().endswith(VIDEO_EXTENSIONS):
                    video_files.append(os.path.join(r, f))

        if not video_files:
            logging.warning("Nem található videófájl.")
            sys.exit(0)

        logging.info(f"Kötegelt feldolgozás: '{args.input_dir}'. Összesen {len(video_files)} videó.")
        for i, video_path in enumerate(video_files):
            project_name = os.path.splitext(os.path.basename(video_path))[0]
            logging.info("\n" + "=" * 60 + f"\nFeldolgozás ({i + 1}/{len(video_files)}): '{project_name}'\nForrás: {video_path}")
            project_path = os.path.join(workdir, project_name)

            create_project_structure(project_path, config['PROJECT_SUBDIRS'])
            upload_dir = os.path.join(project_path, config['PROJECT_SUBDIRS']['upload'])

            has_video = any_file_with_extensions(upload_dir, VIDEO_EXTENSIONS)

            if not has_video:
                logging.info(f"Videófájl másolása a projektbe: {project_name}")
                shutil.copy2(video_path, upload_dir)
            else:
                logging.info("Videófájl már létezik a projektben, a másolás kihagyva.")

            file_handler, log_file = setup_logging(project_path, config['PROJECT_SUBDIRS']['logs'], project_name)
            try:
                if run_dubbing_workflow(project_name, project_path, config, params):
                    copy_results_to_source(project_path, video_path, config)
                else:
                    logging.error(f"Hiba '{project_name}' feldolgozása közben. Folytatás a következővel.")
            except Exception as e:
                logging.critical(f"Váratlan hiba '{project_name}' feldolgozása közben: {e}", exc_info=True)
            finally:
                remove_logging_handler(file_handler)

        logging.info("\n" + "=" * 60 + "\nA kötegelt feldolgozás befejeződött.")

    else:
        project_name = args.project_name
        project_path = os.path.join(workdir, project_name)

        create_project_structure(project_path, config['PROJECT_SUBDIRS'])
        upload_dir = os.path.join(project_path, config['PROJECT_SUBDIRS']['upload'])

        file_handler, log_file = setup_logging(project_path, config['PROJECT_SUBDIRS']['logs'], project_name)
        try:
            has_video = any_file_with_extensions(upload_dir, VIDEO_EXTENSIONS)
            if not has_video:
                print(f"\nKérem, másolja a videót ide: {os.path.abspath(upload_dir)}")
                input("Nyomjon Entert, ha végzett...")
            else:
                logging.info("Videófájl már létezik az 'upload' mappában, a manuális másolás kihagyva.")

            if not args.auto:
                last_params = config.get("LAST_USED_PARAMS", {})
                translation_context = get_input_with_default("Fordítási kontextus", last_params.get("translation_context"))

                tts_base_dir = config['DIRECTORIES']['TTS']
                available_models = [d for d in os.listdir(tts_base_dir) if os.path.isdir(os.path.join(tts_base_dir, d))]

                last_model_name = os.path.basename(last_params.get("tts_model_path", "")) if last_params.get("tts_model_path") else None
                selected_model = select_from_list(available_models, "\nTTS modell (run1-3):", default_value=last_model_name)
                if not selected_model:
                    logging.error("TTS modell választása kötelező!")
                    sys.exit(1)

                last_model_name_run4 = os.path.basename(last_params.get("tts_model_path_run4", "")) if last_params.get("tts_model_path_run4") else selected_model
                default_for_run4 = selected_model if not last_model_name_run4 else last_model_name_run4
                selected_model_run4 = select_from_list(available_models, "\nTTS modell a 4. futtatáshoz (run4):", default_value=default_for_run4)
                if not selected_model_run4:
                    logging.error("TTS modell választása a 4. futtatáshoz kötelező!")
                    sys.exit(1)

                lang_code = select_from_list(["hun", "eng", "ger", "fra"], "\nNyelvi kód:", default_value=last_params.get("lang_code"))
                if not lang_code:
                    logging.error("Nyelvi kód választása kötelező!")
                    sys.exit(1)

                params = {
                    "translation_context": translation_context,
                    "tts_model_path": os.path.join(tts_base_dir, selected_model),
                    "tts_model_path_run4": os.path.join(tts_base_dir, selected_model_run4),
                    "lang_code": lang_code
                }

                for p in ("tts_model_path", "tts_model_path_run4"):
                    if not os.path.isdir(params[p]):
                        logging.error(f"A megadott modell mappa nem létezik: {params[p]}")
                        sys.exit(1)

                config["LAST_USED_PARAMS"] = params
                save_config(config)

            if run_dubbing_workflow(project_name, project_path, config, params):
                logging.info("Az egyszeri futtatás sikeresen befejeződött.")
            else:
                logging.error("Az egyszeri futtatás hibával zárult.")
        finally:
            remove_logging_handler(file_handler)


if __name__ == "__main__":
    main()

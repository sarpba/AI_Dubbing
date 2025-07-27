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
from logging.handlers import MemoryHandler
from datetime import datetime  # <<< JAVÍTÁS: A hiányzó import hozzáadva

VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.webm')

# --- Segédfüggvények ---

_conda_info_cache = None

def get_conda_info():
    """Lefuttatja a `conda info --json` parancsot és gyorsítótárazza az eredményt."""
    global _conda_info_cache
    if _conda_info_cache is not None:
        return _conda_info_cache
    try:
        result = subprocess.run(
            ['conda', 'info', '--json'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        info = json.loads(result.stdout)
        _conda_info_cache = info
        return info
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Hiba a Conda információk lekérdezése közben: {e}")
        return None

def get_conda_python_executable(env_name):
    """Megkeresi a Python végrehajtható fájlt egy Conda környezetben a `conda info` segítségével."""
    info = get_conda_info()
    if not info: return None
    env_path = None
    if env_name == 'base' or env_name == os.path.basename(info.get('root_prefix', '')):
        env_path = info.get('root_prefix')
    else:
        for envs_dir in info.get('envs_dirs', []):
            potential_path = os.path.join(envs_dir, env_name)
            if os.path.isdir(potential_path):
                env_path = potential_path
                break
    if not env_path:
        logging.error(f"A(z) '{env_name}' nevű Conda környezet nem található."); return None
    executable_path = os.path.join(env_path, "python.exe") if sys.platform == "win32" else os.path.join(env_path, "bin", "python")
    if os.path.exists(executable_path):
        logging.debug(f"Python végrehajtható fájl: {executable_path}"); return executable_path
    else:
        logging.error(f"Nem található a Python végrehajtható fájl itt: {executable_path}"); return None

def load_config():
    try:
        with open('config.json', 'r', encoding='utf-8') as f: return json.load(f)
    except FileNotFoundError: print("Hiba: 'config.json' nem található."); sys.exit(1)
    except json.JSONDecodeError: print("Hiba: 'config.json' hibás formátumú."); sys.exit(1)

def save_config(data):
    try:
        with open('config.json', 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info("Beállítások mentve a 'config.json' fájlba.")
    except IOError as e:
        logging.error(f"Hiba a 'config.json' írása közben: {e}")

def setup_logging(project_path, logs_subdir, project_name):
    log_dir = os.path.join(project_path, logs_subdir)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{project_name}_run_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    memory_handler = MemoryHandler(1000, flushLevel=logging.ERROR, target=file_handler)
    logging.getLogger().addHandler(memory_handler)
    logging.info(f"Log fájl ehhez a projekthez: {log_file}")
    return memory_handler, file_handler

def create_project_structure(project_path, subdirs_config):
    logging.info(f"Projekt könyvtár létrehozása: {project_path}")
    os.makedirs(project_path, exist_ok=True)
    for subdir_name in subdirs_config.values():
        os.makedirs(os.path.join(project_path, subdir_name), exist_ok=True)
    logging.info("Projekt struktúra sikeresen létrehozva.")

def run_command(command_list, log_prefix=""):
    command_str = ' '.join(f'"{c}"' if ' ' in c else c for c in command_list)
    logging.info(f"{log_prefix} Parancs futtatása: {command_str}")
    try:
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = "1"
        process = subprocess.Popen(
            command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1, universal_newlines=True, env=env
        )
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                logging.info(f"{log_prefix} {line.strip()}")
        process.wait()
        if process.returncode != 0:
            logging.error(f"{log_prefix} A parancs hibával fejeződött be! Kód: {process.returncode}")
            return False
        logging.info(f"{log_prefix} Parancs sikeresen lefutott.")
        return True
    except Exception as e:
        logging.error(f"Váratlan hiba a parancs futtatása közben: {e}", exc_info=True)
        return False

def run_dubbing_workflow(project_name, config, params, memory_handler):
    logging.info("A Conda környezetek Python végrehajtható fájljainak keresése...")
    sync_python = get_conda_python_executable("sync")
    parakeet_python = get_conda_python_executable("parakeet-fix")
    tts_python = get_conda_python_executable("f5-tts_hun")

    if not all([sync_python, parakeet_python, tts_python]):
        logging.critical("Egy vagy több Conda környezet Pythonja nem található. A folyamat leáll.")
        return False
    
    translation_context, tts_path, lang_code = params['translation_context'], params['tts_model_path'], params['lang_code']
    logging.info(f"--- Dubbing Workflow Indítása: '{project_name}' ---")
    
    steps = [
        ([sync_python, 'scripts/extract_audio_easy.py', project_name], "[extract_audio]", True),
        ([sync_python, 'scripts/separate_audio_easy.py', '-p', project_name], "[separate_audio]", True),
        ([sync_python, 'scripts/unpack_srt_from_mkv_easy.py', project_name], "[unpack_srt]", False),
        ([parakeet_python, 'scripts/Nvidia_asr_eng/parakeet_transcribe_wordts_4.0_easy.py', '-p', project_name], "[parakeet_transcribe]", True),
        ([sync_python, 'scripts/translate_chatgpt_srt_easy.py', '-project_name', project_name, '-context', translation_context], "[translate_chatgpt]", True),
        ([tts_python, 'scripts/f5_tts_easy9_kisérleti.py', project_name, '--phonetic-ref', '--norm', 'hun', '--save-failures', '--double-ref-on-failure', '--model_dir', tts_path], "[f5_tts_run1]", False),
        ([tts_python, 'scripts/f5_tts_easy9_kisérleti.py', project_name, '--phonetic-ref', '--norm', 'hun', '--save-failures', '--double-ref-on-failure', '--model_dir', tts_path], "[f5_tts_run2]", False),
        ([tts_python, 'scripts/f5_tts_easy9_kisérleti.py', project_name, '--phonetic-ref', '--norm', 'hun', '--save-failures', '--double-ref-on-failure', '--normalize-ref-audio', '--model_dir', tts_path], "[f5_tts_run3]", True),
        ([sync_python, 'scripts/normalise_and_cut_json_easy2.py', project_name], "[normalise_and_cut]", True),
        ([sync_python, 'scripts/merge_chunks_with_background_easy.py', project_name], "[merge_chunks]", True),
        ([sync_python, 'scripts/merge_to_video_easy.py', project_name, '-lang', lang_code], "[merge_to_video]", True),
    ]

    for cmd, prefix, is_critical in steps:
        if not run_command(cmd, log_prefix=prefix):
            if is_critical:
                logging.critical(f"{prefix} Kritikus lépés sikertelen. '{project_name}' feldolgozása leáll.")
                memory_handler.flush()
                return False
        
        memory_handler.flush()
        
        if prefix in ("[f5_tts_run1]", "[f5_tts_run2]"):
            logging.info(">>> Várakozás 5 másodpercet a GPU memória felszabadulásáért... <<<")
            time.sleep(5)

    logging.info(f"--- Dubbing Workflow Sikeresen Befejeződött: '{project_name}' ---")
    return True

def copy_results_to_source(project_path, source_video_path, config):
    try:
        logging.info("Eredmények visszamásolása a forráskönyvtárba...")
        destination_dir = os.path.join(os.path.dirname(source_video_path), "AI_szinkron")
        os.makedirs(destination_dir, exist_ok=True)
        project_download_dir = os.path.join(project_path, config['PROJECT_SUBDIRS']['download'])
        if os.path.isdir(project_download_dir):
            for item in os.listdir(project_download_dir):
                if item.lower().endswith(VIDEO_EXTENSIONS): shutil.copy2(os.path.join(project_download_dir, item), destination_dir)
        project_upload_dir = os.path.join(project_path, config['PROJECT_SUBDIRS']['upload'])
        if os.path.isdir(project_upload_dir):
            for item in os.listdir(project_upload_dir):
                if item.lower().endswith('.srt'): shutil.copy2(os.path.join(project_upload_dir, item), destination_dir)
        logging.info("Visszamásolás sikeres.")
    except Exception as e:
        logging.error(f"Hiba az eredmények visszamásolása közben: {e}", exc_info=True)

def get_input_with_default(prompt, default_value):
    prompt_text = f"{prompt} [Alapértelmezett: {default_value}] (Enter): " if default_value else f"{prompt}: "
    user_input = input(prompt_text)
    return default_value if user_input == "" and default_value else user_input

def select_from_list(options, prompt_message, default_value=None):
    if not options: logging.error(f"Nincsenek opciók: {prompt_message}"); return None
    print(prompt_message)
    default_index = -1
    for i, option in enumerate(options, 1):
        if option == default_value: print(f"  {i}) {option}  <-- Alapértelmezett"); default_index = i
        else: print(f"  {i}) {option}")
    prompt = f"Válasszon (1-{len(options)}) [Enter = {default_index}]: " if default_index != -1 else f"Válasszon (1-{len(options)}): "
    while True:
        try:
            choice_input = input(prompt)
            if choice_input == "" and default_index != -1: return options[default_index - 1]
            choice = int(choice_input)
            if 1 <= choice <= len(options): return options[choice - 1]
            else: print("Érvénytelen szám.")
        except ValueError: print("Érvénytelen bemenet.")

def main():
    parser = argparse.ArgumentParser(description="Videó szinkronizálási vezérlő.", formatter_class=argparse.RawTextHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--project_name', type=str, help="Projekt neve (egyszeri futtatás).")
    group.add_argument('-i', '--input_dir', type=str, help="Bemeneti könyvtár videókkal (kötegelt futtatás).")
    parser.add_argument('--auto', action='store_true', help="Automatikus mód: paraméterek olvasása a config.json-ból.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    
    if get_conda_info() is None:
        logging.critical("Nem sikerült lekérdezni a Conda információkat. A program leáll."); sys.exit(1)

    config = load_config()
    workdir = config['DIRECTORIES']['workdir']
    
    if args.auto:
        logging.info("Automatikus mód aktív. Paraméterek betöltése.")
        params = config.get("LAST_USED_PARAMS", {})
        if not all(k in params for k in ["translation_context", "tts_model_path", "lang_code"]):
            logging.error("Nincsenek elmentett paraméterek! Futtasd interaktívan (--auto nélkül)."); sys.exit(1)
    else:
        params = {}
    
    if args.input_dir:
        video_files = [os.path.join(r, f) for r, _, fs in os.walk(args.input_dir) for f in fs if f.lower().endswith(VIDEO_EXTENSIONS)]
        if not video_files: logging.warning("Nem található videófájl."); sys.exit(0)
        
        logging.info(f"Kötegelt feldolgozás: '{args.input_dir}'. Összesen {len(video_files)} videó.")
        for i, video_path in enumerate(video_files):
            project_name = os.path.splitext(os.path.basename(video_path))[0]
            logging.info("\n" + "="*60 + f"\nFeldolgozás ({i+1}/{len(video_files)}): '{project_name}'\nForrás: {video_path}")
            project_path = os.path.join(workdir, project_name)
            if os.path.exists(project_path): logging.warning(f"'{project_name}' projekt már létezik, kihagyva."); continue
            
            memory_handler, file_handler = setup_logging(project_path, config['PROJECT_SUBDIRS']['logs'], project_name)
            try:
                create_project_structure(project_path, config['PROJECT_SUBDIRS'])
                shutil.copy2(video_path, os.path.join(project_path, config['PROJECT_SUBDIRS']['upload']))
                if run_dubbing_workflow(project_name, config, params, memory_handler):
                    copy_results_to_source(project_path, video_path, config)
                else:
                    logging.error(f"Hiba '{project_name}' feldolgozása közben. Folytatás a következővel.")
            except Exception as e:
                logging.critical(f"Váratlan hiba '{project_name}' feldolgozása közben: {e}", exc_info=True)
            finally:
                memory_handler.flush()
                logging.getLogger().removeHandler(memory_handler)
                file_handler.close()
        logging.info("\n" + "="*60 + "\nA kötegelt feldolgozás befejeződött.")

    else:
        project_name = args.project_name
        project_path = os.path.join(workdir, project_name)
        if os.path.exists(project_path): logging.error(f"'{project_name}' projekt már létezik."); sys.exit(1)

        memory_handler, file_handler = setup_logging(project_path, config['PROJECT_SUBDIRS']['logs'], project_name)
        try:
            create_project_structure(project_path, config['PROJECT_SUBDIRS'])
            print(f"\nKérem, másolja a videót ide: {os.path.abspath(os.path.join(project_path, config['PROJECT_SUBDIRS']['upload']))}")
            input("Nyomjon Entert, ha végzett...")
            
            if not args.auto:
                last_params = config.get("LAST_USED_PARAMS", {})
                translation_context = get_input_with_default("Fordítási kontextus", last_params.get("translation_context"))
                tts_base_dir = config['DIRECTORIES']['TTS']
                available_models = [d for d in os.listdir(tts_base_dir) if os.path.isdir(os.path.join(tts_base_dir, d))]
                last_model_name = os.path.basename(last_params.get("tts_model_path", ""))
                selected_model = select_from_list(available_models, "\nTTS modell:", default_value=last_model_name)
                if not selected_model: logging.error("TTS modell választása kötelező!"); sys.exit(1)
                lang_code = select_from_list(["hun", "eng", "ger", "fra"], "\nNyelvi kód:", default_value=last_params.get("lang_code"))
                if not lang_code: logging.error("Nyelvi kód választása kötelező!"); sys.exit(1)
                params = {
                    "translation_context": translation_context,
                    "tts_model_path": os.path.join(tts_base_dir, selected_model),
                    "lang_code": lang_code
                }
                config["LAST_USED_PARAMS"] = params
                save_config(config)

            if run_dubbing_workflow(project_name, config, params, memory_handler):
                 logging.info("Az egyszeri futtatás sikeresen befejeződött.")
            else:
                 logging.error("Az egyszeri futtatás hibával zárult.")
        finally:
            memory_handler.flush()
            logging.getLogger().removeHandler(memory_handler)
            file_handler.close()

if __name__ == "__main__":
    main()

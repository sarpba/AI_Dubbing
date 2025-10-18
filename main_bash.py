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
import base64
import binascii
import getpass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable, Union, Any

from tools.debug_utils import add_debug_argument, configure_debug_mode

VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.webm')

_CONDA_INFO_CACHE: Optional[dict] = None
KEYHOLDER_PATH = Path(__file__).resolve().parent / "keyholder.json"


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


def run_command(
    command_list: List[str],
    log_prefix: str = "",
    timeout: Optional[float] = None,
    env: Optional[dict] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Union[bool, str]:
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
                if should_stop and should_stop():
                    logging.warning(f"{log_prefix} Megszakítási kérés érkezett. A parancs leállítása...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return "cancelled"
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


def load_keyholder_data() -> Dict[str, Any]:
    if not KEYHOLDER_PATH.exists():
        return {}
    try:
        return json.loads(KEYHOLDER_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logging.warning("Nem sikerült beolvasni a keyholder.json fájlt, új tartalom kerül mentésre.")
    except OSError as exc:
        logging.error(f"Nem olvasható a keyholder.json fájl: {exc}")
    return {}


def save_keyholder_data(data: Dict[str, Any]) -> None:
    try:
        KEYHOLDER_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("API kulcsok frissítve a keyholder.json fájlban.")
    except OSError as exc:
        logging.error(f"Nem sikerült elmenteni a keyholder.json fájlt: {exc}")


def decode_key_value(encoded_value: Optional[str]) -> Optional[str]:
    if not encoded_value:
        return None
    encoded_value = encoded_value.strip()
    if not encoded_value:
        return None
    try:
        decoded = base64.b64decode(encoded_value).decode("utf-8")
        return decoded.strip() or None
    except (binascii.Error, UnicodeDecodeError):
        return encoded_value


def prompt_for_secret(prompt_text: str) -> str:
    while True:
        try:
            value = getpass.getpass(prompt_text)
        except KeyboardInterrupt:
            print("\nMegszakítva.")
            sys.exit(1)
        value = value.strip()
        if value:
            return value
        print("A kulcs megadása kötelező. Próbáld újra.")


def ensure_api_key(
    field_names: Tuple[str, ...],
    env_vars: Tuple[str, ...],
    prompt_text: str,
    *,
    keyholder_data: Dict[str, Any],
    required: bool = True,
    store_base64: bool = True,
) -> Tuple[Optional[str], bool]:
    primary_field = field_names[0]

    # Már tárolt kulcs ellenőrzése
    for field in field_names:
        stored_value = decode_key_value(keyholder_data.get(field))
        if stored_value:
            if field != primary_field:
                keyholder_data[primary_field] = keyholder_data[field]
                keyholder_data.pop(field, None)
                return stored_value, True
            return stored_value, False

    # Környezeti változók vizsgálata
    for env_var in env_vars:
        env_value = os.getenv(env_var)
        if env_value and env_value.strip():
            value = env_value.strip()
            encoded_value = (
                base64.b64encode(value.encode("utf-8")).decode("utf-8") if store_base64 else value
            )
            keyholder_data[primary_field] = encoded_value
            for alias in field_names[1:]:
                keyholder_data.pop(alias, None)
            logging.info(f"{env_var} környezeti változóból töltött API kulcs elmentve a keyholder.json fájlba.")
            return value, True

    if not required:
        return None, False

    # Felhasználótól bekérés, ha nincs kulcs
    value = prompt_for_secret(prompt_text if prompt_text.endswith(": ") else f"{prompt_text}: ")
    encoded_value = (
        base64.b64encode(value.encode("utf-8")).decode("utf-8") if store_base64 else value
    )
    keyholder_data[primary_field] = encoded_value
    for alias in field_names[1:]:
        keyholder_data.pop(alias, None)
    return value, True


def has_target_language_srt(upload_dir: str, target_lang: str) -> bool:
    """
    Ellenőrzi, hogy van-e célnyelvi SRT fájl az upload könyvtárban.
    A translate_chatgpt script logikáját követve a '*{lang}.srt' mintára keresünk.
    """
    if not upload_dir or not os.path.isdir(upload_dir):
        return False
    target_lang = (target_lang or "").strip().lower()
    if not target_lang:
        return False
    pattern_suffix = f"{target_lang}.srt"
    try:
        return any(filename.lower().endswith(pattern_suffix) for filename in os.listdir(upload_dir))
    except OSError:
        return False



def gpu_cooldown_wait(seconds: int = 5) -> None:
    logging.info(f">>> Várakozás {seconds} másodpercet a GPU memória felszabadulásáért... <<<")
    time.sleep(seconds)


def run_dubbing_workflow(
    project_name: str,
    project_path: str,
    config: dict,
    params: dict,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Union[bool, str]:
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

    config_defaults = config.get('CONFIG', {}) if isinstance(config.get('CONFIG', {}), dict) else {}
    default_source_lang = str(config_defaults.get('default_source_lang', 'en') or 'en').strip()
    default_target_lang = str(config_defaults.get('default_target_lang', 'hu') or 'hu').strip()

    target_lang_for_srt = default_target_lang.lower() if default_target_lang else 'hu'
    source_lang_for_deepl = default_source_lang.upper() if default_source_lang else 'EN'
    target_lang_for_deepl = default_target_lang.upper() if default_target_lang else 'HU'

    candidate_lang_codes: List[str] = []
    if target_lang_for_srt:
        candidate_lang_codes.append(target_lang_for_srt)
    lang_code_for_srt = (lang_code or '').strip().lower()
    if lang_code_for_srt and lang_code_for_srt not in candidate_lang_codes:
        candidate_lang_codes.append(lang_code_for_srt)

    keyholder_data = load_keyholder_data()
    keyholder_dirty = False
    deepl_key: Optional[str] = None

    has_target_srt = any(has_target_language_srt(upload_dir, code) for code in candidate_lang_codes)
    if has_target_srt:
        logging.info("Célnyelvi SRT fájl elérhető, az igazítás a ChatGPT alapú szkripttel történik.")
        _, changed = ensure_api_key(
            ("chatgpt_api_key",),
            ("OPENAI_API_KEY", "CHATGPT_API_KEY"),
            "Add meg az OpenAI (ChatGPT) API kulcsot",
            keyholder_data=keyholder_data,
        )
        keyholder_dirty |= changed
        translation_step = (
            [sync_python, 'scripts/translate_chatgpt_srt_easy_codex.py', '-project_name', project_name, '-context', translation_context],
            "[translate_chatgpt]",
            True,
            lambda: skip_translate(translated_dir)
        )
    else:
        logging.info("Célnyelvi SRT fájl nem található, DeepL alapú fordításra váltunk.")
        deepl_key, changed = ensure_api_key(
            ("deepL_api_key", "deepl_api_key"),
            ("DEEPL_API_KEY", "DEEPL_AUTH_KEY"),
            "Add meg a DeepL API kulcsot",
            keyholder_data=keyholder_data,
        )
        keyholder_dirty |= changed
        if not deepl_key:
            logging.critical("DeepL API kulcs nem áll rendelkezésre. A fordítás folytatásához add meg a kulcsot a keyholder.json fájlban vagy a DEEPL_API_KEY környezeti változóban.")
            return False
        translation_step = (
            [
                sync_python,
                'scripts/translate.py',
                '-input_dir', speech_dir,
                '-output_dir', translated_dir,
                '-input_language', source_lang_for_deepl,
                '-output_language', target_lang_for_deepl,
                '-auth_key', deepl_key,
            ],
            "[translate_deepl]",
            True,
            lambda: skip_translate(translated_dir)
        )

    if keyholder_dirty:
        save_keyholder_data(keyholder_data)

    Step = Tuple[List[str], str, bool, Optional[Callable[[], bool]]]
    steps: List[Step] = [
        ([sync_python, 'scripts/extract_audio_easy_channels.py', project_name], "[extract_audio]", True, None),
        ([sync_python, 'scripts/separate_audio_easy_codex.py', '-p', project_name], "[separate_audio]", True, lambda: skip_separate_audio(speech_dir, background_dir)),
        ([sync_python, 'scripts/unpack_srt_from_mkv_easy.py', project_name], "[unpack_srt]", False, lambda: skip_unpack_srt(upload_dir)),
        ([parakeet_python, 'scripts/Nvidia_asr_eng/parakeet_transcribe_wordts_4.0_easy.py', '-p', project_name], "[parakeet_transcribe]", True, lambda: skip_parakeet_transcribe(speech_dir)),
        ([sync_python, 'scripts/split_segments_by_speaker_codex.py', '-p', project_name], "[split_by_speaker]", True, lambda: skip_split_by_speaker(speech_dir)),
        translation_step,
        ([tts_python, 'scripts/f5_tts_easy_codex_EQ.py', project_name, '--phonetic-ref', '--norm', 'hun', '--model_dir', tts_path], "[f5_tts_run1]", False, None),
        ([tts_python, 'scripts/f5_tts_easy_codex_EQ.py', project_name, '--phonetic-ref', '--norm', 'hun', '--model_dir', tts_path], "[f5_tts_run2]", False, None),
        ([tts_python, 'scripts/f5_tts_easy_codex_EQ.py', project_name, '--phonetic-ref', '--norm', 'hun', '--save-failures', '--normalize-ref-audio', '--model_dir', tts_path], "[f5_tts_run3]", True, None),
        ([tts_python, 'scripts/f5_tts_easy_codex_EQ.py', project_name, '--phonetic-ref', '--norm', 'hun', '--save-failures', '--normalize-ref-audio', '--keep-best-over-tolerance', '--model_dir', tts_path_run4], "[f5_tts_run4]", True, None),
        ([sync_python, 'scripts/normalise_and_cut_json_easy_codex.py', project_name], "[normalise_and_cut]", True, None),
        ([sync_python, 'scripts/merge_chunks_with_background_easy.py', project_name], "[merge_chunks]", True, None),
        ([sync_python, 'scripts/merge_to_video_easy.py', project_name, '-lang', lang_code], "[merge_to_video]", True, None),
    ]

    for cmd, prefix, is_critical, skip_fn in steps:
        if should_stop and should_stop():
            logging.warning(f"{prefix} Megszakítási kérés érkezett. A workflow leáll.")
            return "cancelled"
        if skip_fn:
            try:
                if skip_fn():
                    logging.info(f"{prefix} Kihagyva (kimeneti fájlok jelen vannak).")
                    continue
            except Exception as e:
                logging.warning(f"{prefix} Skip feltétel ellenőrzése közben hiba: {e}")

        result = run_command(cmd, log_prefix=prefix, should_stop=should_stop)
        if result == "cancelled":
            logging.warning(f"{prefix} Megszakítva a felhasználó kérésére.")
            return "cancelled"
        if result is False:
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
    add_debug_argument(parser)
    args = parser.parse_args()

    log_level = configure_debug_mode(args.debug)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger().setLevel(log_level)

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

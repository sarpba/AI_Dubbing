from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
import json
import subprocess
import base64
import binascii
import logging
import threading
import uuid
import copy
import re
import tempfile
import glob
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import shutil
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import wave
import math
from collections import OrderedDict

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

config_lock = threading.Lock()
workflow_lock = threading.Lock()
theme_config_lock = threading.Lock()
workflow_jobs = {}
workflow_threads = {}
workflow_events = {}

KEYHOLDER_PATH = os.path.join(app.root_path, 'keyholder.json')
CONDA_PYTHON_CACHE = {}

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'}
VIDEO_EXTENSIONS = {
    '.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv', '.flv', '.mts', '.m2ts', '.mpg', '.mpeg'
}


def derive_project_prefix(name: str) -> str:
    tokens = [part for part in re.split(r'[._-]+', name) if part]
    if not tokens:
        return name or 'Egyéb'
    prefix = tokens[0]
    if len(tokens) >= 2 and len(prefix) <= 3:
        prefix = f"{prefix}_{tokens[1]}"
    return prefix


def build_project_entries(projects: List[str], group_threshold: int = 3) -> List[Dict[str, Any]]:
    grouped: "OrderedDict[str, List[str]]" = OrderedDict()
    for project in projects:
        key = derive_project_prefix(project)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(project)

    entries: List[Dict[str, Any]] = []
    for key, names in grouped.items():
        if len(names) >= group_threshold:
            entries.append({
                'type': 'group',
                'key': key,
                'projects': names,
                'count': len(names)
            })
        else:
            for name in names:
                entries.append({
                    'type': 'project',
                    'name': name
                })
    return entries

AUDIO_MIME_MAP = {
    '.wav': 'audio/wav',
    '.mp3': 'audio/mpeg',
    '.ogg': 'audio/ogg',
    '.flac': 'audio/flac',
    '.m4a': 'audio/mp4',
    '.aac': 'audio/aac'
}
VIDEO_MIME_MAP = {
    '.mp4': 'video/mp4',
    '.mkv': 'video/x-matroska',
    '.avi': 'video/x-msvideo',
    '.mov': 'video/quicktime',
    '.webm': 'video/webm',
    '.wmv': 'video/x-ms-wmv',
    '.flv': 'video/x-flv',
    '.mts': 'video/mp2t',
    '.m2ts': 'video/mp2t',
    '.mpg': 'video/mpeg',
    '.mpeg': 'video/mpeg'
}

SCRIPTS_DIR = Path(app.root_path) / 'scripts'
SCRIPTS_CONFIG_PATH = SCRIPTS_DIR / 'scripts.json'
SCRIPTS_CACHE: Dict[str, Any] = {'mtime': None, 'data': []}
SCRIPTS_CACHE_LOCK = threading.Lock()
CONDA_INFO_CACHE: Optional[dict] = None
CONDA_INFO_LOCK = threading.Lock()
WORKFLOWS_DIR = Path(app.root_path) / 'workflows'
WORKFLOW_STATE_FILENAME = 'workflow_state.json'
THEME_CONFIG_PATH = Path(app.root_path) / 'config' / 'theme_colors.json'
THEME_COLOR_KEYS = [
    'primary-color',
    'secondary-color',
    'success-color',
    'background-color',
    'text-color',
    'card-bg',
    'border-color',
    'waveform-bg',
    'timeline-segment-bg',
]
DEFAULT_THEME_COLORS = {
    'light': {
        'primary-color': '#0d6efd',
        'secondary-color': '#6c757d',
        'success-color': '#198754',
        'background-color': '#ffffff',
        'text-color': '#212529',
        'card-bg': '#f8f9fa',
        'border-color': '#dee2e6',
        'waveform-bg': '#f0f0f0',
        'timeline-segment-bg': 'rgba(230, 230, 250, 0.8)',
    },
    'dark': {
        'primary-color': '#0dcaf0',
        'secondary-color': '#6c757d',
        'success-color': '#198754',
        'background-color': '#212529',
        'text-color': '#f8f9fa',
        'card-bg': '#2d3339',
        'border-color': '#495057',
        'waveform-bg': '#343a40',
        'timeline-segment-bg': 'rgba(100, 100, 150, 0.8)',
    },
}

SCRIPT_KEY_REQUIREMENTS = {
    'translate_chatgpt_srt_easy_codex.py': {'chatgpt'},
    'translate.py': {'deepl'},
    'split_segments_by_speaker_codex.py': {'huggingface'},
    'whisx.py': {'huggingface'},
}

SCRIPT_PARAM_KEYHOLDER = {
    'translate_chatgpt_srt_easy_codex.py': {'auth_key': ('chatgpt_api_key', 'api_key')},
    'translate.py': {'auth_key': ('deepL_api_key', 'deepl_api_key')},
    'split_segments_by_speaker_codex.py': {'hf_token': ('hf_token',)},
    'whisx.py': {'hf_token': ('hf_token',)},
}

PROJECT_AUTOFILL_OVERRIDES = {
    'project_name': 'project_name',
    'project': 'project_name',
    'project_dir_name': 'project_name',
    'project_dir': 'project_path',
    'project_path': 'project_path',
}
SECRET_PARAM_NAMES = {'auth_key', 'hf_token'}
ENCODED_SECRET_PREFIX = 'base64:'
SECRET_VALUE_PLACEHOLDER = '***'
ALLOWED_WORKFLOW_WIDGETS = {'reviewContinue', 'cycleWidget'}


class WorkflowValidationError(Exception):
    """Egy workflow lépés konfigurációja érvénytelen."""


def _normalize_theme_colors(data: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, str]]:
    normalized: Dict[str, Dict[str, str]] = {}
    for mode, defaults in DEFAULT_THEME_COLORS.items():
        normalized[mode] = {}
        mode_values = data.get(mode) if isinstance(data, dict) else {}
        mode_values = mode_values if isinstance(mode_values, dict) else {}
        for key, default_value in defaults.items():
            value = mode_values.get(key)
            if isinstance(value, str):
                value = value.strip() or default_value
            else:
                value = default_value
            normalized[mode][key] = value
    return normalized


def load_theme_colors() -> Dict[str, Dict[str, str]]:
    with theme_config_lock:
        if THEME_CONFIG_PATH.exists():
            try:
                with open(THEME_CONFIG_PATH, 'r', encoding='utf-8') as file:
                    raw_data = json.load(file)
            except (OSError, json.JSONDecodeError) as exc:
                logging.warning("Nem sikerült beolvasni a témaszíneket: %s", exc)
                raw_data = None
        else:
            raw_data = None
        return _normalize_theme_colors(raw_data)


def save_theme_colors(data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    normalized = _normalize_theme_colors(data)
    with theme_config_lock:
        try:
            THEME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(THEME_CONFIG_PATH, 'w', encoding='utf-8') as file:
                json.dump(normalized, file, ensure_ascii=False, indent=2)
        except OSError as exc:
            logging.error("Nem sikerült elmenteni a témaszíneket: %s", exc)
            raise
    return normalized


@app.context_processor
def inject_theme_colors():
    try:
        colors = load_theme_colors()
    except Exception as exc:
        logging.error("Nem sikerült betölteni a témaszíneket: %s", exc)
        colors = DEFAULT_THEME_COLORS
    return {
        'theme_colors': colors,
        'default_theme_colors': DEFAULT_THEME_COLORS,
    }


def is_secret_param(name: str) -> bool:
    return name in SECRET_PARAM_NAMES


def mask_workflow_secret_params(steps: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    masked_steps: List[Dict[str, Any]] = copy.deepcopy(steps or [])
    for step in masked_steps:
        params = step.get('params')
        if not isinstance(params, dict):
            continue
        for key, value in list(params.items()):
            if not is_secret_param(key):
                continue
            if isinstance(value, str):
                if value.startswith(ENCODED_SECRET_PREFIX):
                    continue
                encoded_value = encode_keyholder_value(value)
                if encoded_value:
                    params[key] = f"{ENCODED_SECRET_PREFIX}{encoded_value}"
                else:
                    params.pop(key, None)
            elif value is None:
                params.pop(key, None)
    return masked_steps


def unmask_secret_param_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(ENCODED_SECRET_PREFIX):
        decoded = decode_keyholder_value(value[len(ENCODED_SECRET_PREFIX):])
        if decoded is not None:
            return decoded
    return value


def mask_applied_params_for_ui(applied_params: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    masked: List[Dict[str, Any]] = []
    for param in applied_params or []:
        entry = param.copy()
        if is_secret_param(entry.get('name')):
            entry['value'] = SECRET_VALUE_PLACEHOLDER
        masked.append(entry)
    return masked


def mask_command_for_ui(command: Optional[List[str]], applied_params: Optional[List[Dict[str, Any]]], script_meta: Dict[str, Any]) -> List[str]:
    masked = list(command or [])
    secret_flags: Set[str] = set()
    for param_meta in script_meta.get('parameters', []):
        if is_secret_param(param_meta.get('name')):
            for flag in param_meta.get('flags') or []:
                secret_flags.add(flag)
    for idx, token in enumerate(masked[:-1]):
        if token in secret_flags:
            masked[idx + 1] = SECRET_VALUE_PLACEHOLDER
    secret_values = {
        str(param.get('value'))
        for param in applied_params or []
        if is_secret_param(param.get('name')) and param.get('value') is not None
    }
    for idx, token in enumerate(masked):
        if token in secret_values:
            masked[idx] = SECRET_VALUE_PLACEHOLDER
    return masked


def build_masked_command_and_params(
    command: Optional[List[str]],
    applied_params: Optional[List[Dict[str, Any]]],
    script_meta: Dict[str, Any]
) -> Tuple[List[str], List[Dict[str, Any]]]:
    return (
        mask_command_for_ui(command, applied_params, script_meta),
        mask_applied_params_for_ui(applied_params),
    )


def resolve_workspace_path(path_value):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)
    return os.path.abspath(os.path.join(app.root_path, path_value))


def is_subpath(child_path, parent_path):
    try:
        return os.path.commonpath([child_path, parent_path]) == os.path.commonpath([parent_path])
    except ValueError:
        return False


def collect_directory_entries(
    root_path: str,
    target_path: str,
    metadata_directories: Optional[Set[str]] = None,
    highlight_map: Optional[Dict[str, str]] = None,
    failed_generation_directories: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    highlight_map = highlight_map or {}
    metadata_directories = metadata_directories or set()
    failed_generation_directories = failed_generation_directories or set()
    try:
        for name in sorted(os.listdir(target_path)):
            if name.startswith('.'):
                continue
            full_path = os.path.join(target_path, name)
            rel_path = os.path.relpath(full_path, root_path).replace('\\', '/')
            if os.path.isdir(full_path):
                entry = {
                    'name': name,
                    'type': 'directory',
                    'path': rel_path
                }
                highlight_class = highlight_map.get(rel_path)
                if highlight_class:
                    entry['highlight_class'] = highlight_class
                entries.append(entry)
            else:
                file_entry = {
                    'name': name,
                    'type': 'file',
                    'path': rel_path
                }
                if should_enable_failed_move(rel_path, highlight_map):
                    file_entry['enable_failed_move'] = True
                metadata = build_audio_metadata(full_path, rel_path, metadata_directories)
                if metadata:
                    file_entry.update(metadata)
                failed_metadata = build_failed_generation_json_metadata(
                    full_path,
                    rel_path,
                    failed_generation_directories
                )
                if failed_metadata:
                    file_entry.update(failed_metadata)
                entries.append(file_entry)
    except Exception as exc:
        logging.warning("Nem sikerült beolvasni a(z) %s könyvtárat: %s", target_path, exc)
    return entries


def compute_failed_generation_highlights(
    project_dir: str,
    config_snapshot: Dict[str, Any]
) -> Dict[str, str]:
    """
    Térképet készít a failed_generations almappáihoz, amelyeket ki kell emelni.
    Azokat a könyvtárakat jelöljük, amelyekhez létezik azonos bázisnevű fájl
    a translated_splits mappában.
    """
    highlight_map: Dict[str, str] = {}
    failed_root = os.path.join(project_dir, 'failed_generations')
    translated_rel = config_snapshot.get('PROJECT_SUBDIRS', {}).get('translated_splits')
    if not translated_rel:
        return highlight_map

    translated_root = os.path.join(project_dir, translated_rel)
    if not os.path.isdir(failed_root) or not os.path.isdir(translated_root):
        return highlight_map

    translated_basenames: Set[str] = set()
    try:
        for name in os.listdir(translated_root):
            translated_path = os.path.join(translated_root, name)
            if os.path.isdir(translated_path):
                continue
            stem, _ = os.path.splitext(name)
            if stem:
                translated_basenames.add(stem)
    except Exception as exc:
        logging.warning(
            "Nem sikerült beolvasni a translated_splits mappát (%s): %s",
            translated_root,
            exc
        )
        return highlight_map

    try:
        for name in os.listdir(failed_root):
            failed_path = os.path.join(failed_root, name)
            if not os.path.isdir(failed_path):
                continue
            if name not in translated_basenames:
                continue
            rel_path = os.path.relpath(failed_path, project_dir).replace('\\', '/')
            highlight_map[rel_path] = 'fg-has-translated'
    except Exception as exc:
        logging.warning(
            "Nem sikerült beolvasni a failed_generations mappát (%s): %s",
            failed_root,
            exc
        )

    return highlight_map


def should_enable_failed_move(rel_path: str, highlight_map: Dict[str, str]) -> bool:
    if not rel_path:
        return False
    if not rel_path.startswith('failed_generations/'):
        return False
    parent_path = rel_path.rsplit('/', 1)[0] if '/' in rel_path else ''
    if not parent_path:
        return False
    if highlight_map.get(parent_path) == 'fg-has-translated':
        return False
    _, ext = os.path.splitext(rel_path)
    return ext.lower() == '.wav'


FILENAME_RANGE_PATTERN = re.compile(
    r'^(\d{2}-\d{2}-\d{2}-\d{3})_(\d{2}-\d{2}-\d{2}-\d{3})'
)


def get_audio_metadata_directories(config_snapshot: Dict[str, Any]) -> Set[str]:
    project_subdirs = config_snapshot.get('PROJECT_SUBDIRS', {}) if isinstance(config_snapshot, dict) else {}
    candidates = {
        'translated_splits',
        'failed_generations',
        project_subdirs.get('translated_splits') or '',
        project_subdirs.get('failed_generations') or '',
    }
    return {value for value in candidates if isinstance(value, str) and value.strip()}


def should_collect_audio_metadata(rel_path: str, metadata_directories: Set[str]) -> bool:
    if not rel_path or not metadata_directories:
        return False
    normalized = rel_path.replace('\\', '/')
    segments = [segment for segment in normalized.split('/') if segment]
    if len(segments) <= 1:
        return False
    parent_segments = segments[:-1]
    return any(segment in metadata_directories for segment in parent_segments)


def parse_timestamp_to_seconds(value: str) -> Optional[float]:
    if not value:
        return None
    parts = value.split('-')
    if len(parts) != 4:
        return None
    try:
        hours, minutes, seconds, milliseconds = (int(part) for part in parts)
    except ValueError:
        return None
    total_seconds = (hours * 3600) + (minutes * 60) + seconds + (milliseconds / 1000.0)
    return max(total_seconds, 0.0)


def compute_duration_from_filename(filename: str) -> Optional[float]:
    if not filename:
        return None
    stem, _ = os.path.splitext(filename)
    match = FILENAME_RANGE_PATTERN.match(stem or '')
    if not match:
        return None
    start_seconds = parse_timestamp_to_seconds(match.group(1))
    end_seconds = parse_timestamp_to_seconds(match.group(2))
    if start_seconds is None or end_seconds is None:
        return None
    computed = end_seconds - start_seconds
    if computed < 0:
        return None
    return computed


def read_wav_duration_seconds(file_path: str) -> Optional[float]:
    try:
        with wave.open(file_path, 'rb') as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            if not frame_rate:
                return None
            duration = frame_count / float(frame_rate)
            if math.isfinite(duration) and duration >= 0:
                return duration
    except (wave.Error, OSError) as exc:
        logging.debug("Nem sikerült kiolvasni a wav időtartamot (%s): %s", file_path, exc)
    return None


def format_seconds_hundredths(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    rounded = round(max(value, 0) + 1e-9, 2)
    return f"{rounded:.2f}"


def build_audio_metadata(
    full_path: str,
    rel_path: str,
    metadata_directories: Set[str]
) -> Dict[str, Any]:
    _, ext = os.path.splitext(full_path)
    if ext.lower() != '.wav':
        return {}
    if not should_collect_audio_metadata(rel_path, metadata_directories):
        return {}
    filename = os.path.basename(rel_path)
    computed_seconds = compute_duration_from_filename(filename)
    actual_seconds = read_wav_duration_seconds(full_path)
    display_left = format_seconds_hundredths(computed_seconds) or '--'
    display_right = format_seconds_hundredths(actual_seconds) or '--'
    display_value = f"{display_left} / {display_right}"
    return {
        'duration_from_name': computed_seconds,
        'duration_actual': actual_seconds,
        'duration_display': display_value
    }


def get_failed_generation_directories(config_snapshot: Dict[str, Any]) -> Set[str]:
    project_subdirs = config_snapshot.get('PROJECT_SUBDIRS', {}) if isinstance(config_snapshot, dict) else {}
    candidates = {
        'failed_generations',
        project_subdirs.get('failed_generations') or '',
    }
    return {value for value in candidates if isinstance(value, str) and value.strip()}


def should_collect_failed_generation_text(rel_path: str, failed_directories: Set[str]) -> bool:
    if not rel_path or not failed_directories:
        return False
    normalized = rel_path.replace('\\', '/')
    segments = [segment for segment in normalized.split('/') if segment]
    if len(segments) <= 1:
        return False
    parent_segments = segments[:-1]
    return any(segment in failed_directories for segment in parent_segments)


def build_failed_generation_json_metadata(
    full_path: str,
    rel_path: str,
    failed_directories: Set[str]
) -> Dict[str, Any]:
    _, ext = os.path.splitext(full_path)
    if ext.lower() != '.json':
        return {}
    if os.path.basename(rel_path) != 'info.json':
        return {}
    if not should_collect_failed_generation_text(rel_path, failed_directories):
        return {}
    try:
        with open(full_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        logging.debug("Nem sikerült beolvasni az info.json fájlt (%s): %s", full_path, exc)
        return {}
    def extract_text(source: Dict[str, Any], primary: str, secondary: str) -> Optional[str]:
        primary_value = source.get(primary)
        if isinstance(primary_value, str) and primary_value.strip():
            return primary_value
        secondary_value = source.get(secondary)
        if isinstance(secondary_value, str) and secondary_value.strip():
            return secondary_value
        return None

    value = extract_text(data, 'original_text', 'gen_text')
    if value is None:
        failures = data.get('failures')
        if isinstance(failures, list):
            for failure in failures:
                if isinstance(failure, dict):
                    candidate = extract_text(failure, 'original_text', 'gen_text')
                    if isinstance(candidate, str):
                        value = candidate
                        break
    if not isinstance(value, str):
        return {}
    display_value = ' '.join(value.split())
    if not display_value:
        return {}
    return {
        'failed_original_text': value,
        'failed_original_text_display': display_value
    }


def infer_autofill_kind(param_name: str) -> Optional[str]:
    if not param_name:
        return None
    normalized = param_name.strip().lower()
    if normalized in PROJECT_AUTOFILL_OVERRIDES:
        return PROJECT_AUTOFILL_OVERRIDES[normalized]
    if 'project' in normalized:
        if 'dir' in normalized or 'path' in normalized:
            return 'project_path'
        return 'project_name'
    return None


def load_scripts_file() -> List[Dict[str, Any]]:
    entries = rebuild_scripts_config_file()
    return entries


def rebuild_scripts_config_file() -> List[Dict[str, Any]]:
    if not SCRIPTS_DIR.exists():
        logging.warning("scripts könyvtár nem található: %s", SCRIPTS_DIR)
        return []

    collected_entries: List[Dict[str, Any]] = []
    latest_source_mtime = 0.0

    for json_path in SCRIPTS_DIR.rglob('*.json'):
        if json_path == SCRIPTS_CONFIG_PATH:
            continue

        relative_json = json_path.relative_to(SCRIPTS_DIR)
        py_candidate = SCRIPTS_DIR / relative_json.with_suffix('.py')
        if not py_candidate.is_file():
            continue

        try:
            with json_path.open('r', encoding='utf-8') as fp:
                entry = json.load(fp)
        except json.JSONDecodeError as exc:
            logging.error("Hibás JSON fájl: %s (%s)", json_path, exc)
            continue
        except OSError as exc:
            logging.error("Nem olvasható JSON fájl: %s (%s)", json_path, exc)
            continue

        if not isinstance(entry, dict):
            logging.warning("A JSON fájl nem objektum: %s", json_path)
            continue

        # biztosítsuk, hogy a script mező a valós relatív útvonalra mutat
        relative_script = relative_json.with_suffix('.py').as_posix()
        script_needs_update = entry.get('script') != relative_script
        entry['script'] = relative_script

        if script_needs_update:
            try:
                with json_path.open('w', encoding='utf-8') as fp:
                    json.dump(entry, fp, ensure_ascii=False, indent=2)
                    fp.write('\n')
            except OSError as exc:
                logging.error("Nem sikerült frissíteni a script JSON fájlt: %s (%s)", json_path, exc)

        collected_entries.append(entry)

        try:
            latest_source_mtime = max(
                latest_source_mtime,
                json_path.stat().st_mtime,
                py_candidate.stat().st_mtime,
            )
        except OSError:
            continue

    collected_entries.sort(key=lambda item: item.get('script', ''))

    existing_data: Optional[List[Dict[str, Any]]] = None
    target_mtime = 0.0
    if SCRIPTS_CONFIG_PATH.exists():
        try:
            target_mtime = SCRIPTS_CONFIG_PATH.stat().st_mtime
            with SCRIPTS_CONFIG_PATH.open('r', encoding='utf-8') as fp:
                existing_data = json.load(fp)
        except (OSError, json.JSONDecodeError):
            existing_data = None

    if existing_data != collected_entries or target_mtime < latest_source_mtime:
        try:
            SCRIPTS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with SCRIPTS_CONFIG_PATH.open('w', encoding='utf-8') as fp:
                json.dump(collected_entries, fp, ensure_ascii=False, indent=2)
                fp.write('\n')
        except OSError as exc:
            logging.error("Nem sikerült frissíteni a scripts.json fájlt: %s", exc)
            return collected_entries

    return collected_entries


def prepare_script_entry(raw_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    script_name = raw_entry.get('script')
    if not script_name:
        return None
    environment = (raw_entry.get('enviroment') or raw_entry.get('environment') or '') or ''
    api_name = raw_entry.get('api')
    parameters: List[Dict[str, Any]] = []

    def append_params(param_list, required: bool):
        for param in param_list or []:
            name = param.get('name')
            if not name:
                continue
            param_type = param.get('type', 'option')
            flags = param.get('flags') or []
            default_value = param.get('default')
            parameters.append({
                'name': name,
                'type': param_type,
                'flags': flags,
                'required': required,
                'autofill': infer_autofill_kind(name),
                'secret': name in SECRET_PARAM_NAMES,
                'default': default_value
            })

    append_params(raw_entry.get('required'), True)
    append_params(raw_entry.get('optional'), False)

    help_markdown: Optional[str] = None
    try:
        script_path = Path(script_name)
        if script_path.is_absolute():
            script_path = Path(script_path.name)
        help_path = (SCRIPTS_DIR / script_path).with_name(f"{script_path.stem}_help.md")
        if help_path.is_file():
            try:
                help_markdown = help_path.read_text(encoding='utf-8')
            except OSError as exc:
                logging.warning("Nem olvasható segédlet fájl: %s (%s)", help_path, exc)
    except (OSError, ValueError) as exc:
        logging.warning("Hibás segédlet elérési út: %s (%s)", script_name, exc)

    return {
        'id': script_name,
        'script': script_name,
        'display_name': raw_entry.get('name') or script_name,
        'environment': environment,
        'description': raw_entry.get('description'),
        'parameters': parameters,
        'notes': raw_entry.get('notes'),
        'raw': raw_entry,
        'required_keys': sorted(SCRIPT_KEY_REQUIREMENTS.get(script_name, set())),
        'api': api_name,
        'help_markdown': help_markdown,
    }


def get_scripts_catalog(force_reload: bool = False) -> List[Dict[str, Any]]:
    try:
        current_mtime = SCRIPTS_CONFIG_PATH.stat().st_mtime
    except OSError:
        current_mtime = None

    with SCRIPTS_CACHE_LOCK:
        cached_mtime = SCRIPTS_CACHE.get('mtime')
        if not force_reload and cached_mtime == current_mtime and SCRIPTS_CACHE.get('data'):
            return copy.deepcopy(SCRIPTS_CACHE['data'])

        raw_entries = load_scripts_file()
        try:
            current_mtime = SCRIPTS_CONFIG_PATH.stat().st_mtime
        except OSError:
            current_mtime = None
        catalog = []
        for entry in raw_entries:
            prepared = prepare_script_entry(entry)
            if prepared:
                catalog.append(prepared)

        SCRIPTS_CACHE['mtime'] = current_mtime
        SCRIPTS_CACHE['data'] = catalog
        return copy.deepcopy(catalog)


def initialize_scripts_catalog() -> None:
    try:
        rebuild_scripts_config_file()
    except Exception as exc:
        logging.error("Nem sikerült inicializálni a script katalógust: %s", exc)


def get_script_definition(script_id: str) -> Optional[Dict[str, Any]]:
    if not script_id:
        return None
    catalog = get_scripts_catalog()
    for entry in catalog:
        if entry['id'] == script_id:
            return entry
    return None


def ensure_workflows_dir() -> Path:
    try:
        WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logging.error("Nem sikerült létrehozni a workflows könyvtárat: %s", exc)
    return WORKFLOWS_DIR


def sanitize_workflow_id(name: str) -> str:
    candidate = secure_filename(name or '')
    candidate = candidate.replace(' ', '_').strip('_')
    if not candidate:
        candidate = datetime.utcnow().strftime("workflow_%Y%m%d_%H%M%S")
    return candidate.lower()


def _load_workflow_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open('r', encoding='utf-8') as fp:
            payload = json.load(fp)
    except (json.JSONDecodeError, OSError) as exc:
        logging.error("Nem sikerült betölteni a workflow fájlt (%s): %s", path, exc)
        return None

    if isinstance(payload, list):
        steps = payload
        name = path.stem
        description = None
    elif isinstance(payload, dict):
        steps = payload.get('steps')
        name = payload.get('name') or path.stem
        description = payload.get('description')
    else:
        logging.warning("Ismeretlen workflow formátum: %s", path)
        return None

    if not isinstance(steps, list):
        logging.warning("Workflow fájl nem tartalmaz érvényes 'steps' listát: %s", path)
        return None

    masked_steps = mask_workflow_secret_params(steps)

    return {
        'name': name,
        'description': description,
        'steps': masked_steps
    }


def list_workflow_templates() -> List[Dict[str, Any]]:
    directory = ensure_workflows_dir()
    templates: List[Dict[str, Any]] = []
    for file_path in sorted(directory.glob('*.json')):
        template_data = _load_workflow_file(file_path)
        if not template_data:
            continue
        templates.append({
            'id': file_path.stem,
            'name': template_data['name'],
            'filename': file_path.name,
            'description': template_data.get('description')
        })
    return templates


def load_workflow_template(template_id: str) -> Optional[Dict[str, Any]]:
    if not template_id:
        return None
    directory = ensure_workflows_dir()
    candidate = directory / template_id
    if candidate.suffix.lower() != '.json':
        candidate = candidate.with_suffix('.json')
    if not candidate.is_file():
        logging.warning("A kért workflow sablon nem található: %s", candidate)
        return None
    template_data = _load_workflow_file(candidate)
    if not template_data:
        return None
    template_data.update({
        'id': candidate.stem,
        'filename': candidate.name
    })
    return template_data


def save_workflow_template_file(
    name: str,
    steps: List[Dict[str, Any]],
    template_id: Optional[str] = None,
    overwrite: bool = False,
    description: Optional[str] = None
) -> Dict[str, Any]:
    directory = ensure_workflows_dir()
    if template_id:
        file_id = sanitize_workflow_id(Path(template_id).stem)
    else:
        file_id = sanitize_workflow_id(name)

    if not file_id:
        raise WorkflowValidationError("Érvénytelen workflow azonosító.")

    target_path = directory / f"{file_id}.json"
    if target_path.exists() and not overwrite:
        raise WorkflowValidationError("Már létezik ugyanilyen nevű workflow. Engedélyezd a felülírást.")

    masked_steps = mask_workflow_secret_params(steps)

    payload: Dict[str, Any] = {
        'name': name or file_id,
        'steps': masked_steps
    }
    if description:
        payload['description'] = description

    try:
        with target_path.open('w', encoding='utf-8') as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
    except OSError as exc:
        logging.error("Nem sikerült elmenteni a workflow sablont: %s", exc)
        raise WorkflowValidationError(f"Workflow mentése sikertelen: {exc}")

    return {
        'id': file_id,
        'name': payload['name'],
        'filename': target_path.name
    }


def get_project_root_path(project_name: str, config_snapshot: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    sanitized_project = secure_filename(project_name)
    if not sanitized_project:
        return None
    snapshot = config_snapshot or get_config_copy()
    workdir_path = snapshot['DIRECTORIES']['workdir']
    return Path(workdir_path) / sanitized_project


def get_project_workflow_state_path(project_name: str, config_snapshot: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    project_root = get_project_root_path(project_name, config_snapshot=config_snapshot)
    if not project_root:
        return None
    return project_root / WORKFLOW_STATE_FILENAME


def load_project_workflow_state(project_name: str, config_snapshot: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    state_path = get_project_workflow_state_path(project_name, config_snapshot=config_snapshot)
    if not state_path or not state_path.is_file():
        return None
    try:
        with state_path.open('r', encoding='utf-8') as fp:
            raw_state = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Nem sikerült betölteni a workflow állapotot (%s): %s", state_path, exc)
        return None

    if isinstance(raw_state, list):
        steps = mask_workflow_secret_params(raw_state)
        return {'steps': steps, 'template_id': None}
    if isinstance(raw_state, dict):
        steps = mask_workflow_secret_params(raw_state.get('steps') or [])
        state: Dict[str, Any] = {
            'steps': steps,
            'template_id': raw_state.get('template_id')
        }
        if 'saved_at' in raw_state:
            state['saved_at'] = raw_state['saved_at']
        return state
    return None


def save_project_workflow_state(
    project_name: str,
    steps: List[Dict[str, Any]],
    template_id: Optional[str] = None,
    *,
    config_snapshot: Optional[Dict[str, Any]] = None,
    saved_at: Optional[str] = None
) -> Dict[str, Any]:
    state_path = get_project_workflow_state_path(project_name, config_snapshot=config_snapshot)
    if not state_path:
        raise WorkflowValidationError("Érvénytelen projekt azonosító.")
    project_root = state_path.parent
    if not project_root.is_dir():
        raise WorkflowValidationError("A projekt könyvtára nem található.")

    payload: Dict[str, Any] = {
        'steps': mask_workflow_secret_params(steps or [])
    }
    if template_id:
        payload['template_id'] = template_id
    if saved_at:
        payload['saved_at'] = saved_at

    try:
        with state_path.open('w', encoding='utf-8') as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
    except OSError as exc:
        logging.error("Nem sikerült menteni a workflow állapotot (%s): %s", state_path, exc)
        raise WorkflowValidationError("Nem sikerült menteni a workflow állapotot.") from exc
    return payload


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('1', 'true', 'yes', 'on', 'igen'):
            return True
        if normalized in ('0', 'false', 'no', 'off', 'nem'):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def normalize_cycle_widget_params(raw_params: Dict[str, Any]) -> Dict[str, int]:
    if raw_params is None:
        params = {}
    elif isinstance(raw_params, dict):
        params = raw_params
    else:
        raise WorkflowValidationError("A ciklus widget paraméterei hibás formátumúak.")

    def parse_positive_int(value: Any, field_label: str) -> int:
        if value is None:
            candidate = ''
        else:
            candidate = str(value).strip()
        if candidate == '':
            candidate = '1'
        try:
            numeric = int(candidate)
        except (TypeError, ValueError):
            raise WorkflowValidationError(
                f"A ciklus widget {field_label} paramétere csak pozitív egész szám lehet."
            ) from None
        if numeric < 1:
            raise WorkflowValidationError(
                f"A ciklus widget {field_label} paramétere csak pozitív egész szám lehet."
            )
        return numeric

    repeat_count = parse_positive_int(params.get('repeat_count'), 'repeat_count')
    step_back = parse_positive_int(params.get('step_back'), 'step_back')

    return {
        'repeat_count': repeat_count,
        'step_back': step_back
    }


def load_conda_info(force_refresh: bool = False) -> Optional[dict]:
    global CONDA_INFO_CACHE
    with CONDA_INFO_LOCK:
        if CONDA_INFO_CACHE is not None and not force_refresh:
            return CONDA_INFO_CACHE
        try:
            result = subprocess.run(
                ['conda', 'info', '--json'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                check=True
            )
            info = json.loads(result.stdout)
            CONDA_INFO_CACHE = info
            return info
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as exc:
            logging.error("Nem sikerült lekérdezni a Conda információkat: %s", exc)
            return None


def map_env_name_to_path(info: dict) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not info:
        return mapping
    envs = info.get('envs') or []
    root_prefix = info.get('root_prefix')
    if root_prefix:
        mapping['base'] = root_prefix
        mapping[os.path.basename(root_prefix)] = root_prefix
    for env_path in envs:
        if not env_path:
            continue
        env_name = os.path.basename(str(env_path).rstrip(os.sep))
        mapping[env_name] = env_path
    return mapping


def load_keyholder_data():
    if not os.path.exists(KEYHOLDER_PATH):
        return {}
    try:
        with open(KEYHOLDER_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.warning("Nem sikerült beolvasni a keyholder.json fájlt, feltételezzük, hogy üres.")
    except OSError as exc:
        logging.error(f"Nem olvasható a keyholder.json fájl: {exc}")
    return {}


def save_keyholder_data(data):
    try:
        with open(KEYHOLDER_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except OSError as exc:
        logging.error(f"Nem sikerült menteni a keyholder.json fájlt: {exc}")
        return False


def decode_keyholder_value(value):
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        decoded = base64.b64decode(value.encode('utf-8')).decode('utf-8')
        return decoded.strip() or None
    except (binascii.Error, UnicodeDecodeError, AttributeError):
        return value


def encode_keyholder_value(value):
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return base64.b64encode(cleaned.encode('utf-8')).decode('utf-8')


def get_keyholder_value(data: Dict[str, Any], field_names: Tuple[str, ...]) -> Optional[str]:
    for field in field_names:
        if not field:
            continue
        value = decode_keyholder_value(data.get(field))
        if value:
            return value
    return None


def get_conda_python(env_name: str):
    cached = CONDA_PYTHON_CACHE.get(env_name)
    if cached:
        return cached
    if not env_name:
        return None
    info = load_conda_info()
    if not info:
        return None
    env_map = map_env_name_to_path(info)
    env_path = env_map.get(env_name)
    if not env_path:
        for envs_dir in info.get('envs_dirs', []):
            candidate = os.path.join(envs_dir, env_name)
            if os.path.isdir(candidate):
                env_path = candidate
                break
    if not env_path:
        logging.error("A(z) '%s' nevű Conda környezet nem található.", env_name)
        return None
    python_exec = os.path.join(env_path, 'python.exe') if os.name == 'nt' else os.path.join(env_path, 'bin', 'python')
    if os.path.exists(python_exec):
        CONDA_PYTHON_CACHE[env_name] = python_exec
        return python_exec
    logging.error("Nem található Python végrehajtható itt: %s", python_exec)
    return None


def ensure_project_structure(project_path: str, subdirs_config: Dict[str, str]) -> None:
    logging.info("Projekt könyvtár ellenőrzése: %s", project_path)
    os.makedirs(project_path, exist_ok=True)
    for subdir in subdirs_config.values():
        os.makedirs(os.path.join(project_path, subdir), exist_ok=True)


def setup_project_logging(project_path: str, logs_subdir: str, project_name: str) -> Tuple[logging.Handler, str]:
    log_dir = os.path.join(project_path, logs_subdir)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{project_name}_run_{timestamp}.log')

    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    logging.info("Log fájl létrehozva: %s", log_file)
    return handler, log_file


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


def determine_parameter_value(
    param_meta: Dict[str, Any],
    user_params: Dict[str, Any],
    script_meta: Dict[str, Any],
    context: Dict[str, Any]
) -> Any:
    name = param_meta['name']
    value = user_params.get(name)

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            value = None

    if value is None:
        autofill_kind = param_meta.get('autofill')
        if autofill_kind == 'project_name':
            value = context['project_name']
        elif autofill_kind == 'project_path':
            value = context['project_path']

    if value is None:
        key_mapping = SCRIPT_PARAM_KEYHOLDER.get(script_meta['id'], {}).get(name)
        if key_mapping:
            value = get_keyholder_value(context['keyholder'], key_mapping)

    if value is None and 'default' in param_meta:
        default_value = param_meta.get('default')
        if default_value is not None:
            value = default_value

    if param_meta['type'] == 'flag':
        return coerce_bool(value)

    return value


def build_argument_fragment(param_meta: Dict[str, Any], value: Any) -> List[str]:
    param_type = param_meta['type']
    name = param_meta['name']
    flags = param_meta.get('flags') or []

    if param_type == 'flag':
        if not flags:
            return []

        positive_flag = next((flag for flag in flags if not flag.startswith('--no-')), None)
        negative_flag = next((flag for flag in flags if flag.startswith('--no-')), None)

        if value is True:
            return [positive_flag] if positive_flag else []
        if value is False:
            if negative_flag:
                return [negative_flag]
            return []
        return []

    if value is None:
        return []

    if param_type == 'positional':
        return [str(value)]

    if param_type == 'option':
        if flags:
            return [flags[0], str(value)]
        return [str(value)]

    if param_type == 'config_option':
        return [f"{name}={value}"]

    logging.warning("Ismeretlen paraméter típus: %s", param_type)
    return []


def build_command_for_step(
    step_config: Dict[str, Any],
    script_meta: Dict[str, Any],
    context: Dict[str, Any]
) -> Tuple[List[str], List[Dict[str, Any]]]:
    environment = script_meta.get('environment')
    python_exec = get_conda_python(environment)
    if not python_exec:
        raise WorkflowValidationError(f"Nem található Python futtató a(z) '{environment}' környezethez.")

    script_path = Path(app.root_path) / 'scripts' / script_meta['script']
    if not script_path.is_file():
        raise WorkflowValidationError(f"A szkript nem található: {script_path}")

    user_params = step_config.get('params') or {}
    applied_params: List[Dict[str, Any]] = []
    command = [python_exec, str(script_path)]

    for param_meta in script_meta['parameters']:
        value = determine_parameter_value(param_meta, user_params, script_meta, context)
        if value is None and param_meta['required'] and param_meta['type'] != 'flag':
            raise WorkflowValidationError(f"Hiányzó kötelező paraméter: {param_meta['name']} ({script_meta['script']})")
        fragment = build_argument_fragment(param_meta, value)
        if fragment:
            command.extend(fragment)
            applied_params.append({
                'name': param_meta['name'],
                'value': value,
                'type': param_meta['type']
            })

    return command, applied_params


def normalize_workflow_steps(payload: Any) -> Tuple[List[Dict[str, Any]], Set[str], int]:
    if not isinstance(payload, list):
        raise WorkflowValidationError("A workflow lépéseit listában kell megadni.")

    normalized_steps: List[Dict[str, Any]] = []
    required_keys: Set[str] = set()
    enabled_count = 0

    for index, step in enumerate(payload, start=1):
        if not isinstance(step, dict):
            raise WorkflowValidationError(f"A(z) {index}. lépés formátuma hibás.")

        step_type = step.get('type')
        if step_type == 'widget' or (step.get('widget') and not step.get('script')):
            widget_id = step.get('widget')
            if not widget_id:
                raise WorkflowValidationError(f"A(z) {index}. widget lépéshez hiányzik a widget azonosítója.")
            if widget_id not in ALLOWED_WORKFLOW_WIDGETS:
                logging.warning("Ismeretlen workflow widget: %s", widget_id)
            widget_params = step.get('params')
            if widget_params is None:
                widget_params = {}
            if not isinstance(widget_params, dict):
                raise WorkflowValidationError(f"A(z) {index}. widget lépés paraméterei hibás formátumúak.")
            normalized_widget_step: Dict[str, Any] = {
                'type': 'widget',
                'widget': widget_id,
                'enabled': coerce_bool(step.get('enabled', True))
            }
            if widget_id == 'cycleWidget':
                normalized_widget_step['params'] = normalize_cycle_widget_params(widget_params)
            elif widget_params:
                normalized_widget_step['params'] = copy.deepcopy(widget_params)
            normalized_steps.append(normalized_widget_step)
            continue

        script_id = step.get('script')
        script_meta = get_script_definition(script_id)
        if not script_meta:
            raise WorkflowValidationError(f"Ismeretlen szkript: {script_id}")

        enabled = coerce_bool(step.get('enabled', True))
        halt_on_fail = coerce_bool(step.get('halt_on_fail', True))
        params = step.get('params') or {}
        if not isinstance(params, dict):
            raise WorkflowValidationError(f"A(z) {script_meta['display_name']} paraméterei hibás formátumúak.")
        normalized_params: Dict[str, Any] = {}
        for key, value in params.items():
            current_value = value
            if isinstance(current_value, str):
                current_value = current_value.strip()
                if current_value == '':
                    current_value = None
            if is_secret_param(key) and current_value is not None:
                normalized_params[key] = unmask_secret_param_value(current_value)
            else:
                normalized_params[key] = current_value

        normalized_step = {
            'script': script_meta['id'],
            'enabled': enabled,
            'halt_on_fail': halt_on_fail,
            'params': normalized_params
        }
        if enabled:
            enabled_count += 1
            required_keys.update(script_meta.get('required_keys', []))

        normalized_steps.append(normalized_step)

    if enabled_count == 0:
        raise WorkflowValidationError("Legalább egy lépést engedélyezni kell a futtatáshoz.")

    return normalized_steps, required_keys, enabled_count


def run_script_command(
    command_list: List[str],
    log_prefix: str = "",
    env: Optional[Dict[str, str]] = None,
    should_stop: Optional[Callable[[], bool]] = None
) -> Any:
    command_str = ' '.join(f'"{c}"' if ' ' in c else c for c in command_list)
    logging.info("%s Parancs futtatása: %s", log_prefix, command_str)
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
        if proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                logging.info("%s %s", log_prefix, line.rstrip())
                if should_stop and should_stop():
                    logging.warning("%s Megszakítás kérve, a parancs leállítása...", log_prefix)
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return "cancelled"
        rc = proc.wait()
        if rc != 0:
            logging.error("%s A parancs hibával ért véget (%s)", log_prefix, rc)
            return False
        logging.info("%s Parancs sikeresen lefutott.", log_prefix)
        return True
    except Exception as exc:
        logging.error("%s Váratlan hiba a parancs futtatása közben: %s", log_prefix, exc, exc_info=True)
        return False


def determine_workflow_key_status(required_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
    keyholder_data = load_keyholder_data()
    required_keys = set(required_keys or [])

    key_definitions = {
        'chatgpt': {
            'label': 'OpenAI / ChatGPT API kulcs',
            'fields': ('chatgpt_api_key', 'api_key'),
        },
        'deepl': {
            'label': 'DeepL API kulcs',
            'fields': ('deepL_api_key', 'deepl_api_key'),
        },
        'huggingface': {
            'label': 'Hugging Face token',
            'fields': ('hf_token',),
        }
    }

    keys_status = {}
    for key_name, definition in key_definitions.items():
        present = bool(get_keyholder_value(keyholder_data, definition['fields']))
        keys_status[key_name] = {
            'label': definition['label'],
            'present': present,
            'required': key_name in required_keys
        }

    return {
        'keys': keys_status,
        'uses_chatgpt_translation': 'chatgpt' in required_keys
    }

# Statikus fájlok kiszolgálása a workdir mappából
@app.route('/workdir/<path:filename>')
def serve_workdir(filename):
    return send_from_directory('workdir', filename)

# Konfiguráció betöltése
with open('config.json') as config_file:
    config = json.load(config_file)

def get_config_copy():
    with config_lock:
        return copy.deepcopy(config)


def persist_config(updated_config):
    global config
    with config_lock:
        config = updated_config
        with open('config.json', 'w', encoding='utf-8') as config_file:
            json.dump(config, config_file, indent=2, ensure_ascii=False)

def update_workflow_job(job_id, **kwargs):
    with workflow_lock:
        if 'workflow' in kwargs:
            kwargs['workflow'] = mask_workflow_secret_params(kwargs['workflow'])
        if 'execution_steps' in kwargs:
            kwargs['execution_steps'] = mask_workflow_secret_params(kwargs['execution_steps'])
        if job_id in workflow_jobs:
            workflow_jobs[job_id].update(kwargs)


def get_workflow_job(job_id):
    with workflow_lock:
        return workflow_jobs.get(job_id)


def get_project_jobs(project_name):
    sanitized_name = secure_filename(project_name)
    with workflow_lock:
        return [
            job.copy()
            for job in workflow_jobs.values()
            if job.get('project') == sanitized_name
        ]


def register_workflow_job(job_id, job_data):
    with workflow_lock:
        if isinstance(job_data, dict):
            if 'workflow' in job_data:
                job_data['workflow'] = mask_workflow_secret_params(job_data['workflow'])
            if 'execution_steps' in job_data:
                job_data['execution_steps'] = mask_workflow_secret_params(job_data['execution_steps'])
        workflow_jobs[job_id] = job_data
        workflow_events[job_id] = threading.Event()


def set_workflow_thread(job_id, thread):
    with workflow_lock:
        workflow_threads[job_id] = thread


def get_workflow_thread(job_id):
    with workflow_lock:
        return workflow_threads.get(job_id)


def get_workflow_event(job_id):
    with workflow_lock:
        return workflow_events.get(job_id)


def cleanup_workflow_resources(job_id):
    with workflow_lock:
        workflow_threads.pop(job_id, None)
        workflow_events.pop(job_id, None)


def request_workflow_cancel(job_id):
    event = get_workflow_event(job_id)
    if not event:
        return False
    if not event.is_set():
        event.set()
    return True


def is_cancel_requested(job_id):
    event = get_workflow_event(job_id)
    return event.is_set() if event else False


def build_log_links(project_name, logs_subdir, log_filename):
    relative_path = os.path.join(secure_filename(project_name), logs_subdir, log_filename)
    return {
        'relative': relative_path,
        'url': f"/workdir/{relative_path}"
    }


def read_log_tail(log_path, max_bytes=12000):
    if not os.path.exists(log_path):
        return ""
    try:
        with open(log_path, 'rb') as log_file:
            log_file.seek(0, os.SEEK_END)
            file_size = log_file.tell()
            if file_size <= max_bytes:
                log_file.seek(0)
            else:
                log_file.seek(-max_bytes, os.SEEK_END)
            data = log_file.read().decode('utf-8', errors='replace')
            if file_size > max_bytes:
                newline_index = data.find('\n')
                if newline_index != -1:
                    data = data[newline_index + 1:]
        return data
    except Exception as exc:
        logging.error("Log olvasási hiba: %s", exc, exc_info=True)
        return ""


def run_workflow_job(job_id, project_name, workflow_payload):
    sanitized_project = secure_filename(project_name)
    log_handler = None
    cancel_event = get_workflow_event(job_id)

    def should_stop():
        return cancel_event.is_set() if cancel_event else False

    try:
        initial_status = 'cancelling' if should_stop() else 'running'
        initial_message = 'Megszakítás kérve, előkészítés folyamatban...' if initial_status == 'cancelling' else 'Feldolgozás előkészítése...'
        update_workflow_job(
            job_id,
            status=initial_status,
            started_at=datetime.utcnow().isoformat(),
            message=initial_message,
            cancel_requested=should_stop()
        )

        current_config = get_config_copy()
        workdir_path = current_config['DIRECTORIES']['workdir']
        project_path = os.path.join(workdir_path, sanitized_project)

        ensure_project_structure(project_path, current_config['PROJECT_SUBDIRS'])
        log_handler, log_file = setup_project_logging(
            project_path,
            current_config['PROJECT_SUBDIRS']['logs'],
            sanitized_project
        )
        log_filename = os.path.basename(log_file)
        log_links = build_log_links(sanitized_project, current_config['PROJECT_SUBDIRS']['logs'], log_filename)
        update_workflow_job(job_id, log=log_links)

        template_id = workflow_payload.get('template_id')
        steps = workflow_payload.get('steps') or []
        workflow_state = workflow_payload.get('workflow_state') or steps
        masked_steps = mask_workflow_secret_params(steps)
        masked_workflow_state = mask_workflow_secret_params(workflow_state)
        active_steps = [
            step for step in steps
            if step.get('enabled', True) and step.get('type') != 'widget'
        ]
        total_steps = len(active_steps)
        update_workflow_job(
            job_id,
            total_steps=total_steps,
            template_id=template_id,
            workflow=masked_workflow_state,
            execution_steps=masked_steps
        )

        keyholder_snapshot = load_keyholder_data()
        executed_steps: List[Dict[str, Any]] = []

        context = {
            'project_name': sanitized_project,
            'project_path': project_path,
            'keyholder': keyholder_snapshot,
            'config': current_config,
            'template_id': template_id
        }

        for index, step in enumerate(active_steps, start=1):
            if should_stop():
                logging.info("Workflow megszakítás kérve, kilépünk.")
                update_workflow_job(
                    job_id,
                    status='cancelled',
                    finished_at=datetime.utcnow().isoformat(),
                    message='Workflow megszakítva.',
                    cancel_requested=False,
                    current_step=None,
                    results=executed_steps
                )
                return

            script_id = step.get('script')
            script_meta = get_script_definition(script_id)
            if not script_meta:
                raise WorkflowValidationError(f"Ismeretlen szkript: {script_id}")

            try:
                command, applied_params = build_command_for_step(step, script_meta, context)
            except WorkflowValidationError as exc:
                raise

            masked_command, masked_applied_params = build_masked_command_and_params(command, applied_params, script_meta)

            log_prefix = f"[{script_meta['id']}]"
            update_workflow_job(
                job_id,
                status='running',
                message=f"{index}/{total_steps} · {script_meta['display_name']}",
                current_step={
                    'index': index,
                    'total': total_steps,
                    'script': script_meta['id'],
                    'display_name': script_meta['display_name'],
                    'command': masked_command,
                },
                cancel_requested=should_stop()
            )

            result = run_script_command(command, log_prefix=log_prefix, should_stop=should_stop)
            step_record = {
                'script': script_meta['id'],
                'display_name': script_meta['display_name'],
                'command': masked_command,
                'applied_params': masked_applied_params,
            }

            if result == "cancelled":
                step_record['status'] = 'cancelled'
                executed_steps.append(step_record)
                update_workflow_job(
                    job_id,
                    status='cancelled',
                    finished_at=datetime.utcnow().isoformat(),
                    message='Workflow megszakítva.',
                    cancel_requested=False,
                    current_step=None,
                    results=executed_steps
                )
                return

            if result is False:
                step_record['status'] = 'failed'
                executed_steps.append(step_record)
                if step.get('halt_on_fail', True):
                    update_workflow_job(
                        job_id,
                        status='failed',
                        finished_at=datetime.utcnow().isoformat(),
                        message=f"Hiba a(z) {script_meta['display_name']} lépés futtatása közben.",
                        cancel_requested=False,
                        current_step=None,
                        results=executed_steps
                    )
                    return
                else:
                    logging.warning("A(z) %s lépés hibával futott, de a workflow folytatódik.", script_meta['display_name'])
                    continue

            step_record['status'] = 'completed'
            executed_steps.append(step_record)

        update_workflow_job(
            job_id,
            status='completed',
            finished_at=datetime.utcnow().isoformat(),
            message='Workflow sikeresen lefutott.',
            cancel_requested=False,
            current_step=None,
            results=executed_steps
        )
    except WorkflowValidationError as exc:
        logging.error("Workflow konfigurációs hiba: %s", exc)
        update_workflow_job(
            job_id,
            status='failed',
            finished_at=datetime.utcnow().isoformat(),
            message=str(exc),
            cancel_requested=False,
            current_step=None,
            results=locals().get('executed_steps')
        )
    except Exception as exc:
        logging.exception("Workflow futtatási hiba: %s", exc)
        if should_stop():
            update_workflow_job(
                job_id,
                status='cancelled',
                finished_at=datetime.utcnow().isoformat(),
                message='Workflow megszakítva.',
                cancel_requested=False
            )
        else:
            update_workflow_job(
                job_id,
                status='failed',
                finished_at=datetime.utcnow().isoformat(),
                message=f'Hiba: {exc}',
                cancel_requested=False
            )
    finally:
        if log_handler:
            remove_logging_handler(log_handler)
        cleanup_workflow_resources(job_id)

@app.route('/')
def index():
    # Meglévő projektek listázása
    projects = []
    if os.path.exists('workdir'):
        projects = sorted(
            (d for d in os.listdir('workdir') if os.path.isdir(os.path.join('workdir', d))),
            key=str.lower
        )
    project_entries = build_project_entries(projects)
    return render_template('index.html', projects=projects, project_entries=project_entries)


@app.route('/template-editor')
def template_editor():
    return render_template('template_editor.html')

@app.route('/project/<project_name>')
def show_project(project_name):
    # Projekt ellenőrzése
    project_dir = os.path.join('workdir', secure_filename(project_name))
    if not os.path.exists(project_dir):
        return "Project not found", 404
    
    # Projekt adatok összegyűjtése
    project_data = {
        'name': project_name,
        'files': {
            'upload': [],
            'upload_grouped': [],
            'extracted_audio': [],
            'extracted_audio_grouped': [],
            'separated_audio_background': [],
            'separated_audio_background_grouped': [],
            'separated_audio_speech': []
        }
    }

    highlight_map = compute_failed_generation_highlights(project_dir, config)
    metadata_directories = get_audio_metadata_directories(config)
    failed_generation_directories = get_failed_generation_directories(config)

    def group_files_by_extension(file_list):
        grouped = {}
        for filename in sorted(file_list):
            ext = os.path.splitext(filename)[1].lower()
            grouped.setdefault(ext, []).append(filename)
        grouped_list = []
        for ext_key in sorted(grouped.keys()):
            files = grouped[ext_key]
            display = ext_key[1:].upper() if ext_key else 'Nincs kiterjesztés'
            grouped_list.append({
                'extension': ext_key,
                'display': display,
                'files': files,
                'count': len(files)
            })
        return grouped_list

    def build_directory_tree(current_path, relative_path=''):
        entries = []
        try:
            for name in sorted(os.listdir(current_path)):
                if name.startswith('.'):
                    continue
                full_path = os.path.join(current_path, name)
                rel_path = os.path.join(relative_path, name) if relative_path else name
                if os.path.isdir(full_path):
                    normalized_rel_path = rel_path.replace('\\', '/')
                    entry = {
                        'name': name,
                        'type': 'directory',
                        'path': normalized_rel_path,
                        'children': build_directory_tree(full_path, rel_path)
                    }
                    highlight_class = highlight_map.get(normalized_rel_path)
                    if highlight_class:
                        entry['highlight_class'] = highlight_class
                    entries.append(entry)
                else:
                    normalized_file_path = rel_path.replace('\\', '/')
                    file_entry = {
                        'name': name,
                        'type': 'file',
                        'path': normalized_file_path
                    }
                    if should_enable_failed_move(normalized_file_path, highlight_map):
                        file_entry['enable_failed_move'] = True
                    metadata = build_audio_metadata(full_path, normalized_file_path, metadata_directories)
                    if metadata:
                        file_entry.update(metadata)
                    failed_metadata = build_failed_generation_json_metadata(
                        full_path,
                        normalized_file_path,
                        failed_generation_directories
                    )
                    if failed_metadata:
                        file_entry.update(failed_metadata)
                    entries.append(file_entry)
        except Exception as exc:
            logging.warning("Nem sikerült beolvasni a(z) %s könyvtárat: %s", current_path, exc)
        return entries

    # Uploaded files
    upload_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['upload'])
    if os.path.exists(upload_dir_path):
        project_data['files']['upload'] = os.listdir(upload_dir_path)
        project_data['files']['upload_grouped'] = group_files_by_extension(project_data['files']['upload'])

    # Extracted audio files
    extracted_audio_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['extracted_audio'])
    if os.path.exists(extracted_audio_dir_path):
        project_data['files']['extracted_audio'] = os.listdir(extracted_audio_dir_path)
        project_data['files']['extracted_audio_grouped'] = group_files_by_extension(project_data['files']['extracted_audio'])

    # Separated background audio files
    separated_bg_audio_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_background'])
    if os.path.exists(separated_bg_audio_dir_path):
        project_data['files']['separated_audio_background'] = os.listdir(separated_bg_audio_dir_path)
        project_data['files']['separated_audio_background_grouped'] = group_files_by_extension(project_data['files']['separated_audio_background'])

    # Separated speech audio files (and their JSON transcriptions)
    speech_files_data = []
    speech_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_speech'])
    if os.path.exists(speech_dir_path):
        for f_name in sorted(os.listdir(speech_dir_path)): # Sorted for consistent order
            file_data = {'name': f_name, 'segment_count': None, 'is_audio': False, 'is_json': False}
            file_path = os.path.join(speech_dir_path, f_name)
            
            if f_name.lower().endswith('.json'):
                file_data['is_json'] = True
                try:
                    with open(file_path, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                        if 'segments' in data and isinstance(data['segments'], list):
                            file_data['segment_count'] = len(data['segments'])
                except Exception as e:
                    print(f"Error reading or parsing JSON file {file_path}: {e}")
            elif f_name.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                file_data['is_audio'] = True
            speech_files_data.append(file_data)
    project_data['files']['separated_audio_speech'] = speech_files_data
    
    # Ellenőrizzük, van-e felülvizsgálható audio/JSON pár
    can_review = False
    if os.path.exists(speech_dir_path):
        temp_files_list = sorted(os.listdir(speech_dir_path))
        for f_check_name in temp_files_list:
            if f_check_name.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                base_name_check, _ = os.path.splitext(f_check_name)
                if (base_name_check + ".json") in temp_files_list:
                    can_review = True
                    break
    
    # Új flag: van-e transzkribálható (beszéd) audio fájl
    has_transcribable_audio = any(f_info['is_audio'] for f_info in speech_files_data)
    
    return render_template('project.html', 
                         project=project_data,
                         project_tree=build_directory_tree(project_dir),
                         config=config,
                         audio_extensions=AUDIO_EXTENSIONS,
                         video_extensions=VIDEO_EXTENSIONS,
                         audio_mime_map=AUDIO_MIME_MAP,
                         video_mime_map=VIDEO_MIME_MAP,
                         can_review=can_review,
                         has_transcribable_audio=has_transcribable_audio,
                         has_failed_generation_highlights=bool(highlight_map),
                         secret_param_names=sorted(SECRET_PARAM_NAMES))


@app.route('/api/project-tree/<project_name>', methods=['GET'])
def get_project_directory_listing(project_name):
    sanitized_project = secure_filename(project_name)
    config_snapshot = get_config_copy()
    workdir_path = config_snapshot['DIRECTORIES']['workdir']
    base_dir = os.path.join(workdir_path, sanitized_project)
    base_dir_abs = os.path.abspath(base_dir)

    if not os.path.isdir(base_dir_abs):
        return jsonify({'success': False, 'error': 'Projekt nem található'}), 404

    requested_path = (request.args.get('path') or '').strip()
    target_dir = base_dir_abs
    if requested_path:
        target_dir = os.path.abspath(os.path.join(base_dir_abs, requested_path))

    if not is_subpath(target_dir, base_dir_abs):
        return jsonify({'success': False, 'error': 'Érvénytelen útvonal'}), 400
    if not os.path.isdir(target_dir):
        return jsonify({'success': False, 'error': 'A megadott könyvtár nem található'}), 404

    highlight_map = compute_failed_generation_highlights(base_dir_abs, config_snapshot)
    metadata_directories = get_audio_metadata_directories(config_snapshot)
    failed_generation_directories = get_failed_generation_directories(config_snapshot)
    entries = collect_directory_entries(
        base_dir_abs,
        target_dir,
        metadata_directories,
        highlight_map,
        failed_generation_directories
    )
    if target_dir == base_dir_abs:
        current_path_key = ''
    else:
        current_path_key = os.path.relpath(target_dir, base_dir_abs).replace('\\', '/')
    current_highlight = highlight_map.get(current_path_key)

    return jsonify({
        'success': True,
        'entries': entries,
        'current_highlight': current_highlight,
        'has_highlights': bool(highlight_map)
    })


@app.route('/api/project-file/upload', methods=['POST'])
def upload_project_file():
    project_name = (request.form.get('projectName') or '').strip()
    target_path_raw = (request.form.get('targetPath') or '').strip()
    uploaded_file = request.files.get('file')

    if not project_name:
        return jsonify({'success': False, 'error': 'Hiányzó projektnév.'}), 400
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({'success': False, 'error': 'Nem érkezett fájl a feltöltéshez.'}), 400

    sanitized_project = secure_filename(project_name)
    if not sanitized_project:
        return jsonify({'success': False, 'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

    safe_filename = secure_filename(uploaded_file.filename)
    if not safe_filename:
        return jsonify({'success': False, 'error': 'A fájlnév érvénytelen karaktereket tartalmaz.'}), 400

    config_snapshot = get_config_copy()
    workdir_path = config_snapshot['DIRECTORIES']['workdir']
    project_root = os.path.join(workdir_path, sanitized_project)
    project_root_abs = os.path.abspath(project_root)

    if not os.path.isdir(project_root_abs):
        return jsonify({'success': False, 'error': 'A projekt könyvtára nem található.'}), 404

    normalized_target = target_path_raw.strip('/\\')
    destination_dir = project_root_abs
    if normalized_target:
        destination_dir = os.path.abspath(os.path.join(project_root_abs, normalized_target))

    if not is_subpath(destination_dir, project_root_abs):
        return jsonify({'success': False, 'error': 'Érvénytelen célkönyvtár.'}), 400

    try:
        os.makedirs(destination_dir, exist_ok=True)
    except OSError as exc:
        logging.exception("Nem sikerült létrehozni a(z) %s könyvtárat: %s", destination_dir, exc)
        return jsonify({'success': False, 'error': 'Nem sikerült előkészíteni a célkönyvtárat.'}), 500

    destination_path = os.path.join(destination_dir, safe_filename)
    if os.path.exists(destination_path):
        return jsonify({'success': False, 'error': 'Már létezik ilyen nevű fájl ebben a mappában.'}), 409

    try:
        uploaded_file.save(destination_path)
    except OSError as exc:
        logging.exception("Nem sikerült a fájl mentése ide: %s: %s", destination_path, exc)
        return jsonify({'success': False, 'error': 'Nem sikerült menteni a feltöltött fájlt.'}), 500

    relative_saved_path = os.path.relpath(destination_path, project_root_abs).replace('\\', '/')
    return jsonify({
        'success': True,
        'message': 'Fájl sikeresen feltöltve.',
        'path': relative_saved_path
    })


@app.route('/api/project-file/move-failed', methods=['POST'])
def move_failed_generation_file():
    payload = request.get_json(silent=True) or {}
    project_name = (payload.get('projectName') or '').strip()
    source_path = (payload.get('sourcePath') or '').strip()

    if not project_name:
        return jsonify({'success': False, 'error': 'Hiányzó projektnév.'}), 400
    if not source_path:
        return jsonify({'success': False, 'error': 'Hiányzó fájl elérési útvonal.'}), 400

    sanitized_project = secure_filename(project_name)
    if not sanitized_project:
        return jsonify({'success': False, 'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

    config_snapshot = get_config_copy()
    workdir_path = config_snapshot['DIRECTORIES']['workdir']
    project_root = os.path.join(workdir_path, sanitized_project)
    project_root_abs = os.path.abspath(project_root)

    if not os.path.isdir(project_root_abs):
        return jsonify({'success': False, 'error': 'A projekt könyvtára nem található.'}), 404

    source_abs = os.path.abspath(os.path.join(project_root_abs, source_path))
    if not is_subpath(source_abs, project_root_abs):
        return jsonify({'success': False, 'error': 'Érvénytelen forrás elérési útvonal.'}), 400
    if not os.path.isfile(source_abs):
        return jsonify({'success': False, 'error': 'A megadott fájl nem található.'}), 404

    relative_source = os.path.relpath(source_abs, project_root_abs).replace('\\', '/')
    if not relative_source.startswith('failed_generations/'):
        return jsonify({'success': False, 'error': 'Csak failed_generations könyvtárból áthelyezhető.'}), 400

    highlight_map = compute_failed_generation_highlights(project_root_abs, config_snapshot)
    parent_path = relative_source.rsplit('/', 1)[0] if '/' in relative_source else ''
    if highlight_map.get(parent_path) == 'fg-has-translated':
        return jsonify({'success': False, 'error': 'Ez a mappa már rendelkezik fordított szegmenssel.'}), 400

    _, ext = os.path.splitext(source_abs)
    if ext.lower() != '.wav':
        return jsonify({'success': False, 'error': 'Csak WAV fájlok helyezhetők át.'}), 400

    translated_rel = config_snapshot.get('PROJECT_SUBDIRS', {}).get('translated_splits')
    if not translated_rel:
        return jsonify({'success': False, 'error': 'A translated_splits könyvtár nincs konfigurálva.'}), 500

    translated_dir_abs = os.path.abspath(os.path.join(project_root_abs, translated_rel))
    if not is_subpath(translated_dir_abs, project_root_abs):
        return jsonify({'success': False, 'error': 'Érvénytelen célkönyvtár konfiguráció.'}), 500

    try:
        os.makedirs(translated_dir_abs, exist_ok=True)
    except OSError as exc:
        logging.exception("Nem sikerült létrehozni a célkönyvtárat: %s", translated_dir_abs, exc)
        return jsonify({'success': False, 'error': 'Nem sikerült előkészíteni a célkönyvtárat.'}), 500

    filename = os.path.basename(source_abs)
    stem, extension = os.path.splitext(filename)
    attempt_index = stem.find('_attempt_')
    if attempt_index != -1:
        stem = stem[:attempt_index]
    if not stem:
        return jsonify({'success': False, 'error': 'A fájlnév nem értelmezhető az áthelyezéshez.'}), 400

    destination_name = f"{stem}{extension}"
    destination_abs = os.path.abspath(os.path.join(translated_dir_abs, destination_name))
    if not is_subpath(destination_abs, translated_dir_abs):
        return jsonify({'success': False, 'error': 'Érvénytelen célfájl hely.'}), 500

    if os.path.exists(destination_abs):
        return jsonify({'success': False, 'error': 'Már létezik ilyen nevű fájl a translated_splits mappában.'}), 409

    try:
        shutil.move(source_abs, destination_abs)
    except OSError as exc:
        logging.exception("Nem sikerült áthelyezni a fájlt %s -> %s: %s", source_abs, destination_abs, exc)
        return jsonify({'success': False, 'error': 'Nem sikerült áthelyezni a fájlt.'}), 500

    relative_destination = os.path.relpath(destination_abs, project_root_abs).replace('\\', '/')
    return jsonify({
        'success': True,
        'message': 'Fájl sikeresen áthelyezve.',
        'destination_path': relative_destination
    })


@app.route('/api/project-file/<project_name>', methods=['DELETE'])
def delete_project_file(project_name):
    payload = request.get_json(silent=True) or {}
    target_path_raw = (payload.get('path') or '').strip()
    if not target_path_raw:
        return jsonify({'success': False, 'error': 'Hiányzó fájl elérési útvonal.'}), 400

    sanitized_project = secure_filename(project_name)
    if not sanitized_project:
        return jsonify({'success': False, 'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

    config_snapshot = get_config_copy()
    workdir_path = config_snapshot['DIRECTORIES']['workdir']
    project_root = os.path.join(workdir_path, sanitized_project)
    project_root_abs = os.path.abspath(project_root)

    target_abs_path = os.path.abspath(os.path.join(project_root_abs, target_path_raw))
    if not is_subpath(target_abs_path, project_root_abs):
        return jsonify({'success': False, 'error': 'Érvénytelen fájl elérési útvonal.'}), 400

    if not os.path.exists(target_abs_path):
        return jsonify({'success': False, 'error': 'A megadott fájl nem létezik.'}), 404

    if os.path.isdir(target_abs_path):
        return jsonify({'success': False, 'error': 'Könyvtárak törlése nem engedélyezett.'}), 400

    try:
        os.remove(target_abs_path)
    except OSError as exc:
        logging.exception("Nem sikerült törölni a fájlt: %s: %s", target_abs_path, exc)
        return jsonify({'success': False, 'error': 'Nem sikerült törölni a fájlt.'}), 500

    return jsonify({'success': True, 'message': 'Fájl sikeresen törölve.'})


@app.route('/api/project-audio/trim', methods=['POST'])
def trim_project_audio():
    payload = request.get_json(silent=True) or {}
    project_name = (payload.get('projectName') or '').strip()
    file_path_raw = (payload.get('filePath') or '').strip()
    output_name_raw = (payload.get('outputName') or '').strip()

    if not project_name:
        return jsonify({'success': False, 'error': 'Hiányzó projektnév.'}), 400
    if not file_path_raw:
        return jsonify({'success': False, 'error': 'Hiányzó audió elérési útvonal.'}), 400

    try:
        start_value = float(payload.get('start', 0))
        end_value = float(payload.get('end', 0))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Érvénytelen időbélyeg értékek.'}), 400

    start_value = max(0.0, start_value)
    end_value = max(0.0, end_value)
    if end_value <= start_value:
        return jsonify({'success': False, 'error': 'A kijelölés vége nagyobbnak kell lennie a kezdetnél.'}), 400

    sanitized_project = secure_filename(project_name)
    if not sanitized_project:
        return jsonify({'success': False, 'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

    config_snapshot = get_config_copy()
    workdir_path = config_snapshot['DIRECTORIES']['workdir']
    project_root = os.path.join(workdir_path, sanitized_project)
    project_root_abs = os.path.abspath(project_root)

    source_abs = os.path.abspath(os.path.join(project_root_abs, file_path_raw))
    if not is_subpath(source_abs, project_root_abs):
        return jsonify({'success': False, 'error': 'Érvénytelen audió elérési útvonal.'}), 400
    if not os.path.isfile(source_abs):
        return jsonify({'success': False, 'error': 'A megadott audió fájl nem található.'}), 404

    source_ext = os.path.splitext(source_abs)[1].lower()
    if source_ext not in AUDIO_EXTENSIONS:
        return jsonify({'success': False, 'error': 'Csak támogatott hangfájl vágható.'}), 400

    if output_name_raw:
        safe_output_name = secure_filename(output_name_raw)
        if not safe_output_name:
            return jsonify({'success': False, 'error': 'A mentési fájlnév érvénytelen.'}), 400
        if '.' not in safe_output_name:
            safe_output_name = f"{safe_output_name}{source_ext}"
        destination_abs = os.path.abspath(os.path.join(os.path.dirname(source_abs), safe_output_name))
        if not is_subpath(destination_abs, project_root_abs):
            return jsonify({'success': False, 'error': 'Érvénytelen mentési elérési útvonal.'}), 400
    else:
        destination_abs = source_abs

    if destination_abs != source_abs and os.path.exists(destination_abs):
        return jsonify({'success': False, 'error': 'Már létezik ilyen nevű fájl ebben a mappában.'}), 409

    try:
        audio_segment = AudioSegment.from_file(source_abs)
    except Exception as exc:
        logging.exception("Nem sikerült beolvasni a hangfájlt (%s): %s", source_abs, exc)
        return jsonify({'success': False, 'error': 'Nem sikerült beolvasni a hangfájlt.'}), 500

    audio_duration_ms = len(audio_segment)
    if audio_duration_ms <= 0:
        return jsonify({'success': False, 'error': 'A hangfájl nem tartalmaz adatot.'}), 400

    start_ms = int(start_value * 1000)
    end_ms = int(end_value * 1000)
    start_ms = max(0, min(start_ms, audio_duration_ms))
    end_ms = max(0, min(end_ms, audio_duration_ms))

    if end_ms <= start_ms:
        return jsonify({'success': False, 'error': 'A kijelölés túl rövid a mentéshez.'}), 400

    trimmed_segment = audio_segment[start_ms:end_ms]
    if len(trimmed_segment) <= 0:
        return jsonify({'success': False, 'error': 'A kiválasztott szakasz üres.'}), 400

    export_format = source_ext.lstrip('.') or 'wav'
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(suffix=source_ext)
        os.close(fd)
        trimmed_segment.export(temp_path, format=export_format)
        os.replace(temp_path, destination_abs)
    except Exception as exc:
        logging.exception("Nem sikerült menteni a kivágott audiót %s -> %s: %s", source_abs, destination_abs, exc)
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return jsonify({'success': False, 'error': 'Nem sikerült elmenteni a kivágott hangrészletet.'}), 500

    relative_saved_path = os.path.relpath(destination_abs, project_root_abs).replace('\\', '/')
    trimmed_duration = (end_ms - start_ms) / 1000.0

    return jsonify({
        'success': True,
        'message': 'Hangrészlet sikeresen mentve.',
        'saved_path': relative_saved_path,
        'saved_name': os.path.basename(destination_abs),
        'overwrote_original': destination_abs == source_abs,
        'trim_start': start_ms / 1000.0,
        'trim_end': end_ms / 1000.0,
        'trim_duration': trimmed_duration
    })


@app.route('/api/project-directory/<project_name>', methods=['DELETE'])
def clear_project_directory(project_name):
    payload = request.get_json(silent=True) or {}
    target_path_raw = (payload.get('path') or '').strip()
    if not target_path_raw:
        return jsonify({'success': False, 'error': 'Hiányzó könyvtár elérési útvonal.'}), 400

    sanitized_project = secure_filename(project_name)
    if not sanitized_project:
        return jsonify({'success': False, 'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

    config_snapshot = get_config_copy()
    workdir_path = config_snapshot['DIRECTORIES']['workdir']
    project_root = os.path.join(workdir_path, sanitized_project)
    project_root_abs = os.path.abspath(project_root)

    target_abs_path = os.path.abspath(os.path.join(project_root_abs, target_path_raw))
    if not is_subpath(target_abs_path, project_root_abs):
        return jsonify({'success': False, 'error': 'Érvénytelen könyvtár elérési útvonal.'}), 400

    if not os.path.exists(target_abs_path):
        return jsonify({'success': False, 'error': 'A megadott könyvtár nem létezik.'}), 404

    if not os.path.isdir(target_abs_path):
        return jsonify({'success': False, 'error': 'Csak könyvtárak tartalma törölhető ezzel a művelettel.'}), 400

    try:
        with os.scandir(target_abs_path) as entries:
            for entry in entries:
                entry_path = entry.path
                try:
                    if entry.is_dir(follow_symlinks=False):
                        shutil.rmtree(entry_path)
                    else:
                        os.remove(entry_path)
                except OSError as exc:
                    logging.exception("Nem sikerült törölni a könyvtár tartalmát: %s: %s", entry_path, exc)
                    return jsonify({'success': False, 'error': 'Nem sikerült törölni a könyvtár teljes tartalmát.'}), 500
    except OSError as exc:
        logging.exception("Nem sikerült beolvasni a könyvtár tartalmát: %s: %s", target_abs_path, exc)
        return jsonify({'success': False, 'error': 'Nem sikerült elérni a könyvtárat.'}), 500

    return jsonify({'success': True, 'message': 'A könyvtár tartalma sikeresen törölve.'})


@app.route('/review/<project_name>')
def review_project(project_name):
    project_dir = os.path.join('workdir', secure_filename(project_name))
    # Először a "translated" mappában keresünk JSON fájlt
    translated_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['translated'])
    # Ha nincs "translated" JSON, akkor a "separated_audio_speech" mappát használjuk
    speech_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_speech'])
    
    audio_file_name = None
    json_file_name = None
    segments_data = []
    
    # Először a "translated" mappában keresünk
    if os.path.exists(translated_dir_path):
        translated_files = sorted(os.listdir(translated_dir_path))
        # Keressük a legelső audio fájlt, amihez van JSON a "translated" mappában
        for f_name in translated_files:
            if f_name.lower().endswith('.json'):
                base_name, _ = os.path.splitext(f_name)
                # Ellenőrizzük, van-e audio fájl a "separated_audio_speech" mappában
                if os.path.exists(speech_dir_path):
                    speech_files = sorted(os.listdir(speech_dir_path))
                    # Keressük a megfelelő audio fájlt (ugyanaz a base név)
                    audio_candidate = base_name + os.path.splitext(f_name)[1].replace('.json', '')
                    for audio_ext in ['.wav', '.mp3', '.ogg', '.flac']:
                        if audio_candidate + audio_ext in speech_files:
                            audio_file_name = audio_candidate + audio_ext
                            json_file_name = f_name
                            json_full_path = os.path.join(translated_dir_path, json_file_name)
                            try:
                                with open(json_full_path, 'r', encoding='utf-8') as jf:
                                    data = json.load(jf)
                                    segments_data = data.get('segments', [])
                            except Exception as e:
                                print(f"Error reading JSON file {json_full_path} for review: {e}")
                                segments_data = []
                            break
                    if audio_file_name:
                        break
    
    # Ha nem találtunk "translated" JSON-t, akkor az eredeti logikát használjuk
    if not json_file_name and os.path.exists(speech_dir_path):
        files = sorted(os.listdir(speech_dir_path))
        for f_name in files:
            if f_name.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                base_name, _ = os.path.splitext(f_name)
                potential_json_name = base_name + ".json"
                if potential_json_name in files:
                    audio_file_name = f_name
                    json_file_name = potential_json_name
                    json_full_path = os.path.join(speech_dir_path, json_file_name)
                    try:
                        with open(json_full_path, 'r', encoding='utf-8') as jf:
                            data = json.load(jf)
                            segments_data = data.get('segments', [])
                    except Exception as e:
                        print(f"Error reading JSON file {json_full_path} for review: {e}")
                        segments_data = [] # Hiba esetén üres lista
                    break # Első páros megtalálva
    
    audio_url = None
    if audio_file_name:
        audio_url = url_for('serve_workdir', filename=f"{secure_filename(project_name)}/{config['PROJECT_SUBDIRS']['separated_audio_speech']}/{audio_file_name}")

    return render_template('review.html', 
                           project_name=project_name, 
                           audio_file_name=audio_file_name,
                           audio_url=audio_url,
                           segments_data=segments_data,
                           json_file_name=json_file_name,
                           app_config=config)  # Átadjuk a konfigurációt app_config néven

@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    project_name = request.form.get('projectName', '').strip()
    youtube_url = request.form.get('youtubeUrl', '').strip()
    video_file = request.files.get('file')
    has_video_file = bool(video_file and video_file.filename)

    if not project_name:
        return jsonify({'error': 'Üres projektnév nem engedélyezett.'}), 400

    if has_video_file and youtube_url:
        return jsonify({'error': 'Válaszd ki, hogy fájlt töltesz fel, vagy YouTube videót töltesz le.'}), 400

    if not has_video_file and not youtube_url:
        return jsonify({'error': 'Adj meg videó fájlt vagy YouTube hivatkozást.'}), 400

    if youtube_url and not re.match(r'^https?://', youtube_url, re.IGNORECASE):
        return jsonify({'error': 'Adj meg érvényes (http/https) YouTube hivatkozást.'}), 400

    sanitized_project = secure_filename(project_name)
    if not sanitized_project:
        return jsonify({'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

    # Almappák létrehozása a projektben
    project_dir = os.path.join('workdir', sanitized_project)
    try:
        os.makedirs(project_dir, exist_ok=True)
        for subdir in config['PROJECT_SUBDIRS'].values():
            os.makedirs(os.path.join(project_dir, subdir), exist_ok=True)
    except OSError as exc:
        logging.exception("Nem sikerült létrehozni a projekt mappáit: %s", exc)
        return jsonify({'error': 'Nem sikerült létrehozni a projekt könyvtárát.'}), 500

    upload_dir = os.path.join(project_dir, config['PROJECT_SUBDIRS']['upload'])
    subtitle_file = request.files.get('subtitleFile')
    subtitle_suffix_raw = request.form.get('subtitleSuffix', '').strip()
    subtitle_filename = None
    subtitle_suffix_normalized = None

    if subtitle_file and subtitle_file.filename:
        original_subtitle_filename = secure_filename(subtitle_file.filename)
        if not original_subtitle_filename.lower().endswith('.srt'):
            return jsonify({'error': 'Csak .srt felirat fájl tölthető fel.'}), 400

        if not subtitle_suffix_raw:
            return jsonify({'error': 'A felirat feltöltéséhez kötelező kiegészítést megadni (pl. _hu).'}), 400

        if not re.fullmatch(r'_[A-Za-z]{2}', subtitle_suffix_raw):
            return jsonify({'error': 'A felirat kiegészítés formátuma: aláhúzás + kétbetűs nyelvi kód (pl. _hu).'}), 400

        subtitle_suffix_normalized = subtitle_suffix_raw.lower()

    video_filename = None
    video_path = None

    if has_video_file:
        video_filename = secure_filename(video_file.filename)
        if not video_filename:
            return jsonify({'error': 'Érvénytelen videó fájlnév.'}), 400
        video_path = os.path.join(upload_dir, video_filename)
    else:
        if not shutil.which('yt-dlp'):
            return jsonify({'error': 'A yt-dlp nem érhető el a kiszolgálón. Telepítsd a yt-dlp csomagot.'}), 500

        video_filename_base = sanitized_project
        output_template = os.path.join(upload_dir, f"{video_filename_base}.%(ext)s")
        ytdlp_cmd = [
            'yt-dlp',
            youtube_url,
            '--no-playlist',
            '--remux-video', 'mkv',
            '--merge-output-format', 'mkv',
            '-o', output_template,
        ]

        logging.info("yt-dlp parancs futtatása: %s", ' '.join(ytdlp_cmd))
        download_process = subprocess.run(
            ytdlp_cmd,
            capture_output=True,
            text=True,
        )

        if download_process.returncode != 0:
            logging.error("A yt-dlp letöltés sikertelen: %s", download_process.stderr)
            return jsonify({'error': 'Nem sikerült letölteni a YouTube videót.', 'details': download_process.stderr}), 500

        expected_mkv = os.path.join(upload_dir, f"{video_filename_base}.mkv")
        if os.path.isfile(expected_mkv):
            video_filename = os.path.basename(expected_mkv)
            video_path = expected_mkv
        else:
            candidate_files = [
                path for path in glob.glob(os.path.join(upload_dir, f"{video_filename_base}.*"))
                if not path.endswith('.part')
            ]
            if not candidate_files:
                logging.error("A yt-dlp nem hozott létre videó fájlt az elvárt névvel: %s", output_template)
                return jsonify({'error': 'A letöltött videó fájl nem található.'}), 500

            candidate_files.sort(key=os.path.getmtime, reverse=True)
            video_path = candidate_files[0]
            video_filename = os.path.basename(video_path)

    subtitle_path = None
    if subtitle_file and subtitle_file.filename:
        base_video_name = os.path.splitext(video_filename)[0]
        subtitle_filename = secure_filename(f"{base_video_name}{subtitle_suffix_normalized}.srt")
        subtitle_path = os.path.join(upload_dir, subtitle_filename)

    try:
        if has_video_file and video_path:
            video_file.save(video_path)

        if subtitle_filename and subtitle_path:
            subtitle_file.save(subtitle_path)

        return jsonify({
            'success': True,
            'message': 'Projekt sikeresen létrehozva.',
            'project': sanitized_project,
            'video': video_filename,
            'subtitle': subtitle_filename
        })
    except Exception as exc:
        logging.exception("Upload failed for project %s: %s", sanitized_project, exc)
        return jsonify({
            'error': f'Upload failed: {exc}',
            'project': sanitized_project
        }), 500

@app.route('/api/delete-project/<project_name>', methods=['DELETE'])
def delete_project(project_name):
    try:
        project_dir = os.path.join('workdir', secure_filename(project_name))
        if not os.path.exists(project_dir):
            return jsonify({'error': 'Project not found'}), 404
            
        # Remove the entire project directory
        import shutil
        shutil.rmtree(project_dir)
        
        return jsonify({
            'success': True,
            'message': 'Project deleted successfully',
            'project': project_name
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'project': project_name
        }), 500

@app.route('/api/update-segment/<project_name>', methods=['POST'])
def update_segment_api(project_name): 
    app.logger.info(f"update_segment_api called for project: {project_name}")
    try:
        data = request.get_json()
        app.logger.info(f"Received data for update: {data}")
        json_file_name = data.get('json_file_name')
        segment_index = data.get('segment_index')
        new_start = data.get('new_start')
        new_end = data.get('new_end')
        new_text = data.get('new_text', None) # Új szöveg beolvasása, None ha nincs
        new_translated_text = data.get('new_translated_text', None) # Új fordított szöveg

        # new_text nem kötelező, csak a többi
        if None in [json_file_name, segment_index] or new_start is None or new_end is None:
            return jsonify({'success': False, 'error': 'Missing data (json_file_name, segment_index, new_start, or new_end)'}), 400

        project_dir = os.path.join('workdir', secure_filename(project_name))
        # A config['PROJECT_SUBDIRS']['separated_audio_speech'] a separated_audio_speech almappa neve
        speech_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_speech'])
        
        # Fontos: a json_file_name itt már a tényleges fájlnév kell legyen, a secure_filename-t a projekt névre már alkalmaztuk.
        # Ha a json_file_name is tartalmazhat ../ vagy hasonlókat, akkor itt is kell secure_filename.
        # Mivel a json_file_name a szerverről jön (review_project), és ott os.listdir-ből, valószínűleg biztonságos.
        # De a biztonság kedvéért alkalmazhatjuk, ha a fájlnév a kliensről érkező adatból származik közvetlenül.
        # Jelen esetben a kliens a Flask által renderelt {{ json_file_name }} értéket küldi vissza, ami megbízható.
        # Ha a secure_filename itt módosítaná a fájlnevet (pl. speciális karakterek miatt), akkor nem a jó fájlt találná meg.
        # Ezért a secure_filename(json_file_name) helyett csak json_file_name-et használok, feltételezve, hogy az tiszta.
        # Ha a json_file_name a kliens által szabadon beírható lenne, akkor kellene a secure_filename.
        if not json_file_name: # Extra ellenőrzés
             return jsonify({'success': False, 'error': 'json_file_name is missing or empty in payload'}), 400
        
        # Ellenőrizzük, hogy a fájl a "translated" mappában van-e
        translated_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['translated'])
        translated_json_path = os.path.join(translated_dir_path, json_file_name)
        
        # Ha létezik a "translated" mappában, azt használjuk
        if os.path.exists(translated_json_path):
            json_full_path = translated_json_path
            app.logger.info(f"Using translated JSON file at: {json_full_path}")
        elif not os.path.exists(os.path.join(speech_dir_path, json_file_name)):
            app.logger.error(f"JSON file not found at: {json_full_path}")
            return jsonify({'success': False, 'error': f'JSON file not found: {json_file_name}'}), 404
        else:
            json_full_path = os.path.join(speech_dir_path, json_file_name)

        with open(json_full_path, 'r', encoding='utf-8') as f:
            transcription_data = json.load(f)
        
        if 'segments' not in transcription_data or not isinstance(transcription_data['segments'], list):
            return jsonify({'success': False, 'error': 'Invalid JSON structure: "segments" missing or not a list'}), 500
        
        if not (isinstance(segment_index, int) and 0 <= segment_index < len(transcription_data['segments'])):
            return jsonify({'success': False, 'error': f'Invalid segment index: {segment_index}'}), 400

        # Értékek validálása
        if not (isinstance(new_start, (int, float)) and isinstance(new_end, (int, float))):
             return jsonify({'success': False, 'error': 'new_start and new_end must be numbers'}), 400
        if new_start < 0 or new_end < 0 or new_start >= new_end:
             return jsonify({'success': False, 'error': f'Invalid start/end times: start={new_start}, end={new_end}'}), 400
        
        # Szomszédos szegmensek ellenőrzése (opcionális, de ajánlott)
        # Előző szegmens vége < új start
        if segment_index > 0:
            prev_segment_end = transcription_data['segments'][segment_index - 1].get('end')
            if prev_segment_end is not None and new_start < prev_segment_end:
                return jsonify({'success': False, 'error': f'New start time {new_start} overlaps with previous segment end {prev_segment_end}'}), 400
        # Következő szegmens eleje > új end
        if segment_index < len(transcription_data['segments']) - 1:
            next_segment_start = transcription_data['segments'][segment_index + 1].get('start')
            if next_segment_start is not None and new_end > next_segment_start:
                return jsonify({'success': False, 'error': f'New end time {new_end} overlaps with next segment start {next_segment_start}'}), 400


        transcription_data['segments'][segment_index]['start'] = new_start
        transcription_data['segments'][segment_index]['end'] = new_end
        if new_text is not None: # Csak akkor frissítjük a szöveget, ha kaptunk újat
            transcription_data['segments'][segment_index]['text'] = new_text
        if new_translated_text is not None: # Csak akkor frissítjük a fordított szöveget, ha kaptunk újat
            transcription_data['segments'][segment_index]['translated_text'] = new_translated_text
        
        # Minden szegmensből töröljük a 'words' mezőt, ha létezik.
        # Ezt a részt érdemes lehet újragondolni: ha a szöveg változik, a 'words' már nem lesz érvényes.
        # Ha a 'words' mezőt a transzkripció során generáljuk, és utána már nem használjuk fel a szerkesztés során,
        # akkor a törlése rendben van. Ha viszont a 'words' információra később szükség lehet,
        # akkor a szöveg módosításakor ezt is frissíteni kellene, vagy jelezni, hogy elavult.
        # Jelenleg a törlés egyszerűbb, és megakadályozza a konzisztenciahiányt.
        for segment_item in transcription_data['segments']: # Változónév ütközés elkerülése
            if 'words' in segment_item:
                del segment_item['words']

        with open(json_full_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)

        return jsonify({'success': True, 'message': 'Segment updated successfully'})

    except Exception as e:
        # Log the full error for debugging
        app.logger.error(f"Error updating segment for project {project_name}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'An unexpected error occurred on the server.'}), 500

@app.route('/api/add-segment/<project_name>', methods=['POST'])
def add_segment_api(project_name):
    app.logger.info(f"add_segment_api called for project: {project_name}")
    try:
        data = request.get_json()
        app.logger.info(f"Received data for new segment: {data}")
        
        json_file_name = data.get('json_file_name')
        new_start = data.get('start')
        new_end = data.get('end')
        new_text = data.get('text', 'Új szegmens') # Alapértelmezett szöveg

        if None in [json_file_name, new_start, new_end, new_text]:
            return jsonify({'success': False, 'error': 'Missing data (json_file_name, start, end, or text)'}), 400

        project_dir = os.path.join('workdir', secure_filename(project_name))
        speech_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_speech'])
        
        if not json_file_name:
             return jsonify({'success': False, 'error': 'json_file_name is missing or empty in payload'}), 400
        
        # Ellenőrizzük, hogy a fájl a "translated" mappában van-e
        translated_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['translated'])
        translated_json_path = os.path.join(translated_dir_path, json_file_name)
        
        # Ha létezik a "translated" mappában, azt használjuk
        if os.path.exists(translated_json_path):
            json_full_path = translated_json_path
            app.logger.info(f"Using translated JSON file at: {json_full_path}")
        elif not os.path.exists(os.path.join(speech_dir_path, json_file_name)):
            app.logger.error(f"JSON file not found at: {json_full_path}")
            return jsonify({'success': False, 'error': f'JSON file not found: {json_file_name}'}), 404
        else:
            json_full_path = os.path.join(speech_dir_path, json_file_name)

        app.logger.info(f"Attempting to modify JSON file at: {json_full_path}")

        if not os.path.exists(json_full_path):
            app.logger.error(f"JSON file not found at: {json_full_path}")
            return jsonify({'success': False, 'error': f'JSON file not found: {json_file_name}'}), 404

        with open(json_full_path, 'r', encoding='utf-8') as f:
            transcription_data = json.load(f)
        
        if 'segments' not in transcription_data or not isinstance(transcription_data['segments'], list):
            transcription_data['segments'] = [] # Ha nincs segments kulcs, vagy nem lista, hozzunk létre egy üreset

        # Értékek validálása
        if not (isinstance(new_start, (int, float)) and isinstance(new_end, (int, float))):
             return jsonify({'success': False, 'error': 'New start and end times must be numbers'}), 400
        if new_start < 0 or new_end < 0 or new_start >= new_end:
             return jsonify({'success': False, 'error': f'Invalid start/end times for new segment: start={new_start}, end={new_end}'}), 400
        if not isinstance(new_text, str):
            return jsonify({'success': False, 'error': 'New text must be a string'}), 400

        # Átfedés ellenőrzése a meglévő szegmensekkel
        for segment in transcription_data['segments']:
            # Új szegmens teljesen egy meglévőn belül van
            if new_start >= segment['start'] and new_end <= segment['end']:
                return jsonify({'success': False, 'error': f'New segment ({new_start}-{new_end}) is completely within an existing segment ({segment["start"]}-{segment["end"]}).'}), 400
            # Új szegmens teljesen lefedi egy meglévőt
            if new_start <= segment['start'] and new_end >= segment['end']:
                 return jsonify({'success': False, 'error': f'New segment ({new_start}-{new_end}) completely covers an existing segment ({segment["start"]}-{segment["end"]}).'}), 400
            # Új szegmens kezdete egy meglévőbe lóg
            if new_start >= segment['start'] and new_start < segment['end']:
                return jsonify({'success': False, 'error': f'New segment start ({new_start}) overlaps with existing segment ({segment["start"]}-{segment["end"]}).'}), 400
            # Új szegmens vége egy meglévőbe lóg
            if new_end > segment['start'] and new_end <= segment['end']:
                return jsonify({'success': False, 'error': f'New segment end ({new_end}) overlaps with existing segment ({segment["start"]}-{segment["end"]}).'}), 400
        
        new_segment_entry = {
            'start': new_start,
            'end': new_end,
            'text': new_text
            # 'words' kulcsot nem adunk hozzá, mert az a transzkripció eredménye
        }
        transcription_data['segments'].append(new_segment_entry)
        
        # Szegmensek rendezése start idő alapján
        transcription_data['segments'].sort(key=lambda s: s['start'])
        
        # 'words' kulcs törlése minden szegmensből (ha a szöveg vagy időzítés változik, a words elavulttá válik)
        for segment_item in transcription_data['segments']:
            if 'words' in segment_item:
                del segment_item['words']

        with open(json_full_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)

        return jsonify({'success': True, 'message': 'Segment added successfully', 'segments': transcription_data['segments']})

    except Exception as e:
        app.logger.error(f"Error adding segment for project {project_name}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'An unexpected error occurred on the server while adding segment.'}), 500

@app.route('/api/delete-segment/<project_name>', methods=['POST'])
def delete_segment_api(project_name):
    app.logger.info(f"delete_segment_api called for project: {project_name}")
    try:
        data = request.get_json()
        app.logger.info(f"Received data for delete: {data}")
        
        json_file_name = data.get('json_file_name')
        segment_index = data.get('segment_index')

        if None in [json_file_name, segment_index]:
            return jsonify({'success': False, 'error': 'Missing data (json_file_name or segment_index)'}), 400

        project_dir = os.path.join('workdir', secure_filename(project_name))
        speech_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_speech'])
        
        if not json_file_name:
             return jsonify({'success': False, 'error': 'json_file_name is missing or empty in payload'}), 400
        
        # Ellenőrizzük, hogy a fájl a "translated" mappában van-e
        translated_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['translated'])
        translated_json_path = os.path.join(translated_dir_path, json_file_name)
        
        # Ha létezik a "translated" mappában, azt használjuk
        if os.path.exists(translated_json_path):
            json_full_path = translated_json_path
            app.logger.info(f"Using translated JSON file at: {json_full_path}")
        elif not os.path.exists(os.path.join(speech_dir_path, json_file_name)):
            app.logger.error(f"JSON file not found at: {json_full_path}")
            return jsonify({'success': False, 'error': f'JSON file not found: {json_file_name}'}), 404
        else:
            json_full_path = os.path.join(speech_dir_path, json_file_name)

        app.logger.info(f"Attempting to modify JSON file for deletion at: {json_full_path}")

        if not os.path.exists(json_full_path):
            app.logger.error(f"JSON file not found at: {json_full_path}")
            return jsonify({'success': False, 'error': f'JSON file not found: {json_file_name}'}), 404

        with open(json_full_path, 'r', encoding='utf-8') as f:
            transcription_data = json.load(f)
        
        if 'segments' not in transcription_data or not isinstance(transcription_data['segments'], list):
            return jsonify({'success': False, 'error': 'Invalid JSON structure: "segments" missing or not a list'}), 500
        
        if not (isinstance(segment_index, int) and 0 <= segment_index < len(transcription_data['segments'])):
            return jsonify({'success': False, 'error': f'Invalid segment index: {segment_index}'}), 400

        # Szegmens törlése
        del transcription_data['segments'][segment_index]
        
        # 'words' kulcs törlése minden szegmensből (ha a törlés miatt az indexek eltolódnak,
        # és a 'words' információ már nem releváns vagy nehezen karbantartható)
        # Ez a lépés konzisztens a update és add funkciókkal.
        for segment_item in transcription_data['segments']:
            if 'words' in segment_item:
                del segment_item['words']

        with open(json_full_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)

        return jsonify({'success': True, 'message': 'Segment deleted successfully', 'segments': transcription_data['segments']})

    except Exception as e:
        app.logger.error(f"Error deleting segment for project {project_name}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'An unexpected error occurred on the server while deleting segment.'}), 500

@app.route('/save-workflow-keys', methods=['POST'])
def save_workflow_keys():
    data = request.get_json() or {}
    if not data:
        return jsonify({'success': False, 'error': 'Hiányzó kulcs adatok'}), 400

    keyholder_data = load_keyholder_data()
    updated = False

    field_mapping = {
        'chatgpt_api_key': 'chatgpt_api_key',
        'deepl_api_key': 'deepL_api_key',
        'huggingface_token': 'hf_token',
        'hf_token': 'hf_token',
    }

    for incoming_field, storage_field in field_mapping.items():
        raw_value = data.get(incoming_field)
        if not raw_value:
            continue
        encoded = encode_keyholder_value(raw_value)
        if not encoded:
            continue
        keyholder_data[storage_field] = encoded
        if storage_field == 'chatgpt_api_key':
            keyholder_data['api_key'] = encoded
        if storage_field == 'deepL_api_key':
            keyholder_data['deepl_api_key'] = encoded
        updated = True

    if not updated:
        return jsonify({'success': False, 'error': 'Nem került megadásra új kulcs'}), 400

    if not save_keyholder_data(keyholder_data):
        return jsonify({'success': False, 'error': 'Nem sikerült elmenteni a keyholder.json fájlt'}), 500
    try:
        return jsonify({'success': True})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/workflow-key-status/<project_name>', methods=['POST'])
def workflow_key_status(project_name):
    sanitized_project = secure_filename(project_name)
    current_config = get_config_copy()
    workdir_path = current_config['DIRECTORIES']['workdir']
    project_dir = os.path.join(workdir_path, sanitized_project)

    if not os.path.isdir(project_dir):
        return jsonify({'success': False, 'error': 'Projekt nem található.'}), 404

    data = request.get_json() or {}
    steps_payload = data.get('steps')
    if steps_payload in (None, []):
        required_keys = set()
    else:
        try:
            _, required_keys, _ = normalize_workflow_steps(steps_payload)
        except WorkflowValidationError as exc:
            return jsonify({'success': False, 'error': str(exc)}), 400

    status = determine_workflow_key_status(required_keys)
    status['success'] = True
    return jsonify(status)


@app.route('/api/workflow-templates', methods=['GET'])
def get_workflow_templates_api():
    return jsonify({'success': True, 'templates': list_workflow_templates()})


@app.route('/api/workflow-template/<template_id>', methods=['GET'])
def get_workflow_template_api(template_id):
    template = load_workflow_template(template_id)
    if not template:
        return jsonify({'success': False, 'error': 'Workflow sablon nem található.'}), 404
    return jsonify({'success': True, 'template': template})


@app.route('/api/save-workflow-template', methods=['POST'])
def save_workflow_template_api():
    data = request.get_json() or {}
    steps_payload = data.get('steps')
    if steps_payload is None:
        return jsonify({'success': False, 'error': 'Hiányzó workflow lépések.'}), 400

    try:
        normalized_steps, _, _ = normalize_workflow_steps(steps_payload)
    except WorkflowValidationError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400

    overwrite = coerce_bool(data.get('overwrite', False))
    template_id_raw = (data.get('template_id') or '').strip() or None
    name = (data.get('name') or '').strip()
    description = (data.get('description') or '').strip() or None

    if not name and template_id_raw:
        existing = load_workflow_template(template_id_raw)
        if existing:
            name = existing.get('name') or template_id_raw
            if description is None:
                description = existing.get('description')
        else:
            name = template_id_raw

    if not name:
        return jsonify({'success': False, 'error': 'A workflow nevének megadása kötelező.'}), 400

    try:
        saved = save_workflow_template_file(
            name=name,
            steps=normalized_steps,
            template_id=template_id_raw,
            overwrite=overwrite,
            description=description
        )
    except WorkflowValidationError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400

    return jsonify({'success': True, 'template': saved})


@app.route('/api/theme-colors', methods=['GET', 'POST'])
def theme_colors_api():
    if request.method == 'GET':
        return jsonify({'success': True, 'colors': load_theme_colors()})

    if not request.is_json:
        return jsonify({'success': False, 'error': 'Hiányzó JSON payload.'}), 400

    payload = request.get_json(silent=True) or {}
    light_values = payload.get('light')
    dark_values = payload.get('dark')

    if not isinstance(light_values, dict) or not isinstance(dark_values, dict):
        return jsonify({'success': False, 'error': 'Érvénytelen témabeállítások.'}), 400

    normalized_input: Dict[str, Dict[str, Any]] = {
        'light': {key: value for key, value in light_values.items() if key in THEME_COLOR_KEYS},
        'dark': {key: value for key, value in dark_values.items() if key in THEME_COLOR_KEYS},
    }

    try:
        saved = save_theme_colors(normalized_input)
    except OSError:
        return jsonify({'success': False, 'error': 'Nem sikerült elmenteni a témaszíneket.'}), 500

    return jsonify({'success': True, 'colors': saved})


@app.route('/api/workflow-options/<project_name>', methods=['GET'])
def get_workflow_options_api(project_name):
    sanitized_project = secure_filename(project_name)
    current_config = get_config_copy()
    workdir_path = current_config['DIRECTORIES']['workdir']
    project_dir = os.path.join(workdir_path, sanitized_project)
    if not os.path.isdir(project_dir):
        return jsonify({'success': False, 'error': 'Projekt nem található'}), 404

    scripts = get_scripts_catalog()
    templates = list_workflow_templates()

    saved_state = load_project_workflow_state(sanitized_project, config_snapshot=current_config)
    defaults_workflow: List[Dict[str, Any]] = []
    selected_template: Optional[str] = None

    if saved_state and isinstance(saved_state.get('steps'), list):
        defaults_workflow = copy.deepcopy(saved_state['steps'])
        selected_template = saved_state.get('template_id')
    else:
        template_data = load_workflow_template('default')
        if not template_data and templates:
            template_id = templates[0]['id']
            template_data = load_workflow_template(template_id)
        else:
            template_id = 'default'

        if template_data and isinstance(template_data.get('steps'), list):
            defaults_workflow = copy.deepcopy(template_data['steps'])
            selected_template = template_data.get('id') or template_id

        state_path = get_project_workflow_state_path(sanitized_project, config_snapshot=current_config)
        initialize_state = state_path is None or not state_path.is_file()

        if initialize_state:
            try:
                saved_state = save_project_workflow_state(
                    sanitized_project,
                    defaults_workflow,
                    selected_template,
                    config_snapshot=current_config,
                    saved_at=datetime.utcnow().isoformat()
                )
                if saved_state:
                    defaults_workflow = copy.deepcopy(saved_state.get('steps') or [])
                    if saved_state.get('template_id'):
                        selected_template = saved_state['template_id']
            except WorkflowValidationError as exc:
                logging.warning(
                    "Nem sikerült alap workflow állapotot menteni (%s): %s",
                    sanitized_project,
                    exc
                )
            except Exception as exc:  # pragma: no cover - váratlan hibák naplózása
                logging.error(
                    "Váratlan hiba történt az alap workflow állapot mentésekor (%s): %s",
                    sanitized_project,
                    exc,
                    exc_info=True
                )

    defaults = {
        'workflow': mask_workflow_secret_params(defaults_workflow),
        'selected_template': selected_template
    }

    recent_jobs = sorted(
        get_project_jobs(sanitized_project),
        key=lambda job: job.get('started_at') or job.get('created_at') or '',
        reverse=True
    )
    latest_job = recent_jobs[0] if recent_jobs else None

    return jsonify({
        'success': True,
        'scripts': scripts,
        'defaults': defaults,
        'templates': templates,
        'latest_job': latest_job,
        'project': sanitized_project
    })


@app.route('/api/project-workflow-state/<project_name>', methods=['GET', 'POST'])
def project_workflow_state_api(project_name):
    sanitized_project = secure_filename(project_name)
    if not sanitized_project:
        return jsonify({'success': False, 'error': 'Érvénytelen projektnév.'}), 400

    config_snapshot = get_config_copy()
    project_root = get_project_root_path(sanitized_project, config_snapshot=config_snapshot)
    if not project_root or not project_root.is_dir():
        return jsonify({'success': False, 'error': 'Projekt nem található.'}), 404

    if request.method == 'GET':
        state = load_project_workflow_state(sanitized_project, config_snapshot=config_snapshot)
        if not state:
            return jsonify({'success': False, 'error': 'Nincs mentett workflow állapot.'}), 404
        return jsonify({'success': True, 'state': state})

    payload = request.get_json() or {}
    steps_payload = payload.get('steps')
    try:
        normalized_steps, _, _ = normalize_workflow_steps(steps_payload)
    except WorkflowValidationError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400

    template_id = (payload.get('template_id') or '').strip() or None
    saved_at = (payload.get('saved_at') or '').strip() or datetime.utcnow().isoformat()

    try:
        state = save_project_workflow_state(
            sanitized_project,
            normalized_steps,
            template_id,
            config_snapshot=config_snapshot,
            saved_at=saved_at
        )
    except WorkflowValidationError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:  # pragma: no cover - váratlan hibák naplózása
        logging.error("Nem sikerült menteni a workflow állapotot: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': 'Nem sikerült menteni a workflow állapotot.'}), 500

    return jsonify({'success': True, 'state': state})


@app.route('/api/run-workflow/<project_name>', methods=['POST'])
def run_workflow_api(project_name):
    sanitized_project = secure_filename(project_name)
    data = request.get_json() or {}

    steps_payload = data.get('steps')
    try:
        normalized_steps, required_keys, _ = normalize_workflow_steps(steps_payload)
    except WorkflowValidationError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400

    workflow_state_payload = data.get('workflow_state')
    if workflow_state_payload is not None:
        try:
            normalized_full_steps, _, _ = normalize_workflow_steps(workflow_state_payload)
        except WorkflowValidationError as exc:
            return jsonify({'success': False, 'error': str(exc)}), 400
    else:
        normalized_full_steps = normalized_steps

    template_id = (data.get('template_id') or '').strip() or None

    current_config = get_config_copy()
    workdir_path = current_config['DIRECTORIES']['workdir']
    project_dir = os.path.join(workdir_path, sanitized_project)
    if not os.path.isdir(project_dir):
        return jsonify({'success': False, 'error': 'A projekt nem található.'}), 404

    with workflow_lock:
        active_for_project = [
            job_id for job_id, job in workflow_jobs.items()
            if job.get('project') == sanitized_project and job.get('status') in ('queued', 'running', 'cancelling')
        ]
    if active_for_project:
        return jsonify({'success': False, 'error': 'Már fut egy workflow ehhez a projekthez.'}), 409

    key_status = determine_workflow_key_status(required_keys)
    missing_keys = [
        info['label']
        for key, info in key_status['keys'].items()
        if info.get('required') and not info.get('present')
    ]
    if missing_keys:
        return jsonify({
            'success': False,
            'error': 'Hiányzó API kulcsok: ' + ', '.join(missing_keys),
            'missing_keys': missing_keys
        }), 400

    encoded_full_steps = mask_workflow_secret_params(normalized_full_steps)
    encoded_steps = mask_workflow_secret_params(normalized_steps)
    saved_timestamp = datetime.utcnow().isoformat()
    try:
        save_project_workflow_state(
            sanitized_project,
            normalized_full_steps,
            template_id,
            config_snapshot=current_config,
            saved_at=saved_timestamp
        )
    except WorkflowValidationError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500
    except Exception as exc:  # pragma: no cover - váratlan hibák naplózása
        logging.error("Nem sikerült menteni a workflow állapotot: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': 'Nem sikerült menteni a workflow állapotot.'}), 500

    job_id = uuid.uuid4().hex
    job_data = {
        'job_id': job_id,
        'project': sanitized_project,
        'status': 'queued',
        'created_at': datetime.utcnow().isoformat(),
        'message': 'Feladat sorban áll.',
        'log': None,
        'cancel_requested': False,
        'workflow': encoded_full_steps,
        'execution_steps': encoded_steps,
        'required_keys': list(required_keys),
        'template_id': template_id
    }
    register_workflow_job(job_id, job_data)

    thread = threading.Thread(
        target=run_workflow_job,
        args=(job_id, sanitized_project, {'steps': normalized_steps, 'workflow_state': normalized_full_steps, 'template_id': template_id}),
        daemon=True
    )
    set_workflow_thread(job_id, thread)
    thread.start()

    return jsonify({'success': True, 'job_id': job_id})


@app.route('/api/stop-workflow/<job_id>', methods=['POST'])
def stop_workflow(job_id):
    job = get_workflow_job(job_id)
    if not job:
        return jsonify({'success': False, 'error': 'Feladat nem található.'}), 404

    if job.get('status') in ('completed', 'failed', 'cancelled'):
        return jsonify({'success': False, 'error': 'A feladat már befejeződött.'}), 400

    if job.get('cancel_requested'):
        return jsonify({'success': True, 'message': 'Megszakítás már folyamatban.'})

    if not request_workflow_cancel(job_id):
        return jsonify({'success': False, 'error': 'A feladat már nem fut.'}), 409

    update_workflow_job(
        job_id,
        status='cancelling',
        message='Megszakítás kérése folyamatban...',
        cancel_requested=True
    )
    return jsonify({'success': True, 'message': 'Megszakítás kérve. Várakozás a leállásra.'})


@app.route('/api/workflow-log/<job_id>', methods=['GET'])
def get_workflow_log(job_id):
    job = get_workflow_job(job_id)
    if not job:
        return jsonify({'success': False, 'error': 'Feladat nem található.'}), 404

    log_info = job.get('log') or {}
    log_relative = log_info.get('relative')
    log_text = ""
    log_available = False

    if log_relative:
        current_config = get_config_copy()
        workdir_path = current_config['DIRECTORIES']['workdir']
        absolute_path = resolve_workspace_path(os.path.join(workdir_path, log_relative))
        if absolute_path and os.path.exists(absolute_path):
            log_available = True
            log_text = read_log_tail(absolute_path)

    completed = job.get('status') in ('completed', 'failed', 'cancelled')

    return jsonify({
        'success': True,
        'log': log_text,
        'log_available': log_available,
        'status': job.get('status'),
        'completed': completed,
        'cancel_requested': job.get('cancel_requested', False)
    })


@app.route('/api/workflow-status/<job_id>', methods=['GET'])
def get_workflow_status(job_id):
    job = get_workflow_job(job_id)
    if not job:
        return jsonify({'success': False, 'error': 'Feladat nem található'}), 404
    return jsonify({'success': True, 'job': job})

initialize_scripts_catalog()


if __name__ == '__main__':
    import logging
    import os

    debug_flag = os.environ.get('FLASK_DEBUG', '')
    debug_enabled = debug_flag.lower() in {'1', 'true', 'yes', 'on'}

    if not debug_enabled:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    app.run(debug=debug_enabled, host='0.0.0.0')

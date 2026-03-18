from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    url_for,
    send_file,
    after_this_request,
    make_response
)
from flask_compress import Compress
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
import tarfile
import zipfile
import stat
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import wave
import math
from collections import OrderedDict
from services.script_meta import validate_script_meta
from routes.files_api import register_files_api_routes
from routes.pages import register_page_routes
from routes.review_api import register_review_api_routes
from routes.workflow_api import register_workflow_api_routes

app = Flask(__name__)

# Enable gzip compression for large HTML/JSON responses sent to the frontend.
app.config.setdefault(
    "COMPRESS_MIMETYPES",
    [
        "text/html",
        "text/css",
        "text/xml",
        "text/plain",
        "application/json",
        "application/javascript",
        "text/javascript",
    ],
)
app.config.setdefault("COMPRESS_LEVEL", 6)
app.config.setdefault("COMPRESS_MIN_SIZE", 1024)
Compress(app)

logging.basicConfig(level=logging.INFO)

config_lock = threading.Lock()
workflow_lock = threading.Lock()
theme_config_lock = threading.Lock()
workflow_jobs = {}
workflow_threads = {}
workflow_events = {}
review_audio_encoding_jobs: Dict[str, Dict[str, Any]] = {}
review_audio_encoding_lock = threading.Lock()

CONFIG_FILE_PATH = Path(app.root_path) / 'config.json'
CONFIG_MTIME: Optional[float] = None
KEYHOLDER_PATH = os.path.join(app.root_path, 'keyholder.json')
CONDA_PYTHON_CACHE = {}

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'}
VIDEO_EXTENSIONS = {
    '.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv', '.flv', '.mts', '.m2ts', '.mpg', '.mpeg'
}

DEFAULT_UI_LANGUAGE = 'hun'
UI_LANGUAGE_COOKIE = 'ui_language'
language_cache_lock = threading.Lock()
language_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}


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


def sanitize_segment_strings(segments: Any) -> Any:
    """
    Remove stray escape sequences that break JSON loading on the front-end.
    """
    if not isinstance(segments, list):
        return segments

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        for key in ('text', 'translated_text'):
            value = segment.get(key)
            if isinstance(value, str):
                segment[key] = value.replace('\\"', '"')
    return segments


def format_time_for_filename(time_in_seconds: Any) -> str:
    """
    Convert a time value (seconds) into the HH-MM-SS-mmm pattern used for split filenames.
    """
    try:
        time_float = float(time_in_seconds)
    except (TypeError, ValueError):
        return "00-00-00-000"

    total_milliseconds = int(round(time_float * 1000))
    if total_milliseconds < 0:
        total_milliseconds = 0

    hours = total_milliseconds // 3_600_000
    minutes = (total_milliseconds % 3_600_000) // 60_000
    seconds = (total_milliseconds % 60_000) // 1_000
    milliseconds = total_milliseconds % 1_000

    return f"{hours:02d}-{minutes:02d}-{seconds:02d}-{milliseconds:03d}"


def annotate_segments_with_translated_splits(project_dir: str, segments: List[Dict[str, Any]]) -> None:
    """
    Mark each segment with a flag indicating whether a translated split WAV exists.
    """
    try:
        project_subdirs = config.get('PROJECT_SUBDIRS') if isinstance(config, dict) else {}
    except NameError:
        project_subdirs = {}

    translated_splits_subdir = (project_subdirs or {}).get('translated_splits')
    base_dir = Path(project_dir) / translated_splits_subdir if translated_splits_subdir else None
    base_dir_exists = base_dir.exists() if base_dir else False

    for segment in segments:
        has_split = False
        if isinstance(segment, dict):
            start = segment.get('start')
            end = segment.get('end')
            if (
                base_dir_exists
                and isinstance(start, (int, float))
                and isinstance(end, (int, float))
            ):
                filename = f"{format_time_for_filename(start)}_{format_time_for_filename(end)}.wav"
                has_split = (base_dir / filename).is_file()
        segment['has_translated_split'] = has_split


def prepare_segments_for_response(project_dir: str, segments: Any) -> List[Dict[str, Any]]:
    """
    Return a sanitized, annotated copy of the segment list for front-end consumption.
    """
    if not isinstance(segments, list):
        return []
    prepared_segments: List[Dict[str, Any]] = copy.deepcopy(segments)
    sanitize_segment_strings(prepared_segments)
    annotate_segments_with_translated_splits(project_dir, prepared_segments)
    return prepared_segments


def collect_translated_split_progress(project_name: str, config_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    current_config = config_snapshot or get_config_copy()
    directories = current_config.get('DIRECTORIES') or {}
    project_subdirs = current_config.get('PROJECT_SUBDIRS') or {}

    workdir_rel = directories.get('workdir')
    translated_rel = project_subdirs.get('translated')
    translated_splits_rel = project_subdirs.get('translated_splits')
    if not workdir_rel or not translated_rel or not translated_splits_rel:
        raise WorkflowValidationError(
            "Hiányzó config kulcs: DIRECTORIES.workdir, PROJECT_SUBDIRS.translated vagy PROJECT_SUBDIRS.translated_splits."
        )

    safe_project = secure_filename(project_name)
    project_dir = Path(workdir_rel) / safe_project
    if not project_dir.is_dir():
        raise FileNotFoundError(f"A projekt könyvtár nem található: {project_dir}")

    translated_dir = project_dir / translated_rel
    if not translated_dir.is_dir():
        raise FileNotFoundError(f"A translated könyvtár nem található: {translated_dir}")

    translated_json_files = sorted(
        path for path in translated_dir.iterdir()
        if path.is_file() and path.suffix.lower() == '.json'
    )
    if not translated_json_files:
        raise FileNotFoundError(f"Nem található JSON fájl a translated könyvtárban: {translated_dir}")

    selected_json_path = translated_json_files[0]
    try:
        with selected_json_path.open('r', encoding='utf-8') as file:
            payload = json.load(file)
    except OSError as exc:
        raise FileNotFoundError(f"Nem sikerült beolvasni a translated JSON fájlt: {selected_json_path}") from exc
    except json.JSONDecodeError as exc:
        raise WorkflowValidationError(f"Hibás JSON formátum: {selected_json_path.name}") from exc

    segments = payload.get('segments')
    if not isinstance(segments, list):
        raise WorkflowValidationError(
            f'A kiválasztott translated JSON fájl nem tartalmaz érvényes "segments" listát: {selected_json_path.name}'
        )

    expected_segment_stems: List[str] = []
    translated_ready_segments = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        start = segment.get('start')
        end = segment.get('end')
        original_text = str(segment.get('text') or '').strip()
        translated_text = str(segment.get('translated_text') or '').strip()
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if not original_text:
            continue
        expected_segment_stems.append(f"{format_time_for_filename(start)}_{format_time_for_filename(end)}")
        if translated_text:
            translated_ready_segments += 1

    translated_splits_dir = project_dir / translated_splits_rel
    actual_audio_stems: Set[str] = set()
    if translated_splits_dir.is_dir():
        for path in translated_splits_dir.iterdir():
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                actual_audio_stems.add(path.stem)

    completed_segments = sum(1 for stem in expected_segment_stems if stem in actual_audio_stems)
    expected_segments = len(expected_segment_stems)

    return {
        'project_name': safe_project,
        'json_file_name': selected_json_path.name,
        'translated_dir': translated_rel,
        'translated_splits_dir': translated_splits_rel,
        'total_segments': len(segments),
        'expected_segments': expected_segments,
        'translated_ready_segments': translated_ready_segments,
        'completed_segments': completed_segments,
        'missing_segments': max(expected_segments - completed_segments, 0),
        'actual_audio_files': len(actual_audio_stems),
        'translated_splits_exists': translated_splits_dir.is_dir()
    }


def resolve_project_paths(project_name: str) -> Path:
    safe_project = secure_filename(project_name)
    return Path('workdir') / safe_project


def resolve_source_audio_path(project_name: str, audio_file_name: str) -> Optional[Path]:
    if not project_name or not audio_file_name:
        return None
    project_root = resolve_project_paths(project_name)
    speech_subdir = (config.get('PROJECT_SUBDIRS') or {}).get('separated_audio_speech')
    if not speech_subdir:
        return None
    candidate = project_root / speech_subdir / os.path.basename(audio_file_name)
    if candidate.is_file():
        return candidate
    return None


def get_review_encoded_audio_path(project_name: str, audio_file_name: str) -> Optional[Path]:
    temp_subdir = (config.get('PROJECT_SUBDIRS') or {}).get('temp')
    if not temp_subdir or not audio_file_name:
        return None
    project_root = resolve_project_paths(project_name)
    temp_dir = project_root / temp_subdir
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logging.error("Failed to create temp dir %s: %s", temp_dir, exc)
        return None
    source_stem = Path(audio_file_name).stem or Path(audio_file_name).name
    encoded_name = f"{source_stem}_review_preview.mp3"
    return temp_dir / encoded_name


def probe_audio_duration(audio_path: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v',
                'error',
                '-show_entries',
                'format=duration',
                '-of',
                'default=noprint_wrappers=1:nokey=1',
                str(audio_path)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except Exception as exc:
        logging.warning("Failed to probe duration for %s: %s", audio_path, exc)
        return None


def _run_review_audio_encoding_job(project_name: str, source_path: Path, target_path: Path, job: Dict[str, Any]) -> None:
    job['status'] = 'encoding'
    job['progress'] = 0.0
    duration_seconds = probe_audio_duration(source_path)
    command = [
        'ffmpeg',
        '-y',
        '-i',
        str(source_path),
        '-ac',
        '1',
        '-ar',
        '44100',
        '-b:a',
        '128k',
        '-progress',
        'pipe:1',
        '-nostats',
        str(target_path)
    ]
    proc = None
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        if proc.stdout:
            for line in proc.stdout:
                line = line.strip()
                if duration_seconds and line.startswith('out_time_ms='):
                    try:
                        current_ms = float(line.split('=')[1])
                        progress = (current_ms / (duration_seconds * 1000.0)) * 100.0
                        job['progress'] = max(0.0, min(99.0, progress))
                    except (ValueError, ZeroDivisionError):
                        continue
        proc.wait()
        if proc.returncode == 0 and target_path.exists():
            job['progress'] = 100.0
            job['status'] = 'completed'
        else:
            job['status'] = 'failed'
            job['error'] = f"ffmpeg exited with code {proc.returncode}"
            if target_path.exists():
                try:
                    target_path.unlink()
                except OSError:
                    pass
    except Exception as exc:
        job['status'] = 'failed'
        job['error'] = str(exc)
        if target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                pass
    finally:
        if proc and proc.stdout:
            proc.stdout.close()
        with review_audio_encoding_lock:
            review_audio_encoding_jobs.pop(project_name, None)


def find_matching_audio_file(base_name: str, directory: str) -> Optional[str]:
    """
    Find the first audio file in directory whose stem matches base_name.
    Preference order follows PREFERRED_AUDIO_EXTENSIONS.
    """
    for extension in PREFERRED_AUDIO_EXTENSIONS:
        candidate = base_name + extension
        candidate_path = Path(directory) / candidate
        if candidate_path.is_file():
            return candidate
    return None


def delete_translated_split_file(project_dir: str, start: Any, end: Any) -> bool:
    """
    Delete the translated split WAV file for the provided time window, if it exists.
    """
    try:
        project_subdirs = config.get('PROJECT_SUBDIRS') if isinstance(config, dict) else {}
    except NameError:
        project_subdirs = {}

    translated_splits_subdir = (project_subdirs or {}).get('translated_splits')
    if not translated_splits_subdir:
        return False

    try:
        start_float = float(start)
        end_float = float(end)
    except (TypeError, ValueError):
        return False

    base_dir = Path(project_dir) / translated_splits_subdir
    if not base_dir.exists():
        return False

    filename = f"{format_time_for_filename(start_float)}_{format_time_for_filename(end_float)}.wav"
    file_path = (base_dir / filename).resolve()
    project_root = Path(project_dir).resolve()

    if not is_subpath(str(file_path), str(project_root)):
        logging.warning("Skipping translated split deletion outside project scope: %s", file_path)
        return False

    try:
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            logging.info("Deleted translated split file: %s", file_path)
            return True
    except OSError as exc:
        logging.warning("Failed to delete translated split file %s: %s", file_path, exc)
    return False

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
PREFERRED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.ogg', '.flac']

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
SECRET_PARAM_NAMES = {'auth_key', 'api_key', 'hf_token'}
NEGATIVE_FLAG_NAME_PREFIXES: Tuple[str, ...] = ('no_', 'disable_', 'skip_', 'without_')
ENCODED_SECRET_PREFIX = 'base64:'
SECRET_VALUE_PLACEHOLDER = '***'
ALLOWED_WORKFLOW_WIDGETS = {'reviewContinue', 'cycleWidget', 'translatedSplitLoopWidget'}


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


def sanitize_storage_relative_path(path_value: str, *, allow_empty: bool = False) -> str:
    normalized = (path_value or '').strip().strip('/\\')
    if not normalized:
        if allow_empty:
            return ''
        raise ValueError('Adj meg érvényes almappa nevet.')
    segments = [segment for segment in re.split(r'[\\/]+', normalized) if segment]
    if not segments:
        if allow_empty:
            return ''
        raise ValueError('Adj meg érvényes almappa nevet.')
    sanitized_segments = []
    for segment in segments:
        safe_segment = secure_filename(segment)
        if not safe_segment:
            raise ValueError('Az útvonal érvénytelen karaktereket tartalmaz.')
        sanitized_segments.append(safe_segment)
    return '/'.join(sanitized_segments)


def get_tts_root_directory(config_snapshot: Dict[str, Any]) -> Optional[str]:
    directories = config_snapshot.get('DIRECTORIES', {}) if isinstance(config_snapshot, dict) else {}
    tts_dir_value = directories.get('TTS')
    if not tts_dir_value:
        return None
    return resolve_workspace_path(tts_dir_value)


def safe_extract_tar(archive: tarfile.TarFile, destination: str) -> None:
    for member in archive.getmembers():
        member_name = member.name or ''
        if not member_name:
            continue
        member_path = os.path.abspath(os.path.join(destination, member_name))
        if not is_subpath(member_path, destination):
            raise ValueError('Az archívum érvénytelen elérési utakat tartalmaz.')
        if member.issym() or member.islnk():
            raise ValueError('Az archívum szimbolikus linkeket tartalmaz, ami nem támogatott.')
    archive.extractall(path=destination)


def safe_extract_zip(archive: zipfile.ZipFile, destination: str) -> None:
    for info in archive.infolist():
        member_name = info.filename or ''
        if not member_name:
            continue
        if info.create_system == 3:
            permissions = info.external_attr >> 16
            if stat.S_ISLNK(permissions):
                raise ValueError('Az archívum szimbolikus linkeket tartalmaz, ami nem támogatott.')
        member_path = os.path.abspath(os.path.join(destination, member_name))
        if not is_subpath(member_path, destination):
            raise ValueError('Az archívum érvénytelen elérési utakat tartalmaz.')
        if info.is_dir():
            os.makedirs(member_path, exist_ok=True)
            continue
        parent_dir = os.path.dirname(member_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with archive.open(info, 'r') as source, open(member_path, 'wb') as target:
            shutil.copyfileobj(source, target)


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

        issues = validate_script_meta(json_path, entry, SCRIPTS_DIR)
        for issue in issues:
            log_message = "%s: %s"
            if issue.level == 'error':
                logging.error(log_message, issue.path, issue.message)
            else:
                logging.warning(log_message, issue.path, issue.message)

        # biztosítsuk, hogy a script mező a valós relatív útvonalra mutat
        relative_script = relative_json.with_suffix('.py').as_posix()
        entry = copy.deepcopy(entry)
        entry['script'] = relative_script

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

    def humanize_param_name(name: str) -> str:
        return name.replace('_', ' ').strip()

    def strip_negative_prefix(name: str) -> str:
        for prefix in NEGATIVE_FLAG_NAME_PREFIXES:
            if name.startswith(prefix):
                stripped = name[len(prefix):]
                if stripped:
                    return stripped
        return name

    def resolve_flag_mode(name: str, flags: List[str]) -> Tuple[str, str]:
        positive_flag = next((flag for flag in flags if not flag.startswith('--no-')), None)
        negative_flag = next((flag for flag in flags if flag.startswith('--no-')), None)
        if negative_flag and not positive_flag:
            if any(name.startswith(prefix) for prefix in NEGATIVE_FLAG_NAME_PREFIXES):
                return 'negative_only_negative', humanize_param_name(strip_negative_prefix(name))
            return 'negative_only_positive', humanize_param_name(name)
        return 'standard', humanize_param_name(name)

    def append_params(param_list, required: bool):
        for param in param_list or []:
            name = param.get('name')
            if not name:
                continue
            param_type = param.get('type', 'option')
            flags = param.get('flags') or []
            default_value = param.get('default')
            flag_mode = 'standard'
            ui_name = humanize_param_name(name)
            if param_type == 'flag':
                flag_mode, ui_name = resolve_flag_mode(name, flags)
            parameters.append({
                'name': name,
                'ui_name': ui_name,
                'type': param_type,
                'flags': flags,
                'flag_mode': flag_mode,
                'required': required,
                'autofill': infer_autofill_kind(name),
                'secret': name in SECRET_PARAM_NAMES,
                'default': default_value,
                'description': param.get('description')
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


def normalize_translated_split_loop_widget_params(raw_params: Dict[str, Any]) -> Dict[str, int]:
    if raw_params is None:
        params = {}
    elif isinstance(raw_params, dict):
        params = raw_params
    else:
        raise WorkflowValidationError("A translated split loop widget paraméterei hibás formátumúak.")

    value = params.get('allowed_missing_segments')
    candidate = '' if value is None else str(value).strip()
    if candidate == '':
        candidate = '0'
    try:
        numeric = int(candidate)
    except (TypeError, ValueError):
        raise WorkflowValidationError(
            "A translated split loop widget allowed_missing_segments paramétere csak nemnegatív egész szám lehet."
        ) from None
    if numeric < 0:
        raise WorkflowValidationError(
            "A translated split loop widget allowed_missing_segments paramétere csak nemnegatív egész szám lehet."
        )

    return {
        'allowed_missing_segments': numeric
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

        flag_mode = param_meta.get('flag_mode', 'standard')
        positive_flag = next((flag for flag in flags if not flag.startswith('--no-')), None)
        negative_flag = next((flag for flag in flags if flag.startswith('--no-')), None)

        if flag_mode == 'negative_only_negative':
            if value is True and negative_flag:
                return [negative_flag]
            return []

        if flag_mode == 'negative_only_positive':
            if value is False and negative_flag:
                return [negative_flag]
            return []

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
            elif widget_id == 'translatedSplitLoopWidget':
                normalized_widget_step['params'] = normalize_translated_split_loop_widget_params(widget_params)
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
with CONFIG_FILE_PATH.open('r', encoding='utf-8') as config_file:
    config = json.load(config_file)
CONFIG_MTIME = CONFIG_FILE_PATH.stat().st_mtime

def get_config_copy():
    global config, CONFIG_MTIME
    with config_lock:
        current_mtime = CONFIG_FILE_PATH.stat().st_mtime
        if CONFIG_MTIME is None or current_mtime != CONFIG_MTIME:
            with CONFIG_FILE_PATH.open('r', encoding='utf-8') as config_file:
                config = json.load(config_file)
            CONFIG_MTIME = current_mtime
        return copy.deepcopy(config)


def persist_config(updated_config):
    global config, CONFIG_MTIME
    with config_lock:
        config = updated_config
        with CONFIG_FILE_PATH.open('w', encoding='utf-8') as config_file:
            json.dump(config, config_file, indent=2, ensure_ascii=False)
        CONFIG_MTIME = CONFIG_FILE_PATH.stat().st_mtime


def _normalize_language_code(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    normalized = str(code).strip().lower()
    return normalized or None


def get_languages_root_directory() -> Path:
    directories = {}
    try:
        directories = config.get('DIRECTORIES') if isinstance(config, dict) else {}
    except NameError:
        directories = {}
    languages_rel_path = (directories or {}).get('languages', 'languages')
    return Path(app.root_path) / languages_rel_path


def list_language_directories() -> List[str]:
    languages_root = get_languages_root_directory()
    if not languages_root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in languages_root.iterdir()
        if entry.is_dir()
    )


def load_language_payload(lang_code: Optional[str], file_name: str = 'index') -> Optional[Dict[str, Any]]:
    normalized_code = _normalize_language_code(lang_code)
    if not normalized_code:
        return None

    languages_root = get_languages_root_directory()
    lang_file = languages_root / normalized_code / f'{file_name}.json'
    if not lang_file.is_file():
        return None

    file_mtime = lang_file.stat().st_mtime
    cache_key = (normalized_code, file_name)
    with language_cache_lock:
        cached = language_cache.get(cache_key)
        if cached and cached.get('mtime') == file_mtime:
            return cached.get('data')

    try:
        with lang_file.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Nem sikerült beolvasni a(z) %s nyelvi fájlt: %s", lang_file, exc)
        return None

    with language_cache_lock:
        language_cache[cache_key] = {
            'mtime': file_mtime,
            'data': payload
        }

    return payload


def get_language_strings(lang_code: Optional[str], file_name: str = 'index') -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = load_language_payload(lang_code, file_name=file_name)
    if not isinstance(payload, dict):
        return {}, {}

    strings = payload.get('strings')
    if not isinstance(strings, dict):
        strings = {key: value for key, value in payload.items() if key != 'meta'}

    meta = payload.get('meta')
    if not isinstance(meta, dict):
        meta = {}

    return strings, meta


def list_language_options() -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []
    for code in list_language_directories():
        payload = load_language_payload(code, file_name='index')
        meta = payload.get('meta') if isinstance(payload, dict) else {}
        label = None
        if isinstance(meta, dict):
            label = meta.get('label') or meta.get('name') or meta.get('native_label')
        options.append({
            'code': code,
            'label': label or code.upper()
        })
    return options


def build_translation_helper(strings: Optional[Dict[str, Any]]):
    safe_strings = strings or {}

    def translate(key: str, default: Optional[str] = None, **kwargs):
        value: Any = safe_strings
        for part in key.split('.'):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break

        if value is None:
            value = default if default is not None else key

        if isinstance(value, str) and kwargs:
            try:
                value = value.format(**kwargs)
            except KeyError:
                pass
        return value

    return translate


def resolve_language_context(preferred_code: Optional[str], file_name: str = 'index') -> Dict[str, Any]:
    available_codes = list_language_directories()
    normalized_preference = _normalize_language_code(preferred_code)
    requested_code = normalized_preference if normalized_preference in available_codes else None

    strings: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    resolved_code: Optional[str] = None

    if requested_code:
        candidate_strings, candidate_meta = get_language_strings(requested_code, file_name=file_name)
        if candidate_strings:
            strings = candidate_strings
            meta = candidate_meta
            resolved_code = requested_code

    if not resolved_code:
        fallback_priority: List[str] = []
        if DEFAULT_UI_LANGUAGE in available_codes:
            fallback_priority.append(DEFAULT_UI_LANGUAGE)
        fallback_priority.extend(code for code in available_codes if code not in fallback_priority)
        if requested_code and requested_code not in fallback_priority:
            fallback_priority.append(requested_code)

        for candidate in fallback_priority:
            candidate_strings, candidate_meta = get_language_strings(candidate, file_name=file_name)
            if candidate_strings:
                strings = candidate_strings
                meta = candidate_meta
                resolved_code = candidate
                break

    if not strings:
        strings = {}
    if not meta:
        meta = {}
    if not resolved_code:
        resolved_code = requested_code or DEFAULT_UI_LANGUAGE

    return {
        'requested': requested_code,
        'resolved': resolved_code,
        'strings': strings,
        'meta': meta
    }


def render_with_language(template_name: str, *, file_name: str = 'index', **context):
    preferred_language = request.args.get('lang') or request.cookies.get(UI_LANGUAGE_COOKIE)
    language_context = resolve_language_context(preferred_language, file_name=file_name)
    translation_helper = build_translation_helper(language_context.get('strings'))
    language_options = list_language_options()

    active_language_code = (
        language_context.get('requested')
        or language_context.get('resolved')
        or DEFAULT_UI_LANGUAGE
    )
    resolved_language_code = language_context.get('resolved') or DEFAULT_UI_LANGUAGE
    language_meta = language_context.get('meta') or {}
    html_lang = language_meta.get('html_lang') or language_meta.get('code') or resolved_language_code

    context.update({
        'translations': language_context.get('strings') or {},
        'translate': translation_helper,
        'language_options': language_options,
        'active_language': active_language_code,
        'resolved_language': resolved_language_code,
        'language_meta': language_meta,
        'html_lang': html_lang
    })

    response = make_response(render_template(template_name, **context))
    cookie_value = request.cookies.get(UI_LANGUAGE_COOKIE)
    should_update_cookie = (
        active_language_code
        and active_language_code != cookie_value
        and (request.args.get('lang') or not cookie_value)
    )
    if should_update_cookie:
        response.set_cookie(
            UI_LANGUAGE_COOKIE,
            active_language_code,
            max_age=60 * 60 * 24 * 365,
            samesite='Lax'
        )
    return response

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


register_page_routes(
    app,
    {
        'workdir_path': 'workdir',
        'build_project_entries': build_project_entries,
        'render_with_language': render_with_language,
        'config': config,
        'secure_filename': secure_filename,
        'compute_failed_generation_highlights': compute_failed_generation_highlights,
        'get_audio_metadata_directories': get_audio_metadata_directories,
        'get_failed_generation_directories': get_failed_generation_directories,
        'should_enable_failed_move': should_enable_failed_move,
        'build_audio_metadata': build_audio_metadata,
        'build_failed_generation_json_metadata': build_failed_generation_json_metadata,
        'get_tts_root_directory': get_tts_root_directory,
        'audio_extensions': AUDIO_EXTENSIONS,
        'video_extensions': VIDEO_EXTENSIONS,
        'audio_mime_map': AUDIO_MIME_MAP,
        'video_mime_map': VIDEO_MIME_MAP,
        'secret_param_names': sorted(SECRET_PARAM_NAMES),
        'prepare_segments_for_response': prepare_segments_for_response,
        'get_review_encoded_audio_path': get_review_encoded_audio_path,
    },
)

register_review_api_routes(
    app,
    {
        'config': config,
        'secure_filename': secure_filename,
        'resolve_source_audio_path': resolve_source_audio_path,
        'get_review_encoded_audio_path': get_review_encoded_audio_path,
        'review_audio_encoding_lock': review_audio_encoding_lock,
        'review_audio_encoding_jobs': review_audio_encoding_jobs,
        '_run_review_audio_encoding_job': _run_review_audio_encoding_job,
        'prepare_segments_for_response': prepare_segments_for_response,
        'delete_translated_split_file': delete_translated_split_file,
        'get_config_copy': get_config_copy,
        'sanitize_segment_strings': sanitize_segment_strings,
        'find_matching_audio_file': find_matching_audio_file,
        'get_project_workflow_state_path': get_project_workflow_state_path,
        'normalize_workflow_steps': normalize_workflow_steps,
        'WorkflowValidationError': WorkflowValidationError,
        'workflow_lock': workflow_lock,
        'workflow_jobs': workflow_jobs,
        'determine_workflow_key_status': determine_workflow_key_status,
        'mask_workflow_secret_params': mask_workflow_secret_params,
        'register_workflow_job': register_workflow_job,
        'run_workflow_job': run_workflow_job,
        'set_workflow_thread': set_workflow_thread,
    },
)

register_workflow_api_routes(
    app,
    {
        'secure_filename': secure_filename,
        'get_config_copy': get_config_copy,
        'normalize_workflow_steps': normalize_workflow_steps,
        'WorkflowValidationError': WorkflowValidationError,
        'determine_workflow_key_status': determine_workflow_key_status,
        'list_workflow_templates': list_workflow_templates,
        'load_workflow_template': load_workflow_template,
        'coerce_bool': coerce_bool,
        'save_workflow_template_file': save_workflow_template_file,
        'get_scripts_catalog': get_scripts_catalog,
        'load_project_workflow_state': load_project_workflow_state,
        'get_project_workflow_state_path': get_project_workflow_state_path,
        'save_project_workflow_state': save_project_workflow_state,
        'mask_workflow_secret_params': mask_workflow_secret_params,
        'get_project_jobs': get_project_jobs,
        'get_project_root_path': get_project_root_path,
        'workflow_lock': workflow_lock,
        'workflow_jobs': workflow_jobs,
        'register_workflow_job': register_workflow_job,
        'run_workflow_job': run_workflow_job,
        'set_workflow_thread': set_workflow_thread,
        'get_workflow_job': get_workflow_job,
        'request_workflow_cancel': request_workflow_cancel,
        'update_workflow_job': update_workflow_job,
        'resolve_workspace_path': resolve_workspace_path,
        'read_log_tail': read_log_tail,
        'load_keyholder_data': load_keyholder_data,
        'encode_keyholder_value': encode_keyholder_value,
        'save_keyholder_data': save_keyholder_data,
    },
)

register_files_api_routes(
    app,
    {
        'secure_filename': secure_filename,
        'get_config_copy': get_config_copy,
        'is_subpath': is_subpath,
        'get_project_jobs': get_project_jobs,
        'compute_failed_generation_highlights': compute_failed_generation_highlights,
        'get_audio_metadata_directories': get_audio_metadata_directories,
        'get_failed_generation_directories': get_failed_generation_directories,
        'collect_directory_entries': collect_directory_entries,
        'get_tts_root_directory': get_tts_root_directory,
        'sanitize_storage_relative_path': sanitize_storage_relative_path,
        'safe_extract_tar': safe_extract_tar,
        'safe_extract_zip': safe_extract_zip,
        'audio_extensions': AUDIO_EXTENSIONS,
    },
)


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


@app.route('/api/translated-split-progress/<project_name>', methods=['GET'])
def translated_split_progress_api(project_name):
    sanitized_project = secure_filename(project_name)
    try:
        progress = collect_translated_split_progress(sanitized_project, config_snapshot=get_config_copy())
    except FileNotFoundError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 404
    except WorkflowValidationError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:  # pragma: no cover - váratlan hibák naplózása
        logging.error(
            "Nem sikerült lekérdezni a translated split előrehaladást (%s): %s",
            sanitized_project,
            exc,
            exc_info=True
        )
        return jsonify({'success': False, 'error': 'Nem sikerült lekérdezni a translated split előrehaladást.'}), 500

    return jsonify({'success': True, 'progress': progress})


initialize_scripts_catalog()


if __name__ == '__main__':
    import logging
    import os

    debug_flag = os.environ.get('FLASK_DEBUG', '')
    debug_enabled = debug_flag.lower() in {'1', 'true', 'yes', 'on'}

    if not debug_enabled:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    app.run(debug=debug_enabled, host='0.0.0.0')

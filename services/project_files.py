from __future__ import annotations

import json
import logging
import math
import os
import re
import wave
from typing import Any, Callable, Dict, List, Optional, Set


FILENAME_RANGE_PATTERN = re.compile(r'^(\d{2}-\d{2}-\d{2}-\d{3})_(\d{2}-\d{2}-\d{2}-\d{3})')


def sanitize_storage_relative_path(
    path_value: str,
    secure_filename: Callable[[str], str],
    *,
    allow_empty: bool = False,
) -> str:
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


def get_tts_root_directory(
    config_snapshot: Dict[str, Any],
    resolve_workspace_path: Callable[[str], Optional[str]],
) -> Optional[str]:
    directories = config_snapshot.get('DIRECTORIES', {}) if isinstance(config_snapshot, dict) else {}
    tts_dir_value = directories.get('TTS')
    if not tts_dir_value:
        return None
    return resolve_workspace_path(tts_dir_value)


def should_enable_failed_move(rel_path: str, highlight_map: Dict[str, str]) -> bool:
    if not rel_path or not rel_path.startswith('failed_generations/'):
        return False
    parent_path = rel_path.rsplit('/', 1)[0] if '/' in rel_path else ''
    if not parent_path or highlight_map.get(parent_path) == 'fg-has-translated':
        return False
    _, ext = os.path.splitext(rel_path)
    return ext.lower() == '.wav'


def compute_failed_generation_highlights(project_dir: str, config_snapshot: Dict[str, Any]) -> Dict[str, str]:
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
        logging.warning("Nem sikerült beolvasni a translated_splits mappát (%s): %s", translated_root, exc)
        return highlight_map

    try:
        for name in os.listdir(failed_root):
            failed_path = os.path.join(failed_root, name)
            if not os.path.isdir(failed_path) or name not in translated_basenames:
                continue
            rel_path = os.path.relpath(failed_path, project_dir).replace('\\', '/')
            highlight_map[rel_path] = 'fg-has-translated'
    except Exception as exc:
        logging.warning("Nem sikerült beolvasni a failed_generations mappát (%s): %s", failed_root, exc)

    return highlight_map


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
    return None if computed < 0 else computed


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
    if value is None or not math.isfinite(value):
        return None
    rounded = round(max(value, 0) + 1e-9, 2)
    return f"{rounded:.2f}"


def build_audio_metadata(full_path: str, rel_path: str, metadata_directories: Set[str]) -> Dict[str, Any]:
    _, ext = os.path.splitext(full_path)
    if ext.lower() != '.wav' or not should_collect_audio_metadata(rel_path, metadata_directories):
        return {}
    filename = os.path.basename(rel_path)
    computed_seconds = compute_duration_from_filename(filename)
    actual_seconds = read_wav_duration_seconds(full_path)
    display_left = format_seconds_hundredths(computed_seconds) or '--'
    display_right = format_seconds_hundredths(actual_seconds) or '--'
    return {
        'duration_from_name': computed_seconds,
        'duration_actual': actual_seconds,
        'duration_display': f"{display_left} / {display_right}",
    }


def get_failed_generation_directories(config_snapshot: Dict[str, Any]) -> Set[str]:
    project_subdirs = config_snapshot.get('PROJECT_SUBDIRS', {}) if isinstance(config_snapshot, dict) else {}
    candidates = {'failed_generations', project_subdirs.get('failed_generations') or ''}
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


def build_failed_generation_json_metadata(full_path: str, rel_path: str, failed_directories: Set[str]) -> Dict[str, Any]:
    _, ext = os.path.splitext(full_path)
    if ext.lower() != '.json' or os.path.basename(rel_path) != 'info.json':
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
    return {'failed_original_text': value, 'failed_original_text_display': display_value}


def collect_directory_entries(
    root_path: str,
    target_path: str,
    *,
    metadata_directories: Optional[Set[str]] = None,
    highlight_map: Optional[Dict[str, str]] = None,
    failed_generation_directories: Optional[Set[str]] = None,
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
                entry: Dict[str, Any] = {'name': name, 'type': 'directory', 'path': rel_path}
                highlight_class = highlight_map.get(rel_path)
                if highlight_class:
                    entry['highlight_class'] = highlight_class
                entries.append(entry)
                continue

            file_entry: Dict[str, Any] = {'name': name, 'type': 'file', 'path': rel_path}
            if should_enable_failed_move(rel_path, highlight_map):
                file_entry['enable_failed_move'] = True
            metadata = build_audio_metadata(full_path, rel_path, metadata_directories)
            if metadata:
                file_entry.update(metadata)
            failed_metadata = build_failed_generation_json_metadata(full_path, rel_path, failed_generation_directories)
            if failed_metadata:
                file_entry.update(failed_metadata)
            entries.append(file_entry)
    except Exception as exc:
        logging.warning("Nem sikerült beolvasni a(z) %s könyvtárat: %s", target_path, exc)
    return entries

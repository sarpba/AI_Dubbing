from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def build_index_context(workdir_path: str, build_project_entries: Callable[[List[str]], List[Dict[str, Any]]]) -> Dict[str, Any]:
    projects: List[str] = []
    if os.path.exists(workdir_path):
        projects = sorted(
            (d for d in os.listdir(workdir_path) if os.path.isdir(os.path.join(workdir_path, d))),
            key=str.lower,
        )
    return {
        'projects': projects,
        'project_entries': build_project_entries(projects),
    }


def build_project_page_context(
    project_name: str,
    config: Dict[str, Any],
    secure_filename: Callable[[str], str],
    compute_failed_generation_highlights: Callable[[str, Dict[str, Any]], Dict[str, str]],
    get_audio_metadata_directories: Callable[[Dict[str, Any]], Any],
    get_failed_generation_directories: Callable[[Dict[str, Any]], Any],
    should_enable_failed_move: Callable[[str, Dict[str, str]], bool],
    build_audio_metadata: Callable[[str, str, Any], Optional[Dict[str, Any]]],
    build_failed_generation_json_metadata: Callable[[str, str, Any], Optional[Dict[str, Any]]],
    get_tts_root_directory: Callable[[Dict[str, Any]], Optional[str]],
) -> Optional[Dict[str, Any]]:
    project_dir = os.path.join('workdir', secure_filename(project_name))
    if not os.path.exists(project_dir):
        return None

    project_data = {
        'name': project_name,
        'files': {
            'upload': [],
            'upload_grouped': [],
            'extracted_audio': [],
            'extracted_audio_grouped': [],
            'separated_audio_background': [],
            'separated_audio_background_grouped': [],
            'separated_audio_speech': [],
        },
    }

    highlight_map = compute_failed_generation_highlights(project_dir, config)
    metadata_directories = get_audio_metadata_directories(config)
    failed_generation_directories = get_failed_generation_directories(config)

    def group_files_by_extension(file_list: List[str]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[str]] = {}
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
                'count': len(files),
            })
        return grouped_list

    def build_directory_tree(current_path: str, relative_path: str = '') -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        try:
            for name in sorted(os.listdir(current_path)):
                if name.startswith('.'):
                    continue
                full_path = os.path.join(current_path, name)
                rel_path = os.path.join(relative_path, name) if relative_path else name
                if os.path.isdir(full_path):
                    normalized_rel_path = rel_path.replace('\\', '/')
                    entry: Dict[str, Any] = {
                        'name': name,
                        'type': 'directory',
                        'path': normalized_rel_path,
                        'children': build_directory_tree(full_path, rel_path),
                    }
                    highlight_class = highlight_map.get(normalized_rel_path)
                    if highlight_class:
                        entry['highlight_class'] = highlight_class
                    entries.append(entry)
                else:
                    normalized_file_path = rel_path.replace('\\', '/')
                    file_entry: Dict[str, Any] = {
                        'name': name,
                        'type': 'file',
                        'path': normalized_file_path,
                    }
                    if should_enable_failed_move(normalized_file_path, highlight_map):
                        file_entry['enable_failed_move'] = True
                    metadata = build_audio_metadata(full_path, normalized_file_path, metadata_directories)
                    if metadata:
                        file_entry.update(metadata)
                    failed_metadata = build_failed_generation_json_metadata(
                        full_path,
                        normalized_file_path,
                        failed_generation_directories,
                    )
                    if failed_metadata:
                        file_entry.update(failed_metadata)
                    entries.append(file_entry)
        except Exception as exc:
            logging.warning("Nem sikerült beolvasni a(z) %s könyvtárat: %s", current_path, exc)
        return entries

    upload_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['upload'])
    if os.path.exists(upload_dir_path):
        project_data['files']['upload'] = os.listdir(upload_dir_path)
        project_data['files']['upload_grouped'] = group_files_by_extension(project_data['files']['upload'])

    extracted_audio_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['extracted_audio'])
    if os.path.exists(extracted_audio_dir_path):
        project_data['files']['extracted_audio'] = os.listdir(extracted_audio_dir_path)
        project_data['files']['extracted_audio_grouped'] = group_files_by_extension(project_data['files']['extracted_audio'])

    separated_bg_audio_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_background'])
    if os.path.exists(separated_bg_audio_dir_path):
        project_data['files']['separated_audio_background'] = os.listdir(separated_bg_audio_dir_path)
        project_data['files']['separated_audio_background_grouped'] = group_files_by_extension(project_data['files']['separated_audio_background'])

    speech_files_data: List[Dict[str, Any]] = []
    speech_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_speech'])
    if os.path.exists(speech_dir_path):
        for file_name in sorted(os.listdir(speech_dir_path)):
            file_data: Dict[str, Any] = {'name': file_name, 'segment_count': None, 'is_audio': False, 'is_json': False}
            file_path = os.path.join(speech_dir_path, file_name)

            if file_name.lower().endswith('.json'):
                file_data['is_json'] = True
                try:
                    with open(file_path, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                        if 'segments' in data and isinstance(data['segments'], list):
                            file_data['segment_count'] = len(data['segments'])
                except Exception as exc:
                    logging.warning("Error reading or parsing JSON file %s: %s", file_path, exc)
            elif file_name.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                file_data['is_audio'] = True
            speech_files_data.append(file_data)
    project_data['files']['separated_audio_speech'] = speech_files_data

    can_review = False
    if os.path.exists(speech_dir_path):
        temp_files_list = sorted(os.listdir(speech_dir_path))
        for file_name in temp_files_list:
            if file_name.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                base_name, _ = os.path.splitext(file_name)
                if f'{base_name}.json' in temp_files_list:
                    can_review = True
                    break

    has_transcribable_audio = any(file_info['is_audio'] for file_info in speech_files_data)

    tts_root_abs = get_tts_root_directory(config)
    tts_directories: List[str] = []
    if tts_root_abs and os.path.isdir(tts_root_abs):
        try:
            for entry in sorted(os.listdir(tts_root_abs)):
                full_path = os.path.join(tts_root_abs, entry)
                if os.path.isdir(full_path):
                    tts_directories.append(os.path.abspath(full_path))
        except OSError as exc:
            logging.warning("Nem sikerült beolvasni a TTS könyvtárat: %s", exc)

    return {
        'project': project_data,
        'project_tree': build_directory_tree(project_dir),
        'config': config,
        'tts_root_path': tts_root_abs,
        'tts_directories': tts_directories,
        'can_review': can_review,
        'has_transcribable_audio': has_transcribable_audio,
        'has_failed_generation_highlights': bool(highlight_map),
    }


def build_review_page_context(
    project_name: str,
    config: Dict[str, Any],
    secure_filename: Callable[[str], str],
    prepare_segments_for_response: Callable[[str, Any], List[Dict[str, Any]]],
    get_review_encoded_audio_path: Callable[[str, str], Optional[Path]],
    build_audio_url: Callable[[Path], str],
) -> Dict[str, Any]:
    project_dir = os.path.join('workdir', secure_filename(project_name))
    translated_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['translated'])
    speech_dir_path = os.path.join(project_dir, config['PROJECT_SUBDIRS']['separated_audio_speech'])

    audio_file_name = None
    json_file_name = None
    segments_data: Any = []

    if os.path.exists(translated_dir_path):
        translated_files = sorted(os.listdir(translated_dir_path))
        for file_name in translated_files:
            if file_name.lower().endswith('.json'):
                base_name, _ = os.path.splitext(file_name)
                if os.path.exists(speech_dir_path):
                    speech_files = sorted(os.listdir(speech_dir_path))
                    audio_candidate = base_name
                    for audio_ext in ['.wav', '.mp3', '.ogg', '.flac']:
                        if audio_candidate + audio_ext in speech_files:
                            audio_file_name = audio_candidate + audio_ext
                            json_file_name = file_name
                            json_full_path = os.path.join(translated_dir_path, json_file_name)
                            try:
                                with open(json_full_path, 'r', encoding='utf-8') as jf:
                                    data = json.load(jf)
                                    segments_data = data.get('segments', [])
                            except Exception as exc:
                                logging.warning("Error reading JSON file %s for review: %s", json_full_path, exc)
                                segments_data = []
                            break
                    if audio_file_name:
                        break

    if not json_file_name and os.path.exists(speech_dir_path):
        files = sorted(os.listdir(speech_dir_path))
        for file_name in files:
            if file_name.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                base_name, _ = os.path.splitext(file_name)
                potential_json_name = f'{base_name}.json'
                if potential_json_name in files:
                    audio_file_name = file_name
                    json_file_name = potential_json_name
                    json_full_path = os.path.join(speech_dir_path, json_file_name)
                    try:
                        with open(json_full_path, 'r', encoding='utf-8') as jf:
                            data = json.load(jf)
                            segments_data = data.get('segments', [])
                    except Exception as exc:
                        logging.warning("Error reading JSON file %s for review: %s", json_full_path, exc)
                        segments_data = []
                    break

    segments_data = prepare_segments_for_response(project_dir, segments_data)

    audio_url = None
    needs_audio_encoding = False
    if audio_file_name:
        encoded_audio_path = get_review_encoded_audio_path(project_name, audio_file_name)
        if encoded_audio_path and encoded_audio_path.exists():
            audio_url = build_audio_url(encoded_audio_path)
        else:
            needs_audio_encoding = True

    return {
        'project_name': project_name,
        'audio_file_name': audio_file_name,
        'audio_url': audio_url,
        'needs_audio_encoding': needs_audio_encoding,
        'segments_data': segments_data,
        'json_file_name': json_file_name,
        'app_config': config,
    }

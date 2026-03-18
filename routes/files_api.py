from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from flask import after_this_request, jsonify, request, send_file
from pydub import AudioSegment


def register_files_api_routes(app, deps: Dict[str, Any]) -> None:
    @app.route('/api/project/<project_name>', methods=['DELETE'])
    def delete_project(project_name):
        sanitized_project = deps['secure_filename'](project_name)
        if not sanitized_project:
            return jsonify({'success': False, 'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

        config_snapshot = deps['get_config_copy']()
        workdir_path = config_snapshot['DIRECTORIES']['workdir']
        workdir_abs = os.path.abspath(workdir_path)
        project_root = os.path.join(workdir_path, sanitized_project)
        project_root_abs = os.path.abspath(project_root)

        if not deps['is_subpath'](project_root_abs, workdir_abs):
            return jsonify({'success': False, 'error': 'Érvénytelen projekt útvonal.'}), 400
        if not os.path.isdir(project_root_abs):
            return jsonify({'success': False, 'error': 'A projekt nem található.'}), 404

        active_statuses = {'queued', 'running', 'cancelling'}
        active_jobs = [
            job for job in deps['get_project_jobs'](sanitized_project)
            if isinstance(job, dict) and job.get('status') in active_statuses
        ]
        if active_jobs:
            return jsonify({'success': False, 'error': 'A projekt feldolgozása folyamatban van, törlés nem engedélyezett.'}), 409

        try:
            shutil.rmtree(project_root_abs)
        except OSError as exc:
            logging.exception("Nem sikerült törölni a projekt könyvtárát: %s: %s", project_root_abs, exc)
            return jsonify({'success': False, 'error': 'Nem sikerült törölni a projektet.'}), 500

        return jsonify({'success': True, 'message': 'Projekt sikeresen törölve.'})

    @app.route('/api/project-tree/<project_name>', methods=['GET'])
    def get_project_directory_listing(project_name):
        sanitized_project = deps['secure_filename'](project_name)
        config_snapshot = deps['get_config_copy']()
        workdir_path = config_snapshot['DIRECTORIES']['workdir']
        base_dir = os.path.join(workdir_path, sanitized_project)
        base_dir_abs = os.path.abspath(base_dir)

        if not os.path.isdir(base_dir_abs):
            return jsonify({'success': False, 'error': 'Projekt nem található'}), 404

        requested_path = (request.args.get('path') or '').strip()
        target_dir = base_dir_abs
        if requested_path:
            target_dir = os.path.abspath(os.path.join(base_dir_abs, requested_path))

        if not deps['is_subpath'](target_dir, base_dir_abs):
            return jsonify({'success': False, 'error': 'Érvénytelen útvonal'}), 400
        if not os.path.isdir(target_dir):
            return jsonify({'success': False, 'error': 'A megadott könyvtár nem található'}), 404

        highlight_map = deps['compute_failed_generation_highlights'](base_dir_abs, config_snapshot)
        metadata_directories = deps['get_audio_metadata_directories'](config_snapshot)
        failed_generation_directories = deps['get_failed_generation_directories'](config_snapshot)
        entries = deps['collect_directory_entries'](
            base_dir_abs,
            target_dir,
            metadata_directories,
            highlight_map,
            failed_generation_directories,
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
            'has_highlights': bool(highlight_map),
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

        sanitized_project = deps['secure_filename'](project_name)
        if not sanitized_project:
            return jsonify({'success': False, 'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

        safe_filename = deps['secure_filename'](uploaded_file.filename)
        if not safe_filename:
            return jsonify({'success': False, 'error': 'A fájlnév érvénytelen karaktereket tartalmaz.'}), 400

        config_snapshot = deps['get_config_copy']()
        workdir_path = config_snapshot['DIRECTORIES']['workdir']
        project_root = os.path.join(workdir_path, sanitized_project)
        project_root_abs = os.path.abspath(project_root)

        if not os.path.isdir(project_root_abs):
            return jsonify({'success': False, 'error': 'A projekt könyvtára nem található.'}), 404

        normalized_target = target_path_raw.strip('/\\')
        destination_dir = project_root_abs
        if normalized_target:
            destination_dir = os.path.abspath(os.path.join(project_root_abs, normalized_target))

        if not deps['is_subpath'](destination_dir, project_root_abs):
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
        return jsonify({'success': True, 'message': 'Fájl sikeresen feltöltve.', 'path': relative_saved_path})

    @app.route('/api/project-file/move-failed', methods=['POST'])
    def move_failed_generation_file():
        payload = request.get_json(silent=True) or {}
        project_name = (payload.get('projectName') or '').strip()
        source_path = (payload.get('sourcePath') or '').strip()

        if not project_name:
            return jsonify({'success': False, 'error': 'Hiányzó projektnév.'}), 400
        if not source_path:
            return jsonify({'success': False, 'error': 'Hiányzó fájl elérési útvonal.'}), 400

        sanitized_project = deps['secure_filename'](project_name)
        if not sanitized_project:
            return jsonify({'success': False, 'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

        config_snapshot = deps['get_config_copy']()
        workdir_path = config_snapshot['DIRECTORIES']['workdir']
        project_root = os.path.join(workdir_path, sanitized_project)
        project_root_abs = os.path.abspath(project_root)

        if not os.path.isdir(project_root_abs):
            return jsonify({'success': False, 'error': 'A projekt könyvtára nem található.'}), 404

        source_abs = os.path.abspath(os.path.join(project_root_abs, source_path))
        if not deps['is_subpath'](source_abs, project_root_abs):
            return jsonify({'success': False, 'error': 'Érvénytelen forrás elérési útvonal.'}), 400
        if not os.path.isfile(source_abs):
            return jsonify({'success': False, 'error': 'A megadott fájl nem található.'}), 404

        relative_source = os.path.relpath(source_abs, project_root_abs).replace('\\', '/')
        if not relative_source.startswith('failed_generations/'):
            return jsonify({'success': False, 'error': 'Csak failed_generations könyvtárból áthelyezhető.'}), 400

        highlight_map = deps['compute_failed_generation_highlights'](project_root_abs, config_snapshot)
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
        if not deps['is_subpath'](translated_dir_abs, project_root_abs):
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
        if not deps['is_subpath'](destination_abs, translated_dir_abs):
            return jsonify({'success': False, 'error': 'Érvénytelen célfájl hely.'}), 500

        if os.path.exists(destination_abs):
            return jsonify({'success': False, 'error': 'Már létezik ilyen nevű fájl a translated_splits mappában.'}), 409

        try:
            shutil.move(source_abs, destination_abs)
        except OSError as exc:
            logging.exception("Nem sikerült áthelyezni a fájlt %s -> %s: %s", source_abs, destination_abs, exc)
            return jsonify({'success': False, 'error': 'Nem sikerült áthelyezni a fájlt.'}), 500

        relative_destination = os.path.relpath(destination_abs, project_root_abs).replace('\\', '/')
        return jsonify({'success': True, 'message': 'Fájl sikeresen áthelyezve.', 'destination_path': relative_destination})

    @app.route('/api/project-backup', methods=['POST'])
    def create_project_backup():
        payload = request.get_json(silent=True) or {}
        project_name = (payload.get('projectName') or '').strip()

        if not project_name:
            return jsonify({'success': False, 'error': 'Hiányzó projektnév.'}), 400

        sanitized_project = deps['secure_filename'](project_name)
        if not sanitized_project:
            return jsonify({'success': False, 'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

        config_snapshot = deps['get_config_copy']()
        workdir_path = config_snapshot['DIRECTORIES']['workdir']
        project_root = os.path.join(workdir_path, sanitized_project)
        project_root_abs = os.path.abspath(project_root)

        if not os.path.isdir(project_root_abs):
            return jsonify({'success': False, 'error': 'A projekt könyvtára nem található.'}), 404

        temp_dir = tempfile.mkdtemp(prefix='project-backup-')
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        archive_base = os.path.join(temp_dir, f"{sanitized_project}_backup_{timestamp}")
        archive_path = f"{archive_base}.tar.gz"

        @after_this_request
        def cleanup_temp_dir(response):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as exc:
                logging.exception("Nem sikerült törölni az ideiglenes könyvtárat (%s): %s", temp_dir, exc)
            return response

        archive_created = False
        tar_executable = shutil.which('tar')
        pigz_executable = shutil.which('pigz')
        pigz_threads = max(1, min((os.cpu_count() or 1), 16))

        if tar_executable:
            tar_cmd = [tar_executable]
            if pigz_executable:
                tar_cmd.extend(['-cf', archive_path, '-I', f'{pigz_executable} -p {pigz_threads}'])
            else:
                tar_cmd.extend(['-czf', archive_path])
            tar_cmd.extend(['-C', project_root_abs, '.'])
            try:
                subprocess.run(tar_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                archive_created = True
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                logging.exception("Tar archívum készítése sikertelen (pigz=%s): %s", bool(pigz_executable), exc)

        if not archive_created:
            if os.path.exists(archive_path):
                try:
                    os.remove(archive_path)
                except OSError:
                    pass
            try:
                archive_path = shutil.make_archive(archive_base, 'gztar', root_dir=project_root_abs)
                archive_created = True
            except Exception as exc:
                logging.exception("Python alapú archívum készítése sikertelen: %s", exc)
                return jsonify({'success': False, 'error': 'Nem sikerült létrehozni a biztonsági mentést.'}), 500

        if not archive_created or not os.path.exists(archive_path):
            return jsonify({'success': False, 'error': 'A biztonsági mentés nem található.'}), 500

        download_name = os.path.basename(archive_path)
        try:
            return send_file(
                archive_path,
                mimetype='application/gzip',
                as_attachment=True,
                download_name=download_name,
            )
        except FileNotFoundError:
            logging.exception("A létrehozott archívum nem található: %s", archive_path)
            return jsonify({'success': False, 'error': 'A biztonsági mentés nem található.'}), 500

    @app.route('/api/tts-directory', methods=['POST'])
    def create_tts_directory():
        payload = request.get_json(silent=True) or {}
        target_path_raw = (payload.get('path') or '').strip()
        if not target_path_raw:
            return jsonify({'success': False, 'error': 'Adj meg almappa nevet.'}), 400

        config_snapshot = deps['get_config_copy']()
        tts_root_abs = deps['get_tts_root_directory'](config_snapshot)
        if not tts_root_abs:
            return jsonify({'success': False, 'error': 'A TTS könyvtár nincs konfigurálva.'}), 500

        try:
            os.makedirs(tts_root_abs, exist_ok=True)
        except OSError as exc:
            logging.exception("Nem sikerült előkészíteni a TTS könyvtárat: %s", exc)
            return jsonify({'success': False, 'error': 'Nem sikerült elérni a TTS könyvtárat.'}), 500

        try:
            sanitized_relative = deps['sanitize_storage_relative_path'](target_path_raw, allow_empty=False)
        except ValueError as exc:
            return jsonify({'success': False, 'error': str(exc)}), 400

        target_abs = os.path.abspath(os.path.join(tts_root_abs, sanitized_relative))
        if not deps['is_subpath'](target_abs, tts_root_abs):
            return jsonify({'success': False, 'error': 'Érvénytelen TTS almappa útvonal.'}), 400

        if os.path.isdir(target_abs):
            relative_existing = os.path.relpath(target_abs, tts_root_abs).replace('\\', '/')
            return jsonify({'success': True, 'message': 'A megadott almappa már létezik.', 'path': relative_existing})

        try:
            os.makedirs(target_abs, exist_ok=True)
        except OSError as exc:
            logging.exception("Nem sikerült létrehozni a TTS almappát (%s): %s", target_abs, exc)
            return jsonify({'success': False, 'error': 'Nem sikerült létrehozni az almappát.'}), 500

        relative_created = os.path.relpath(target_abs, tts_root_abs).replace('\\', '/')
        return jsonify({'success': True, 'message': 'Almappa sikeresen létrehozva.', 'path': relative_created})

    @app.route('/api/tts-upload', methods=['POST'])
    def upload_tts_file():
        target_path_raw = (request.form.get('targetPath') or '').strip()
        chunk_upload_id = (request.form.get('chunkUploadId') or '').strip()
        chunk_file_key_raw = (request.form.get('chunkFileKey') or '').strip()

        def _parse_int(value: Any) -> Optional[int]:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        chunk_index = _parse_int(request.form.get('chunkIndex'))
        total_chunks = _parse_int(request.form.get('totalChunks'))
        _parse_int(request.form.get('chunkSize'))
        requested_filename = (request.form.get('fileName') or '').strip()

        if not target_path_raw:
            return jsonify({'success': False, 'error': 'Add meg a cél almappát (pl. f5_models/custom).'}), 400

        config_snapshot = deps['get_config_copy']()
        tts_root_abs = deps['get_tts_root_directory'](config_snapshot)
        if not tts_root_abs:
            return jsonify({'success': False, 'error': 'A TTS könyvtár nincs konfigurálva.'}), 500

        try:
            os.makedirs(tts_root_abs, exist_ok=True)
        except OSError as exc:
            logging.exception("Nem sikerült előkészíteni a TTS könyvtárat: %s", exc)
            return jsonify({'success': False, 'error': 'Nem sikerült elérni a TTS könyvtárat.'}), 500

        try:
            sanitized_relative = deps['sanitize_storage_relative_path'](target_path_raw, allow_empty=False)
        except ValueError as exc:
            return jsonify({'success': False, 'error': str(exc)}), 400

        destination_dir = tts_root_abs
        if sanitized_relative:
            destination_dir = os.path.abspath(os.path.join(tts_root_abs, sanitized_relative))
            if not deps['is_subpath'](destination_dir, tts_root_abs):
                return jsonify({'success': False, 'error': 'Érvénytelen célkönyvtár a TTS mappában.'}), 400
            try:
                os.makedirs(destination_dir, exist_ok=True)
            except OSError as exc:
                logging.exception("Nem sikerült létrehozni a célkönyvtárat (%s): %s", destination_dir, exc)
                return jsonify({'success': False, 'error': 'Nem sikerült előkészíteni a célkönyvtárat.'}), 500

        chunk_file_key = chunk_file_key_raw.lower()
        chunk_mode = bool(chunk_upload_id) and chunk_file_key in {'model', 'vocab', 'config'} and chunk_index is not None and total_chunks is not None and total_chunks > 0
        chunk_labels = {'model': 'model fájl', 'vocab': 'vocab.txt fájl', 'config': 'konfigurációs (JSON) fájl'}

        if chunk_mode:
            file_storage = request.files.get('file') or request.files.get('chunk') or request.files.get('modelFile') or request.files.get('vocabFile') or request.files.get('configFile')
            if not file_storage or not file_storage.filename:
                return jsonify({'success': False, 'error': f'Hiányzik a(z) {chunk_labels[chunk_file_key]} darab fájl.'}), 400

            chunk_id_safe = deps['secure_filename'](chunk_upload_id) or f"{chunk_file_key}_chunk"
            chunk_root_base = os.path.join(tts_root_abs, '.chunk_uploads')
            chunk_parent_root = os.path.join(chunk_root_base, chunk_file_key)
            chunk_base_dir = os.path.join(chunk_parent_root, chunk_id_safe)
            try:
                os.makedirs(chunk_base_dir, exist_ok=True)
            except OSError as exc:
                logging.exception("Nem sikerült létrehozni a TTS chunk könyvtárat: %s", exc)
                return jsonify({'success': False, 'error': 'Nem sikerült előkészíteni a chunk könyvtárat.'}), 500

            chunk_filename = os.path.join(chunk_base_dir, f"{chunk_index:06d}.part")
            try:
                file_storage.save(chunk_filename)
            except Exception as exc:
                logging.exception("Nem sikerült menteni a TTS chunkot: %s", exc)
                return jsonify({'success': False, 'error': 'Nem sikerült menteni a chunk fájlt.'}), 500

            if chunk_index + 1 < total_chunks:
                return jsonify({'success': True, 'completed': False, 'fileKey': chunk_file_key})

            safe_filename = deps['secure_filename'](requested_filename or file_storage.filename) or f"{chunk_file_key}.bin"
            destination_abs = os.path.abspath(os.path.join(destination_dir, safe_filename))
            if not deps['is_subpath'](destination_abs, tts_root_abs):
                shutil.rmtree(chunk_base_dir, ignore_errors=True)
                return jsonify({'success': False, 'error': 'Érvénytelen célfájl elérési útvonal.'}), 400
            if os.path.exists(destination_abs):
                shutil.rmtree(chunk_base_dir, ignore_errors=True)
                return jsonify({'success': False, 'error': f'Már létezik ilyen nevű fájl: {safe_filename}'}), 409

            try:
                with open(destination_abs, 'wb') as destination:
                    for idx in range(total_chunks):
                        part_path = os.path.join(chunk_base_dir, f"{idx:06d}.part")
                        if not os.path.exists(part_path):
                            shutil.rmtree(chunk_base_dir, ignore_errors=True)
                            return jsonify({'success': False, 'error': f'Hiányzó mentett chunk: {idx}.'}), 400
                        with open(part_path, 'rb') as part_file:
                            shutil.copyfileobj(part_file, destination)
            except Exception as exc:
                logging.exception("Nem sikerült összeilleszteni a TTS chunkokat: %s", exc)
                shutil.rmtree(chunk_base_dir, ignore_errors=True)
                if os.path.exists(destination_abs):
                    try:
                        os.remove(destination_abs)
                    except OSError:
                        pass
                return jsonify({'success': False, 'error': 'Nem sikerült a chunkok összeillesztése.'}), 500
            finally:
                shutil.rmtree(chunk_base_dir, ignore_errors=True)
                if os.path.isdir(chunk_parent_root):
                    shutil.rmtree(chunk_parent_root, ignore_errors=True)
                if os.path.isdir(chunk_root_base):
                    try:
                        os.rmdir(chunk_root_base)
                    except OSError:
                        pass

            relative_saved_path = os.path.relpath(destination_abs, tts_root_abs).replace('\\', '/')
            return jsonify({
                'success': True,
                'completed': True,
                'fileKey': chunk_file_key,
                'path': relative_saved_path,
                'message': f'A(z) {chunk_labels[chunk_file_key]} feltöltése befejeződött.',
            })

        uploaded_files = {
            'model': request.files.get('modelFile'),
            'vocab': request.files.get('vocabFile'),
            'config': request.files.get('configFile'),
        }
        required_labels = {'model': 'model fájl', 'vocab': 'vocab.txt fájl', 'config': 'konfigurációs (JSON) fájl'}
        for key, file_storage in uploaded_files.items():
            if not file_storage or not file_storage.filename:
                return jsonify({'success': False, 'error': f'Hiányzik a(z) {required_labels[key]}.'}), 400

        saved_paths: List[Tuple[str, str]] = []
        try:
            for key, file_storage in uploaded_files.items():
                safe_filename = deps['secure_filename'](file_storage.filename)
                if not safe_filename:
                    raise ValueError(f'A(z) {required_labels[key]} neve érvénytelen karaktereket tartalmaz.')
                destination_abs = os.path.abspath(os.path.join(destination_dir, safe_filename))
                if not deps['is_subpath'](destination_abs, tts_root_abs):
                    raise ValueError('Érvénytelen célfájl elérési útvonal.')
                if os.path.exists(destination_abs):
                    raise FileExistsError(f'Már létezik ilyen nevű fájl: {safe_filename}')
                file_storage.save(destination_abs)
                relative_saved_path = os.path.relpath(destination_abs, tts_root_abs).replace('\\', '/')
                saved_paths.append((safe_filename, relative_saved_path))
        except ValueError as exc:
            for _, rel_path in saved_paths:
                try:
                    os.remove(os.path.join(tts_root_abs, rel_path))
                except OSError:
                    pass
            return jsonify({'success': False, 'error': str(exc)}), 400
        except FileExistsError as exc:
            for _, rel_path in saved_paths:
                try:
                    os.remove(os.path.join(tts_root_abs, rel_path))
                except OSError:
                    pass
            return jsonify({'success': False, 'error': str(exc)}), 409
        except OSError as exc:
            logging.exception("Nem sikerült menteni a TTS fájlokat: %s", exc)
            for _, rel_path in saved_paths:
                try:
                    os.remove(os.path.join(tts_root_abs, rel_path))
                except OSError:
                    pass
            return jsonify({'success': False, 'error': 'Nem sikerült menteni a feltöltött fájlokat.'}), 500

        return jsonify({'success': True, 'message': 'F5-TTS modell csomag sikeresen feltöltve.', 'paths': [path for _, path in saved_paths]})

    @app.route('/api/project-file/<project_name>', methods=['DELETE'])
    def delete_project_file(project_name):
        payload = request.get_json(silent=True) or {}
        target_path_raw = (payload.get('path') or '').strip()
        if not target_path_raw:
            return jsonify({'success': False, 'error': 'Hiányzó fájl elérési útvonal.'}), 400

        sanitized_project = deps['secure_filename'](project_name)
        if not sanitized_project:
            return jsonify({'success': False, 'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

        config_snapshot = deps['get_config_copy']()
        workdir_path = config_snapshot['DIRECTORIES']['workdir']
        project_root = os.path.join(workdir_path, sanitized_project)
        project_root_abs = os.path.abspath(project_root)

        target_abs_path = os.path.abspath(os.path.join(project_root_abs, target_path_raw))
        if not deps['is_subpath'](target_abs_path, project_root_abs):
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

        sanitized_project = deps['secure_filename'](project_name)
        if not sanitized_project:
            return jsonify({'success': False, 'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

        config_snapshot = deps['get_config_copy']()
        workdir_path = config_snapshot['DIRECTORIES']['workdir']
        project_root = os.path.join(workdir_path, sanitized_project)
        project_root_abs = os.path.abspath(project_root)

        source_abs = os.path.abspath(os.path.join(project_root_abs, file_path_raw))
        if not deps['is_subpath'](source_abs, project_root_abs):
            return jsonify({'success': False, 'error': 'Érvénytelen audió elérési útvonal.'}), 400
        if not os.path.isfile(source_abs):
            return jsonify({'success': False, 'error': 'A megadott audió fájl nem található.'}), 404

        source_ext = os.path.splitext(source_abs)[1].lower()
        if source_ext not in deps['audio_extensions']:
            return jsonify({'success': False, 'error': 'Csak támogatott hangfájl vágható.'}), 400

        if output_name_raw:
            safe_output_name = deps['secure_filename'](output_name_raw)
            if not safe_output_name:
                return jsonify({'success': False, 'error': 'A mentési fájlnév érvénytelen.'}), 400
            if '.' not in safe_output_name:
                safe_output_name = f"{safe_output_name}{source_ext}"
            destination_abs = os.path.abspath(os.path.join(os.path.dirname(source_abs), safe_output_name))
            if not deps['is_subpath'](destination_abs, project_root_abs):
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
            'trim_duration': trimmed_duration,
        })

    @app.route('/api/project-directory/<project_name>', methods=['DELETE'])
    def clear_project_directory(project_name):
        payload = request.get_json(silent=True) or {}
        target_path_raw = (payload.get('path') or '').strip()
        if not target_path_raw:
            return jsonify({'success': False, 'error': 'Hiányzó könyvtár elérési útvonal.'}), 400

        sanitized_project = deps['secure_filename'](project_name)
        if not sanitized_project:
            return jsonify({'success': False, 'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

        config_snapshot = deps['get_config_copy']()
        workdir_path = config_snapshot['DIRECTORIES']['workdir']
        project_root = os.path.join(workdir_path, sanitized_project)
        project_root_abs = os.path.abspath(project_root)

        target_abs_path = os.path.abspath(os.path.join(project_root_abs, target_path_raw))
        if not deps['is_subpath'](target_abs_path, project_root_abs):
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

    @app.route('/api/restore-project', methods=['POST'])
    def restore_project_backup():
        project_name = (request.form.get('projectName') or '').strip()
        backup_file = request.files.get('backupFile')
        chunk_upload_id = (request.form.get('chunkUploadId') or '').strip()

        def _parse_int(value: Any) -> Optional[int]:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        chunk_index = _parse_int(request.form.get('chunkIndex'))
        total_chunks = _parse_int(request.form.get('totalChunks'))
        _parse_int(request.form.get('chunkSize'))
        requested_backup_name = request.form.get('fileName') or (backup_file.filename if backup_file else '')
        chunk_mode = bool(chunk_upload_id) and chunk_index is not None and total_chunks is not None and total_chunks > 0

        if not project_name:
            return jsonify({'success': False, 'error': 'Add meg a projekt nevét!'}), 400
        if not backup_file or not backup_file.filename:
            return jsonify({'success': False, 'error': 'Nem érkezett archívum a visszaállításhoz.'}), 400

        sanitized_project = deps['secure_filename'](project_name)
        if not sanitized_project:
            return jsonify({'success': False, 'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

        config_snapshot = deps['get_config_copy']()
        workdir_path = config_snapshot['DIRECTORIES']['workdir']
        workdir_abs = os.path.abspath(workdir_path)
        try:
            os.makedirs(workdir_abs, exist_ok=True)
        except OSError as exc:
            logging.exception("Nem sikerült elérni a workdir könyvtárat: %s", exc)
            return jsonify({'success': False, 'error': 'Nem sikerült elérni a workdir könyvtárat.'}), 500

        project_root = os.path.join(workdir_abs, sanitized_project)
        project_root_abs = os.path.abspath(project_root)
        if not deps['is_subpath'](project_root_abs, workdir_abs):
            return jsonify({'success': False, 'error': 'A projektnév pályája érvénytelen.'}), 400

        project_dir_preexisted = os.path.isdir(project_root_abs)
        if project_dir_preexisted:
            try:
                has_entries = any(os.scandir(project_root_abs))
            except OSError as exc:
                logging.exception("Nem sikerült beolvasni a projekt könyvtárát: %s", exc)
                return jsonify({'success': False, 'error': 'Nem sikerült elérni a projekt könyvtárát.'}), 500
            if has_entries:
                return jsonify({'success': False, 'error': 'Már létezik ilyen nevű projekt. Töröld vagy válassz másik nevet a visszaállításhoz.'}), 409
        else:
            try:
                os.makedirs(project_root_abs, exist_ok=True)
            except OSError as exc:
                logging.exception("Nem sikerült létrehozni a projekt könyvtárát: %s", exc)
                return jsonify({'success': False, 'error': 'Nem sikerült létrehozni a projekt könyvtárát.'}), 500

        def reset_project_directory():
            try:
                shutil.rmtree(project_root_abs, ignore_errors=True)
                if project_dir_preexisted:
                    os.makedirs(project_root_abs, exist_ok=True)
            except Exception as exc:
                logging.exception("Nem sikerült visszaállítani a projekt könyvtárat: %s", exc)

        chunk_base_dir = None
        assembled_archive_path = None
        chunk_parent_root = os.path.join(workdir_abs, '.restore_chunks', sanitized_project)

        if chunk_mode:
            if chunk_index < 0 or chunk_index >= total_chunks:
                return jsonify({'success': False, 'error': 'Érvénytelen darab index.'}), 400

            chunk_id_safe = deps['secure_filename'](chunk_upload_id) or f"{sanitized_project}_chunk"
            chunk_base_dir = os.path.join(chunk_parent_root, chunk_id_safe)
            try:
                os.makedirs(chunk_base_dir, exist_ok=True)
            except OSError as exc:
                logging.exception("Nem sikerült létrehozni a chunk könyvtárat: %s", exc)
                return jsonify({'success': False, 'error': 'Nem sikerült előkészíteni a chunk könyvtárat.'}), 500

            chunk_filename = os.path.join(chunk_base_dir, f"{chunk_index:06d}.part")
            try:
                backup_file.save(chunk_filename)
            except Exception as exc:
                logging.exception("Nem sikerült menteni a mentés darabot: %s", exc)
                return jsonify({'success': False, 'error': 'Nem sikerült menteni a mentés darabot.'}), 500

            if chunk_index + 1 < total_chunks:
                return jsonify({'success': True, 'chunkIndex': chunk_index, 'totalChunks': total_chunks, 'completed': False})

            assembled_name = deps['secure_filename'](requested_backup_name) or f"{sanitized_project}_backup.tar"
            assembled_archive_path = os.path.join(chunk_base_dir, assembled_name)

            try:
                with open(assembled_archive_path, 'wb') as destination:
                    for idx in range(total_chunks):
                        part_path = os.path.join(chunk_base_dir, f"{idx:06d}.part")
                        if not os.path.exists(part_path):
                            logging.error("Hiányzik a(z) %s chunk a mentés összefűzéséhez.", part_path)
                            shutil.rmtree(chunk_base_dir, ignore_errors=True)
                            return jsonify({'success': False, 'error': f'Hiányzó mentés darab: {idx}.'}), 400
                        with open(part_path, 'rb') as part_file:
                            shutil.copyfileobj(part_file, destination)
            except Exception as exc:
                logging.exception("Nem sikerült a chunkokat összeilleszteni: %s", exc)
                shutil.rmtree(chunk_base_dir, ignore_errors=True)
                return jsonify({'success': False, 'error': 'Nem sikerült a mentés darabjainak összeillesztése.'}), 500

        temp_dir = tempfile.mkdtemp(prefix='project-restore-')
        extract_dir = os.path.join(temp_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)

        if assembled_archive_path:
            archive_path = assembled_archive_path
        else:
            archive_name = deps['secure_filename'](backup_file.filename) or 'project-backup'
            archive_path = os.path.join(temp_dir, archive_name)
            try:
                backup_file.save(archive_path)
            except Exception as exc:
                logging.exception("Nem sikerült menteni a feltöltött archívumot: %s", exc)
                reset_project_directory()
                shutil.rmtree(temp_dir, ignore_errors=True)
                return jsonify({'success': False, 'error': 'Nem sikerült elmenteni a feltöltött fájlt.'}), 500

        def resolve_payload_root(base_dir: str) -> str:
            try:
                entries = [name for name in os.listdir(base_dir) if name not in ('.', '..', '__MACOSX')]
            except OSError as exc:
                logging.exception("Nem sikerült beolvasni a kicsomagolt archívumot: %s", exc)
                raise
            if len(entries) == 1:
                only_entry_path = os.path.join(base_dir, entries[0])
                if os.path.isdir(only_entry_path):
                    return only_entry_path
            return base_dir

        try:
            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path) as archive:
                    deps['safe_extract_zip'](archive, extract_dir)
            elif tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path, 'r:*') as archive:
                    deps['safe_extract_tar'](archive, extract_dir)
            else:
                reset_project_directory()
                return jsonify({'success': False, 'error': 'A feltöltött fájl nem támogatott archívum (csak .tar, .tar.gz, .tgz vagy .zip).'}), 400

            payload_root = resolve_payload_root(extract_dir)
            try:
                entries_to_move = [entry for entry in os.listdir(payload_root) if entry not in ('__MACOSX',)]
            except OSError as exc:
                logging.exception("Nem sikerült beolvasni a kicsomagolt fájlokat: %s", exc)
                reset_project_directory()
                return jsonify({'success': False, 'error': 'Nem sikerült feldolgozni a kicsomagolt fájlokat.'}), 500

            if not entries_to_move:
                reset_project_directory()
                return jsonify({'success': False, 'error': 'Az archívum nem tartalmaz visszaállítható fájlokat.'}), 400

            for entry in entries_to_move:
                source_path = os.path.join(payload_root, entry)
                destination_path = os.path.join(project_root_abs, entry)
                shutil.move(source_path, destination_path)

            return jsonify({'success': True, 'message': 'A projekt mentés sikeresen visszaállítva.', 'project': sanitized_project})
        except ValueError as exc:
            logging.exception("Érvénytelen archívumtartalom: %s", exc)
            reset_project_directory()
            return jsonify({'success': False, 'error': 'Az archívum érvénytelen vagy tiltott útvonalakat tartalmaz.'}), 400
        except (zipfile.BadZipFile, tarfile.TarError) as exc:
            logging.exception("Nem sikerült kibontani a mentést: %s", exc)
            reset_project_directory()
            return jsonify({'success': False, 'error': 'Nem sikerült kibontani a mentést. Ellenőrizd, hogy sértetlen-e a fájl.'}), 400
        except Exception as exc:
            logging.exception("Ismeretlen hiba a visszaállítás közben: %s", exc)
            reset_project_directory()
            return jsonify({'success': False, 'error': 'Nem sikerült visszaállítani a projektet.'}), 500
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if chunk_parent_root:
                shutil.rmtree(chunk_parent_root, ignore_errors=True)

    @app.route('/api/upload-video', methods=['POST'])
    def upload_video():
        project_name = request.form.get('projectName', '').strip()
        youtube_url = request.form.get('youtubeUrl', '').strip()
        video_file = request.files.get('file')
        chunk_upload_id = (request.form.get('chunkUploadId') or '').strip()

        def _parse_int(value: Any) -> Optional[int]:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        chunk_index = _parse_int(request.form.get('chunkIndex'))
        total_chunks = _parse_int(request.form.get('totalChunks'))
        _parse_int(request.form.get('chunkSize'))
        has_video_file = bool(video_file and video_file.filename)

        if not project_name:
            return jsonify({'error': 'Üres projektnév nem engedélyezett.'}), 400
        if has_video_file and youtube_url:
            return jsonify({'error': 'Válaszd ki, hogy fájlt töltesz fel, vagy YouTube videót töltesz le.'}), 400
        if not has_video_file and not youtube_url:
            return jsonify({'error': 'Adj meg videó fájlt vagy YouTube hivatkozást.'}), 400
        if youtube_url and not re.match(r'^https?://', youtube_url, re.IGNORECASE):
            return jsonify({'error': 'Adj meg érvényes (http/https) YouTube hivatkozást.'}), 400

        sanitized_project = deps['secure_filename'](project_name)
        if not sanitized_project:
            return jsonify({'error': 'A projektnév érvénytelen karaktereket tartalmaz.'}), 400

        current_config = deps['get_config_copy']()
        project_dir = os.path.join('workdir', sanitized_project)
        try:
            os.makedirs(project_dir, exist_ok=True)
            for subdir in current_config['PROJECT_SUBDIRS'].values():
                os.makedirs(os.path.join(project_dir, subdir), exist_ok=True)
        except OSError as exc:
            logging.exception("Nem sikerült létrehozni a projekt mappáit: %s", exc)
            return jsonify({'error': 'Nem sikerült létrehozni a projekt könyvtárát.'}), 500

        upload_dir = os.path.join(project_dir, current_config['PROJECT_SUBDIRS']['upload'])
        subtitle_file = request.files.get('subtitleFile')
        subtitle_suffix_raw = request.form.get('subtitleSuffix', '').strip()
        subtitle_filename = None
        subtitle_suffix_normalized = None
        chunk_upload_handled = False

        if subtitle_file and subtitle_file.filename:
            original_subtitle_filename = deps['secure_filename'](subtitle_file.filename)
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
            chunk_mode = bool(chunk_upload_id) and chunk_index is not None and total_chunks is not None and total_chunks > 0
            if chunk_mode:
                if chunk_index < 0 or chunk_index >= total_chunks:
                    return jsonify({'error': 'Érvénytelen darab index.'}), 400

                chunk_id_safe = deps['secure_filename'](chunk_upload_id) or f"{sanitized_project}_chunk"
                chunk_parent_root = os.path.join(upload_dir, '.chunks')
                chunk_base_dir = os.path.join(chunk_parent_root, chunk_id_safe)

                try:
                    os.makedirs(chunk_base_dir, exist_ok=True)
                except OSError as exc:
                    logging.exception("Nem sikerült létrehozni a chunk könyvtárat: %s", exc)
                    return jsonify({'error': 'Nem sikerült előkészíteni a chunk könyvtárat.'}), 500

                chunk_filename = os.path.join(chunk_base_dir, f"{chunk_index:06d}.part")
                try:
                    video_file.save(chunk_filename)
                except Exception as exc:
                    logging.exception("Nem sikerült menteni a videó chunkot: %s", exc)
                    return jsonify({'error': 'Nem sikerült menteni a videó darabot.'}), 500

                if chunk_index + 1 < total_chunks:
                    return jsonify({'success': True, 'chunkIndex': chunk_index, 'totalChunks': total_chunks, 'completed': False})

                requested_name = request.form.get('fileName') or video_file.filename
                video_filename = deps['secure_filename'](requested_name) or f"{sanitized_project}_video.mkv"
                video_path = os.path.join(upload_dir, video_filename)

                try:
                    with open(video_path, 'wb') as destination:
                        for idx in range(total_chunks):
                            part_path = os.path.join(chunk_base_dir, f"{idx:06d}.part")
                            if not os.path.exists(part_path):
                                logging.error("A chunk fájl hiányzik: %s", part_path)
                                return jsonify({'error': f'Hiányzó videó darab: {idx}.'}), 400
                            with open(part_path, 'rb') as part_file:
                                shutil.copyfileobj(part_file, destination)
                except Exception as exc:
                    logging.exception("Nem sikerült összeilleszteni a chunkokat: %s", exc)
                    return jsonify({'error': 'Nem sikerült a videó chunkjainak összeillesztése.'}), 500
                finally:
                    shutil.rmtree(chunk_base_dir, ignore_errors=True)

                chunk_upload_handled = True
            else:
                video_filename = deps['secure_filename'](video_file.filename)
                if not video_filename:
                    return jsonify({'error': 'Érvénytelen videó fájlnév.'}), 400
                video_path = os.path.join(upload_dir, video_filename)
        else:
            if not shutil.which('yt-dlp'):
                return jsonify({'error': 'A yt-dlp nem érhető el a kiszolgálón. Telepítsd a yt-dlp csomagot.'}), 500

            video_filename_base = sanitized_project
            output_template = os.path.join(upload_dir, f"{video_filename_base}.%(ext)s")
            ytdlp_cmd = ['yt-dlp', youtube_url, '--no-playlist', '--remux-video', 'mkv', '--merge-output-format', 'mkv', '-o', output_template]

            logging.info("yt-dlp parancs futtatása: %s", ' '.join(ytdlp_cmd))
            download_process = subprocess.run(ytdlp_cmd, capture_output=True, text=True)

            if download_process.returncode != 0:
                logging.error("A yt-dlp letöltés sikertelen: %s", download_process.stderr)
                return jsonify({'error': 'Nem sikerült letölteni a YouTube videót.', 'details': download_process.stderr}), 500

            expected_mkv = os.path.join(upload_dir, f"{video_filename_base}.mkv")
            if os.path.isfile(expected_mkv):
                video_filename = os.path.basename(expected_mkv)
                video_path = expected_mkv
            else:
                candidate_files = [path for path in glob.glob(os.path.join(upload_dir, f"{video_filename_base}.*")) if not path.endswith('.part')]
                if not candidate_files:
                    logging.error("A yt-dlp nem hozott létre videó fájlt az elvárt névvel: %s", output_template)
                    return jsonify({'error': 'A letöltött videó fájl nem található.'}), 500
                candidate_files.sort(key=os.path.getmtime, reverse=True)
                video_path = candidate_files[0]
                video_filename = os.path.basename(video_path)

        subtitle_path = None
        if subtitle_file and subtitle_file.filename:
            base_video_name = os.path.splitext(video_filename)[0]
            subtitle_filename = deps['secure_filename'](f"{base_video_name}{subtitle_suffix_normalized}.srt")
            subtitle_path = os.path.join(upload_dir, subtitle_filename)

        try:
            if has_video_file and video_path and not chunk_upload_handled:
                video_file.save(video_path)
            if subtitle_filename and subtitle_path:
                subtitle_file.save(subtitle_path)
            return jsonify({'success': True, 'message': 'Projekt sikeresen létrehozva.', 'project': sanitized_project, 'video': video_filename, 'subtitle': subtitle_filename})
        except Exception as exc:
            logging.exception("Upload failed for project %s: %s", sanitized_project, exc)
            return jsonify({'error': f'Upload failed: {exc}', 'project': sanitized_project}), 500

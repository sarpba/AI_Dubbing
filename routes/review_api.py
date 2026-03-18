from __future__ import annotations

import copy
import json
import math
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from flask import jsonify, request, url_for


def _resolve_review_json_path(
    project_dir: str,
    json_file_name: str,
    project_subdirs: Dict[str, str],
) -> Optional[str]:
    speech_dir_path = os.path.join(project_dir, project_subdirs['separated_audio_speech'])
    translated_dir_path = os.path.join(project_dir, project_subdirs['translated'])
    translated_json_path = os.path.join(translated_dir_path, json_file_name)
    speech_json_path = os.path.join(speech_dir_path, json_file_name)

    if os.path.isfile(translated_json_path):
        return translated_json_path
    if os.path.isfile(speech_json_path):
        return speech_json_path
    return None


def register_review_api_routes(app, deps: Dict[str, Any]) -> None:
    @app.route('/api/review-audio-status/<project_name>')
    def review_audio_status(project_name):
        audio_file_name = (request.args.get('audio_file') or '').strip()
        if not audio_file_name:
            return jsonify({'success': False, 'error': 'missing_audio_file'}), 400
        source_path = deps['resolve_source_audio_path'](project_name, audio_file_name)
        if not source_path:
            return jsonify({'success': False, 'error': 'audio_not_found'}), 404
        encoded_audio_path = deps['get_review_encoded_audio_path'](project_name, audio_file_name)
        if not encoded_audio_path:
            return jsonify({'success': False, 'error': 'temp_unavailable'}), 500
        file_exists = encoded_audio_path.exists()

        with deps['review_audio_encoding_lock']:
            job = deps['review_audio_encoding_jobs'].get(project_name)
            if not file_exists and not job:
                job = {'status': 'encoding', 'progress': 0.0, 'error': None}
                deps['review_audio_encoding_jobs'][project_name] = job
                thread = threading.Thread(
                    target=deps['_run_review_audio_encoding_job'],
                    args=(project_name, source_path, encoded_audio_path, job),
                    daemon=True,
                )
                job['thread'] = thread
                thread.start()

        job_active = job is not None
        if file_exists and not job_active:
            try:
                relative_path = encoded_audio_path.relative_to(Path('workdir'))
            except ValueError:
                relative_path = encoded_audio_path
            audio_url = url_for('serve_workdir', filename=str(relative_path).replace('\\', '/'))
            return jsonify({
                'success': True,
                'status': 'available',
                'progress': 100.0,
                'audio_url': audio_url,
            })

        response = {
            'success': True,
            'status': job.get('status', 'encoding'),
            'progress': max(0.0, min(100.0, float(job.get('progress', 0.0)))),
        }
        if job.get('status') == 'failed':
            response['error'] = job.get('error') or 'encoding_failed'
        return jsonify(response)

    @app.route('/api/update-segment/<project_name>', methods=['POST'])
    def update_segment_api(project_name):
        app.logger.info("update_segment_api called for project: %s", project_name)
        try:
            data = request.get_json()
            app.logger.info("Received data for update: %s", data)
            json_file_name = data.get('json_file_name')
            segment_index = data.get('segment_index')
            new_start = data.get('new_start')
            new_end = data.get('new_end')
            new_text = data.get('new_text', None)
            new_translated_text = data.get('new_translated_text', None)

            if None in [json_file_name, segment_index] or new_start is None or new_end is None:
                return jsonify({'success': False, 'error': 'Missing data (json_file_name, segment_index, new_start, or new_end)'}), 400

            project_dir = os.path.join('workdir', deps['secure_filename'](project_name))
            if not json_file_name:
                return jsonify({'success': False, 'error': 'json_file_name is missing or empty in payload'}), 400

            json_full_path = _resolve_review_json_path(project_dir, json_file_name, deps['config']['PROJECT_SUBDIRS'])
            if not json_full_path:
                app.logger.error("JSON file not found for update: %s", json_file_name)
                return jsonify({'success': False, 'error': f'JSON file not found: {json_file_name}'}), 404

            with open(json_full_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)

            if 'segments' not in transcription_data or not isinstance(transcription_data['segments'], list):
                return jsonify({'success': False, 'error': 'Invalid JSON structure: "segments" missing or not a list'}), 500

            if not (isinstance(segment_index, int) and 0 <= segment_index < len(transcription_data['segments'])):
                return jsonify({'success': False, 'error': f'Invalid segment index: {segment_index}'}), 400

            if not (isinstance(new_start, (int, float)) and isinstance(new_end, (int, float))):
                return jsonify({'success': False, 'error': 'new_start and new_end must be numbers'}), 400
            if new_start < 0 or new_end < 0 or new_start >= new_end:
                return jsonify({'success': False, 'error': f'Invalid start/end times: start={new_start}, end={new_end}'}), 400

            if segment_index > 0:
                prev_segment_end = transcription_data['segments'][segment_index - 1].get('end')
                if prev_segment_end is not None and new_start < prev_segment_end:
                    return jsonify({'success': False, 'error': f'New start time {new_start} overlaps with previous segment end {prev_segment_end}'}), 400
            if segment_index < len(transcription_data['segments']) - 1:
                next_segment_start = transcription_data['segments'][segment_index + 1].get('start')
                if next_segment_start is not None and new_end > next_segment_start:
                    return jsonify({'success': False, 'error': f'New end time {new_end} overlaps with next segment start {next_segment_start}'}), 400

            original_segment = transcription_data['segments'][segment_index] if segment_index < len(transcription_data['segments']) else {}
            original_start = original_segment.get('start')
            original_end = original_segment.get('end')
            original_text = original_segment.get('text')
            original_translated = original_segment.get('translated_text')

            def _time_changed(original_value: Any, new_value: Any) -> bool:
                try:
                    return not math.isclose(float(original_value), float(new_value), rel_tol=1e-6, abs_tol=1e-6)
                except (TypeError, ValueError):
                    return original_value != new_value

            start_changed = _time_changed(original_start, new_start)
            end_changed = _time_changed(original_end, new_end)
            text_changed = (new_text is not None and new_text != original_text)
            translated_changed = (new_translated_text is not None and new_translated_text != original_translated)
            should_delete_split = start_changed or end_changed or text_changed or translated_changed

            transcription_data['segments'][segment_index]['start'] = new_start
            transcription_data['segments'][segment_index]['end'] = new_end
            if new_text is not None:
                transcription_data['segments'][segment_index]['text'] = new_text
            if new_translated_text is not None:
                transcription_data['segments'][segment_index]['translated_text'] = new_translated_text

            for segment_item in transcription_data['segments']:
                if 'words' in segment_item:
                    del segment_item['words']

            with open(json_full_path, 'w', encoding='utf-8') as f:
                json.dump(transcription_data, f, indent=2, ensure_ascii=False)

            if should_delete_split and original_start is not None and original_end is not None:
                deps['delete_translated_split_file'](project_dir, original_start, original_end)

            prepared_segments = deps['prepare_segments_for_response'](project_dir, transcription_data['segments'])
            prepared_segment = prepared_segments[segment_index] if 0 <= segment_index < len(prepared_segments) else None
            response_payload: Dict[str, Any] = {'success': True, 'message': 'Segment updated successfully'}
            if prepared_segment is not None:
                response_payload.update({
                    'segment_index': segment_index,
                    'segment': prepared_segment,
                })

            return jsonify(response_payload)
        except Exception as exc:
            app.logger.error("Error updating segment for project %s: %s", project_name, exc, exc_info=True)
            return jsonify({'success': False, 'error': 'An unexpected error occurred on the server.'}), 500

    @app.route('/api/add-segment/<project_name>', methods=['POST'])
    def add_segment_api(project_name):
        app.logger.info("add_segment_api called for project: %s", project_name)
        try:
            data = request.get_json()
            app.logger.info("Received data for new segment: %s", data)

            json_file_name = data.get('json_file_name')
            new_start = data.get('start')
            new_end = data.get('end')
            new_text = data.get('text', 'Új szegmens')

            if None in [json_file_name, new_start, new_end, new_text]:
                return jsonify({'success': False, 'error': 'Missing data (json_file_name, start, end, or text)'}), 400

            project_dir = os.path.join('workdir', deps['secure_filename'](project_name))
            if not json_file_name:
                return jsonify({'success': False, 'error': 'json_file_name is missing or empty in payload'}), 400

            json_full_path = _resolve_review_json_path(project_dir, json_file_name, deps['config']['PROJECT_SUBDIRS'])
            if not json_full_path:
                app.logger.error("JSON file not found for add: %s", json_file_name)
                return jsonify({'success': False, 'error': f'JSON file not found: {json_file_name}'}), 404

            app.logger.info("Attempting to modify JSON file at: %s", json_full_path)
            with open(json_full_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)

            if 'segments' not in transcription_data or not isinstance(transcription_data['segments'], list):
                transcription_data['segments'] = []

            if not (isinstance(new_start, (int, float)) and isinstance(new_end, (int, float))):
                return jsonify({'success': False, 'error': 'New start and end times must be numbers'}), 400
            if new_start < 0 or new_end < 0 or new_start >= new_end:
                return jsonify({'success': False, 'error': f'Invalid start/end times for new segment: start={new_start}, end={new_end}'}), 400
            if not isinstance(new_text, str):
                return jsonify({'success': False, 'error': 'New text must be a string'}), 400

            for segment in transcription_data['segments']:
                if new_start >= segment['start'] and new_end <= segment['end']:
                    return jsonify({'success': False, 'error': f'New segment ({new_start}-{new_end}) is completely within an existing segment ({segment["start"]}-{segment["end"]}).'}), 400
                if new_start <= segment['start'] and new_end >= segment['end']:
                    return jsonify({'success': False, 'error': f'New segment ({new_start}-{new_end}) completely covers an existing segment ({segment["start"]}-{segment["end"]}).'}), 400
                if new_start >= segment['start'] and new_start < segment['end']:
                    return jsonify({'success': False, 'error': f'New segment start ({new_start}) overlaps with existing segment ({segment["start"]}-{segment["end"]}).'}), 400
                if new_end > segment['start'] and new_end <= segment['end']:
                    return jsonify({'success': False, 'error': f'New segment end ({new_end}) overlaps with existing segment ({segment["start"]}-{segment["end"]}).'}), 400

            transcription_data['segments'].append({
                'start': new_start,
                'end': new_end,
                'text': new_text,
            })
            transcription_data['segments'].sort(key=lambda s: s['start'])

            for segment_item in transcription_data['segments']:
                if 'words' in segment_item:
                    del segment_item['words']

            with open(json_full_path, 'w', encoding='utf-8') as f:
                json.dump(transcription_data, f, indent=2, ensure_ascii=False)

            prepared_segments = deps['prepare_segments_for_response'](project_dir, transcription_data['segments'])
            return jsonify({'success': True, 'message': 'Segment added successfully', 'segments': prepared_segments})
        except Exception as exc:
            app.logger.error("Error adding segment for project %s: %s", project_name, exc, exc_info=True)
            return jsonify({'success': False, 'error': 'An unexpected error occurred on the server while adding segment.'}), 500

    @app.route('/api/delete-segment/<project_name>', methods=['POST'])
    def delete_segment_api(project_name):
        app.logger.info("delete_segment_api called for project: %s", project_name)
        try:
            data = request.get_json()
            app.logger.info("Received data for delete: %s", data)

            json_file_name = data.get('json_file_name')
            segment_index = data.get('segment_index')

            if None in [json_file_name, segment_index]:
                return jsonify({'success': False, 'error': 'Missing data (json_file_name or segment_index)'}), 400

            project_dir = os.path.join('workdir', deps['secure_filename'](project_name))
            if not json_file_name:
                return jsonify({'success': False, 'error': 'json_file_name is missing or empty in payload'}), 400

            json_full_path = _resolve_review_json_path(project_dir, json_file_name, deps['config']['PROJECT_SUBDIRS'])
            if not json_full_path:
                app.logger.error("JSON file not found for delete: %s", json_file_name)
                return jsonify({'success': False, 'error': f'JSON file not found: {json_file_name}'}), 404

            app.logger.info("Attempting to modify JSON file for deletion at: %s", json_full_path)
            with open(json_full_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)

            if 'segments' not in transcription_data or not isinstance(transcription_data['segments'], list):
                return jsonify({'success': False, 'error': 'Invalid JSON structure: "segments" missing or not a list'}), 500

            if not (isinstance(segment_index, int) and 0 <= segment_index < len(transcription_data['segments'])):
                return jsonify({'success': False, 'error': f'Invalid segment index: {segment_index}'}), 400

            segment_to_delete = transcription_data['segments'][segment_index]
            original_start = segment_to_delete.get('start') if isinstance(segment_to_delete, dict) else None
            original_end = segment_to_delete.get('end') if isinstance(segment_to_delete, dict) else None

            del transcription_data['segments'][segment_index]

            for segment_item in transcription_data['segments']:
                if 'words' in segment_item:
                    del segment_item['words']

            with open(json_full_path, 'w', encoding='utf-8') as f:
                json.dump(transcription_data, f, indent=2, ensure_ascii=False)

            if original_start is not None and original_end is not None:
                deps['delete_translated_split_file'](project_dir, original_start, original_end)

            prepared_segments = deps['prepare_segments_for_response'](project_dir, transcription_data['segments'])
            return jsonify({'success': True, 'message': 'Segment deleted successfully', 'segments': prepared_segments})
        except Exception as exc:
            app.logger.error("Error deleting segment for project %s: %s", project_name, exc, exc_info=True)
            return jsonify({'success': False, 'error': 'An unexpected error occurred on the server while deleting segment.'}), 500

    @app.route('/api/get-segments/<project_name>', methods=['POST'])
    def get_segments_api(project_name):
        sanitized_project = deps['secure_filename'](project_name)
        if not sanitized_project:
            return jsonify({'success': False, 'error': 'Érvénytelen projektnév.'}), 400

        payload = request.get_json() or {}
        json_file_name = payload.get('json_file_name')
        if not json_file_name or not isinstance(json_file_name, str):
            return jsonify({'success': False, 'error': 'Hiányzó vagy érvénytelen JSON fájlnév.'}), 400

        current_config = deps['get_config_copy']()
        workdir_path = current_config['DIRECTORIES']['workdir']
        project_dir = os.path.join(workdir_path, sanitized_project)
        if not os.path.isdir(project_dir):
            return jsonify({'success': False, 'error': 'A projekt könyvtára nem található.'}), 404

        project_subdirs = current_config.get('PROJECT_SUBDIRS', {})
        translated_subdir = project_subdirs.get('translated')
        speech_subdir = project_subdirs.get('separated_audio_speech')
        if not translated_subdir or not speech_subdir:
            return jsonify({'success': False, 'error': 'A konfiguráció nem tartalmazza a szükséges almappákat.'}), 500

        json_full_path = _resolve_review_json_path(project_dir, json_file_name, project_subdirs)
        if not json_full_path:
            return jsonify({'success': False, 'error': f'A megadott JSON fájl nem található: {json_file_name}'}), 404

        try:
            with open(json_full_path, 'r', encoding='utf-8') as fp:
                transcription_data = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            app.logger.error("Nem sikerült beolvasni a JSON fájlt a szegmens lekérdezéshez: %s", exc, exc_info=True)
            return jsonify({'success': False, 'error': 'Nem sikerült beolvasni a JSON fájlt.'}), 500

        segments = transcription_data.get('segments', [])
        prepared_segments = deps['prepare_segments_for_response'](project_dir, segments)
        return jsonify({'success': True, 'segments': prepared_segments})

    @app.route('/api/regenerate-segment/<project_name>', methods=['POST'])
    def regenerate_segment_api(project_name):
        app.logger.info("Regenerate segment requested for project: %s", project_name)
        payload = request.get_json() or {}
        json_file_name = payload.get('json_file_name')
        segment_index = payload.get('segment_index')

        if not json_file_name or not isinstance(json_file_name, str):
            return jsonify({'success': False, 'error': 'Hiányzó vagy érvénytelen JSON fájlnév.'}), 400
        if not isinstance(segment_index, int):
            return jsonify({'success': False, 'error': 'Hiányzó vagy érvénytelen szegmens index.'}), 400

        sanitized_project = deps['secure_filename'](project_name)
        if not sanitized_project:
            return jsonify({'success': False, 'error': 'Érvénytelen projektnév.'}), 400

        current_config = deps['get_config_copy']()
        workdir_path = current_config['DIRECTORIES']['workdir']
        project_dir = os.path.join(workdir_path, sanitized_project)
        if not os.path.isdir(project_dir):
            return jsonify({'success': False, 'error': 'A projekt könyvtára nem található.'}), 404

        project_subdirs = current_config.get('PROJECT_SUBDIRS', {})
        translated_subdir = project_subdirs.get('translated')
        speech_subdir = project_subdirs.get('separated_audio_speech')
        temp_subdir = project_subdirs.get('temp')
        if not all([translated_subdir, speech_subdir, temp_subdir]):
            return jsonify({'success': False, 'error': 'A konfiguráció nem tartalmazza a szükséges almappákat.'}), 500

        speech_dir_path = os.path.join(project_dir, speech_subdir)
        temp_dir_path = os.path.join(project_dir, temp_subdir)
        os.makedirs(temp_dir_path, exist_ok=True)

        json_full_path = _resolve_review_json_path(project_dir, json_file_name, project_subdirs)
        if not json_full_path:
            return jsonify({'success': False, 'error': f'A megadott JSON fájl nem található: {json_file_name}'}), 404

        try:
            with open(json_full_path, 'r', encoding='utf-8') as source_fp:
                transcription_data = json.load(source_fp)
        except (OSError, json.JSONDecodeError) as exc:
            app.logger.error("Nem sikerült beolvasni a JSON fájlt regeneráláshoz: %s", exc, exc_info=True)
            return jsonify({'success': False, 'error': 'Nem sikerült beolvasni a JSON fájlt a regeneráláshoz.'}), 500

        segments = transcription_data.get('segments')
        if not isinstance(segments, list) or not segments:
            return jsonify({'success': False, 'error': 'A JSON fájl nem tartalmaz szegmens listát.'}), 400
        if not (0 <= segment_index < len(segments)):
            return jsonify({'success': False, 'error': f'Érvénytelen szegmens index: {segment_index}'}), 400

        target_segment = copy.deepcopy(segments[segment_index])
        if isinstance(target_segment, dict):
            target_segment.pop('words', None)

        deps['sanitize_segment_strings']([target_segment])
        regenerate_payload = {
            'segments': [target_segment],
            'segment_index': segment_index,
            'source_json': json_file_name,
        }

        temp_json_path = os.path.join(temp_dir_path, json_file_name)
        try:
            with open(temp_json_path, 'w', encoding='utf-8') as temp_fp:
                json.dump(regenerate_payload, temp_fp, ensure_ascii=False, indent=2)
        except OSError as exc:
            app.logger.error("Nem sikerült létrehozni a regenerációs JSON fájlt: %s", exc, exc_info=True)
            return jsonify({'success': False, 'error': 'Nem sikerült létrehozni a regenerációs JSON fájlt.'}), 500

        base_name, _ = os.path.splitext(json_file_name)
        matching_audio_name = deps['find_matching_audio_file'](base_name, speech_dir_path)
        if not matching_audio_name:
            return jsonify({'success': False, 'error': 'Nem található a JSON fájlhoz tartozó referencia WAV.'}), 404

        source_audio_path = os.path.join(speech_dir_path, matching_audio_name)
        temp_audio_path = os.path.join(temp_dir_path, matching_audio_name)
        if not os.path.exists(temp_audio_path):
            try:
                os.symlink(os.path.abspath(source_audio_path), temp_audio_path)
            except FileExistsError:
                pass
            except (AttributeError, NotImplementedError):
                app.logger.error("A környezet nem támogatja a szimbolikus linket, audio fájl nem érhető el másolás nélkül.")
                return jsonify({'success': False, 'error': 'A környezet nem támogatja a szimbolikus linkek létrehozását, a referencia WAV nem érhető el.'}), 500
            except OSError as exc:
                app.logger.error("Nem sikerült létrehozni a referencia WAV szimbolikus linkjét: %s", exc, exc_info=True)
                return jsonify({'success': False, 'error': 'Nem sikerült előkészíteni a referencia WAV fájlt.'}), 500

        workflow_state_path = deps['get_project_workflow_state_path'](sanitized_project, config_snapshot=current_config)
        if not workflow_state_path or not workflow_state_path.is_file():
            return jsonify({'success': False, 'error': 'A projekt nem tartalmaz workflow_state.json fájlt.'}), 404

        try:
            with workflow_state_path.open('r', encoding='utf-8') as workflow_fp:
                workflow_state_raw = json.load(workflow_fp)
        except (OSError, json.JSONDecodeError) as exc:
            app.logger.error("Nem sikerült beolvasni a workflow_state.json fájlt: %s", exc, exc_info=True)
            return jsonify({'success': False, 'error': 'Nem sikerült beolvasni a workflow_state.json fájlt.'}), 500

        if isinstance(workflow_state_raw, dict):
            workflow_steps = workflow_state_raw.get('steps') or []
            template_id = workflow_state_raw.get('template_id')
        elif isinstance(workflow_state_raw, list):
            workflow_steps = workflow_state_raw
            template_id = None
        else:
            workflow_steps = []
            template_id = None

        tts_steps = [
            step for step in workflow_steps
            if isinstance(step, dict) and isinstance(step.get('script'), str) and step['script'].startswith('TTS/')
        ]
        if not tts_steps:
            return jsonify({'success': False, 'error': 'A workflow-ban nem található TTS szkript. Regenerálás nem indítható.'}), 400
        if len(tts_steps) > 1:
            return jsonify({'success': False, 'error': 'Egynél több TTS szkript található a workflow-ban. Csak egy TTS lépés engedélyezett a regeneráláshoz.'}), 400

        original_tts_step = tts_steps[0]
        step_params = dict(original_tts_step.get('params') or {})
        step_params['input_directory_override'] = 'true'
        step_params['max_retries'] = '1'
        step_params['pitch_retry'] = '1'
        step_params['save_failures'] = True

        regenerate_step = {
            'script': original_tts_step.get('script'),
            'enabled': True,
            'halt_on_fail': bool(original_tts_step.get('halt_on_fail', True)),
            'params': step_params,
        }

        try:
            normalized_steps, required_keys, _ = deps['normalize_workflow_steps']([regenerate_step])
        except deps['WorkflowValidationError'] as exc:
            return jsonify({'success': False, 'error': str(exc)}), 400

        with deps['workflow_lock']:
            active_for_project = [
                job_id for job_id, job in deps['workflow_jobs'].items()
                if job.get('project') == sanitized_project and job.get('status') in ('queued', 'running', 'cancelling')
            ]
        if active_for_project:
            return jsonify({'success': False, 'error': 'Már fut egy workflow ehhez a projekthez. Várd meg a befejezést.'}), 409

        key_status = deps['determine_workflow_key_status'](required_keys)
        missing_keys = [
            info['label']
            for info in key_status['keys'].values()
            if info['required'] and not info['present']
        ]
        if missing_keys:
            missing_list = ', '.join(missing_keys)
            return jsonify({'success': False, 'error': f'Hiányzó API kulcs(ok) a TTS futtatásához: {missing_list}'}), 400

        job_id = uuid.uuid4().hex
        encoded_steps = deps['mask_workflow_secret_params'](normalized_steps)
        job_data = {
            'job_id': job_id,
            'project': sanitized_project,
            'status': 'queued',
            'created_at': datetime.utcnow().isoformat(),
            'message': 'Regeneráció sorban áll.',
            'log': None,
            'cancel_requested': False,
            'workflow': encoded_steps,
            'execution_steps': encoded_steps,
            'required_keys': list(required_keys),
            'template_id': template_id,
        }
        deps['register_workflow_job'](job_id, job_data)

        thread_payload = {
            'steps': normalized_steps,
            'workflow_state': normalized_steps,
            'template_id': template_id,
        }
        thread = threading.Thread(
            target=deps['run_workflow_job'],
            args=(job_id, sanitized_project, thread_payload),
            daemon=True,
        )
        deps['set_workflow_thread'](job_id, thread)
        thread.start()

        message = (
            f'Szegmens regeneráció elindítva. A feldolgozás a(z) "{matching_audio_name}" referencia WAV alapján történik. '
            'Az állapot a felülvizsgálati nézetben is követhető.'
        )
        return jsonify({
            'success': True,
            'message': message,
            'job_id': job_id,
            'temp_json_path': os.path.relpath(temp_json_path, project_dir),
        })

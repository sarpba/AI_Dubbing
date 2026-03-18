from __future__ import annotations

import copy
import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import jsonify, request


def register_workflow_api_routes(app, deps: Dict[str, Any]) -> None:
    @app.route('/save-workflow-keys', methods=['POST'])
    def save_workflow_keys():
        data = request.get_json() or {}
        if not data:
            return jsonify({'success': False, 'error': 'Hiányzó kulcs adatok'}), 400

        keyholder_data = deps['load_keyholder_data']()
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
            encoded = deps['encode_keyholder_value'](raw_value)
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

        if not deps['save_keyholder_data'](keyholder_data):
            return jsonify({'success': False, 'error': 'Nem sikerült elmenteni a keyholder.json fájlt'}), 500
        try:
            return jsonify({'success': True})
        except Exception as exc:
            return jsonify({'success': False, 'error': str(exc)}), 500

    @app.route('/api/workflow-key-status/<project_name>', methods=['POST'])
    def workflow_key_status(project_name):
        sanitized_project = deps['secure_filename'](project_name)
        current_config = deps['get_config_copy']()
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
                _, required_keys, _ = deps['normalize_workflow_steps'](steps_payload)
            except deps['WorkflowValidationError'] as exc:
                return jsonify({'success': False, 'error': str(exc)}), 400

        status = deps['determine_workflow_key_status'](required_keys)
        status['success'] = True
        return jsonify(status)

    @app.route('/api/workflow-templates', methods=['GET'])
    def get_workflow_templates_api():
        return jsonify({'success': True, 'templates': deps['list_workflow_templates']()})

    @app.route('/api/workflow-template/<template_id>', methods=['GET'])
    def get_workflow_template_api(template_id):
        template = deps['load_workflow_template'](template_id)
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
            normalized_steps, _, _ = deps['normalize_workflow_steps'](steps_payload)
        except deps['WorkflowValidationError'] as exc:
            return jsonify({'success': False, 'error': str(exc)}), 400

        overwrite = deps['coerce_bool'](data.get('overwrite', False))
        template_id_raw = (data.get('template_id') or '').strip() or None
        name = (data.get('name') or '').strip()
        description = (data.get('description') or '').strip() or None

        if not name and template_id_raw:
            existing = deps['load_workflow_template'](template_id_raw)
            if existing:
                name = existing.get('name') or template_id_raw
                if description is None:
                    description = existing.get('description')
            else:
                name = template_id_raw

        if not name:
            return jsonify({'success': False, 'error': 'A workflow nevének megadása kötelező.'}), 400

        try:
            saved = deps['save_workflow_template_file'](
                name=name,
                steps=normalized_steps,
                template_id=template_id_raw,
                overwrite=overwrite,
                description=description,
            )
        except deps['WorkflowValidationError'] as exc:
            return jsonify({'success': False, 'error': str(exc)}), 400

        return jsonify({'success': True, 'template': saved})

    @app.route('/api/workflow-options/<project_name>', methods=['GET'])
    def get_workflow_options_api(project_name):
        sanitized_project = deps['secure_filename'](project_name)
        current_config = deps['get_config_copy']()
        workdir_path = current_config['DIRECTORIES']['workdir']
        project_dir = os.path.join(workdir_path, sanitized_project)
        if not os.path.isdir(project_dir):
            return jsonify({'success': False, 'error': 'Projekt nem található'}), 404

        scripts = deps['get_scripts_catalog']()
        templates = deps['list_workflow_templates']()

        saved_state = deps['load_project_workflow_state'](sanitized_project, config_snapshot=current_config)
        defaults_workflow: List[Dict[str, Any]] = []
        selected_template: Optional[str] = None

        if saved_state and isinstance(saved_state.get('steps'), list):
            defaults_workflow = copy.deepcopy(saved_state['steps'])
            selected_template = saved_state.get('template_id')
        else:
            template_data = deps['load_workflow_template']('default')
            if not template_data and templates:
                template_id = templates[0]['id']
                template_data = deps['load_workflow_template'](template_id)
            else:
                template_id = 'default'

            if template_data and isinstance(template_data.get('steps'), list):
                defaults_workflow = copy.deepcopy(template_data['steps'])
                selected_template = template_data.get('id') or template_id

            state_path = deps['get_project_workflow_state_path'](sanitized_project, config_snapshot=current_config)
            initialize_state = state_path is None or not state_path.is_file()

            if initialize_state:
                try:
                    saved_state = deps['save_project_workflow_state'](
                        sanitized_project,
                        defaults_workflow,
                        selected_template,
                        config_snapshot=current_config,
                        saved_at=datetime.utcnow().isoformat(),
                    )
                    if saved_state:
                        defaults_workflow = copy.deepcopy(saved_state.get('steps') or [])
                        if saved_state.get('template_id'):
                            selected_template = saved_state['template_id']
                except deps['WorkflowValidationError'] as exc:
                    logging.warning(
                        "Nem sikerült alap workflow állapotot menteni (%s): %s",
                        sanitized_project,
                        exc,
                    )
                except Exception as exc:
                    logging.error(
                        "Váratlan hiba történt az alap workflow állapot mentésekor (%s): %s",
                        sanitized_project,
                        exc,
                        exc_info=True,
                    )

        defaults = {
            'workflow': deps['mask_workflow_secret_params'](defaults_workflow),
            'selected_template': selected_template,
        }

        recent_jobs = sorted(
            deps['get_project_jobs'](sanitized_project),
            key=lambda job: job.get('started_at') or job.get('created_at') or '',
            reverse=True,
        )
        latest_job = recent_jobs[0] if recent_jobs else None

        return jsonify({
            'success': True,
            'scripts': scripts,
            'defaults': defaults,
            'templates': templates,
            'latest_job': latest_job,
            'project': sanitized_project,
        })

    @app.route('/api/project-workflow-state/<project_name>', methods=['GET', 'POST'])
    def project_workflow_state_api(project_name):
        sanitized_project = deps['secure_filename'](project_name)
        if not sanitized_project:
            return jsonify({'success': False, 'error': 'Érvénytelen projektnév.'}), 400

        config_snapshot = deps['get_config_copy']()
        project_root = deps['get_project_root_path'](sanitized_project, config_snapshot=config_snapshot)
        if not project_root or not project_root.is_dir():
            return jsonify({'success': False, 'error': 'Projekt nem található.'}), 404

        if request.method == 'GET':
            state = deps['load_project_workflow_state'](sanitized_project, config_snapshot=config_snapshot)
            if not state:
                return jsonify({'success': False, 'error': 'Nincs mentett workflow állapot.'}), 404
            return jsonify({'success': True, 'state': state})

        payload = request.get_json() or {}
        steps_payload = payload.get('steps')
        try:
            normalized_steps, _, _ = deps['normalize_workflow_steps'](steps_payload)
        except deps['WorkflowValidationError'] as exc:
            return jsonify({'success': False, 'error': str(exc)}), 400

        template_id = (payload.get('template_id') or '').strip() or None
        saved_at = (payload.get('saved_at') or '').strip() or datetime.utcnow().isoformat()

        try:
            state = deps['save_project_workflow_state'](
                sanitized_project,
                normalized_steps,
                template_id,
                config_snapshot=config_snapshot,
                saved_at=saved_at,
            )
        except deps['WorkflowValidationError'] as exc:
            return jsonify({'success': False, 'error': str(exc)}), 400
        except Exception as exc:
            logging.error("Nem sikerült menteni a workflow állapotot: %s", exc, exc_info=True)
            return jsonify({'success': False, 'error': 'Nem sikerült menteni a workflow állapotot.'}), 500

        return jsonify({'success': True, 'state': state})

    @app.route('/api/run-workflow/<project_name>', methods=['POST'])
    def run_workflow_api(project_name):
        sanitized_project = deps['secure_filename'](project_name)
        data = request.get_json() or {}

        steps_payload = data.get('steps')
        try:
            normalized_steps, required_keys, _ = deps['normalize_workflow_steps'](steps_payload)
        except deps['WorkflowValidationError'] as exc:
            return jsonify({'success': False, 'error': str(exc)}), 400

        workflow_state_payload = data.get('workflow_state')
        if workflow_state_payload is not None:
            try:
                normalized_full_steps, _, _ = deps['normalize_workflow_steps'](workflow_state_payload)
            except deps['WorkflowValidationError'] as exc:
                return jsonify({'success': False, 'error': str(exc)}), 400
        else:
            normalized_full_steps = normalized_steps

        template_id = (data.get('template_id') or '').strip() or None

        current_config = deps['get_config_copy']()
        workdir_path = current_config['DIRECTORIES']['workdir']
        project_dir = os.path.join(workdir_path, sanitized_project)
        if not os.path.isdir(project_dir):
            return jsonify({'success': False, 'error': 'A projekt nem található.'}), 404

        with deps['workflow_lock']:
            active_for_project = [
                job_id for job_id, job in deps['workflow_jobs'].items()
                if job.get('project') == sanitized_project and job.get('status') in ('queued', 'running', 'cancelling')
            ]
        if active_for_project:
            return jsonify({'success': False, 'error': 'Már fut egy workflow ehhez a projekthez.'}), 409

        key_status = deps['determine_workflow_key_status'](required_keys)
        missing_keys = [
            info['label']
            for key, info in key_status['keys'].items()
            if info.get('required') and not info.get('present')
        ]
        if missing_keys:
            return jsonify({
                'success': False,
                'error': 'Hiányzó API kulcsok: ' + ', '.join(missing_keys),
                'missing_keys': missing_keys,
            }), 400

        encoded_full_steps = deps['mask_workflow_secret_params'](normalized_full_steps)
        encoded_steps = deps['mask_workflow_secret_params'](normalized_steps)
        saved_timestamp = datetime.utcnow().isoformat()
        try:
            deps['save_project_workflow_state'](
                sanitized_project,
                normalized_full_steps,
                template_id,
                config_snapshot=current_config,
                saved_at=saved_timestamp,
            )
        except deps['WorkflowValidationError'] as exc:
            return jsonify({'success': False, 'error': str(exc)}), 500
        except Exception as exc:
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
            'template_id': template_id,
        }
        deps['register_workflow_job'](job_id, job_data)

        thread = threading.Thread(
            target=deps['run_workflow_job'],
            args=(job_id, sanitized_project, {'steps': normalized_steps, 'workflow_state': normalized_full_steps, 'template_id': template_id}),
            daemon=True,
        )
        deps['set_workflow_thread'](job_id, thread)
        thread.start()

        return jsonify({'success': True, 'job_id': job_id})

    @app.route('/api/stop-workflow/<job_id>', methods=['POST'])
    def stop_workflow(job_id):
        job = deps['get_workflow_job'](job_id)
        if not job:
            return jsonify({'success': False, 'error': 'Feladat nem található.'}), 404

        if job.get('status') in ('completed', 'failed', 'cancelled'):
            return jsonify({'success': False, 'error': 'A feladat már befejeződött.'}), 400

        if job.get('cancel_requested'):
            return jsonify({'success': True, 'message': 'Megszakítás már folyamatban.'})

        if not deps['request_workflow_cancel'](job_id):
            return jsonify({'success': False, 'error': 'A feladat már nem fut.'}), 409

        deps['update_workflow_job'](
            job_id,
            status='cancelling',
            message='Megszakítás kérése folyamatban...',
            cancel_requested=True,
        )
        return jsonify({'success': True, 'message': 'Megszakítás kérve. Várakozás a leállásra.'})

    @app.route('/api/workflow-log/<job_id>', methods=['GET'])
    def get_workflow_log(job_id):
        job = deps['get_workflow_job'](job_id)
        if not job:
            return jsonify({'success': False, 'error': 'Feladat nem található.'}), 404

        log_info = job.get('log') or {}
        log_relative = log_info.get('relative')
        log_text = ""
        log_available = False

        if log_relative:
            current_config = deps['get_config_copy']()
            workdir_path = current_config['DIRECTORIES']['workdir']
            absolute_path = deps['resolve_workspace_path'](os.path.join(workdir_path, log_relative))
            if absolute_path and os.path.exists(absolute_path):
                log_available = True
                log_text = deps['read_log_tail'](absolute_path)

        completed = job.get('status') in ('completed', 'failed', 'cancelled')

        return jsonify({
            'success': True,
            'log': log_text,
            'log_available': log_available,
            'status': job.get('status'),
            'message': job.get('message'),
            'completed': completed,
            'cancel_requested': job.get('cancel_requested', False),
        })

    @app.route('/api/workflow-status/<job_id>', methods=['GET'])
    def get_workflow_status(job_id):
        job = deps['get_workflow_job'](job_id)
        if not job:
            return jsonify({'success': False, 'error': 'Feladat nem található'}), 404
        return jsonify({'success': True, 'job': job})

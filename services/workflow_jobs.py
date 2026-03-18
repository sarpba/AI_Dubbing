from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple


def update_workflow_job(
    job_id: str,
    workflow_lock,
    workflow_jobs: Dict[str, Dict[str, Any]],
    mask_workflow_secret_params: Callable[[Optional[List[Dict[str, Any]]]], List[Dict[str, Any]]],
    **kwargs,
) -> None:
    with workflow_lock:
        if 'workflow' in kwargs:
            kwargs['workflow'] = mask_workflow_secret_params(kwargs['workflow'])
        if 'execution_steps' in kwargs:
            kwargs['execution_steps'] = mask_workflow_secret_params(kwargs['execution_steps'])
        if job_id in workflow_jobs:
            workflow_jobs[job_id].update(kwargs)


def get_workflow_job(job_id: str, workflow_lock, workflow_jobs: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    with workflow_lock:
        return workflow_jobs.get(job_id)


def get_project_jobs(
    project_name: str,
    workflow_lock,
    workflow_jobs: Dict[str, Dict[str, Any]],
    secure_filename: Callable[[str], str],
) -> List[Dict[str, Any]]:
    sanitized_name = secure_filename(project_name)
    with workflow_lock:
        return [job.copy() for job in workflow_jobs.values() if job.get('project') == sanitized_name]


def register_workflow_job(
    job_id: str,
    job_data: Dict[str, Any],
    workflow_lock,
    workflow_jobs: Dict[str, Dict[str, Any]],
    workflow_events: Dict[str, threading.Event],
    mask_workflow_secret_params: Callable[[Optional[List[Dict[str, Any]]]], List[Dict[str, Any]]],
) -> None:
    with workflow_lock:
        if isinstance(job_data, dict):
            if 'workflow' in job_data:
                job_data['workflow'] = mask_workflow_secret_params(job_data['workflow'])
            if 'execution_steps' in job_data:
                job_data['execution_steps'] = mask_workflow_secret_params(job_data['execution_steps'])
        workflow_jobs[job_id] = job_data
        workflow_events[job_id] = threading.Event()


def set_workflow_thread(job_id: str, thread, workflow_lock, workflow_threads: Dict[str, Any]) -> None:
    with workflow_lock:
        workflow_threads[job_id] = thread


def get_workflow_thread(job_id: str, workflow_lock, workflow_threads: Dict[str, Any]):
    with workflow_lock:
        return workflow_threads.get(job_id)


def get_workflow_event(job_id: str, workflow_lock, workflow_events: Dict[str, threading.Event]):
    with workflow_lock:
        return workflow_events.get(job_id)


def cleanup_workflow_resources(job_id: str, workflow_lock, workflow_threads: Dict[str, Any], workflow_events: Dict[str, threading.Event]) -> None:
    with workflow_lock:
        workflow_threads.pop(job_id, None)
        workflow_events.pop(job_id, None)


def request_workflow_cancel(job_id: str, workflow_lock, workflow_events: Dict[str, threading.Event]) -> bool:
    event = get_workflow_event(job_id, workflow_lock, workflow_events)
    if not event:
        return False
    if not event.is_set():
        event.set()
    return True


def read_log_tail(log_path: str, max_bytes: int = 12000) -> str:
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


def build_log_links(project_name: str, logs_subdir: str, log_filename: str, secure_filename: Callable[[str], str]) -> Dict[str, str]:
    relative_path = os.path.join(secure_filename(project_name), logs_subdir, log_filename)
    return {'relative': relative_path, 'url': f"/workdir/{relative_path}"}


def run_workflow_job(
    job_id: str,
    project_name: str,
    workflow_payload: Dict[str, Any],
    *,
    secure_filename: Callable[[str], str],
    get_workflow_event_fn: Callable[[str], Any],
    update_workflow_job_fn: Callable[..., None],
    get_config_copy: Callable[[], Dict[str, Any]],
    ensure_project_structure: Callable[[str, Dict[str, str]], None],
    setup_project_logging: Callable[[str, str, str], Tuple[logging.Handler, str]],
    build_log_links_fn: Callable[[str, str, str], Dict[str, str]],
    mask_workflow_secret_params: Callable[[Optional[List[Dict[str, Any]]]], List[Dict[str, Any]]],
    load_keyholder_data: Callable[[], Dict[str, Any]],
    get_script_definition: Callable[[str], Optional[Dict[str, Any]]],
    build_command_for_step: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Tuple[List[str], List[Dict[str, Any]]]],
    build_masked_command_and_params: Callable[[Optional[List[str]], Optional[List[Dict[str, Any]]], Dict[str, Any]], Tuple[List[str], List[Dict[str, Any]]]],
    run_script_command: Callable[..., Any],
    workflow_validation_error: type[Exception],
    remove_logging_handler: Callable[[logging.Handler], None],
    cleanup_workflow_resources_fn: Callable[[str], None],
) -> None:
    sanitized_project = secure_filename(project_name)
    log_handler = None
    cancel_event = get_workflow_event_fn(job_id)

    def should_stop():
        return cancel_event.is_set() if cancel_event else False

    try:
        initial_status = 'cancelling' if should_stop() else 'running'
        initial_message = 'Megszakítás kérve, előkészítés folyamatban...' if initial_status == 'cancelling' else 'Feldolgozás előkészítése...'
        update_workflow_job_fn(
            job_id,
            status=initial_status,
            started_at=datetime.now(timezone.utc).isoformat(),
            message=initial_message,
            cancel_requested=should_stop(),
        )

        current_config = get_config_copy()
        workdir_path = current_config['DIRECTORIES']['workdir']
        project_path = os.path.join(workdir_path, sanitized_project)

        ensure_project_structure(project_path, current_config['PROJECT_SUBDIRS'])
        log_handler, log_file = setup_project_logging(project_path, current_config['PROJECT_SUBDIRS']['logs'], sanitized_project)
        log_links = build_log_links_fn(sanitized_project, current_config['PROJECT_SUBDIRS']['logs'], os.path.basename(log_file))
        update_workflow_job_fn(job_id, log=log_links)

        template_id = workflow_payload.get('template_id')
        steps = workflow_payload.get('steps') or []
        workflow_state = workflow_payload.get('workflow_state') or steps
        masked_steps = mask_workflow_secret_params(steps)
        masked_workflow_state = mask_workflow_secret_params(workflow_state)
        active_steps = [step for step in steps if step.get('enabled', True) and step.get('type') != 'widget']
        total_steps = len(active_steps)
        update_workflow_job_fn(
            job_id,
            total_steps=total_steps,
            template_id=template_id,
            workflow=masked_workflow_state,
            execution_steps=masked_steps,
        )

        keyholder_snapshot = load_keyholder_data()
        executed_steps: List[Dict[str, Any]] = []
        context = {
            'project_name': sanitized_project,
            'project_path': project_path,
            'keyholder': keyholder_snapshot,
            'config': current_config,
            'template_id': template_id,
        }

        for index, step in enumerate(active_steps, start=1):
            if should_stop():
                logging.info("Workflow megszakítás kérve, kilépünk.")
                update_workflow_job_fn(
                    job_id,
                    status='cancelled',
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    message='Workflow megszakítva.',
                    cancel_requested=False,
                    current_step=None,
                    results=executed_steps,
                )
                return

            script_id = step.get('script')
            script_meta = get_script_definition(script_id)
            if not script_meta:
                raise workflow_validation_error(f"Ismeretlen szkript: {script_id}")

            command, applied_params = build_command_for_step(step, script_meta, context)
            masked_command, masked_applied_params = build_masked_command_and_params(command, applied_params, script_meta)

            update_workflow_job_fn(
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
                cancel_requested=should_stop(),
            )

            result = run_script_command(command, log_prefix=f"[{script_meta['id']}]", should_stop=should_stop)
            step_record = {
                'script': script_meta['id'],
                'display_name': script_meta['display_name'],
                'command': masked_command,
                'applied_params': masked_applied_params,
            }

            if result == "cancelled":
                step_record['status'] = 'cancelled'
                executed_steps.append(step_record)
                update_workflow_job_fn(
                    job_id,
                    status='cancelled',
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    message='Workflow megszakítva.',
                    cancel_requested=False,
                    current_step=None,
                    results=executed_steps,
                )
                return

            if result is False:
                step_record['status'] = 'failed'
                executed_steps.append(step_record)
                if step.get('halt_on_fail', True):
                    update_workflow_job_fn(
                        job_id,
                        status='failed',
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        message=f"Hiba a(z) {script_meta['display_name']} lépés futtatása közben.",
                        cancel_requested=False,
                        current_step=None,
                        results=executed_steps,
                    )
                    return
                logging.warning("A(z) %s lépés hibával futott, de a workflow folytatódik.", script_meta['display_name'])
                continue

            step_record['status'] = 'completed'
            executed_steps.append(step_record)

        update_workflow_job_fn(
            job_id,
            status='completed',
            finished_at=datetime.now(timezone.utc).isoformat(),
            message='Workflow sikeresen lefutott.',
            cancel_requested=False,
            current_step=None,
            results=executed_steps,
        )
    except workflow_validation_error as exc:
        logging.error("Workflow konfigurációs hiba: %s", exc)
        update_workflow_job_fn(
            job_id,
            status='failed',
            finished_at=datetime.now(timezone.utc).isoformat(),
            message=str(exc),
            cancel_requested=False,
            current_step=None,
            results=locals().get('executed_steps'),
        )
    except Exception as exc:
        logging.exception("Workflow futtatási hiba: %s", exc)
        if should_stop():
            update_workflow_job_fn(
                job_id,
                status='cancelled',
                finished_at=datetime.now(timezone.utc).isoformat(),
                message='Workflow megszakítva.',
                cancel_requested=False,
            )
        else:
            update_workflow_job_fn(
                job_id,
                status='failed',
                finished_at=datetime.now(timezone.utc).isoformat(),
                message=f'Hiba: {exc}',
                cancel_requested=False,
            )
    finally:
        if log_handler:
            remove_logging_handler(log_handler)
        cleanup_workflow_resources_fn(job_id)

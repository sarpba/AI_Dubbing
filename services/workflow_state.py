from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def ensure_workflows_dir(workflows_dir: Path) -> Path:
    try:
        workflows_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logging.error("Nem sikerült létrehozni a workflows könyvtárat: %s", exc)
    return workflows_dir


def sanitize_workflow_id(name: str, secure_filename: Callable[[str], str]) -> str:
    candidate = secure_filename(name or '')
    candidate = candidate.replace(' ', '_').strip('_')
    if not candidate:
        candidate = datetime.now(timezone.utc).strftime("workflow_%Y%m%d_%H%M%S")
    return candidate.lower()


def load_workflow_file(path: Path, mask_workflow_secret_params: Callable[[Optional[List[Dict[str, Any]]]], List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
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

    return {
        'name': name,
        'description': description,
        'steps': mask_workflow_secret_params(steps),
    }


def list_workflow_templates(
    workflows_dir: Path,
    mask_workflow_secret_params: Callable[[Optional[List[Dict[str, Any]]]], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    directory = ensure_workflows_dir(workflows_dir)
    templates: List[Dict[str, Any]] = []
    for file_path in sorted(directory.glob('*.json')):
        template_data = load_workflow_file(file_path, mask_workflow_secret_params)
        if not template_data:
            continue
        templates.append({
            'id': file_path.stem,
            'name': template_data['name'],
            'filename': file_path.name,
            'description': template_data.get('description'),
        })
    return templates


def load_workflow_template(
    workflows_dir: Path,
    template_id: str,
    mask_workflow_secret_params: Callable[[Optional[List[Dict[str, Any]]]], List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if not template_id:
        return None
    directory = ensure_workflows_dir(workflows_dir)
    candidate = directory / template_id
    if candidate.suffix.lower() != '.json':
        candidate = candidate.with_suffix('.json')
    if not candidate.is_file():
        logging.warning("A kért workflow sablon nem található: %s", candidate)
        return None
    template_data = load_workflow_file(candidate, mask_workflow_secret_params)
    if not template_data:
        return None
    template_data.update({'id': candidate.stem, 'filename': candidate.name})
    return template_data


def save_workflow_template_file(
    workflows_dir: Path,
    name: str,
    steps: List[Dict[str, Any]],
    mask_workflow_secret_params: Callable[[Optional[List[Dict[str, Any]]]], List[Dict[str, Any]]],
    secure_filename: Callable[[str], str],
    workflow_validation_error: type[Exception],
    template_id: Optional[str] = None,
    overwrite: bool = False,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    directory = ensure_workflows_dir(workflows_dir)
    file_id = sanitize_workflow_id(Path(template_id).stem if template_id else name, secure_filename)
    if not file_id:
        raise workflow_validation_error("Érvénytelen workflow azonosító.")

    target_path = directory / f"{file_id}.json"
    if target_path.exists() and not overwrite:
        raise workflow_validation_error("Már létezik ugyanilyen nevű workflow. Engedélyezd a felülírást.")

    payload: Dict[str, Any] = {
        'name': name or file_id,
        'steps': mask_workflow_secret_params(steps),
    }
    if description:
        payload['description'] = description

    try:
        with target_path.open('w', encoding='utf-8') as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
    except OSError as exc:
        logging.error("Nem sikerült elmenteni a workflow sablont: %s", exc)
        raise workflow_validation_error(f"Workflow mentése sikertelen: {exc}")

    return {
        'id': file_id,
        'name': payload['name'],
        'filename': target_path.name,
    }


def get_project_root_path(
    project_name: str,
    get_config_copy: Callable[[], Dict[str, Any]],
    secure_filename: Callable[[str], str],
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    sanitized_project = secure_filename(project_name)
    if not sanitized_project:
        return None
    snapshot = config_snapshot or get_config_copy()
    return Path(snapshot['DIRECTORIES']['workdir']) / sanitized_project


def get_project_workflow_state_path(
    project_name: str,
    workflow_state_filename: str,
    get_config_copy: Callable[[], Dict[str, Any]],
    secure_filename: Callable[[str], str],
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    project_root = get_project_root_path(project_name, get_config_copy, secure_filename, config_snapshot=config_snapshot)
    if not project_root:
        return None
    return project_root / workflow_state_filename


def load_project_workflow_state(
    project_name: str,
    workflow_state_filename: str,
    get_config_copy: Callable[[], Dict[str, Any]],
    secure_filename: Callable[[str], str],
    mask_workflow_secret_params: Callable[[Optional[List[Dict[str, Any]]]], List[Dict[str, Any]]],
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    state_path = get_project_workflow_state_path(
        project_name,
        workflow_state_filename,
        get_config_copy,
        secure_filename,
        config_snapshot=config_snapshot,
    )
    if not state_path or not state_path.is_file():
        return None
    try:
        with state_path.open('r', encoding='utf-8') as fp:
            raw_state = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Nem sikerült betölteni a workflow állapotot (%s): %s", state_path, exc)
        return None

    if isinstance(raw_state, list):
        return {'steps': mask_workflow_secret_params(raw_state), 'template_id': None}
    if isinstance(raw_state, dict):
        state: Dict[str, Any] = {
            'steps': mask_workflow_secret_params(raw_state.get('steps') or []),
            'template_id': raw_state.get('template_id'),
        }
        if 'saved_at' in raw_state:
            state['saved_at'] = raw_state['saved_at']
        return state
    return None


def save_project_workflow_state(
    project_name: str,
    steps: List[Dict[str, Any]],
    mask_workflow_secret_params: Callable[[Optional[List[Dict[str, Any]]]], List[Dict[str, Any]]],
    workflow_state_filename: str,
    get_config_copy: Callable[[], Dict[str, Any]],
    secure_filename: Callable[[str], str],
    workflow_validation_error: type[Exception],
    template_id: Optional[str] = None,
    *,
    config_snapshot: Optional[Dict[str, Any]] = None,
    saved_at: Optional[str] = None,
) -> Dict[str, Any]:
    state_path = get_project_workflow_state_path(
        project_name,
        workflow_state_filename,
        get_config_copy,
        secure_filename,
        config_snapshot=config_snapshot,
    )
    if not state_path:
        raise workflow_validation_error("Érvénytelen projekt azonosító.")
    if not state_path.parent.is_dir():
        raise workflow_validation_error("A projekt könyvtára nem található.")

    payload: Dict[str, Any] = {'steps': mask_workflow_secret_params(steps or [])}
    if template_id:
        payload['template_id'] = template_id
    if saved_at:
        payload['saved_at'] = saved_at

    try:
        with state_path.open('w', encoding='utf-8') as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
    except OSError as exc:
        logging.error("Nem sikerült menteni a workflow állapotot (%s): %s", state_path, exc)
        raise workflow_validation_error("Nem sikerült menteni a workflow állapotot.") from exc
    return payload

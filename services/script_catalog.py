from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


def infer_autofill_kind(param_name: str, project_autofill_overrides: Dict[str, str]) -> Optional[str]:
    if not param_name:
        return None
    normalized = param_name.strip().lower()
    if normalized in project_autofill_overrides:
        return project_autofill_overrides[normalized]
    if 'project' in normalized:
        if 'dir' in normalized or 'path' in normalized:
            return 'project_path'
        return 'project_name'
    return None


def rebuild_scripts_config_file(
    scripts_dir: Path,
    scripts_config_path: Path,
    validate_script_meta: Callable[[Path, Dict[str, Any], Path], List[Any]],
) -> List[Dict[str, Any]]:
    if not scripts_dir.exists():
        logging.warning("scripts könyvtár nem található: %s", scripts_dir)
        return []

    collected_entries: List[Dict[str, Any]] = []
    latest_source_mtime = 0.0

    for json_path in scripts_dir.rglob('*.json'):
        if json_path == scripts_config_path:
            continue

        relative_json = json_path.relative_to(scripts_dir)
        py_candidate = scripts_dir / relative_json.with_suffix('.py')
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

        issues = validate_script_meta(json_path, entry, scripts_dir)
        for issue in issues:
            log_message = "%s: %s"
            if issue.level == 'error':
                logging.error(log_message, issue.path, issue.message)
            else:
                logging.warning(log_message, issue.path, issue.message)

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
    if scripts_config_path.exists():
        try:
            target_mtime = scripts_config_path.stat().st_mtime
            with scripts_config_path.open('r', encoding='utf-8') as fp:
                existing_data = json.load(fp)
        except (OSError, json.JSONDecodeError):
            existing_data = None

    if existing_data != collected_entries or target_mtime < latest_source_mtime:
        try:
            scripts_config_path.parent.mkdir(parents=True, exist_ok=True)
            with scripts_config_path.open('w', encoding='utf-8') as fp:
                json.dump(collected_entries, fp, ensure_ascii=False, indent=2)
                fp.write('\n')
        except OSError as exc:
            logging.error("Nem sikerült frissíteni a scripts.json fájlt: %s", exc)
            return collected_entries

    return collected_entries


def prepare_script_entry(
    raw_entry: Dict[str, Any],
    *,
    scripts_dir: Path,
    negative_flag_name_prefixes: Tuple[str, ...],
    secret_param_names: Set[str],
    script_key_requirements: Dict[str, Set[str]],
    project_autofill_overrides: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    script_name = raw_entry.get('script')
    if not script_name:
        return None
    environment = (raw_entry.get('enviroment') or raw_entry.get('environment') or '') or ''
    api_name = raw_entry.get('api')
    parameters: List[Dict[str, Any]] = []

    def humanize_param_name(name: str) -> str:
        return name.replace('_', ' ').strip()

    def strip_negative_prefix(name: str) -> str:
        for prefix in negative_flag_name_prefixes:
            if name.startswith(prefix):
                stripped = name[len(prefix):]
                if stripped:
                    return stripped
        return name

    def resolve_flag_mode(name: str, flags: List[str]) -> Tuple[str, str]:
        positive_flag = next((flag for flag in flags if not flag.startswith('--no-')), None)
        negative_flag = next((flag for flag in flags if flag.startswith('--no-')), None)
        if negative_flag and not positive_flag:
            if any(name.startswith(prefix) for prefix in negative_flag_name_prefixes):
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
                'autofill': infer_autofill_kind(name, project_autofill_overrides),
                'secret': name in secret_param_names,
                'default': default_value,
                'description': param.get('description'),
            })

    append_params(raw_entry.get('required'), True)
    append_params(raw_entry.get('optional'), False)

    help_markdown: Optional[str] = None
    try:
        script_path = Path(script_name)
        if script_path.is_absolute():
            script_path = Path(script_path.name)
        help_path = (scripts_dir / script_path).with_name(f"{script_path.stem}_help.md")
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
        'required_keys': sorted(script_key_requirements.get(script_name, set())),
        'api': api_name,
        'help_markdown': help_markdown,
    }


def get_scripts_catalog(
    *,
    force_reload: bool,
    scripts_config_path: Path,
    scripts_cache: Dict[str, Any],
    scripts_cache_lock,
    load_scripts_file: Callable[[], List[Dict[str, Any]]],
    prepare_script_entry_fn: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    try:
        current_mtime = scripts_config_path.stat().st_mtime
    except OSError:
        current_mtime = None

    with scripts_cache_lock:
        cached_mtime = scripts_cache.get('mtime')
        if not force_reload and cached_mtime == current_mtime and scripts_cache.get('data'):
            return copy.deepcopy(scripts_cache['data'])

        raw_entries = load_scripts_file()
        try:
            current_mtime = scripts_config_path.stat().st_mtime
        except OSError:
            current_mtime = None

        catalog = []
        for entry in raw_entries:
            prepared = prepare_script_entry_fn(entry)
            if prepared:
                catalog.append(prepared)

        scripts_cache['mtime'] = current_mtime
        scripts_cache['data'] = catalog
        return copy.deepcopy(catalog)


def get_script_definition(
    script_id: str,
    get_scripts_catalog_fn: Callable[[], List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if not script_id:
        return None
    catalog = get_scripts_catalog_fn()
    for entry in catalog:
        if entry['id'] == script_id:
            return entry
    return None

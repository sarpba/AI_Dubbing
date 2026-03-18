from __future__ import annotations

import copy
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


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


def mask_workflow_secret_params(
    steps: Optional[List[Dict[str, Any]]],
    *,
    secret_param_names: Set[str],
    encoded_secret_prefix: str,
    encode_keyholder_value: Callable[[str], Optional[str]],
) -> List[Dict[str, Any]]:
    masked_steps: List[Dict[str, Any]] = copy.deepcopy(steps or [])
    for step in masked_steps:
        params = step.get('params')
        if not isinstance(params, dict):
            continue
        for key, value in list(params.items()):
            if key not in secret_param_names:
                continue
            if isinstance(value, str):
                if value.startswith(encoded_secret_prefix):
                    continue
                encoded_value = encode_keyholder_value(value)
                if encoded_value:
                    params[key] = f"{encoded_secret_prefix}{encoded_value}"
                else:
                    params.pop(key, None)
            elif value is None:
                params.pop(key, None)
    return masked_steps


def unmask_secret_param_value(
    value: Any,
    *,
    encoded_secret_prefix: str,
    decode_keyholder_value: Callable[[str], Optional[str]],
) -> Any:
    if isinstance(value, str) and value.startswith(encoded_secret_prefix):
        decoded = decode_keyholder_value(value[len(encoded_secret_prefix):])
        if decoded is not None:
            return decoded
    return value


def mask_applied_params_for_ui(
    applied_params: Optional[List[Dict[str, Any]]],
    *,
    secret_param_names: Set[str],
    secret_value_placeholder: str,
) -> List[Dict[str, Any]]:
    masked: List[Dict[str, Any]] = []
    for param in applied_params or []:
        entry = param.copy()
        if entry.get('name') in secret_param_names:
            entry['value'] = secret_value_placeholder
        masked.append(entry)
    return masked


def mask_command_for_ui(
    command: Optional[List[str]],
    applied_params: Optional[List[Dict[str, Any]]],
    script_meta: Dict[str, Any],
    *,
    secret_param_names: Set[str],
    secret_value_placeholder: str,
) -> List[str]:
    masked = list(command or [])
    secret_flags: Set[str] = set()
    for param_meta in script_meta.get('parameters', []):
        if param_meta.get('name') in secret_param_names:
            for flag in param_meta.get('flags') or []:
                secret_flags.add(flag)
    for idx, token in enumerate(masked[:-1]):
        if token in secret_flags:
            masked[idx + 1] = secret_value_placeholder
    secret_values = {
        str(param.get('value'))
        for param in applied_params or []
        if param.get('name') in secret_param_names and param.get('value') is not None
    }
    for idx, token in enumerate(masked):
        if token in secret_values:
            masked[idx] = secret_value_placeholder
    return masked


def build_masked_command_and_params(
    command: Optional[List[str]],
    applied_params: Optional[List[Dict[str, Any]]],
    script_meta: Dict[str, Any],
    *,
    secret_param_names: Set[str],
    secret_value_placeholder: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    return (
        mask_command_for_ui(
            command,
            applied_params,
            script_meta,
            secret_param_names=secret_param_names,
            secret_value_placeholder=secret_value_placeholder,
        ),
        mask_applied_params_for_ui(
            applied_params,
            secret_param_names=secret_param_names,
            secret_value_placeholder=secret_value_placeholder,
        ),
    )


def determine_parameter_value(
    param_meta: Dict[str, Any],
    user_params: Dict[str, Any],
    script_meta: Dict[str, Any],
    context: Dict[str, Any],
    *,
    script_param_keyholder: Dict[str, Dict[str, Tuple[str, ...]]],
    get_keyholder_value: Callable[[Dict[str, Any], Tuple[str, ...]], Optional[str]],
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
        key_mapping = script_param_keyholder.get(script_meta['id'], {}).get(name)
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
    context: Dict[str, Any],
    *,
    get_conda_python: Callable[[str], Optional[str]],
    app_root_path: str,
    workflow_validation_error: type[Exception],
    script_param_keyholder: Dict[str, Dict[str, Tuple[str, ...]]],
    get_keyholder_value: Callable[[Dict[str, Any], Tuple[str, ...]], Optional[str]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    environment = script_meta.get('environment')
    python_exec = get_conda_python(environment)
    if not python_exec:
        raise workflow_validation_error(f"Nem található Python futtató a(z) '{environment}' környezethez.")

    script_path = Path(app_root_path) / 'scripts' / script_meta['script']
    if not script_path.is_file():
        raise workflow_validation_error(f"A szkript nem található: {script_path}")

    user_params = step_config.get('params') or {}
    applied_params: List[Dict[str, Any]] = []
    command = [python_exec, str(script_path)]

    for param_meta in script_meta['parameters']:
        value = determine_parameter_value(
            param_meta,
            user_params,
            script_meta,
            context,
            script_param_keyholder=script_param_keyholder,
            get_keyholder_value=get_keyholder_value,
        )
        if value is None and param_meta['required'] and param_meta['type'] != 'flag':
            raise workflow_validation_error(f"Hiányzó kötelező paraméter: {param_meta['name']} ({script_meta['script']})")
        fragment = build_argument_fragment(param_meta, value)
        if fragment:
            command.extend(fragment)
            applied_params.append({
                'name': param_meta['name'],
                'value': value,
                'type': param_meta['type'],
            })

    return command, applied_params


def normalize_workflow_steps(
    payload: Any,
    *,
    workflow_validation_error: type[Exception],
    allowed_workflow_widgets: Set[str],
    get_script_definition: Callable[[str], Optional[Dict[str, Any]]],
    normalize_cycle_widget_params: Callable[[Dict[str, Any]], Dict[str, int]],
    normalize_translated_split_loop_widget_params: Callable[[Dict[str, Any]], Dict[str, int]],
    secret_param_names: Set[str],
    unmask_secret_param_value_fn: Callable[[Any], Any],
) -> Tuple[List[Dict[str, Any]], Set[str], int]:
    if not isinstance(payload, list):
        raise workflow_validation_error("A workflow lépéseit listában kell megadni.")

    normalized_steps: List[Dict[str, Any]] = []
    required_keys: Set[str] = set()
    enabled_count = 0

    for index, step in enumerate(payload, start=1):
        if not isinstance(step, dict):
            raise workflow_validation_error(f"A(z) {index}. lépés formátuma hibás.")

        step_type = step.get('type')
        if step_type == 'widget' or (step.get('widget') and not step.get('script')):
            widget_id = step.get('widget')
            if not widget_id:
                raise workflow_validation_error(f"A(z) {index}. widget lépéshez hiányzik a widget azonosítója.")
            if widget_id not in allowed_workflow_widgets:
                logging.warning("Ismeretlen workflow widget: %s", widget_id)
            widget_params = step.get('params')
            if widget_params is None:
                widget_params = {}
            if not isinstance(widget_params, dict):
                raise workflow_validation_error(f"A(z) {index}. widget lépés paraméterei hibás formátumúak.")
            normalized_widget_step: Dict[str, Any] = {
                'type': 'widget',
                'widget': widget_id,
                'enabled': coerce_bool(step.get('enabled', True)),
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
            raise workflow_validation_error(f"Ismeretlen szkript: {script_id}")

        enabled = coerce_bool(step.get('enabled', True))
        halt_on_fail = coerce_bool(step.get('halt_on_fail', True))
        params = step.get('params') or {}
        if not isinstance(params, dict):
            raise workflow_validation_error(f"A(z) {script_meta['display_name']} paraméterei hibás formátumúak.")
        normalized_params: Dict[str, Any] = {}
        for key, value in params.items():
            current_value = value
            if isinstance(current_value, str):
                current_value = current_value.strip()
                if current_value == '':
                    current_value = None
            if key in secret_param_names and current_value is not None:
                normalized_params[key] = unmask_secret_param_value_fn(current_value)
            else:
                normalized_params[key] = current_value

        normalized_step = {
            'script': script_meta['id'],
            'enabled': enabled,
            'halt_on_fail': halt_on_fail,
            'params': normalized_params,
        }
        if enabled:
            enabled_count += 1
            required_keys.update(script_meta.get('required_keys', []))

        normalized_steps.append(normalized_step)

    if enabled_count == 0:
        raise workflow_validation_error("Legalább egy lépést engedélyezni kell a futtatáshoz.")

    return normalized_steps, required_keys, enabled_count


def run_script_command(
    command_list: List[str],
    *,
    log_prefix: str = "",
    env: Optional[Dict[str, str]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
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

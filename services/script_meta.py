from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ScriptMetaIssue:
    path: str
    level: str
    message: str


def _format_path(path: Path) -> str:
    return path.as_posix()


def _validate_param(param: Any, group_name: str, path: Path) -> List[ScriptMetaIssue]:
    issues: List[ScriptMetaIssue] = []
    if not isinstance(param, dict):
        issues.append(ScriptMetaIssue(_format_path(path), 'error', f'A(z) {group_name} paraméter nem objektum.'))
        return issues

    name = param.get('name')
    if not isinstance(name, str) or not name.strip():
        issues.append(ScriptMetaIssue(_format_path(path), 'error', f'Hiányzó vagy üres paraméternév a(z) {group_name} listában.'))

    param_type = param.get('type', 'option')
    if param_type not in {'option', 'flag', 'positional', 'config_option'}:
        issues.append(ScriptMetaIssue(_format_path(path), 'warning', f'Ismeretlen paramétertípus: {param_type}'))

    if 'default' not in param:
        issues.append(ScriptMetaIssue(_format_path(path), 'warning', f'A(z) {name or "ismeretlen"} paraméteren nincs explicit default mező.'))

    flags = param.get('flags')
    if flags is not None and not isinstance(flags, list):
        issues.append(ScriptMetaIssue(_format_path(path), 'error', f'A(z) {name or "ismeretlen"} paraméter flags mezője nem lista.'))

    if param_type == 'flag' and isinstance(flags, list) and not flags:
        issues.append(ScriptMetaIssue(_format_path(path), 'warning', f'A(z) {name or "ismeretlen"} flag paraméterhez nincs kapcsoló megadva.'))

    return issues


def validate_script_meta(path: Path, data: Any, scripts_dir: Path) -> List[ScriptMetaIssue]:
    issues: List[ScriptMetaIssue] = []
    if not isinstance(data, dict):
        return [ScriptMetaIssue(_format_path(path), 'error', 'A script meta JSON gyökéreleme nem objektum.')]

    rel_path = path.relative_to(scripts_dir)
    expected_script = rel_path.with_suffix('.py').as_posix()
    script_value = data.get('script')
    if script_value != expected_script:
        issues.append(
            ScriptMetaIssue(
                _format_path(path),
                'warning',
                f'A script mező eltér a várt relatív útvonaltól. Várt: {expected_script}, aktuális: {script_value!r}',
            )
        )

    environment = data.get('environment')
    if environment is None:
        if 'enviroment' in data:
            issues.append(ScriptMetaIssue(_format_path(path), 'warning', 'Az enviroment kulcs elavult, environment kulcsot használj helyette.'))
        else:
            issues.append(ScriptMetaIssue(_format_path(path), 'warning', 'Hiányzik az environment kulcs.'))

    help_path = path.with_name(f'{path.stem}_help.md')
    if not help_path.is_file():
        issues.append(ScriptMetaIssue(_format_path(path), 'warning', f'Hiányzó help fájl: {help_path.name}'))

    for group_name in ('required', 'optional'):
        group = data.get(group_name)
        if group is None:
            continue
        if not isinstance(group, list):
            issues.append(ScriptMetaIssue(_format_path(path), 'error', f'A(z) {group_name} mező nem lista.'))
            continue
        for param in group:
            issues.extend(_validate_param(param, group_name, path))

    return issues


def load_script_meta(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as fp:
        return json.load(fp)


def collect_script_meta_issues(scripts_dir: Path) -> List[ScriptMetaIssue]:
    issues: List[ScriptMetaIssue] = []
    for json_path in sorted(scripts_dir.rglob('*.json')):
        if json_path.name == 'scripts.json':
            continue
        py_candidate = scripts_dir / json_path.relative_to(scripts_dir).with_suffix('.py')
        if not py_candidate.is_file():
            continue
        try:
            data = load_script_meta(json_path)
        except json.JSONDecodeError as exc:
            issues.append(ScriptMetaIssue(_format_path(json_path), 'error', f'Hibás JSON: {exc}'))
            continue
        except OSError as exc:
            issues.append(ScriptMetaIssue(_format_path(json_path), 'error', f'Nem olvasható fájl: {exc}'))
            continue
        issues.extend(validate_script_meta(json_path, data, scripts_dir))
    return issues


def has_errors(issues: List[ScriptMetaIssue]) -> bool:
    return any(issue.level == 'error' for issue in issues)

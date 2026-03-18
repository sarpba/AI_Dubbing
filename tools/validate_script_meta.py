from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.script_meta import collect_script_meta_issues, has_errors


def main() -> int:
    scripts_dir = PROJECT_ROOT / 'scripts'
    issues = collect_script_meta_issues(scripts_dir)

    if not issues:
        print('A script meta fájlok rendben vannak.')
        return 0

    for issue in issues:
        print(f'[{issue.level.upper()}] {issue.path}: {issue.message}')

    return 1 if has_errors(issues) else 0


if __name__ == '__main__':
    sys.exit(main())

from __future__ import annotations

import unittest
from pathlib import Path

from services.script_meta import collect_script_meta_issues


class ScriptMetaValidationTests(unittest.TestCase):
    def test_script_meta_has_no_validation_errors(self) -> None:
        scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
        issues = collect_script_meta_issues(scripts_dir)
        errors = [issue for issue in issues if issue.level == 'error']
        self.assertEqual([], errors, f'Script meta validation errors: {errors}')


if __name__ == '__main__':
    unittest.main()

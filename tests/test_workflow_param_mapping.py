from __future__ import annotations

import sys
import types
import unittest

if 'pydub' not in sys.modules:
    pydub_stub = types.ModuleType('pydub')
    pydub_stub.AudioSegment = object
    sys.modules['pydub'] = pydub_stub

from main_app import build_argument_fragment, prepare_script_entry


class WorkflowParamMappingTests(unittest.TestCase):
    def test_negative_only_flag_is_not_emitted_for_default_enabled_state(self) -> None:
        script_meta = prepare_script_entry(
            {
                'script': 'demo/example.py',
                'environment': 'sync',
                'optional': [
                    {
                        'name': 'sync_loudness',
                        'type': 'flag',
                        'flags': ['--no-sync-loudness'],
                        'default': True,
                    }
                ],
            }
        )
        param_meta = script_meta['parameters'][0]
        self.assertEqual([], build_argument_fragment(param_meta, True))
        self.assertEqual(['--no-sync-loudness'], build_argument_fragment(param_meta, False))

    def test_negative_name_flag_is_humanized_and_emitted_only_when_enabled(self) -> None:
        script_meta = prepare_script_entry(
            {
                'script': 'demo/example.py',
                'environment': 'sync',
                'optional': [
                    {
                        'name': 'no_backup',
                        'type': 'flag',
                        'flags': ['--no-backup'],
                        'default': False,
                    }
                ],
            }
        )
        param_meta = script_meta['parameters'][0]
        self.assertEqual('backup', param_meta['ui_name'])
        self.assertEqual([], build_argument_fragment(param_meta, False))
        self.assertEqual(['--no-backup'], build_argument_fragment(param_meta, True))


if __name__ == '__main__':
    unittest.main()

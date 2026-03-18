from __future__ import annotations

import json
import shutil
import sys
import types
import unittest
from pathlib import Path

if 'pydub' not in sys.modules:
    pydub_stub = types.ModuleType('pydub')
    pydub_stub.AudioSegment = object
    sys.modules['pydub'] = pydub_stub

from main_app import app, config


class RouteSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_name = 'test_smoke_project'
        cls.workdir = Path(config['DIRECTORIES']['workdir'])
        cls.project_dir = cls.workdir / cls.project_name
        cls.project_dir.mkdir(parents=True, exist_ok=True)
        for subdir in config['PROJECT_SUBDIRS'].values():
            (cls.project_dir / subdir).mkdir(parents=True, exist_ok=True)
        translated_dir = cls.project_dir / config['PROJECT_SUBDIRS']['translated']
        cls.fixture_json_name = 'segments_fixture.json'
        with (translated_dir / cls.fixture_json_name).open('w', encoding='utf-8') as fp:
            json.dump(
                {
                    'segments': [
                        {'start': 0.0, 'end': 1.0, 'text': 'Teszt szegmens'},
                    ]
                },
                fp,
                ensure_ascii=False,
                indent=2,
            )
        cls.workflow_state_path = cls.project_dir / 'workflow_state.json'
        with cls.workflow_state_path.open('w', encoding='utf-8') as fp:
            json.dump(
                {
                    'steps': [],
                    'template_id': 'default',
                },
                fp,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.project_dir.exists():
            shutil.rmtree(cls.project_dir)

    def setUp(self) -> None:
        app.testing = True
        self.client = app.test_client()

    def test_index_route(self) -> None:
        response = self.client.get('/')
        self.assertEqual(200, response.status_code)

    def test_template_editor_route(self) -> None:
        response = self.client.get('/template-editor')
        self.assertEqual(200, response.status_code)

    def test_project_route(self) -> None:
        response = self.client.get(f'/project/{self.project_name}')
        self.assertEqual(200, response.status_code)

    def test_review_route(self) -> None:
        response = self.client.get(f'/review/{self.project_name}')
        self.assertEqual(200, response.status_code)

    def test_workflow_templates_api(self) -> None:
        response = self.client.get('/api/workflow-templates')
        self.assertEqual(200, response.status_code)

    def test_project_tree_api(self) -> None:
        response = self.client.get(f'/api/project-tree/{self.project_name}')
        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertIn('entries', payload)

    def test_workflow_options_api(self) -> None:
        response = self.client.get(f'/api/workflow-options/{self.project_name}')
        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual(self.project_name, payload['project'])

    def test_project_workflow_state_api(self) -> None:
        response = self.client.get(f'/api/project-workflow-state/{self.project_name}')
        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertIn('state', payload)

    def test_get_segments_api(self) -> None:
        response = self.client.post(
            f'/api/get-segments/{self.project_name}',
            json={'json_file_name': self.fixture_json_name},
        )
        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual(1, len(payload['segments']))


if __name__ == '__main__':
    unittest.main()

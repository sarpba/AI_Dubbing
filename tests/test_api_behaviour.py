from __future__ import annotations

import io
import json
import shutil
import sys
import threading
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

if 'pydub' not in sys.modules:
    pydub_stub = types.ModuleType('pydub')
    pydub_stub.AudioSegment = object
    sys.modules['pydub'] = pydub_stub

from main_app import app, config, review_audio_encoding_jobs, workflow_events, workflow_jobs, workflow_threads


class _FakeTrimmedSegment:
    def __init__(self, duration_ms: int) -> None:
        self.duration_ms = duration_ms

    def __len__(self) -> int:
        return self.duration_ms

    def export(self, path: str, format: str | None = None) -> None:
        Path(path).write_bytes(b'fake-audio')


class _FakeAudioSegment:
    def __init__(self, duration_ms: int = 3000) -> None:
        self.duration_ms = duration_ms

    def __len__(self) -> int:
        return self.duration_ms

    def __getitem__(self, key):
        start = 0 if key.start is None else key.start
        end = self.duration_ms if key.stop is None else key.stop
        return _FakeTrimmedSegment(max(end - start, 0))


class ApiBehaviourTests(unittest.TestCase):
    VALID_WORKFLOW_STEP = {
        'script': 'AUDIO-VIDEO/unpack_srt_from_mkv/unpack_srt_from_mkv_easy.py',
        'enabled': True,
        'halt_on_fail': True,
        'params': {},
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.project_name = 'test_api_project'
        cls.workdir = Path(config['DIRECTORIES']['workdir'])
        cls.project_dir = cls.workdir / cls.project_name
        cls.project_dir.mkdir(parents=True, exist_ok=True)
        for subdir in config['PROJECT_SUBDIRS'].values():
            (cls.project_dir / subdir).mkdir(parents=True, exist_ok=True)

        cls.translated_dir = cls.project_dir / config['PROJECT_SUBDIRS']['translated']
        cls.speech_dir = cls.project_dir / config['PROJECT_SUBDIRS']['separated_audio_speech']
        cls.upload_dir = cls.project_dir / config['PROJECT_SUBDIRS']['upload']
        cls.logs_dir = cls.project_dir / config['PROJECT_SUBDIRS']['logs']

        cls.fixture_json_name = 'segments.json'
        cls.speech_only_json_name = 'speech_only.json'
        cls.no_audio_json_name = 'no_audio.json'
        cls.audio_name = 'source.wav'
        cls.audio_path = cls.upload_dir / cls.audio_name
        cls.audio_path.write_bytes(b'original-audio')
        cls.regenerate_audio_name = 'segments.wav'

        cls.template_id = 'test_api_template'
        cls.template_path = Path('workflows') / f'{cls.template_id}.json'

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.project_dir.exists():
            shutil.rmtree(cls.project_dir)
        if cls.template_path.exists():
            cls.template_path.unlink()

    def setUp(self) -> None:
        app.testing = True
        self.client = app.test_client()
        self.project_dir.mkdir(parents=True, exist_ok=True)
        for subdir in config['PROJECT_SUBDIRS'].values():
            (self.project_dir / subdir).mkdir(parents=True, exist_ok=True)
        self._write_segments_fixture(
            self.translated_dir / self.fixture_json_name,
            [
                {'start': 0.0, 'end': 1.0, 'text': 'Első', 'translated_text': 'First'},
                {'start': 1.5, 'end': 2.5, 'text': 'Második', 'translated_text': 'Second'},
            ],
        )
        self._write_segments_fixture(
            self.speech_dir / self.speech_only_json_name,
            [{'start': 0.0, 'end': 1.0, 'text': 'Speech only'}],
        )
        self._write_segments_fixture(
            self.translated_dir / self.no_audio_json_name,
            [{'start': 0.0, 'end': 1.0, 'text': 'No audio'}],
        )
        (self.speech_dir / self.regenerate_audio_name).write_bytes(b'fake-wav')
        workflow_state_path = self.project_dir / 'workflow_state.json'
        workflow_state_path.write_text(
            json.dumps({'steps': [], 'template_id': 'default'}, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        if self.template_path.exists():
            self.template_path.unlink()
        for extra_project in ('restored_api_project', 'uploaded_video_project'):
            extra_dir = self.workdir / extra_project
            if extra_dir.exists():
                shutil.rmtree(extra_dir, ignore_errors=True)
        trimmed_path = self.upload_dir / 'trimmed.wav'
        if trimmed_path.exists():
            trimmed_path.unlink()
        temp_dir = self.project_dir / config['PROJECT_SUBDIRS']['temp']
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        workflow_jobs.clear()
        workflow_threads.clear()
        workflow_events.clear()
        review_audio_encoding_jobs.clear()

    def _write_segments_fixture(self, path: Path, segments) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({'segments': segments}, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _read_segments(self, path: Path):
        return json.loads(path.read_text(encoding='utf-8'))['segments']

    def test_update_segment_updates_json_content(self) -> None:
        response = self.client.post(
            f'/api/update-segment/{self.project_name}',
            json={
                'json_file_name': self.fixture_json_name,
                'segment_index': 0,
                'new_start': 0.1,
                'new_end': 1.1,
                'new_text': 'Frissitett',
                'new_translated_text': 'Updated',
            },
        )
        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])

        segments = self._read_segments(self.translated_dir / self.fixture_json_name)
        self.assertEqual(0.1, segments[0]['start'])
        self.assertEqual(1.1, segments[0]['end'])
        self.assertEqual('Frissitett', segments[0]['text'])
        self.assertEqual('Updated', segments[0]['translated_text'])

    def test_update_segment_rejects_overlap(self) -> None:
        response = self.client.post(
            f'/api/update-segment/{self.project_name}',
            json={
                'json_file_name': self.fixture_json_name,
                'segment_index': 1,
                'new_start': 0.5,
                'new_end': 2.0,
            },
        )
        self.assertEqual(400, response.status_code)
        self.assertIn('overlaps', response.get_json()['error'])

    def test_add_segment_inserts_sorted_segment(self) -> None:
        response = self.client.post(
            f'/api/add-segment/{self.project_name}',
            json={
                'json_file_name': self.fixture_json_name,
                'start': 1.1,
                'end': 1.4,
                'text': 'Koztes',
            },
        )
        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual([0.0, 1.1, 1.5], [segment['start'] for segment in payload['segments']])

    def test_add_segment_rejects_overlap(self) -> None:
        response = self.client.post(
            f'/api/add-segment/{self.project_name}',
            json={
                'json_file_name': self.fixture_json_name,
                'start': 0.8,
                'end': 1.2,
                'text': 'Atfedo',
            },
        )

        self.assertEqual(400, response.status_code)
        self.assertIn('overlaps', response.get_json()['error'])

    def test_delete_segment_removes_entry(self) -> None:
        response = self.client.post(
            f'/api/delete-segment/{self.project_name}',
            json={
                'json_file_name': self.fixture_json_name,
                'segment_index': 0,
            },
        )
        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual(1, len(payload['segments']))
        segments = self._read_segments(self.translated_dir / self.fixture_json_name)
        self.assertEqual(1, len(segments))
        self.assertEqual('Második', segments[0]['text'])

    def test_delete_segment_rejects_invalid_index(self) -> None:
        response = self.client.post(
            f'/api/delete-segment/{self.project_name}',
            json={
                'json_file_name': self.fixture_json_name,
                'segment_index': 99,
            },
        )

        self.assertEqual(400, response.status_code)
        self.assertIn('Invalid segment index', response.get_json()['error'])

    def test_get_segments_uses_speech_directory_fallback(self) -> None:
        response = self.client.post(
            f'/api/get-segments/{self.project_name}',
            json={'json_file_name': self.speech_only_json_name},
        )
        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual(1, len(payload['segments']))
        self.assertEqual('Speech only', payload['segments'][0]['text'])

    def test_review_audio_status_rejects_missing_audio_file_param(self) -> None:
        response = self.client.get(f'/api/review-audio-status/{self.project_name}')

        self.assertEqual(400, response.status_code)
        self.assertEqual('missing_audio_file', response.get_json()['error'])

    def test_review_audio_status_returns_available_for_existing_encoded_audio(self) -> None:
        encoded_path = self.project_dir / config['PROJECT_SUBDIRS']['temp'] / 'segments_review_preview.mp3'
        encoded_path.parent.mkdir(parents=True, exist_ok=True)
        encoded_path.write_bytes(b'encoded-audio')

        response = self.client.get(
            f'/api/review-audio-status/{self.project_name}?audio_file={self.regenerate_audio_name}'
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual('available', payload['status'])
        self.assertEqual(100.0, payload['progress'])
        self.assertIn('segments_review_preview.mp3', payload['audio_url'])

    def test_review_audio_status_starts_encoding_job_when_preview_missing(self) -> None:
        fake_thread = mock.Mock()
        with mock.patch('routes.review_api.threading.Thread', return_value=fake_thread):
            response = self.client.get(
                f'/api/review-audio-status/{self.project_name}?audio_file={self.regenerate_audio_name}'
            )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual('encoding', payload['status'])
        self.assertEqual(0.0, payload['progress'])
        self.assertIn(self.project_name, review_audio_encoding_jobs)
        fake_thread.start.assert_called_once()

    def test_review_audio_status_returns_not_found_for_missing_source_audio(self) -> None:
        response = self.client.get(
            f'/api/review-audio-status/{self.project_name}?audio_file=missing.wav'
        )

        self.assertEqual(404, response.status_code)
        self.assertEqual('audio_not_found', response.get_json()['error'])

    def test_project_file_upload_and_delete(self) -> None:
        upload_response = self.client.post(
            '/api/project-file/upload',
            data={
                'projectName': self.project_name,
                'targetPath': config['PROJECT_SUBDIRS']['upload'],
                'file': (io.BytesIO(b'hello world'), 'uploaded.txt'),
            },
            content_type='multipart/form-data',
        )
        self.assertEqual(200, upload_response.status_code)
        payload = upload_response.get_json()
        self.assertTrue(payload['success'])
        uploaded_path = payload['path']
        self.assertTrue((self.project_dir / uploaded_path).exists())

        delete_response = self.client.delete(
            f'/api/project-file/{self.project_name}',
            json={'path': uploaded_path},
        )
        self.assertEqual(200, delete_response.status_code)
        self.assertFalse((self.project_dir / uploaded_path).exists())

    def test_project_file_upload_rejects_invalid_target_path(self) -> None:
        response = self.client.post(
            '/api/project-file/upload',
            data={
                'projectName': self.project_name,
                'targetPath': '../../outside',
                'file': (io.BytesIO(b'blocked'), 'blocked.txt'),
            },
            content_type='multipart/form-data',
        )

        self.assertEqual(400, response.status_code)
        self.assertIn('Érvénytelen célkönyvtár', response.get_json()['error'])

    def test_project_audio_trim_creates_output_file(self) -> None:
        with mock.patch('routes.files_api.AudioSegment') as audio_segment_cls:
            audio_segment_cls.from_file.return_value = _FakeAudioSegment(3000)
            response = self.client.post(
                '/api/project-audio/trim',
                json={
                    'projectName': self.project_name,
                    'filePath': f"{config['PROJECT_SUBDIRS']['upload']}/{self.audio_name}",
                    'outputName': 'trimmed.wav',
                    'start': 0.5,
                    'end': 1.5,
                },
            )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual('trimmed.wav', payload['saved_name'])
        self.assertTrue((self.upload_dir / 'trimmed.wav').exists())

    def test_project_workflow_state_post_roundtrip(self) -> None:
        new_state = {
            'steps': [self.VALID_WORKFLOW_STEP],
            'template_id': 'custom-template',
            'saved_at': '2026-03-18T10:00:00',
        }
        post_response = self.client.post(
            f'/api/project-workflow-state/{self.project_name}',
            json=new_state,
        )
        self.assertEqual(200, post_response.status_code)
        self.assertTrue(post_response.get_json()['success'])

        get_response = self.client.get(f'/api/project-workflow-state/{self.project_name}')
        self.assertEqual(200, get_response.status_code)
        returned_state = get_response.get_json()['state']
        self.assertEqual('custom-template', returned_state['template_id'])
        self.assertEqual('2026-03-18T10:00:00', returned_state['saved_at'])

    def test_run_workflow_rejects_invalid_steps_payload(self) -> None:
        response = self.client.post(
            f'/api/run-workflow/{self.project_name}',
            json={'steps': 'invalid-steps'},
        )
        self.assertEqual(400, response.status_code)
        self.assertFalse(response.get_json()['success'])

    def test_run_workflow_registers_job_and_status_endpoint_returns_it(self) -> None:
        fake_thread = mock.Mock()
        with mock.patch('routes.workflow_api.threading.Thread', return_value=fake_thread):
            response = self.client.post(
                f'/api/run-workflow/{self.project_name}',
                json={
                    'steps': [self.VALID_WORKFLOW_STEP],
                    'template_id': 'run-template',
                },
            )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        job_id = payload['job_id']
        self.assertIn(job_id, workflow_jobs)
        self.assertEqual('queued', workflow_jobs[job_id]['status'])
        self.assertEqual('run-template', workflow_jobs[job_id]['template_id'])
        fake_thread.start.assert_called_once()

        status_response = self.client.get(f'/api/workflow-status/{job_id}')
        self.assertEqual(200, status_response.status_code)
        status_payload = status_response.get_json()
        self.assertTrue(status_payload['success'])
        self.assertEqual(job_id, status_payload['job']['job_id'])
        self.assertEqual(self.project_name, status_payload['job']['project'])
        self.assertEqual('queued', status_payload['job']['status'])

    def test_stop_workflow_sets_cancel_event_and_updates_job_state(self) -> None:
        job_id = 'cancel-me'
        workflow_jobs[job_id] = {
            'job_id': job_id,
            'project': self.project_name,
            'status': 'running',
            'cancel_requested': False,
        }
        workflow_events[job_id] = threading.Event()

        response = self.client.post(f'/api/stop-workflow/{job_id}')

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertTrue(workflow_events[job_id].is_set())
        self.assertEqual('cancelling', workflow_jobs[job_id]['status'])
        self.assertTrue(workflow_jobs[job_id]['cancel_requested'])

    def test_stop_workflow_rejects_already_finished_job(self) -> None:
        job_id = 'finished-job'
        workflow_jobs[job_id] = {
            'job_id': job_id,
            'project': self.project_name,
            'status': 'completed',
            'cancel_requested': False,
        }

        response = self.client.post(f'/api/stop-workflow/{job_id}')

        self.assertEqual(400, response.status_code)
        self.assertIn('befejez', response.get_json()['error'])

    def test_save_workflow_template_creates_file(self) -> None:
        response = self.client.post(
            '/api/save-workflow-template',
            json={
                'template_id': self.template_id,
                'name': 'API Test Template',
                'description': 'created by tests',
                'steps': [self.VALID_WORKFLOW_STEP],
                'overwrite': True,
            },
        )
        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertTrue(self.template_path.exists())

    def test_workflow_log_and_stop_return_404_for_missing_job(self) -> None:
        missing_job_id = 'missing-job-id'
        log_response = self.client.get(f'/api/workflow-log/{missing_job_id}')
        self.assertEqual(404, log_response.status_code)
        stop_response = self.client.post(f'/api/stop-workflow/{missing_job_id}')
        self.assertEqual(404, stop_response.status_code)

    def test_workflow_log_returns_tail_for_existing_completed_job(self) -> None:
        job_id = 'logged-job'
        log_name = 'workflow.log'
        log_path = self.logs_dir / log_name
        log_path.write_text('elso sor\nmasodik sor\nharmadik sor\n', encoding='utf-8')
        workflow_jobs[job_id] = {
            'job_id': job_id,
            'project': self.project_name,
            'status': 'completed',
            'message': 'Workflow sikeresen lefutott.',
            'cancel_requested': False,
            'log': {
                'relative': f'{self.project_name}/{config["PROJECT_SUBDIRS"]["logs"]}/{log_name}',
                'url': f'/workdir/{self.project_name}/{config["PROJECT_SUBDIRS"]["logs"]}/{log_name}',
            },
        }

        response = self.client.get(f'/api/workflow-log/{job_id}')

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertTrue(payload['log_available'])
        self.assertTrue(payload['completed'])
        self.assertEqual('completed', payload['status'])
        self.assertIn('harmadik sor', payload['log'])

    def test_workflow_log_handles_missing_log_file_for_existing_job(self) -> None:
        job_id = 'missing-log-job'
        workflow_jobs[job_id] = {
            'job_id': job_id,
            'project': self.project_name,
            'status': 'running',
            'message': 'Fut',
            'cancel_requested': False,
            'log': {
                'relative': f'{self.project_name}/{config["PROJECT_SUBDIRS"]["logs"]}/does-not-exist.log',
                'url': f'/workdir/{self.project_name}/{config["PROJECT_SUBDIRS"]["logs"]}/does-not-exist.log',
            },
        }

        response = self.client.get(f'/api/workflow-log/{job_id}')

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertFalse(payload['log_available'])
        self.assertFalse(payload['completed'])
        self.assertEqual('', payload['log'])

    def test_regenerate_segment_starts_job_and_creates_temp_json(self) -> None:
        (self.project_dir / 'workflow_state.json').write_text(
            json.dumps(
                {
                    'steps': [
                        {
                            'script': 'TTS/f5-tts-narrator/f5_tts_narrator.py',
                            'enabled': True,
                            'halt_on_fail': True,
                            'params': {},
                        }
                    ],
                    'template_id': 'regen-template',
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )

        fake_thread = mock.Mock()
        with mock.patch('routes.review_api.threading.Thread', return_value=fake_thread):
            response = self.client.post(
                f'/api/regenerate-segment/{self.project_name}',
                json={
                    'json_file_name': self.fixture_json_name,
                    'segment_index': 0,
                },
            )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertIn('job_id', payload)
        self.assertIn(payload['job_id'], workflow_jobs)
        fake_thread.start.assert_called_once()

        temp_json_path = self.project_dir / payload['temp_json_path']
        self.assertTrue(temp_json_path.exists())
        regenerate_payload = json.loads(temp_json_path.read_text(encoding='utf-8'))
        self.assertEqual(0, regenerate_payload['segment_index'])
        self.assertEqual(self.fixture_json_name, regenerate_payload['source_json'])
        self.assertEqual(1, len(regenerate_payload['segments']))

    def test_regenerate_segment_rejects_missing_reference_audio(self) -> None:
        (self.project_dir / 'workflow_state.json').write_text(
            json.dumps(
                {
                    'steps': [
                        {
                            'script': 'TTS/f5-tts-narrator/f5_tts_narrator.py',
                            'enabled': True,
                            'halt_on_fail': True,
                            'params': {},
                        }
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )

        response = self.client.post(
            f'/api/regenerate-segment/{self.project_name}',
            json={
                'json_file_name': self.no_audio_json_name,
                'segment_index': 0,
            },
        )

        self.assertEqual(404, response.status_code)
        self.assertIn('referencia WAV', response.get_json()['error'])

    def test_regenerate_segment_rejects_workflow_without_tts(self) -> None:
        (self.project_dir / 'workflow_state.json').write_text(
            json.dumps(
                {
                    'steps': [self.VALID_WORKFLOW_STEP],
                    'template_id': 'no-tts-template',
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )

        response = self.client.post(
            f'/api/regenerate-segment/{self.project_name}',
            json={
                'json_file_name': self.fixture_json_name,
                'segment_index': 0,
            },
        )

        self.assertEqual(400, response.status_code)
        self.assertIn('nem található TTS', response.get_json()['error'])

    def test_restore_project_from_zip_backup(self) -> None:
        restored_project = 'restored_api_project'
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('payload/readme.txt', 'restored-content')
            archive.writestr('payload/nested/data.json', '{"ok": true}')
        archive_buffer.seek(0)

        response = self.client.post(
            '/api/restore-project',
            data={
                'projectName': restored_project,
                'backupFile': (archive_buffer, 'backup.zip'),
            },
            content_type='multipart/form-data',
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        restored_dir = self.workdir / restored_project
        self.assertTrue((restored_dir / 'readme.txt').exists())
        self.assertEqual('restored-content', (restored_dir / 'readme.txt').read_text(encoding='utf-8'))
        self.assertTrue((restored_dir / 'nested' / 'data.json').exists())

    def test_restore_project_rejects_unsupported_archive(self) -> None:
        restored_project = 'restored_api_project'
        response = self.client.post(
            '/api/restore-project',
            data={
                'projectName': restored_project,
                'backupFile': (io.BytesIO(b'not-an-archive'), 'broken.bin'),
            },
            content_type='multipart/form-data',
        )

        self.assertEqual(400, response.status_code)
        self.assertIn('nem támogatott archívum', response.get_json()['error'])

    def test_restore_project_supports_chunked_zip_upload(self) -> None:
        restored_project = 'restored_api_project'
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('payload/readme.txt', 'chunk-restored')
        archive_bytes = archive_buffer.getvalue()
        midpoint = len(archive_bytes) // 2
        chunks = [archive_bytes[:midpoint], archive_bytes[midpoint:]]

        first_response = self.client.post(
            '/api/restore-project',
            data={
                'projectName': restored_project,
                'chunkUploadId': 'restore-chunk-test',
                'chunkIndex': '0',
                'totalChunks': '2',
                'fileName': 'backup.zip',
                'backupFile': (io.BytesIO(chunks[0]), 'backup.zip'),
            },
            content_type='multipart/form-data',
        )

        self.assertEqual(200, first_response.status_code)
        first_payload = first_response.get_json()
        self.assertTrue(first_payload['success'])
        self.assertFalse(first_payload['completed'])

        final_response = self.client.post(
            '/api/restore-project',
            data={
                'projectName': restored_project,
                'chunkUploadId': 'restore-chunk-test',
                'chunkIndex': '1',
                'totalChunks': '2',
                'fileName': 'backup.zip',
                'backupFile': (io.BytesIO(chunks[1]), 'backup.zip'),
            },
            content_type='multipart/form-data',
        )

        self.assertEqual(200, final_response.status_code)
        final_payload = final_response.get_json()
        self.assertTrue(final_payload['success'])
        restored_dir = self.workdir / restored_project
        self.assertEqual('chunk-restored', (restored_dir / 'readme.txt').read_text(encoding='utf-8'))

    def test_upload_video_with_subtitle_file(self) -> None:
        uploaded_project = 'uploaded_video_project'
        response = self.client.post(
            '/api/upload-video',
            data={
                'projectName': uploaded_project,
                'subtitleSuffix': '_hu',
                'file': (io.BytesIO(b'fake-video-bytes'), 'demo_video.mkv'),
                'subtitleFile': (io.BytesIO(b'1\n00:00:00,000 --> 00:00:01,000\nSzia\n'), 'subtitle.srt'),
            },
            content_type='multipart/form-data',
        )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual(uploaded_project, payload['project'])
        self.assertEqual('demo_video.mkv', payload['video'])
        self.assertEqual('demo_video_hu.srt', payload['subtitle'])

        upload_dir = self.workdir / uploaded_project / config['PROJECT_SUBDIRS']['upload']
        self.assertTrue((upload_dir / 'demo_video.mkv').exists())
        self.assertTrue((upload_dir / 'demo_video_hu.srt').exists())

    def test_upload_video_youtube_rejects_when_ytdlp_missing(self) -> None:
        with mock.patch('routes.files_api.shutil.which', return_value=None):
            response = self.client.post(
                '/api/upload-video',
                data={
                    'projectName': 'uploaded_video_project',
                    'youtubeUrl': 'https://youtube.com/watch?v=test',
                },
                content_type='multipart/form-data',
            )

        self.assertEqual(500, response.status_code)
        self.assertIn('yt-dlp nem érhető el', response.get_json()['error'])

    def test_upload_video_youtube_returns_error_when_download_fails(self) -> None:
        failed_process = mock.Mock(returncode=1, stderr='download failed')
        with mock.patch('routes.files_api.shutil.which', return_value='/usr/bin/yt-dlp'):
            with mock.patch('routes.files_api.subprocess.run', return_value=failed_process):
                response = self.client.post(
                    '/api/upload-video',
                    data={
                        'projectName': 'uploaded_video_project',
                        'youtubeUrl': 'https://youtube.com/watch?v=test',
                    },
                    content_type='multipart/form-data',
                )

        self.assertEqual(500, response.status_code)
        payload = response.get_json()
        self.assertIn('Nem sikerült letölteni a YouTube videót', payload['error'])
        self.assertEqual('download failed', payload['details'])

    def test_upload_video_youtube_succeeds_when_download_creates_expected_file(self) -> None:
        uploaded_project = 'uploaded_video_project'
        upload_dir = self.workdir / uploaded_project / config['PROJECT_SUBDIRS']['upload']

        def fake_run(*args, **kwargs):
            upload_dir.mkdir(parents=True, exist_ok=True)
            (upload_dir / f'{uploaded_project}.mkv').write_bytes(b'youtube-video')
            return mock.Mock(returncode=0, stderr='')

        with mock.patch('routes.files_api.shutil.which', return_value='/usr/bin/yt-dlp'):
            with mock.patch('routes.files_api.subprocess.run', side_effect=fake_run):
                response = self.client.post(
                    '/api/upload-video',
                    data={
                        'projectName': uploaded_project,
                        'youtubeUrl': 'https://youtube.com/watch?v=test',
                    },
                    content_type='multipart/form-data',
                )

        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual(uploaded_project, payload['project'])
        self.assertEqual(f'{uploaded_project}.mkv', payload['video'])
        self.assertTrue((upload_dir / f'{uploaded_project}.mkv').exists())

    def test_upload_video_rejects_file_and_youtube_url_together(self) -> None:
        response = self.client.post(
            '/api/upload-video',
            data={
                'projectName': 'uploaded_video_project',
                'youtubeUrl': 'https://youtube.com/watch?v=test',
                'file': (io.BytesIO(b'fake-video-bytes'), 'demo_video.mkv'),
            },
            content_type='multipart/form-data',
        )

        self.assertEqual(400, response.status_code)
        self.assertIn('fájlt töltesz fel, vagy YouTube videót', response.get_json()['error'])

    def test_upload_video_rejects_missing_video_and_url(self) -> None:
        response = self.client.post(
            '/api/upload-video',
            data={'projectName': 'uploaded_video_project'},
            content_type='multipart/form-data',
        )

        self.assertEqual(400, response.status_code)
        self.assertIn('Adj meg videó fájlt vagy YouTube hivatkozást', response.get_json()['error'])

    def test_upload_video_rejects_invalid_subtitle_suffix(self) -> None:
        response = self.client.post(
            '/api/upload-video',
            data={
                'projectName': 'uploaded_video_project',
                'subtitleSuffix': 'hu',
                'file': (io.BytesIO(b'fake-video-bytes'), 'demo_video.mkv'),
                'subtitleFile': (io.BytesIO(b'1\n00:00:00,000 --> 00:00:01,000\nSzia\n'), 'subtitle.srt'),
            },
            content_type='multipart/form-data',
        )

        self.assertEqual(400, response.status_code)
        self.assertIn('aláhúzás + kétbetűs', response.get_json()['error'])

    def test_upload_video_rejects_non_srt_subtitle(self) -> None:
        response = self.client.post(
            '/api/upload-video',
            data={
                'projectName': 'uploaded_video_project',
                'subtitleSuffix': '_hu',
                'file': (io.BytesIO(b'fake-video-bytes'), 'demo_video.mkv'),
                'subtitleFile': (io.BytesIO(b'plain text'), 'subtitle.txt'),
            },
            content_type='multipart/form-data',
        )

        self.assertEqual(400, response.status_code)
        self.assertIn('Csak .srt felirat fájl', response.get_json()['error'])

    def test_upload_video_supports_chunked_file_upload(self) -> None:
        uploaded_project = 'uploaded_video_project'
        video_bytes = b'first-chunk-second-chunk'
        midpoint = len(video_bytes) // 2
        chunks = [video_bytes[:midpoint], video_bytes[midpoint:]]

        first_response = self.client.post(
            '/api/upload-video',
            data={
                'projectName': uploaded_project,
                'chunkUploadId': 'video-chunk-test',
                'chunkIndex': '0',
                'totalChunks': '2',
                'fileName': 'chunked_video.mkv',
                'file': (io.BytesIO(chunks[0]), 'chunked_video.mkv'),
            },
            content_type='multipart/form-data',
        )

        self.assertEqual(200, first_response.status_code)
        first_payload = first_response.get_json()
        self.assertTrue(first_payload['success'])
        self.assertFalse(first_payload['completed'])

        final_response = self.client.post(
            '/api/upload-video',
            data={
                'projectName': uploaded_project,
                'chunkUploadId': 'video-chunk-test',
                'chunkIndex': '1',
                'totalChunks': '2',
                'fileName': 'chunked_video.mkv',
                'file': (io.BytesIO(chunks[1]), 'chunked_video.mkv'),
            },
            content_type='multipart/form-data',
        )

        self.assertEqual(200, final_response.status_code)
        final_payload = final_response.get_json()
        self.assertTrue(final_payload['success'])
        self.assertEqual('chunked_video.mkv', final_payload['video'])

        upload_dir = self.workdir / uploaded_project / config['PROJECT_SUBDIRS']['upload']
        self.assertEqual(video_bytes, (upload_dir / 'chunked_video.mkv').read_bytes())


if __name__ == '__main__':
    unittest.main()

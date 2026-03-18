from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from flask import url_for

from services.page_views import build_index_context, build_project_page_context, build_review_page_context


def register_page_routes(app, deps: Dict[str, Any]) -> None:
    @app.route('/')
    def index():
        context = build_index_context(deps['workdir_path'], deps['build_project_entries'])
        return deps['render_with_language'](
            'index.html',
            file_name='index',
            **context,
        )

    @app.route('/template-editor')
    def template_editor():
        return deps['render_with_language']('template_editor.html', file_name='template_editor')

    @app.route('/project/<project_name>')
    def show_project(project_name):
        context = build_project_page_context(
            project_name=project_name,
            config=deps['config'],
            secure_filename=deps['secure_filename'],
            compute_failed_generation_highlights=deps['compute_failed_generation_highlights'],
            get_audio_metadata_directories=deps['get_audio_metadata_directories'],
            get_failed_generation_directories=deps['get_failed_generation_directories'],
            should_enable_failed_move=deps['should_enable_failed_move'],
            build_audio_metadata=deps['build_audio_metadata'],
            build_failed_generation_json_metadata=deps['build_failed_generation_json_metadata'],
            get_tts_root_directory=deps['get_tts_root_directory'],
        )
        if context is None:
            return "Project not found", 404

        return deps['render_with_language'](
            'project.html',
            file_name='project',
            audio_extensions=deps['audio_extensions'],
            video_extensions=deps['video_extensions'],
            audio_mime_map=deps['audio_mime_map'],
            video_mime_map=deps['video_mime_map'],
            secret_param_names=deps['secret_param_names'],
            **context,
        )

    @app.route('/review/<project_name>')
    def review_project(project_name):
        def build_audio_url(encoded_audio_path: Path) -> str:
            try:
                relative_path = encoded_audio_path.relative_to(Path('workdir'))
            except ValueError:
                relative_path = encoded_audio_path
            return url_for('serve_workdir', filename=str(relative_path).replace('\\', '/'))

        context = build_review_page_context(
            project_name=project_name,
            config=deps['config'],
            secure_filename=deps['secure_filename'],
            prepare_segments_for_response=deps['prepare_segments_for_response'],
            get_review_encoded_audio_path=deps['get_review_encoded_audio_path'],
            build_audio_url=build_audio_url,
        )

        return deps['render_with_language'](
            'review.html',
            file_name='review',
            **context,
        )

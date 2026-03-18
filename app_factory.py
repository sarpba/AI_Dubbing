from __future__ import annotations

from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_compress import Compress

from routes.files_api import register_files_api_routes
from routes.pages import register_page_routes
from routes.review_api import register_review_api_routes
from routes.workflow_api import register_workflow_api_routes


def create_flask_app(import_name: str) -> Flask:
    app = Flask(import_name)
    app.config.setdefault(
        "COMPRESS_MIMETYPES",
        [
            "text/html",
            "text/css",
            "text/xml",
            "text/plain",
            "application/json",
            "application/javascript",
            "text/javascript",
        ],
    )
    app.config.setdefault("COMPRESS_LEVEL", 6)
    app.config.setdefault("COMPRESS_MIN_SIZE", 1024)
    Compress(app)
    return app


def register_app_routes(app, deps: Dict[str, Any]) -> None:
    register_page_routes(app, deps['pages'])
    register_review_api_routes(app, deps['review'])
    register_workflow_api_routes(app, deps['workflow'])
    register_files_api_routes(app, deps['files'])

    @app.route('/api/theme-colors', methods=['GET', 'POST'])
    def theme_colors_api():
        if request.method == 'GET':
            return jsonify({'success': True, 'colors': deps['load_theme_colors']()})

        if not request.is_json:
            return jsonify({'success': False, 'error': 'Hiányzó JSON payload.'}), 400

        payload = request.get_json(silent=True) or {}
        light_values = payload.get('light')
        dark_values = payload.get('dark')
        if not isinstance(light_values, dict) or not isinstance(dark_values, dict):
            return jsonify({'success': False, 'error': 'Érvénytelen témabeállítások.'}), 400

        normalized_input = {
            'light': {key: value for key, value in light_values.items() if key in deps['theme_color_keys']},
            'dark': {key: value for key, value in dark_values.items() if key in deps['theme_color_keys']},
        }

        try:
            saved = deps['save_theme_colors'](normalized_input)
        except OSError:
            return jsonify({'success': False, 'error': 'Nem sikerült elmenteni a témaszíneket.'}), 500

        return jsonify({'success': True, 'colors': saved})

    @app.route('/api/translated-split-progress/<project_name>', methods=['GET'])
    def translated_split_progress_api(project_name):
        sanitized_project = deps['secure_filename'](project_name)
        try:
            progress = deps['collect_translated_split_progress'](sanitized_project, config_snapshot=deps['get_config_copy']())
        except FileNotFoundError as exc:
            return jsonify({'success': False, 'error': str(exc)}), 404
        except deps['workflow_validation_error'] as exc:
            return jsonify({'success': False, 'error': str(exc)}), 400
        except Exception as exc:  # pragma: no cover
            deps['logger'].error(
                "Nem sikerült lekérdezni a translated split előrehaladást (%s): %s",
                sanitized_project,
                exc,
                exc_info=True,
            )
            return jsonify({'success': False, 'error': 'Nem sikerült lekérdezni a translated split előrehaladást.'}), 500

        return jsonify({'success': True, 'progress': progress})

    deps['initialize_scripts_catalog']()

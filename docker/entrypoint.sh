#!/usr/bin/env bash
set -euo pipefail

APP_ENV="${APP_ENV:-sync}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-2018}"
export FLASK_APP="${FLASK_APP:-main_app:app}"
AUTO_UPDATE="${AUTO_UPDATE:-1}"
REPO_URL="${REPO_URL:-}"
REPO_BRANCH="${REPO_BRANCH:-main}"

if [ "${AUTO_UPDATE}" = "1" ]; then
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        CURRENT_REMOTE="$(git remote get-url origin 2>/dev/null || true)"
        if [ -z "${CURRENT_REMOTE}" ] && [ -n "${REPO_URL}" ]; then
            git remote add origin "${REPO_URL}"
        fi
        git fetch origin "${REPO_BRANCH}"
        git reset --hard "origin/${REPO_BRANCH}"
        git clean -fd
    elif [ -n "${REPO_URL}" ]; then
        temp_dir="$(mktemp -d)"
        git clone --branch "${REPO_BRANCH}" --depth 1 "${REPO_URL}" "${temp_dir}/repo"
        find . -mindepth 1 -maxdepth 1 -exec rm -rf {} +
        cp -a "${temp_dir}/repo/." .
        rm -rf "${temp_dir}"
    else
        echo "AUTO_UPDATE enabled but repository metadata is missing and REPO_URL is empty; skipping update." >&2
    fi
    if [ -f requirements.txt ]; then
        conda run --no-capture-output -n "${APP_ENV}" python -m pip install --upgrade -r requirements.txt
    fi
    if [ -f docker/entrypoint.sh ]; then
        chmod +x docker/entrypoint.sh
    fi
fi

WHISPERX_CUDNN_PATH="$(conda run --no-capture-output -n whisperx python - <<'PY' 2>/dev/null || true
import site
import glob
import os
for base in site.getsitepackages():
    candidate = os.path.join(base, "nvidia", "cudnn", "lib")
    matches = glob.glob(candidate)
    if matches:
        print(matches[0])
        break
PY
)"
if [ -n "${WHISPERX_CUDNN_PATH}" ] && [ -d "${WHISPERX_CUDNN_PATH}" ]; then
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${WHISPERX_CUDNN_PATH}"
    else
        export LD_LIBRARY_PATH="${WHISPERX_CUDNN_PATH}"
    fi
fi

exec conda run --no-capture-output -n "${APP_ENV}" flask run \
    --host "${APP_HOST}" \
    --port "${APP_PORT}"

#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
DEFAULT_ENTRYPOINT="/opt/base_entrypoint.sh"
TARGET_ENTRYPOINT="${WORKSPACE_DIR}/docker/entrypoint.sh"

if [ ! -d "${WORKSPACE_DIR}" ]; then
    mkdir -p "${WORKSPACE_DIR}"
fi

if [ ! -f "${TARGET_ENTRYPOINT}" ]; then
    mkdir -p "$(dirname "${TARGET_ENTRYPOINT}")"
    cp "${DEFAULT_ENTRYPOINT}" "${TARGET_ENTRYPOINT}"
fi

chmod +x "${TARGET_ENTRYPOINT}"

cd "${WORKSPACE_DIR}"

exec "${TARGET_ENTRYPOINT}" "$@"

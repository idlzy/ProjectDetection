#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
user_home="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)"
python_bin=""
for candidate in \
    "${TRAIN_PYTHON:-}" \
    "${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}" \
    "${user_home:+${user_home}/miniconda3/envs/ai/bin/python}" \
    "${user_home:+${user_home}/anaconda3/envs/ai/bin/python}" \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
        python_bin="${candidate}"
        break
    fi
done
if [[ -z "${python_bin}" ]]; then
    echo "No Python interpreter found. Activate the environment or set TRAIN_PYTHON." >&2
    exit 1
fi
"${python_bin}" "${PROJECT_ROOT}/tools/train_runtime.py" list \
    --runtime-root "${PROJECT_ROOT}/.runtime"

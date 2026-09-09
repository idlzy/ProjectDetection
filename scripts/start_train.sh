#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${PROJECT_ROOT}/.runtime"
PID_FILE="${RUNTIME_DIR}/DetectionTrain.pid"

mkdir -p "${RUNTIME_DIR}"
if [[ -f "${PID_FILE}" ]]; then
    running_pid="$(<"${PID_FILE}")"
    if [[ "${running_pid}" =~ ^[0-9]+$ ]] && kill -0 "${running_pid}" 2>/dev/null; then
        echo "DetectionTrain is already running (PID=${running_pid})"
        exit 1
    fi
    rm -f "${PID_FILE}"
fi

if [[ $# -eq 0 ]]; then
    set -- --config configs/experiments/fcos3d_exp_pgda.yaml
fi

cd "${PROJECT_ROOT}"

resolve_python() {
    local candidate
    local conda_base
    local candidates=("${TRAIN_PYTHON:-}" "$(command -v python3 2>/dev/null || true)")
    if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base 2>/dev/null || true)"
        candidates+=("${conda_base}/envs/ai/bin/python")
    fi
    for candidate in "${candidates[@]}"; do
        if [[ -n "${candidate}" && -x "${candidate}" ]] && \
           "${candidate}" -c 'import torch' >/dev/null 2>&1; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

if ! python_bin="$(resolve_python)"; then
    echo "No Python interpreter with PyTorch was found. Activate the training environment or set TRAIN_PYTHON." >&2
    exit 1
fi

echo "Using Python: ${python_bin}"
"${python_bin}" -u tools/train.py "$@" --validate-only

nohup bash -c 'exec -a DetectionTrain bash "$1" "${@:2}"' \
    DetectionTrain "${PROJECT_ROOT}/scripts/train_supervisor.sh" "${python_bin}" "$@" \
    </dev/null >/dev/null 2>&1 &
train_pid=$!
echo "${train_pid}" > "${PID_FILE}"

sleep 1
if ! kill -0 "${train_pid}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "DetectionTrain failed during startup; run the foreground command once to inspect configuration errors"
    exit 1
fi

echo "DetectionTrain started in background (PID=${train_pid})"
echo "Application logs are written by Python under outputs/<experiment>/logs/train.log"

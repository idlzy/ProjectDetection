#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${TRAIN_RUNTIME_DIR:-${PROJECT_ROOT}/.runtime}"
STATUS_FILE="${RUNTIME_DIR}/DetectionTrain.status"
EVENT_FILE="${RUNTIME_DIR}/DetectionTrain.events.log"
PYTHON_BIN="$1"
shift
TRAIN_ARGS=("$@")
MAX_RESTARTS="${TRAIN_MAX_RESTARTS:-5}"
RESTART_DELAY="${TRAIN_RESTART_DELAY:-30}"
restart_count=0
child_pid=""

record_status() {
    local state="$1" exit_code="$2" signal_name="$3"
    local timestamp temporary
    timestamp="$(date --iso-8601=seconds)"
    temporary="${STATUS_FILE}.tmp"
    printf 'timestamp=%s state=%s exit_code=%s signal=%s restart_count=%s max_restarts=%s\n' \
        "${timestamp}" "${state}" "${exit_code}" "${signal_name}" \
        "${restart_count}" "${MAX_RESTARTS}" > "${temporary}"
    mv "${temporary}" "${STATUS_FILE}"
    printf '%s state=%s exit_code=%s signal=%s restart_count=%s/%s\n' \
        "${timestamp}" "${state}" "${exit_code}" "${signal_name}" \
        "${restart_count}" "${MAX_RESTARTS}" >> "${EVENT_FILE}"
}

forward_signal() {
    local signal_name="$1"
    record_status "stopping" "" "${signal_name}"
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
        kill -"${signal_name}" "${child_pid}" 2>/dev/null || true
        wait "${child_pid}" 2>/dev/null || true
    fi
    exit 0
}

trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap 'forward_signal HUP' HUP

mkdir -p "${RUNTIME_DIR}"
cd "${PROJECT_ROOT}"

while true; do
    command=("${PYTHON_BIN}" -u tools/train.py "${TRAIN_ARGS[@]}")
    if (( restart_count > 0 )); then
        command+=(--auto-resume)
    fi
    record_status "running" "" ""
    "${command[@]}" &
    child_pid=$!
    wait "${child_pid}"
    exit_code=$?
    child_pid=""
    if (( exit_code == 0 )); then
        record_status "completed" "0" ""
        exit 0
    fi
    signal_name=""
    if (( exit_code > 128 )); then
        signal_name="$(kill -l "$((exit_code - 128))" 2>/dev/null || true)"
    fi
    restart_count=$((restart_count + 1))
    record_status "crashed" "${exit_code}" "${signal_name}"
    if (( restart_count > MAX_RESTARTS )); then
        record_status "failed" "${exit_code}" "${signal_name}"
        exit "${exit_code}"
    fi
    sleep "${RESTART_DELAY}"
done

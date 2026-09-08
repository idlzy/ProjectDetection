#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${PROJECT_ROOT}/.runtime/DetectionTrain.pid"

if [[ ! -f "${PID_FILE}" ]]; then
    echo "DetectionTrain is not running (PID file not found)"
    exit 0
fi

train_pid="$(<"${PID_FILE}")"
if [[ ! "${train_pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${train_pid}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "DetectionTrain is not running (removed stale PID file)"
    exit 0
fi

process_name="$(tr -d '\0' < "/proc/${train_pid}/comm" 2>/dev/null || true)"
process_args="$(tr '\0' ' ' < "/proc/${train_pid}/cmdline" 2>/dev/null || true)"
if [[ "${process_name}" != "DetectionTrain" && "${process_args}" != *"DetectionTrain"* ]]; then
    echo "Refusing to stop PID=${train_pid}: it is not DetectionTrain"
    exit 1
fi

kill -TERM "${train_pid}"
for _ in {1..30}; do
    if ! kill -0 "${train_pid}" 2>/dev/null; then
        rm -f "${PID_FILE}"
        echo "DetectionTrain stopped (PID=${train_pid})"
        exit 0
    fi
    sleep 1
done

echo "DetectionTrain did not exit within 30 seconds; PID=${train_pid} was left untouched"
exit 1

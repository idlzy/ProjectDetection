#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${TRAIN_RUNTIME_DIR:-${PROJECT_ROOT}/.runtime}"
JOB_DIR="${TRAIN_JOB_DIR:-${RUNTIME_DIR}}"
if [[ -n "${TRAIN_JOB_DIR:-}" ]]; then
    STATUS_FILE="${JOB_DIR}/status"
    EVENT_FILE="${JOB_DIR}/events.log"
    CHILD_PID_FILE="${JOB_DIR}/child.pid"
else
    # Backward-compatible paths for direct invocation and older jobs/tests.
    STATUS_FILE="${RUNTIME_DIR}/DetectionTrain.status"
    EVENT_FILE="${RUNTIME_DIR}/DetectionTrain.events.log"
    CHILD_PID_FILE="${RUNTIME_DIR}/DetectionTrain.child.pid"
fi
PYTHON_BIN="$1"
shift
TRAIN_ARGS=("$@")
MAX_RESTARTS="${TRAIN_MAX_RESTARTS:-5}"
RESTART_DELAY="${TRAIN_RESTART_DELAY:-30}"
GPU_CLEANUP_TIMEOUT="${TRAIN_GPU_CLEANUP_TIMEOUT:-60}"
WORLD_SIZE="${TRAIN_WORLD_SIZE:-1}"
if [[ ! "${WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid TRAIN_WORLD_SIZE: ${WORLD_SIZE}" >&2
    exit 2
fi
restart_count=0
child_pid=""
worker_monitor_pid=""
KNOWN_WORKER_FILE="${JOB_DIR}/worker.pids"

sync_registry_state() {
    local state="$1"
    [[ -n "${TRAIN_JOB_DIR:-}" ]] || return 0
    "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/train_runtime.py" state \
        --job-dir "${JOB_DIR}" --value "${state}" >/dev/null 2>&1 || true
}

sync_child_process() {
    local action="$1"
    [[ -n "${TRAIN_JOB_DIR:-}" ]] || return 0
    if [[ "${action}" == "start" ]]; then
        "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/train_runtime.py" child-start \
            --job-dir "${JOB_DIR}" --pid "${child_pid}" >/dev/null 2>&1 || true
    else
        "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/train_runtime.py" child-clear \
            --job-dir "${JOB_DIR}" >/dev/null 2>&1 || true
    fi
}

sync_worker_processes() {
    local pids="${1:-}"
    [[ -n "${TRAIN_JOB_DIR:-}" ]] || return 0
    "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/train_runtime.py" workers-set \
        --job-dir "${JOB_DIR}" --pids "${pids}" >/dev/null 2>&1 || true
}

capture_worker_pids() {
    local pids
    [[ -n "${child_pid}" ]] || return 0
    pids="$(
        ps -eo pid=,pgid= 2>/dev/null | awk -v group="${child_pid}" \
            '$2 == group && $1 != group {print $1}' | paste -sd, -
    )"
    if [[ -n "${pids}" ]]; then
        tr ',' '\n' <<< "${pids}" >> "${KNOWN_WORKER_FILE}"
        sort -nu "${KNOWN_WORKER_FILE}" -o "${KNOWN_WORKER_FILE}"
    fi
    sync_worker_processes "${pids}"
}

monitor_workers() {
    while [[ -n "${child_pid}" ]] && kill -0 -- "-${child_pid}" 2>/dev/null; do
        capture_worker_pids
        sleep 1
    done
}

stop_worker_monitor() {
    if [[ -n "${worker_monitor_pid}" ]]; then
        kill -TERM "${worker_monitor_pid}" 2>/dev/null || true
        wait "${worker_monitor_pid}" 2>/dev/null || true
        worker_monitor_pid=""
    fi
}

record_status() {
    local state="$1" exit_code="$2" signal_name="$3"
    local timestamp temporary
    timestamp="$(date --iso-8601=seconds)"
    temporary="${STATUS_FILE}.tmp"
    printf 'timestamp=%s state=%s exit_code=%s signal=%s restart_count=%s max_restarts=%s experiment=%s gpu=%s\n' \
        "${timestamp}" "${state}" "${exit_code}" "${signal_name}" \
        "${restart_count}" "${MAX_RESTARTS}" "${TRAIN_EXPERIMENT:-unknown}" \
        "${TRAIN_VISIBLE_DEVICES:-unknown}" > "${temporary}"
    mv "${temporary}" "${STATUS_FILE}"
    printf '%s state=%s exit_code=%s signal=%s restart_count=%s/%s\n' \
        "${timestamp}" "${state}" "${exit_code}" "${signal_name}" \
        "${restart_count}" "${MAX_RESTARTS}" >> "${EVENT_FILE}"
    sync_registry_state "${state}"
}

terminate_child_group() {
    local signal_name="${1:-TERM}"
    [[ -n "${child_pid}" ]] || return 0
    kill -"${signal_name}" -- "-${child_pid}" 2>/dev/null || true
}

cleanup_child_group() {
    local index
    [[ -n "${child_pid}" ]] || return 0
    # The train process is a separate session/group leader. Remove DataLoader
    # workers that survived an abnormal parent exit before restarting it.
    terminate_child_group TERM
    for index in {1..20}; do
        if ! kill -0 -- "-${child_pid}" 2>/dev/null; then
            return 0
        fi
        sleep 0.1
    done
    terminate_child_group KILL
}

gpu_pid_is_listed() {
    local target_pid="$1"
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | awk -v target="${target_pid}" \
        '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); if ($0 == target) found=1} END {exit !found}'
}

wait_for_child_resources() {
    local target_pid="$1"
    local elapsed=0
    local group_alive=0
    local gpu_alive=0
    local tracked_pid

    while (( elapsed <= GPU_CLEANUP_TIMEOUT )); do
        group_alive=0
        gpu_alive=0
        kill -0 -- "-${target_pid}" 2>/dev/null && group_alive=1
        gpu_pid_is_listed "${target_pid}" && gpu_alive=1
        if [[ -f "${KNOWN_WORKER_FILE}" ]]; then
            while read -r tracked_pid; do
                [[ "${tracked_pid}" =~ ^[0-9]+$ ]] || continue
                if gpu_pid_is_listed "${tracked_pid}"; then
                    gpu_alive=1
                    break
                fi
            done < "${KNOWN_WORKER_FILE}"
        fi
        if (( group_alive == 0 && gpu_alive == 0 )); then
            return 0
        fi
        if (( elapsed == GPU_CLEANUP_TIMEOUT )); then
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    printf '%s GPU/process cleanup timed out | child_pid=%s | process_group_alive=%s | cuda_pid_listed=%s | timeout=%ss\n' \
        "$(date --iso-8601=seconds)" "${target_pid}" "${group_alive}" \
        "${gpu_alive}" "${GPU_CLEANUP_TIMEOUT}" >&2
    return 1
}

forward_signal() {
    local signal_name="$1"
    record_status "stopping" "" "${signal_name}"
    stop_worker_monitor
    if [[ -n "${child_pid}" ]]; then
        terminate_child_group "${signal_name}"
        wait "${child_pid}" 2>/dev/null || true
        cleanup_child_group
    fi
    rm -f "${CHILD_PID_FILE}"
    sync_child_process clear
    sync_worker_processes ""
    record_status "stopped" "0" "${signal_name}"
    exit 0
}

trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap 'forward_signal HUP' HUP

mkdir -p "${JOB_DIR}"
cd "${PROJECT_ROOT}"

while true; do
    if (( WORLD_SIZE > 1 )); then
        command=(
            "${PYTHON_BIN}" -u -m torch.distributed.run
            --standalone --nproc_per_node="${WORLD_SIZE}"
            tools/train.py "${TRAIN_ARGS[@]}"
        )
    else
        command=("${PYTHON_BIN}" -u tools/train.py "${TRAIN_ARGS[@]}")
    fi
    if (( restart_count > 0 )); then
        command+=(--auto-resume)
    fi
    record_status "running" "" ""
    setsid "${command[@]}" &
    child_pid=$!
    : > "${KNOWN_WORKER_FILE}"
    printf '%s\n' "${child_pid}" > "${CHILD_PID_FILE}"
    sync_child_process start
    monitor_workers &
    worker_monitor_pid=$!
    exit_code=0
    wait "${child_pid}" || exit_code=$?
    capture_worker_pids
    stop_worker_monitor
    cleanup_child_group
    exited_child_pid="${child_pid}"
    if ! wait_for_child_resources "${exited_child_pid}"; then
        restart_count=$((restart_count + 1))
        record_status "cleanup_failed" "${exit_code}" "GPU_CONTEXT_STILL_ACTIVE"
        # Restarting while the old CUDA context still owns memory only creates
        # an OOM loop. Keep the recovery checkpoint intact and require GPU
        # cleanup/reset before a new launch.
        record_status "failed" "70" "GPU_CONTEXT_STILL_ACTIVE"
        exit 70
    fi
    rm -f "${CHILD_PID_FILE}"
    sync_child_process clear
    sync_worker_processes ""
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
    record_status "restarting" "${exit_code}" "${signal_name}"
    sleep "${RESTART_DELAY}"
done

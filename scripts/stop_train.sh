#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${PROJECT_ROOT}/.runtime"
RUNTIME_TOOL="${PROJECT_ROOT}/tools/train_runtime.py"
selector="${1:-}"
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
lookup_command=("${python_bin}" "${RUNTIME_TOOL}" lookup --runtime-root "${RUNTIME_DIR}")
if [[ -n "${selector}" ]]; then
    lookup_command+=("${selector}")
fi

if ! lookup_output="$("${lookup_command[@]}")"; then
    exit 1
fi
mapfile -t job_fields <<< "${lookup_output}"
job_dir="${job_fields[0]}"
experiment="${job_fields[1]}"
train_pid="${job_fields[6]}"
expected_start_ticks="${job_fields[7]}"
child_pid="${job_fields[8]:-}"
child_start_ticks="${job_fields[9]:-}"

process_matches() {
    local pid="$1" expected="$2" state current
    [[ "${pid}" =~ ^[0-9]+$ && -n "${expected}" && -r "/proc/${pid}/stat" ]] || return 1
    read -r state current < <(
        sed -E 's/^.*\) //' "/proc/${pid}/stat" 2>/dev/null | awk '{print $1, $20}'
    )
    [[ "${state}" != "Z" && "${current}" == "${expected}" ]]
}

target_kind=""
target_pid=""
if process_matches "${train_pid}" "${expected_start_ticks}"; then
    target_kind="supervisor"
    target_pid="${train_pid}"
elif process_matches "${child_pid}" "${child_start_ticks}"; then
    target_kind="child-group"
    target_pid="${child_pid}"
elif [[ "${child_pid}" =~ ^[0-9]+$ ]] && \
     kill -0 -- "-${child_pid}" 2>/dev/null; then
    # The distributed launcher may have exited while rank/DataLoader workers
    # remain in its process group.
    target_kind="child-group"
    target_pid="${child_pid}"
else
    echo "Training job is no longer running: ${experiment}" >&2
    exit 1
fi

if [[ "${target_kind}" == "supervisor" ]]; then
    kill -TERM "${target_pid}"
else
    kill -TERM -- "-${target_pid}"
fi
for _ in {1..30}; do
    if [[ "${target_kind}" == "supervisor" ]]; then
        process_matches "${target_pid}" "${expected_start_ticks}" || stopped=1
    else
        kill -0 -- "-${target_pid}" 2>/dev/null || stopped=1
    fi
    if [[ "${stopped:-0}" == "1" ]]; then
        "${python_bin}" "${RUNTIME_TOOL}" state --job-dir "${job_dir}" \
            --value stopped >/dev/null 2>&1 || true
        echo "DetectionTrain stopped: ${experiment} (${target_kind} PID=${target_pid})"
        exit 0
    fi
    sleep 1
done

echo "DetectionTrain did not exit within 30 seconds; ${experiment} ${target_kind} PID=${target_pid} was left running" >&2
exit 1

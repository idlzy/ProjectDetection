#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${PROJECT_ROOT}/.runtime"
RUNTIME_TOOL="${PROJECT_ROOT}/tools/train_runtime.py"
mkdir -p "${RUNTIME_DIR}/jobs"

if [[ $# -eq 0 ]]; then
    set -- --config configs/experiments/fcos3d_exp_pgda.yaml
fi

raw_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a requested_devices <<< "${raw_visible_devices}"
declare -A seen_devices=()
normalized_devices=()
for device_token in "${requested_devices[@]}"; do
    device_token="${device_token//[[:space:]]/}"
    if [[ -z "${device_token}" ]]; then
        echo "CUDA_VISIBLE_DEVICES contains an empty device entry: ${raw_visible_devices}" >&2
        exit 2
    fi
    if [[ ! "${device_token}" =~ ^[0-9]+$ && \
          ! "${device_token}" =~ ^(GPU|MIG)-[A-Za-z0-9-]+$ ]]; then
        echo "Unsupported CUDA device token: ${device_token}" >&2
        exit 2
    fi
    if [[ -n "${seen_devices[${device_token}]:-}" ]]; then
        echo "CUDA_VISIBLE_DEVICES contains a duplicate device: ${device_token}" >&2
        exit 2
    fi
    seen_devices["${device_token}"]=1
    normalized_devices+=("${device_token}")
done
if (( ${#normalized_devices[@]} == 0 )); then
    normalized_devices=(0)
fi
CUDA_VISIBLE_DEVICES="$(IFS=','; echo "${normalized_devices[*]}")"
export CUDA_VISIBLE_DEVICES
world_size="${#normalized_devices[@]}"
launch_mode="single"
if (( world_size > 1 )); then
    launch_mode="ddp"
fi

cd "${PROJECT_ROOT}"

# Detection batches contain many tensors. With several DataLoader workers,
# PyTorch can exceed a small inherited soft file-descriptor limit while
# transferring them between processes.
requested_open_files="${TRAIN_OPEN_FILES_LIMIT:-65535}"
current_open_files="$(ulimit -Sn)"
hard_open_files="$(ulimit -Hn)"
if [[ "${requested_open_files}" =~ ^[0-9]+$ && \
      "${current_open_files}" =~ ^[0-9]+$ ]] && \
   (( current_open_files < requested_open_files )); then
    effective_open_files="${requested_open_files}"
    if [[ "${hard_open_files}" =~ ^[0-9]+$ ]] && \
       (( effective_open_files > hard_open_files )); then
        effective_open_files="${hard_open_files}"
    fi
    if ! ulimit -Sn "${effective_open_files}"; then
        echo "Warning: failed to raise open-file limit; DataLoader workers may fail" >&2
    fi
fi
echo "Open-file limit: $(ulimit -Sn)"

resolve_python() {
    local candidate
    local conda_base
    local user_home
    user_home="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)"
    local candidates=(
        "${TRAIN_PYTHON:-}"
        "${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}"
        "${user_home:+${user_home}/miniconda3/envs/ai/bin/python}"
        "${user_home:+${user_home}/anaconda3/envs/ai/bin/python}"
        "$(command -v python3 2>/dev/null || true)"
        "$(command -v python 2>/dev/null || true)"
    )
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

if ! registry_output="$(
    "${python_bin}" "${RUNTIME_TOOL}" reserve \
        --runtime-root "${RUNTIME_DIR}" -- "$@"
)"; then
    exit 1
fi
mapfile -t job_fields <<< "${registry_output}"
if (( ${#job_fields[@]} < 6 )); then
    echo "Failed to reserve a training runtime entry" >&2
    exit 1
fi
job_dir="${job_fields[0]}"
experiment="${job_fields[1]}"
job_id="${job_fields[2]}"
visible_devices="${job_fields[3]}"
output_dir="${job_fields[5]}"
supervisor_log="${job_dir}/supervisor.log"

TRAIN_JOB_DIR="${job_dir}" \
TRAIN_JOB_ID="${job_id}" \
TRAIN_EXPERIMENT="${experiment}" \
TRAIN_VISIBLE_DEVICES="${visible_devices}" \
TRAIN_WORLD_SIZE="${world_size}" \
TRAIN_LAUNCH_MODE="${launch_mode}" \
nohup bash -c 'exec -a "$1" bash "$2" "${@:3}"' \
    _ "DetectionTrain:${job_id}" "${PROJECT_ROOT}/scripts/train_supervisor.sh" \
    "${python_bin}" "$@" </dev/null >>"${supervisor_log}" 2>&1 &
train_pid=$!

if ! "${python_bin}" "${RUNTIME_TOOL}" activate \
    --job-dir "${job_dir}" --pid "${train_pid}"; then
    kill -TERM "${train_pid}" 2>/dev/null || true
    "${python_bin}" "${RUNTIME_TOOL}" state \
        --job-dir "${job_dir}" --value failed 2>/dev/null || true
    echo "DetectionTrain failed while activating its runtime entry" >&2
    exit 1
fi

sleep 1
if ! kill -0 "${train_pid}" 2>/dev/null; then
    if grep -q ' state=completed ' "${job_dir}/status" 2>/dev/null; then
        echo "DetectionTrain completed during startup: ${experiment}"
        echo "Application log: ${output_dir}/logs/train.log"
        exit 0
    fi
    "${python_bin}" "${RUNTIME_TOOL}" state \
        --job-dir "${job_dir}" --value failed 2>/dev/null || true
    echo "DetectionTrain failed during startup; inspect ${supervisor_log}" >&2
    tail -n 20 "${supervisor_log}" 2>/dev/null || true
    exit 1
fi

echo "DetectionTrain started in background (PID=${train_pid})"
echo "Experiment: ${experiment}"
echo "GPU: ${visible_devices}"
echo "World size: ${world_size}"
echo "Launch mode: ${launch_mode}"
echo "Runtime: ${job_dir}"
echo "Application log: ${output_dir}/logs/train.log"

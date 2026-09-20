#!/usr/bin/env bash
# Visualize and evaluate an MW3D split with one reproducible command.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

SPLIT="${SPLIT:-test}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)
            if [[ $# -lt 2 ]]; then
                echo "[mw3d-test] ERROR: --split requires train, val or test" >&2
                exit 2
            fi
            SPLIT="$2"
            shift 2
            ;;
        --split=*)
            SPLIT="${1#--split=}"
            shift
            ;;
        *)
            echo "[mw3d-test] ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done
if [[ "${SPLIT}" != "train" && "${SPLIT}" != "val" && "${SPLIT}" != "test" ]]; then
    echo "[mw3d-test] ERROR: split must be train, val or test: ${SPLIT}" >&2
    exit 2
fi

resolve_python() {
    local candidate
    local conda_base
    local candidates=("${TEST_PYTHON:-}" "$(command -v python3 2>/dev/null || true)")
    if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base 2>/dev/null || true)"
        candidates+=("${conda_base}/envs/ai/bin/python")
    fi
    for candidate in "${candidates[@]}"; do
        if [[ -n "${candidate}" && -x "${candidate}" ]] && \
           "${candidate}" -c 'import cv2, matplotlib, torch, yaml' >/dev/null 2>&1; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

if ! PYTHON_BIN="$(resolve_python)"; then
    echo "[mw3d-test] ERROR: no Python with PyTorch, OpenCV, Matplotlib and PyYAML found" >&2
    echo "[mw3d-test] Activate the ai environment or set TEST_PYTHON" >&2
    exit 1
fi

CONFIG="${CONFIG:-configs/experiments/fcos3d_r101_fpn_dcn_full_pgd_local.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
DEVICE_ID="${DEVICE_ID:-0}"
VIZ_SCORE_THR="${VIZ_SCORE_THR:-0.35}"
EVAL_SCORE_THR="${EVAL_SCORE_THR:-0.05}"
EVAL_CLASSES="${EVAL_CLASSES:-}"
DATA_ROOT="${DATA_ROOT:-}"
READY_ROOT="${READY_ROOT:-${DATA_ROOT}}"
VIZ_MAX_IMAGES="${VIZ_MAX_IMAGES:-}"
VIZ_START_INDEX="${VIZ_START_INDEX:-0}"
MAX_DETECTIONS="${MAX_DETECTIONS:-30}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TAG="${TAG:-mw3d_${SPLIT}_${STAMP}}"
RUN_DIR="${RUN_DIR:-outputs/test_runs/${TAG}}"
VIZ_DIR="${VIZ_DIR:-${RUN_DIR}/visualizations}"
METRIC_JSON="${METRIC_JSON:-${RUN_DIR}/metrics.json}"
PLOT_DIR="${PLOT_DIR:-${RUN_DIR}/plots}"

if [[ -z "${CHECKPOINT}" ]]; then
    echo "[mw3d-test] ERROR: set CHECKPOINT=/path/to/best.pth" >&2
    exit 1
fi
for required in "${CONFIG}" "${CHECKPOINT}"; do
    if [[ ! -f "${required}" ]]; then
        echo "[mw3d-test] ERROR: not found: ${required}" >&2
        exit 1
    fi
done

mkdir -p "${VIZ_DIR}" "$(dirname "${METRIC_JSON}")" "${PLOT_DIR}"

echo "[mw3d-test] config=${CONFIG}"
echo "[mw3d-test] checkpoint=${CHECKPOINT}"
echo "[mw3d-test] split=${SPLIT}"
echo "[mw3d-test] python=${PYTHON_BIN}"
echo "[mw3d-test] device=${DEVICE_ID}"
echo "[mw3d-test] visualizations=${VIZ_DIR}"
echo "[mw3d-test] metric_json=${METRIC_JSON}"
echo "[mw3d-test] plots=${PLOT_DIR}"
if [[ -n "${EVAL_CLASSES}" ]]; then
    echo "[mw3d-test] evaluated classes=${EVAL_CLASSES}"
fi

if [[ "${SKIP_VIZ:-0}" != "1" ]]; then
    viz_args=(
        --config "${CONFIG}"
        --checkpoint "${CHECKPOINT}"
        --split "${SPLIT}"
        --output-dir "${VIZ_DIR}"
        --max-detections "${MAX_DETECTIONS}"
        --layout grouped
        --sampling sequential
        --start-index "${VIZ_START_INDEX}"
        --skip-summary-json
        --set "evaluation.nms_pre=50"
              "evaluation.max_per_image=${MAX_DETECTIONS}"
              "evaluation.score_threshold=${VIZ_SCORE_THR}"
    )
    if [[ -n "${DATA_ROOT}" ]]; then
        viz_args+=("data.data_root=${DATA_ROOT}")
    fi
    if [[ -n "${READY_ROOT}" ]]; then
        viz_args+=("data.ready_root=${READY_ROOT}")
    fi
    if [[ -n "${EVAL_CLASSES}" ]]; then
        viz_args+=(--set "evaluation.classes=${EVAL_CLASSES}")
    fi
    if [[ -n "${VIZ_MAX_IMAGES}" ]]; then
        viz_args+=(--max-images "${VIZ_MAX_IMAGES}")
    fi
    if [[ "${VIZ_OVERWRITE:-0}" == "1" ]]; then
        viz_args+=(--overwrite)
    fi
    if [[ "${SAVE_BEV:-0}" == "1" ]]; then
        viz_args+=(--save-bev)
    fi
    if [[ "${DRAW_GROUND_TRUTH:-0}" == "1" ]]; then
        viz_args+=(--draw-ground-truth)
    fi
    CUDA_VISIBLE_DEVICES="${DEVICE_ID}" "${PYTHON_BIN}" tools/visualize_test.py "${viz_args[@]}"
else
    echo "[mw3d-test] SKIP_VIZ=1: visualization skipped"
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    eval_args=(
        --config "${CONFIG}"
        --checkpoint "${CHECKPOINT}"
        --split "${SPLIT}"
        --output "${METRIC_JSON}"
        --plot-dir "${PLOT_DIR}"
        --set "evaluation.score_threshold=${EVAL_SCORE_THR}"
    )
    if [[ -n "${DATA_ROOT}" ]]; then
        eval_args+=(--set "data.data_root=${DATA_ROOT}")
    fi
    if [[ -n "${READY_ROOT}" ]]; then
        eval_args+=(--set "data.ready_root=${READY_ROOT}")
    fi
    if [[ -n "${EVAL_MAX_SAMPLES:-}" ]]; then
        eval_args+=(--set "runtime.max_${SPLIT}_samples=${EVAL_MAX_SAMPLES}")
        echo "[mw3d-test] WARNING: evaluation limited to ${EVAL_MAX_SAMPLES} samples"
    fi
    if [[ -n "${EVAL_CLASSES}" ]]; then
        eval_args+=(--set "evaluation.classes=${EVAL_CLASSES}")
    fi
    CUDA_VISIBLE_DEVICES="${DEVICE_ID}" "${PYTHON_BIN}" tools/test.py "${eval_args[@]}"
else
    echo "[mw3d-test] SKIP_EVAL=1: formal evaluation skipped"
fi

echo "[mw3d-test] done"
echo "[mw3d-test] visualizations: ${VIZ_DIR}"
echo "[mw3d-test] metrics:        ${METRIC_JSON}"
echo "[mw3d-test] plots:          ${PLOT_DIR}"

#!/usr/bin/env bash
set -euo pipefail
torchrun --standalone --nproc_per_node=2 tools/train.py --config configs/experiments/fcos3d_exp_pgda.yaml "$@"

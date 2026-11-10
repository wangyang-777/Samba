#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${1:-1}"
if [[ $# -gt 0 ]]; then shift; fi
export PYTHONPATH=dataset:transform:.
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec ../.venv-samba/bin/python train/train_rgbd.py \
  --epoch 50 --batch_size 2 --num_thread 4 --depth_aux_weight 0.2 \
  --save_fold ./checkpoints "$@"

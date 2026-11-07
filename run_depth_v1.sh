#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=.:dataset:transform:models/encoders/selective_scan
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

exec ../.venv-samba/bin/python -u train/train_rgbd.py \
  --epoch 50 \
  --batch_size 2 \
  --num_thread 4 \
  --save_fold ./checkpoints/depth_quality_v1 \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_gpu="${1:-0}"
if (( $# > 0 )); then
    shift
fi
cd "$experiment_dir"
export CUDA_VISIBLE_DEVICES="$experiment_gpu"
export PYTHONPATH="$experiment_dir/dataset:$experiment_dir/transform:$experiment_dir"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

exec "$experiment_dir/../.venv-samba/bin/python" train/train_rgbd.py \
    --epoch 50 --batch_size 2 --num_thread 8 --depth_aux_weight 0.2 \
    --test_dataset NJU2K "$@"

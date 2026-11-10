# Retrain selective-guidance A on the exact ACINet_best data source

## Data protocol

Source root: `/home/wmy/project/Datasets/RGBD_dataset`, as configured by
`/home/wmy/project/ACINet_best/options.py` and its official test entry point.
RGB, depth and GT all come from this same source version. The aliases below
are directory symlinks; source images retain their original resolution and
8-bit values until the inherited Samba data augmentation and resizing.

| Samba alias | Actual source relative to that root | Samples |
|---|---|---:|
| train_DUT | RGB-D_train | 2985 |
| DUT-RGBD | RGB-D_test/DUT | 400 |
| NJU2K | RGB-D_test/NJUD | 500 |
| NLPR | RGB-D_test/NLPR | 300 |
| SIP | RGB-D_test/SIP | 929 |
| STERE | RGB-D_test/STERE | 1000 |

`prepare_acinet_rgbd.py` verifies all triplets, dimensions, bit depths,
grayscale channels and spatial variation after reading. All 6114 triplets
passed. Train/test identifiers are disjoint. The manifest and file
path/size/mtime fingerprints are saved to `rgbd_dataset/data_protocol.json`.

Checkpoint selection continues to use NJU2K (500 images), from the ACINet_best
test root, matching the earlier Samba/A experimental protocol. NJU2K is therefore
a selection set; report the four other sets separately as well.

## Model and training

Baseline model parent: `66ca592` (original Samba architecture). The model,
dataset loader, transformations and objectives are byte-identical to that
parent. For A, use the companion `experiment/selective-acinet-data` branch,
whose model parent is `f92cac445986244a0ff4372ab203204efaa6fd3e`.

Both experiments use their original model implementations with VMamba ImageNet
pretraining, seed 1024, 50 epochs, input 448x448, batch 2, accumulation 4,
AdamW lr 1e-4 / weight decay 0.01, cosine T_max=10 / eta_min=1e-5,
DataLoader workers 4. A uses its existing synthetic-repair loss with weight 0.2.
Original optimizer grouping and unscaled gradient accumulation are retained.
These are the Samba/A training settings, rather than ACINet's optimizer settings.

Each run starts from VMamba pretraining and writes to its own new worktree.
Source commit, pretrained SHA256, options and data provenance are written to
`checkpoints/Samba/rgbd_datasets/run_config.json`. Previous experiments and
weights remain available in their original directories.

The changed data includes resolution and depth polarity differences as well as
bit depth. Compare the two new runs with each other; their scores are a separate
protocol from earlier runs on the resized mixed-bit-depth dataset.

## Run and verification

```bash
../.venv-samba/bin/python prepare_acinet_rgbd.py
ln -s ../datasets_acinet_best rgbd_dataset
bash run_acinet_data.sh 1
```

For the companion A worktree, use GPU argument `1`. The existing shared
`.venv-samba` environment and `models/pretrained` link are required.

Verification: eight real training batches, two optimizer updates, strict
checkpoint reload and real-image inference. The formerly saturated `0002`
depth example now retains spatial variation. Results are recorded in
`verification/acinet_smoke.json`.

Training-code edits add provenance and first-update logging. The smoke-only
stop path avoids an extra optimizer step at an accumulation boundary.
Full-training objectives, schedules and update rules follow the respective
baseline/A implementation. All outputs and datasets are excluded from Git.

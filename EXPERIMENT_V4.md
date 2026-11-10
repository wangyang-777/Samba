# V4: A + depth-specific adapters

## Research question

Does adapting geometry inside the shared RGB-pretrained encoder improve the
best selective-guidance A variant? This is an adaptation of the representation
mismatch motivation of [DFormer++, IEEE TPAMI 2026](https://doi.org/10.1109/TPAMI.2026.3658114),
not a reproduction of its architecture, ImageNet RGB-D pretraining, or weights.

## Implementation

- Parent: A, `f92cac445986244a0ff4372ab203204efaa6fd3e`.
- Keep the A input depth enhancer and all its auxiliary losses.
- Insert a depth-only adapter before each of the four shared encoder stages,
  after patch embedding / the preceding stage downsampling.
- Each adapter: LayerNorm -> channel bottleneck -> depthwise 3x3 convolution ->
  GELU -> channel projection -> residual addition. The RGB path bypasses it.
- Bottleneck widths: 16, 24, 48, 96; total additional parameters: 202,952.
- Output projections initialize to zero. At initialization the encoder output
  matches A. Adapter construction preserves the original RNG stream, so all
  865 common state tensors match A under seed 1024.
- The original shared Mamba backbone and fourth-stage MFM remain in use.
- No stage 2/3 cross-modal residual injections from V3 are included.
- All new adapter parameters use standard Linear/Conv/LayerNorm grouping, and
  the training script asserts that every adapter parameter is in the optimizer.

## Controlled training protocol

Train from the same ImageNet VMamba pretrained checkpoint as A, not from its
50-epoch SOD weights. Epochs 50; seed 1024; input 448; batch size 2; accumulation
4; AdamW lr 1e-4, weight decay 0.01; cosine T_max=10, eta_min=1e-5; depth auxiliary
weight 0.2. Both SOD prediction losses and synthetic degradation follow A.
The inherited unscaled accumulation and optimizer grouping of original A
parameters are preserved. DataLoader workers: 4.

The existing 2985 training samples are reused. NJU2K (500 images) is used for
exploratory checkpoint selection, matching A; it is not an independent final
test set. Report the same five datasets, including averages excluding NJU2K.
These are single-seed exploratory results until replicated.

## Run

Use the existing sibling `.venv-samba` environment and the existing RGB-D and
pretraining symlinks. Launch from this worktree:

```bash
bash run_depth_adapter_v4.sh 1
```

Checkpoints: `checkpoints/Samba/rgbd_datasets/`; exact options and source commit
are written to `run_config.json` in that directory. Each worktree has separate
checkpoints, losses and logs.

## Verification

Eight real training batches, optimizer coverage of all new parameters, updates
of both adapter bottleneck and output weights at all four stages, initial
identity against the encoder bypass, unchanged A parameter initialization,
strict checkpoint reload, and real-image inference are checked with
`scripts/verify_depth_adapter.py`. Results are in `verification/adapter_smoke.json`.

The `--max_steps` smoke-only path avoids an extra optimizer step when an
accumulation boundary coincides with stopping. Full training is unchanged.
Datasets, large checkpoints and runtime outputs are not part of Git backup.

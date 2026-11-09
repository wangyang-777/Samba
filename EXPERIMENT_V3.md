# A plus depth residual fusion at stages 2 and 3

## Hypothesis and architecture

The A experiment enhances depth at the input, while the original Samba fusion
uses depth features only at stage 4. This experiment also injects enhanced-depth
features at stages 2 and 3, after their RGB saliency-guided Mamba blocks and
before the decoder. Stage 1 and the existing stage 4 Mamba fusion follow A.

For each new fusion stage:

`F_out = F_rgb + tanh(scale) * Linear(LayerNorm(F_depth))`

Features are channel-last. Stage 2 has 192 channels and stage 3 has 384 channels
(56x56 and 28x28 spatial sizes with a 448x448 input). The two modules add 186,048
parameters. Channel scales start at zero, so the initial computation matches A.
The new modules are initialized inside a CPU RNG fork to preserve initialization
of every common parameter under the same seed.

The inherited optimizer grouping helper does not collect custom scalar/vector
parameters. The two new channel-scale vectors are explicitly registered in an
additional no-decay optimizer group, and all new fusion parameters are checked
for optimizer membership before training.

This is the multiscale fusion ablation. It has learned channel scales but no
new spatial reliability predictor, boundary objective, or teacher distillation.
The input depth enhancer and its auxiliary loss are exactly those of A.

## Training protocol

Train for 50 epochs from the same ImageNet-pretrained VMamba checkpoint used by
A, with seed 1024, input 448x448, batch size 2, gradient accumulation 4,
AdamW at initial learning rate 1e-4, the inherited cosine schedule T_max=10,
and depth auxiliary weight 0.2. The trained A checkpoint is not used to initialize
this experiment, to keep the training budget comparable with A.

Use the existing 2,985-image train split. Keep the existing exploratory NJU2K
checkpoint selection, and evaluate the best checkpoint on the same five RGB-D
test sets. NJU2K participates in checkpoint selection and therefore is not an
independent final test set. The reproduction gap with the paper remains a
separate issue; this experiment isolates the added stage 2/3 depth fusion.

## Run and outputs

```bash
cd /home/wmy/project/aug_deepqulity/Samba-depth-multiscale
bash run_depth_multiscale_v3.sh 0
```

The first argument is the physical CUDA device index. Additional arguments are
forwarded to the training script. Best weights are written to
`checkpoints/Samba/rgbd_datasets/`; epoch losses and selection scores are written
to `train_loss_rgbd.txt` and `Smeasure_rgbd.txt` in this worktree.
Local datasets, pretrained weights and training outputs are excluded from Git.

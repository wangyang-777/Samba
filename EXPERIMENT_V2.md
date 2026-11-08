# Experiment A: degradation-aware selective RGB guidance

## Hypothesis

The previous enhancer perturbed depth even where RGB texture did not reflect geometry. This variant estimates a degradation map and transfers RGB high-frequency information only through a learned gate. A clean-region identity loss and a texture-leak penalty aim to preserve reliable measured depth and improve SOD boundaries.

This is an adaptation of the selective guidance idea in [SigNet, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_Completion_as_Enhancement_A_Degradation-Aware_Selective_Image_Guided_Network_for_CVPR_2025_paper.html), not a reproduction of its conditional Mamba or depth-completion pipeline. Samba remains the Mamba backbone.

## Changes from depth-quality-v1

- Seven-channel depth/RGB cues: raw depth, coarse depth, depth detail, depth edge, RGB detail, RGB edge, and edge disagreement.
- Separate predicted quality and RGB-transfer gates. The correction is quality-gated and amplitude bounded.
- An edge restoration loss in artificially damaged patches and a penalty for RGB-only edges in clean regions.
- The same synthetic corruption types and the same SOD objective as v1.

## Training protocol

Use the existing RGB-D train split and VMamba pretrained checkpoint. Train from scratch for 50 epochs with seed 1024, batch size 2, accumulation 4, input 448, and depth auxiliary weight 0.2. Use NJU2K for exploratory checkpoint selection as in v1, then report all five test sets with the same evaluation script. Because NJU2K selects the checkpoint, it is not an independent final test set.

No high-quality depth ground truth is used: the auxiliary target is the original sensor depth before synthetic corruption.

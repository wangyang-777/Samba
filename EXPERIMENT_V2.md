# Experiment B: decoupled geometry and detail repair

## Hypothesis

The previous single-scale correction did not preserve both broad geometry and object boundaries: max-F and max-E fell on all five test sets. This variant separates a low-resolution geometry branch from a full-resolution detail branch and uses depth discontinuities to restrict coarse corrections near boundaries.

The geometry/detail separation is inspired by [GDNet, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zheng_Decoupling_Fine_Detail_and_Global_Geometry_for_Compressed_Depth_Map_CVPR_2025_paper.html). The use of depth geometry as an explicit prior is conceptually related to [DFormerv2, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_DFormerv2_Geometry_Self-Attention_for_RGBD_Semantic_Segmentation_CVPR_2025_paper.html). This is a lightweight Samba adaptation, not either paper's architecture or attention mechanism.

## Changes from depth-quality-v1

- A quarter-resolution depth branch predicts a broad geometry correction.
- A full-resolution RGB/depth branch predicts local detail, controlled by a learned gate.
- A depth-edge prior suppresses coarse corrections at measured discontinuities.
- Separate low- and high-frequency restoration losses on synthetically damaged regions, plus a clean-region identity loss.
- The same synthetic corruption types and the same SOD objective as v1.

## Training protocol

Use the existing RGB-D train split and VMamba pretrained checkpoint. Train from scratch for 50 epochs with seed 1024, batch size 2, accumulation 4, input 448, and depth auxiliary weight 0.2. Use NJU2K for exploratory checkpoint selection as in v1, then report all five test sets with the same evaluation script. Because NJU2K selects the checkpoint, it is not an independent final test set.

No high-quality depth ground truth is used: the auxiliary target is the original sensor depth before synthetic corruption.

# Experiment summary

## Purpose

Run3 baseline: train with only BSRGAN+ synthetic degradations and no explicitly fully masked source-token batches.

## Model and conditioning

- Generator: `mage_vit_base_patch16` MAGE token generator with frozen VQGAN codec.
- Content condition: frozen `mocov3_vit_base` encoding of HR-A.
- Degradation condition: frozen DeResNet encoding of non-overlapping LR-B, followed by the trainable degradation projector and AdaLN conditioning.
- Source tokens: LR-A tokens, masked with a truncated-normal ratio in [0.50, 1.00]. No batches are forced fully masked.

## Training data

- Dataset: `/scratch/suleymanyildirim22/datasets/DRealSR_Train_x4`.
- Each branch uses matched source A and degradation-condition B crops. Real and synthetic branches are trained together.
- Synthetic recipe: BSRGAN+ only (100%).

## Objective

- Masked token cross-entropy: scale 1.0.
- Wavelet reconstruction of the fully predicted LR image: scale 0.06.
- Degradation loss: scale 10.0.
  - Absolute target: source LR-A DeResNet embedding.
  - Rank target: source LR-A against the opposite-domain LR-A anchor.
  - Adaptive rank margin: `min(0.10, 0.5 * anchor separation)`; rank weight: 0.25.

## Optimisation

- Per-GPU batch size: 32; gradient accumulation: 1; distributed world size: 6.
- Base learning rate: 0.00015; warm-up: 10 epochs; weight decay: 0.05; gradient clip: 3.
- Sampling configuration for previews/evaluation: temperature 6, 20 iterations.

The complete machine-readable invocation is in `args.json`; per-epoch metrics are in `log.txt`.

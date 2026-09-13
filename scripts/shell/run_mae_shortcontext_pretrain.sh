#!/bin/bash
# MAE 2D pretraining on shortcontext (200-timestep) patches — exp026.
#
# Architecture: SpectralPatchEncoder vit_depth=12
#   - time_patch=10: 200/10 = 20 temporal tokens
#   - freq_patch=20: 100/20 = 5 freq tokens
#   - Total: 20 × 5 = 100 patches (same as #023)
# Training: batch_size=1280, mask_ratio=0.5, 2d_patch masking
# Data: len200_stride40_kaggle_daily_unlabeled.zarr (~5× more patches than #023)

set -euo pipefail
cd "$(dirname "$0")/../.."

PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run python scripts/train/pretrain_mae.py \
  experiment=pretraining/spectral_patch_mae_shortcontext_daily \
  "model.backbone.vit_depth=12" \
  "+train.logger.name=mae_shortcontext_daily" \
  "+train.logger.tags=[ssl_pretrain,mae_2d,shortcontext,exp026]" \
  "+train.logger.group=pretrain_mae_shortcontext_exp026"

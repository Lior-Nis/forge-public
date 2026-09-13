#!/bin/bash
# MAE 2D pretraining on medcontext (500-timestep) data.
# Requires generate_medcontext_zarrs.sh to have been run first.
#
# Usage: bash scripts/shell/run_mae_medcontext_pretrain.sh

set -e
exec > >(tee logs/mae_medcontext_pretrain.log) 2>&1

echo "=== MAE medcontext pretraining start $(date) ==="

PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run python scripts/train/pretrain_mae.py \
    experiment=pretraining/spectral_patch_mae_medcontext_daily \
    "+train.logger.name=mae_medcontext_daily" \
    "+train.logger.tags=[ssl_pretrain,mae_2d,medcontext,exp025]"

echo "=== MAE medcontext pretraining done $(date) ==="

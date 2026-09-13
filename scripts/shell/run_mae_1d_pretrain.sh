#!/bin/bash
# MAE pretraining with 1D temporal masking.
# Part of the fair SSL comparison (#024 series) — all hyperparameters identical
# to #023 (2d_patch), only mask_mode changes.
#
# Config: spectral_patch_mae_daily (ssl_pretrain trainer: accum=4, bs=320, limit=null)
# WandB project: fog-pretraining
#
# Usage:
#   Step 1 — start pretraining (this script):
#     bash logs/run_mae_1d_pretrain.sh
#
#   Step 2 — once first epoch checkpoint appears, start probe loop in a second terminal:
#     bash scripts/probe_loop.sh \
#         --ckpt-dir checkpoints/mae/<wandb-run-name> \
#         --wandb-group probe_mae_1d \
#         --ssl-method mae_1d \
#         --experiment classification/spectral_patch_mae_finetune_defog \
#         --n-folds 3 \
#         --interval 20
#
#   Step 3 — once best epoch is identified from WandB probe results, run downstream eval:
#     bash logs/run_mae_1d_downstream.sh   (fill in CKPT and MODEL first)

set -e
exec > >(tee logs/mae_1d_pretrain.log) 2>&1

echo "=== MAE 1D pretraining start $(date) ==="

PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run python scripts/train/pretrain_mae.py \
    experiment=pretraining/spectral_patch_mae_daily \
    model.mask_mode=1d \
    "+train.logger.name=mae_1d_daily" \
    "+train.logger.tags=[ssl_pretrain,mae_1d,fogcount_eval]"

echo "=== MAE 1D pretraining done $(date) ==="

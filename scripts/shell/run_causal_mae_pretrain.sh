#!/bin/bash
# #024c — Causal MAE pretraining.
# Identical setup to #023 (ssl_pretrain trainer, bs=320, accum×4, full dataset).
# Fix applied: SpectralPatchEncoder now uses temporal causal attn_mask in causal mode
# (tokens can only attend to same/earlier time positions — bidirectional shortcut removed).
#
# Usage:
#   Step 1 — start pretraining:
#     bash scripts/shell/run_causal_mae_pretrain.sh
#
#   Step 2 — once first checkpoint appears, start probe loop in a second terminal:
#     bash scripts/shell/probe_loop.sh \
#         --ckpt-dir checkpoints/mae/causal_mae_daily \
#         --wandb-group probe_causal_mae \
#         --ssl-method causal_mae \
#         --experiment classification/spectral_patch_mae_finetune_defog \
#         --n-folds 3 \
#         --interval 20
#
#   Step 3 — once best epoch identified, run downstream eval:
#     bash scripts/shell/run_causal_mae_downstream.sh  (fill in CKPT and MODEL)

set -e
exec > >(tee logs/causal_mae_pretrain.log) 2>&1

echo "=== Causal MAE pretraining start $(date) ==="

PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run python scripts/train/pretrain_mae.py \
    experiment=pretraining/spectral_patch_causal_mae_daily \
    "+train.logger.name=causal_mae_daily" \
    "+train.logger.tags=[ssl_pretrain,causal_mae,fogcount_eval]"

echo "=== Causal MAE pretraining done $(date) ==="

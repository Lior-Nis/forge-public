#!/bin/bash
# #024d — I-JEPA pretraining.
# Identical setup to #023 (ssl_pretrain trainer, bs=320, accum×4, full dataset).
# Fix applied: EMA schedule corrected to 0.996 → 0.9999 (was 0.9999 → 0.99999).
# Previous run (#019) collapsed because EMA started too high (target barely updated).
#
# Usage:
#   Step 1 — start pretraining:
#     bash scripts/shell/run_jepa_pretrain.sh
#
#   Step 2 — once first checkpoint appears, start probe loop in a second terminal:
#     bash scripts/shell/probe_loop.sh \
#         --ckpt-dir checkpoints/jepa/jepa_daily \
#         --wandb-group probe_jepa \
#         --ssl-method jepa \
#         --experiment classification/spectral_patch_mae_finetune_defog \
#         --n-folds 3 \
#         --interval 20
#
#   Step 3 — once best epoch identified, run downstream eval:
#     bash scripts/shell/run_jepa_downstream.sh  (fill in CKPT and MODEL)

set -e
exec > >(tee logs/jepa_pretrain.log) 2>&1

echo "=== I-JEPA pretraining start $(date) ==="

PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run python scripts/train/pretrain_jepa.py \
    experiment=pretraining/spectral_patch_jepa_daily \
    "+train.logger.name=jepa_daily" \
    "+train.logger.tags=[ssl_pretrain,jepa,fogcount_eval]"

echo "=== I-JEPA pretraining done $(date) ==="

#!/bin/bash
# Downstream evaluation for MAE 1D masking pretraining.
# Mirrors run_ep7_fogcount.sh — frozen probe + direct finetune + eval,
# all on fogcount 3-fold splits.
#
# Fill in CKPT and MODEL before running:
#   CKPT: path to best epoch checkpoint (e.g. checkpoints/mae/<run>/eepoch=06.ckpt)
#   MODEL: short name used for checkpoint dirs and WandB runs (e.g. mae1d_ep6)
#
# Usage:
#   bash logs/run_mae_1d_downstream.sh

set -e
exec > >(tee logs/mae_1d_downstream.log) 2>&1

# ep7: best val_loss (0.072); ep8: marginally best probe AP (0.124) — difference negligible
CKPT="checkpoints/mae/mae_1d_daily/eepoch=07.ckpt"
MODEL="mae1d_ep07"

SPLIT_PREFIX="kaggle_labeled/kfold_defog_fogcount_3fold"

echo "=== MAE 1D downstream eval: ${MODEL} $(date) ==="
echo "    Checkpoint : ${CKPT}"
echo "    Splits     : ${SPLIT_PREFIX}0..2"

# ── 1. Frozen probe (3-fold) ────────────────────────────────
echo "--- Frozen probe (fogcount 3-fold) ---"
for fold in 0 1 2; do
    RUN_NAME="probe_${MODEL}_fogcount${fold}"
    OUT_DIR="checkpoints/classification/${RUN_NAME}"
    if [ -f "${OUT_DIR}/last.ckpt" ]; then echo "[fold $fold] skip"; continue; fi
    uv run python scripts/train/train_classification.py \
        experiment=classification/spectral_patch_mae_finetune_defog \
        "train.weights.load_from='${CKPT}'" \
        "data/splits=${SPLIT_PREFIX}${fold}" \
        "+train.logger.name=${RUN_NAME}" \
        "train.logger.tags=[probe,frozen,mae_1d,${MODEL},fogcount,fold${fold}]" \
        "data.dataloader.batch_size=128"
done

echo "--- Eval: frozen probe ---"
uv run python scripts/eval/eval_pooled_ap.py \
    --ckpt-pattern "checkpoints/classification/probe_${MODEL}_fogcount{fold}/last.ckpt" \
    --model-name "probe_${MODEL}_fogcount" \
    --split-prefix "${SPLIT_PREFIX}" \
    --n-folds 3 --output-dir logs/ssl_comparison

uv run python scripts/eval/eval_fogathome.py \
    --ckpt-pattern "checkpoints/classification/probe_${MODEL}_fogcount{fold}/last.ckpt" \
    --model-name "probe_${MODEL}_fogcount" \
    --n-folds 3 --output-dir logs/ssl_comparison

# ── 2. Direct finetuning, backbone LR=1e-6 (3-fold) ────────
echo "--- Direct finetuning backbone LR=1e-6 (fogcount 3-fold) ---"
for fold in 0 1 2; do
    RUN_NAME="finetune_${MODEL}_direct_fogcount${fold}"
    OUT_DIR="checkpoints/classification/${RUN_NAME}"
    if [ -f "${OUT_DIR}/last.ckpt" ]; then echo "[fold $fold] skip"; continue; fi
    uv run python scripts/train/train_classification.py \
        experiment=classification/spectral_patch_mae_finetune_defog \
        "train.weights.load_from='${CKPT}'" \
        "train.weights.freeze_backbone=false" \
        "train.weights.unfreeze_schedule.enabled=false" \
        "train/optimizer=adamw_differential_lr_1e6" \
        "data/splits=${SPLIT_PREFIX}${fold}" \
        "+train.logger.name=${RUN_NAME}" \
        "train.logger.tags=[finetune,unfrozen,mae_1d,${MODEL},fogcount,fold${fold},lr_1e6]" \
        "data.dataloader.batch_size=128" \
        "train.trainer.max_epochs=50" \
        "train.scheduler.T_max=50"
done

echo "--- Eval: direct finetuning 1e-6 ---"
uv run python scripts/eval/eval_pooled_ap.py \
    --ckpt-pattern "checkpoints/classification/finetune_${MODEL}_direct_fogcount{fold}/last.ckpt" \
    --model-name "finetune_${MODEL}_direct_fogcount" \
    --split-prefix "${SPLIT_PREFIX}" \
    --n-folds 3 --output-dir logs/ssl_comparison

uv run python scripts/eval/eval_fogathome.py \
    --ckpt-pattern "checkpoints/classification/finetune_${MODEL}_direct_fogcount{fold}/last.ckpt" \
    --model-name "finetune_${MODEL}_direct_fogcount" \
    --n-folds 3 --output-dir logs/ssl_comparison

echo "=== ALL DONE $(date) ==="
echo "Results in logs/ssl_comparison/"

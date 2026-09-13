#!/bin/bash
# #024b — Random init vit12 baseline (no pretraining).
# Provides the floor for the fair SSL comparison (#024 series).
# Same downstream setup as all other methods: fogcount 3-fold, backbone_lr=1e-6.
#
# Usage:
#   bash scripts/shell/run_random_init_downstream.sh

set -e
exec > >(tee logs/random_init_downstream.log) 2>&1

MODEL="random_init_vit12"
SPLIT_PREFIX="kaggle_labeled/kfold_defog_fogcount_3fold"

echo "=== Random init downstream eval: ${MODEL} $(date) ==="
echo "    No checkpoint — random initialization"
echo "    Splits     : ${SPLIT_PREFIX}0..2"

# ── 1. Frozen probe (3-fold) ────────────────────────────────
echo "--- Frozen probe (fogcount 3-fold) ---"
for fold in 0 1 2; do
    RUN_NAME="probe_${MODEL}_fogcount${fold}"
    OUT_DIR="checkpoints/classification/${RUN_NAME}"
    if [ -f "${OUT_DIR}/last.ckpt" ]; then echo "[fold $fold] skip"; continue; fi
    uv run python scripts/train/train_classification.py \
        experiment=classification/spectral_patch_mae_finetune_defog \
        "train.weights.load_from=null" \
        "data/splits=${SPLIT_PREFIX}${fold}" \
        "+train.logger.name=${RUN_NAME}" \
        "train.logger.tags=[probe,frozen,random_init,${MODEL},fogcount,fold${fold}]" \
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
        "train.weights.load_from=null" \
        "train.weights.freeze_backbone=false" \
        "train.weights.unfreeze_schedule.enabled=false" \
        "train/optimizer=adamw_differential_lr_1e6" \
        "data/splits=${SPLIT_PREFIX}${fold}" \
        "+train.logger.name=${RUN_NAME}" \
        "train.logger.tags=[finetune,unfrozen,random_init,${MODEL},fogcount,fold${fold},lr_1e6]" \
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

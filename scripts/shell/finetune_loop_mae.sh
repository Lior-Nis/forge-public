#!/bin/bash
# Full finetuning (unfrozen backbone) for one SSL checkpoint, with differential LR.
#
# Usage:
#   bash scripts/finetune_loop_mae.sh <ckpt_path> <model_name> [n_folds] [split_prefix] [experiment] [ssl_method]
#
# Examples:
#   # MAE ep7, fogcount 3-fold (primary evaluation)
#   bash scripts/finetune_loop_mae.sh checkpoints/mae/worldly-moon-229/eepoch=07.ckpt vit12_ep7
#
#   # SimCLR checkpoint
#   bash scripts/finetune_loop_mae.sh checkpoints/simclr/run-xyz/last.ckpt simclr_run 3 \
#       kaggle_labeled/kfold_defog_fogcount_3fold \
#       classification/spectral_patch_simclr_finetune_defog \
#       simclr
#
# Defaults: n_folds=3, split=fogcount 3-fold, experiment=mae_finetune_defog, ssl_method=mae_2d
# WandB tags auto-derived: finetune, unfrozen, <ssl_method>, ep<N>, fogcount/fogstrat, fold<N>, lr_1e6

set -e

CKPT_PATH="$1"
MODEL_NAME="$2"
N_FOLDS="${3:-3}"
SPLIT_PREFIX="${4:-kaggle_labeled/kfold_defog_fogcount_3fold}"
EXPERIMENT="${5:-classification/spectral_patch_mae_finetune_defog}"
SSL_METHOD="${6:-mae_2d}"

# Derive epoch tag from checkpoint filename (e.g. eepoch=07.ckpt → ep7)
EPOCH_NUM=$(echo "$CKPT_PATH" | grep -oP '(?<=epoch=)\d+' | sed 's/^0*//' || echo "")
EPOCH_TAG="${EPOCH_NUM:+ep${EPOCH_NUM}}"
EPOCH_TAG="${EPOCH_TAG:-last}"

# Derive split tag from split prefix
if echo "$SPLIT_PREFIX" | grep -q "fogcount"; then
    SPLIT_TAG="fogcount"
elif echo "$SPLIT_PREFIX" | grep -q "fogstrat"; then
    SPLIT_TAG="fogstrat"
else
    SPLIT_TAG="unknown_split"
fi

echo "=== Finetuning $MODEL_NAME from $CKPT_PATH ==="
echo "    SSL method : ${SSL_METHOD}"
echo "    Epoch      : ${EPOCH_TAG}"
echo "    Splits     : ${SPLIT_PREFIX}0..$((N_FOLDS-1))  [${SPLIT_TAG}]"
echo "    Experiment : ${EXPERIMENT}"

for fold in $(seq 0 $((N_FOLDS - 1))); do
    OUT_DIR="checkpoints/classification/finetune_${MODEL_NAME}_fold${fold}"
    if [ -f "${OUT_DIR}/last.ckpt" ]; then
        echo "[fold $fold] already done, skipping"
        continue
    fi
    echo "[fold $fold] training..."
    uv run python scripts/train/train_classification.py \
        experiment="${EXPERIMENT}" \
        "train.weights.load_from='${CKPT_PATH}'" \
        "train.weights.freeze_backbone=false" \
        "train.weights.unfreeze_schedule.enabled=false" \
        "train/optimizer=adamw_differential_lr_1e6" \
        "data/splits=${SPLIT_PREFIX}${fold}" \
        "+train.logger.name=finetune_${MODEL_NAME}_fold${fold}" \
        "train.logger.tags=[finetune,unfrozen,${SSL_METHOD},${EPOCH_TAG},${SPLIT_TAG},fold${fold},lr_1e6]" \
        "data.dataloader.batch_size=128" \
        "train.trainer.max_epochs=50" \
        "train.scheduler.T_max=50"
    echo "[fold $fold] done"
done

echo "=== All folds done for finetune $MODEL_NAME ==="

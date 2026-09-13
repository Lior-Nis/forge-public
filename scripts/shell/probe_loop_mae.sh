#!/bin/bash
# Run k-fold linear probes for one SSL checkpoint (frozen backbone).
#
# Usage:
#   bash scripts/probe_loop_mae.sh <ckpt_path> <model_name> [n_folds] [split_prefix] [ssl_method] [experiment]
#
# Examples:
#   bash scripts/probe_loop_mae.sh checkpoints/mae/worldly-moon-229/eepoch=07.ckpt vit12_ep7
#   bash scripts/probe_loop_mae.sh checkpoints/simclr/run-xyz/last.ckpt simclr_run 3 \
#       kaggle_labeled/kfold_defog_fogcount_3fold simclr \
#       classification/spectral_patch_simclr_finetune_defog
#
# WandB tags auto-derived: probe, frozen, <ssl_method>, ep<N>, fogcount/fogstrat, fold<N>

set -e

CKPT_PATH="$1"
MODEL_NAME="$2"
N_FOLDS="${3:-3}"
SPLIT_PREFIX="${4:-kaggle_labeled/kfold_defog_fogcount_3fold}"
SSL_METHOD="${5:-mae_2d}"
EXPERIMENT="${6:-classification/spectral_patch_mae_finetune_defog}"

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

echo "=== Probing $MODEL_NAME from $CKPT_PATH ==="
echo "    SSL method : ${SSL_METHOD}"
echo "    Epoch      : ${EPOCH_TAG}"
echo "    Splits     : ${SPLIT_PREFIX}0..$((N_FOLDS-1))  [${SPLIT_TAG}]"
echo "    Experiment : ${EXPERIMENT}"

for fold in $(seq 0 $((N_FOLDS - 1))); do
    OUT_DIR="checkpoints/classification/probe_${MODEL_NAME}_fold${fold}"
    if [ -f "${OUT_DIR}/last.ckpt" ]; then
        echo "[fold $fold] already done, skipping"
        continue
    fi
    echo "[fold $fold] training..."
    uv run python scripts/train/train_classification.py \
        experiment="${EXPERIMENT}" \
        "train.weights.load_from='${CKPT_PATH}'" \
        "data/splits=${SPLIT_PREFIX}${fold}" \
        "+train.logger.name=probe_${MODEL_NAME}_fold${fold}" \
        "train.logger.tags=[probe,frozen,${SSL_METHOD},${EPOCH_TAG},${SPLIT_TAG},fold${fold}]" \
        "data.dataloader.batch_size=512"
    echo "[fold $fold] done"
done

echo "=== All folds done for $MODEL_NAME ==="

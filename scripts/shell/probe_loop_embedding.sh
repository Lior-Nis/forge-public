#!/bin/bash
# Precompute embeddings once, then train GRU head per fold — GPU-efficient probe.
# Usage: bash scripts/probe_loop_embedding.sh <ckpt_path> <model_name> [n_folds]
# Example:
#   bash scripts/probe_loop_embedding.sh checkpoints/mae/worldly-moon-229/eepoch=00.ckpt vit12_ep6 5

set -e

CKPT_PATH="$1"
MODEL_NAME="$2"
N_FOLDS="${3:-5}"
REF_CKPT="${4:-checkpoints/classification/probe_vit12_ep6_fold0/last.ckpt}"
EMB_DIR="checkpoints/embeddings/${MODEL_NAME}"
OUT_BASE="checkpoints/classification"

echo "=== Embedding probe: $MODEL_NAME ==="

# Step 1: precompute embeddings for all folds in one pass (model loaded once)
echo "[precompute] all ${N_FOLDS} folds..."
uv run python scripts/embed/precompute_mae_embeddings.py \
    --mae-ckpt "${CKPT_PATH}" \
    --ref-ckpt "${REF_CKPT}" \
    --output-dir "${EMB_DIR}" \
    --n-folds "${N_FOLDS}" \
    --gpu-batch-size 1024 \
    --io-batch-size 256
echo "[precompute] done"

# Step 2: train GRU head on precomputed embeddings (skips if done)
for fold in $(seq 0 $((N_FOLDS - 1))); do
    OUT_DIR="${OUT_BASE}/emb_probe_${MODEL_NAME}_fold${fold}"
    if [ -f "${OUT_DIR}/last.ckpt" ]; then
        echo "[fold $fold] head already trained, skipping"
        continue
    fi
    echo "[fold $fold] training GRU head..."
    uv run python scripts/embed/train_embedding_probe.py \
        --emb-dir "${EMB_DIR}" \
        --fold "${fold}" \
        --model-name "${MODEL_NAME}" \
        --output-dir "${OUT_DIR}" \
        --epochs 30 \
        --batch-size 2048 \
        --lr 1e-3
    echo "[fold $fold] done"
done

echo "=== All folds done for $MODEL_NAME ==="

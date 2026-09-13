#!/usr/bin/env bash
# run_forge_fogcount_probes.sh — NIX-163
#
# Frozen linear probes for all 7 FORGE epoch checkpoints using fogcount 3-fold splits.
# Consolidates all FORGE evaluations onto the single fogcount protocol for paper Tables 1–3.
#
# FORGE checkpoint mapping (local epoch → global epoch):
#   colorful-blaze-219  eepoch=00 → ep1
#   lunar-lake-221      eepoch=00 → ep3
#   lunar-lake-221      eepoch=01 → ep4
#   lunar-lake-221      eepoch=02 → ep5
#   worldly-moon-229    eepoch=00 → ep6
#   worldly-moon-229    eepoch=01 → ep7  ← already done
#   worldly-moon-229    eepoch=02 → ep8
#
# Probe config: SpectralPatchEncoder vit12 (frozen), BiGRU head, 30 epochs,
#               AdamW LR=1e-3, batch=128, focal α=[0.25,2.0] γ=2.0, fogcount 3-fold.
#               (Matches existing ep7 fogcount probes for consistency — NIX-163 inherits these params.)
#
# Usage:
#   bash scripts/shell/run_forge_fogcount_probes.sh
#   bash scripts/shell/run_forge_fogcount_probes.sh --epochs ep1,ep3   # subset
#   bash scripts/shell/run_forge_fogcount_probes.sh --force             # re-run existing

set -euo pipefail

FORCE=false
EPOCHS_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)        FORCE=true; shift ;;
        --epochs)       EPOCHS_FILTER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Checkpoint map: global_epoch → path ─────────────────────────────────────
declare -A CKPTS
CKPTS["ep1"]="checkpoints/mae/colorful-blaze-219/eepoch=00.ckpt"
CKPTS["ep3"]="checkpoints/mae/lunar-lake-221/eepoch=00.ckpt"
CKPTS["ep4"]="checkpoints/mae/lunar-lake-221/eepoch=01.ckpt"
CKPTS["ep5"]="checkpoints/mae/lunar-lake-221/eepoch=02.ckpt"
CKPTS["ep6"]="checkpoints/mae/worldly-moon-229/eepoch=00.ckpt"
CKPTS["ep7"]="checkpoints/mae/worldly-moon-229/eepoch=01.ckpt"
CKPTS["ep8"]="checkpoints/mae/worldly-moon-229/eepoch=02.ckpt"

# Ordered epoch list
EPOCHS=(ep1 ep3 ep4 ep5 ep6 ep7 ep8)

N_FOLDS=3
SPLIT_PREFIX="kaggle_labeled/kfold_defog_fogcount_3fold"
EXPERIMENT="classification/spectral_patch_mae_finetune_defog"
WANDB_GROUP="forge_fogcount_probe_trajectory"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Filter epochs if requested ───────────────────────────────────────────────
if [[ -n "$EPOCHS_FILTER" ]]; then
    IFS=',' read -ra EPOCHS <<< "$EPOCHS_FILTER"
fi

log "=== FORGE fogcount probe sweep (NIX-163) ==="
log "  Epochs   : ${EPOCHS[*]}"
log "  Folds    : fogcount 0–$((N_FOLDS-1))"
log "  Group    : $WANDB_GROUP"
log "  Force    : $FORCE"
echo ""

# ── Main loop ────────────────────────────────────────────────────────────────
for epoch in "${EPOCHS[@]}"; do
    CKPT="${CKPTS[$epoch]}"

    if [[ ! -f "$CKPT" ]]; then
        log "WARNING: $epoch checkpoint not found at $CKPT — skipping"
        continue
    fi

    log "── $epoch ── ($CKPT)"

    for fold in $(seq 0 $((N_FOLDS - 1))); do
        RUN_NAME="probe_vit12_${epoch}_fogcount${fold}"
        OUT_DIR="checkpoints/classification/${RUN_NAME}"

        if [[ "$FORCE" = false && -f "${OUT_DIR}/last.ckpt" ]]; then
            log "  [fold $fold] already done ($OUT_DIR) — skipping"
            continue
        fi

        log "  [fold $fold] training → $RUN_NAME ..."

        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        uv run python scripts/train/train_classification.py \
            experiment="${EXPERIMENT}" \
            "train.weights.load_from='${CKPT}'" \
            "data/splits=${SPLIT_PREFIX}${fold}" \
            "+train.logger.name=${RUN_NAME}" \
            "+train.logger.group=${WANDB_GROUP}" \
            "train.logger.tags=[probe,frozen,mae_2d,${epoch},fogcount,fold${fold}]" \
            "data.dataloader.batch_size=128" \
            "hydra.run.dir=logs/forge_fogcount_probes/${epoch}_fold${fold}"

        log "  [fold $fold] done."
    done

    log "$epoch complete."
    echo ""
done

log "=== All FORGE fogcount probes done ==="
log ""
log "Next step: bash scripts/shell/eval_forge_fogcount.sh"

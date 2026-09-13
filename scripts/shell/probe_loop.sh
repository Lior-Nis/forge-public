#!/usr/bin/env bash
# ============================================================
# probe_loop.sh — Periodic linear probe on pretraining checkpoints
#
# Polls a checkpoint directory every INTERVAL minutes.
# For each new (unprobed) epoch checkpoint, runs a 3-fold
# linear probe and logs results to WandB.
#
# Usage:
#   bash scripts/probe_loop.sh \
#       --ckpt-dir checkpoints/mae/sleek-monkey-199 \
#       --wandb-group probe_sleek-monkey-199 \
#       --ssl-method mae_2d \
#       [--interval 15] \
#       [--experiment classification/spectral_patch_mae_finetune_defog] \
#       [--n-folds 3] \
#       [--split-prefix kaggle_labeled/kfold_defog_fogcount_3fold]
#
# WandB tags auto-derived per checkpoint: probe, frozen, <ssl_method>, ep<N>, fogcount/fogstrat, fold<N>
# ============================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────
INTERVAL=15          # minutes between polls
N_FOLDS=3
BATCH_SIZE=256
EXPERIMENT="classification/spectral_patch_mae_finetune_defog"
SPLIT_PREFIX="kaggle_labeled/kfold_defog_fogcount_3fold"
SSL_METHOD="mae_2d"

# ── Argument parsing ─────────────────────────────────────────
CKPT_DIR=""
WANDB_GROUP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt-dir)       CKPT_DIR="$2";       shift 2 ;;
        --wandb-group)    WANDB_GROUP="$2";    shift 2 ;;
        --ssl-method)     SSL_METHOD="$2";     shift 2 ;;
        --interval)       INTERVAL="$2";       shift 2 ;;
        --experiment)     EXPERIMENT="$2";     shift 2 ;;
        --n-folds)        N_FOLDS="$2";        shift 2 ;;
        --split-prefix)   SPLIT_PREFIX="$2";   shift 2 ;;
        --batch-size)     BATCH_SIZE="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CKPT_DIR" || -z "$WANDB_GROUP" ]]; then
    echo "Usage: $0 --ckpt-dir <dir> --wandb-group <group> --ssl-method <method> [options]"
    exit 1
fi

# ── Derive split tag from prefix ─────────────────────────────
if echo "$SPLIT_PREFIX" | grep -q "fogcount"; then
    SPLIT_TAG="fogcount"
elif echo "$SPLIT_PREFIX" | grep -q "fogstrat"; then
    SPLIT_TAG="fogstrat"
else
    SPLIT_TAG="unknown_split"
fi

# ── State tracking ───────────────────────────────────────────
STATE_FILE="${CKPT_DIR}/.probed_checkpoints"
touch "$STATE_FILE"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

probe_checkpoint() {
    local ckpt="$1"
    local epoch="$2"

    log "Probing checkpoint: $(basename "$ckpt") (epoch $epoch)"

    # Normalize epoch tag (strip leading zeros: 07 → 7)
    local epoch_num epoch_tag
    epoch_num=$(echo "$epoch" | sed 's/^0*//')
    epoch_tag="ep${epoch_num:-0}"

    for fold in $(seq 0 $((N_FOLDS - 1))); do
        log "  Running fold $fold ..."
        local run_name="probe_ep${epoch}_fold${fold}"

        PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/train/train_classification.py \
            experiment="${EXPERIMENT}" \
            "data/splits=${SPLIT_PREFIX}${fold}" \
            "train.weights.load_from='${ckpt}'" \
            "+train.logger.name=${run_name}" \
            "+train.logger.group=${WANDB_GROUP}" \
            "train.logger.tags=[probe,frozen,${SSL_METHOD},${epoch_tag},${SPLIT_TAG},fold${fold}]" \
            "hydra.run.dir=logs/probe_loop/${WANDB_GROUP}/${run_name}" \
            "data.dataloader.batch_size=${BATCH_SIZE}" \
            "model.backbone.vit_depth=12" \
            2>&1 | tail -5

        if [[ $? -ne 0 ]]; then
            log "  WARNING: fold $fold failed — skipping remaining folds for this checkpoint"
            return 1
        fi

        log "  Fold $fold done."
    done

    log "Checkpoint epoch $epoch complete — all $N_FOLDS folds done."
}

# ── Main poll loop ───────────────────────────────────────────
log "Starting probe loop"
log "  Checkpoint dir : $CKPT_DIR"
log "  WandB group    : $WANDB_GROUP"
log "  SSL method     : $SSL_METHOD"
log "  Experiment     : $EXPERIMENT"
log "  Split prefix   : $SPLIT_PREFIX (folds 0-$((N_FOLDS-1)))  [$SPLIT_TAG]"
log "  Poll interval  : ${INTERVAL}m"
log "  State file     : $STATE_FILE"
log "  WandB tags     : probe,frozen,${SSL_METHOD},ep<N>,${SPLIT_TAG},fold<N>"
echo ""

while true; do
    # Find epoch checkpoints (not last.ckpt — it's a moving target)
    while IFS= read -r ckpt; do
        basename_ckpt=$(basename "$ckpt")

        # Skip if already probed
        if grep -qF "$basename_ckpt" "$STATE_FILE"; then
            continue
        fi

        # Extract epoch number from filename (e.g. eepoch=07.ckpt → 07)
        epoch=$(echo "$basename_ckpt" | grep -oP '(?<=epoch=)\d+' || echo "??")

        probe_checkpoint "$ckpt" "$epoch"

        # Mark as probed regardless of outcome (avoid retry loops on bad checkpoints)
        echo "$basename_ckpt" >> "$STATE_FILE"

    done < <(find "$CKPT_DIR" -maxdepth 1 -name "*.ckpt" ! -name "last*.ckpt" | sort)

    log "Poll done. Next check in ${INTERVAL}m. (probed: $(wc -l < "$STATE_FILE") checkpoints)"
    sleep $((INTERVAL * 60))
done

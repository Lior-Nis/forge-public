#!/usr/bin/env bash
# eval_forge_fogcount.sh — NIX-163
#
# Evaluates all FORGE epoch probes trained on fogcount 3-fold:
#   1. Defog pooled AP  (eval_pooled_ap.py)
#   2. FogAtHome AP     (eval_fogathome.py)
#
# Requires probe checkpoints produced by run_forge_fogcount_probes.sh.
# Results are written to logs/forge_fogcount_eval/<epoch>/.
#
# Usage:
#   bash scripts/shell/eval_forge_fogcount.sh
#   bash scripts/shell/eval_forge_fogcount.sh --epochs ep7     # single epoch

set -euo pipefail

EPOCHS_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs) EPOCHS_FILTER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

EPOCHS=(ep1 ep3 ep4 ep5 ep6 ep7 ep8)
N_FOLDS=3
SPLIT_PREFIX="kaggle_labeled/kfold_defog_fogcount_3fold"
EVAL_DIR="logs/forge_fogcount_eval"

if [[ -n "$EPOCHS_FILTER" ]]; then
    IFS=',' read -ra EPOCHS <<< "$EPOCHS_FILTER"
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== FORGE fogcount evaluation sweep (NIX-163) ==="
log "  Epochs : ${EPOCHS[*]}"
log "  Output : $EVAL_DIR"
echo ""

for epoch in "${EPOCHS[@]}"; do
    CKPT_PATTERN="checkpoints/classification/probe_vit12_${epoch}_fogcount{fold}/last.ckpt"
    OUT_DIR="${EVAL_DIR}/${epoch}"

    # Check at least fold 0 exists
    if [[ ! -f "checkpoints/classification/probe_vit12_${epoch}_fogcount0/last.ckpt" ]]; then
        log "WARNING: $epoch probe checkpoints not found — skipping (run run_forge_fogcount_probes.sh first)"
        continue
    fi

    log "── $epoch ──"
    mkdir -p "$OUT_DIR"

    log "  [defog pooled AP] ..."
    uv run python scripts/eval/eval_pooled_ap.py \
        --model-name "vit12_${epoch}_fogcount" \
        --ckpt-pattern "$CKPT_PATTERN" \
        --split-prefix "$SPLIT_PREFIX" \
        --n-folds "$N_FOLDS" \
        --output-dir "$OUT_DIR"

    log "  [FogAtHome AP] ..."
    uv run python scripts/eval/eval_fogathome.py \
        --ckpt-pattern "$CKPT_PATTERN" \
        --model-name "vit12_${epoch}_fogcount" \
        --n-folds "$N_FOLDS" \
        --output-dir "$OUT_DIR"

    log "$epoch eval done → $OUT_DIR"
    echo ""
done

log "=== All evaluations complete ==="
log ""
log "Collect results:"
log "  grep 'pooled_ap\|fogathome_ap' ${EVAL_DIR}/*/summary.txt 2>/dev/null || true"
log ""
log "Update PROGRESS.md epoch_trajectory table with fogcount results (NIX-163)."

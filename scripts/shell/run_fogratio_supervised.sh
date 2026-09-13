#!/bin/bash
# Soft-label supervised baselines — fog_ratio BCE targets.
#
# Trains SpectralPatchEncoder from scratch (no pretraining) for LC, MC, SC using
# fog_ratio soft labels (BCEWithLogitsLoss against continuous fog_ratio target).
# Architecture identical to probe/finetune counterparts (vit_depth=4).
#
# Usage: bash scripts/shell/run_fogratio_supervised.sh [lc|mc|sc|all]

set -e
cd "$(dirname "$0")/../.."

CONTEXT="${1:-all}"

LC_EXP="classification/supervised_lc_fogr025_defog"
MC_EXP="classification/supervised_mc_fogr025_defog"
SC_EXP="classification/supervised_sc_fogr025_defog"

LC_SPLIT="kaggle_labeled/kfold_defog_fogcount_valid_lc"
MC_SPLIT="kaggle_labeled/kfold_defog_fogcount_valid_mc"
SC_SPLIT="kaggle_labeled/kfold_defog_fogcount_valid_sc"

LC_BS=32
MC_BS=128
SC_BS=256

run_supervised() {
    local name="$1" exp="$2" split="$3" bs="$4"
    echo ""
    echo "=========================================="
    echo "SUPERVISED (soft): $name  (batch_size=$bs)"
    echo "=========================================="
    for fold in 0 1 2; do
        OUT_DIR="checkpoints/classification/soft_supervised_${name}_fold${fold}"
        if [ -f "${OUT_DIR}/last.ckpt" ]; then
            echo "[fold $fold] already done, skipping"
            continue
        fi
        echo "[fold $fold] training..."
        uv run python scripts/train/train_classification.py \
            experiment="${exp}" \
            "data/dataset=fog_ratio_dataset" \
            "train/loss=fog_ratio" \
            "data/splits=${split}${fold}" \
            "+train.logger.name=soft_supervised_${name}_fold${fold}" \
            "train.logger.tags=[supervised,scratch,fog_ratio,fogcount,fold${fold}]" \
            "data.dataloader.batch_size=${bs}"
        echo "[fold $fold] done"
    done
    echo "=== Supervised done: $name ==="
}

if [[ "$CONTEXT" == "lc" || "$CONTEXT" == "all" ]]; then
    run_supervised "lc" "$LC_EXP" "$LC_SPLIT" "$LC_BS"
fi

if [[ "$CONTEXT" == "mc" || "$CONTEXT" == "all" ]]; then
    run_supervised "mc" "$MC_EXP" "$MC_SPLIT" "$MC_BS"
fi

if [[ "$CONTEXT" == "sc" || "$CONTEXT" == "all" ]]; then
    run_supervised "sc" "$SC_EXP" "$SC_SPLIT" "$SC_BS"
fi

echo ""
echo "=== All supervised runs complete $(date) ==="
echo ""
echo "Next: clear old pred CSV caches and run comprehensive eval"
echo "  rm -f logs/comprehensive_eval_cache/*_fold*_preds.csv"
echo "  uv run python scripts/eval/eval_comprehensive.py"

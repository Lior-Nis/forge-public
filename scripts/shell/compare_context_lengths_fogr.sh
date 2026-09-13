#!/bin/bash
# Fair context-length comparison — fog_ratio labeling, validity-only filter.
#
# Runs probe (frozen) + finetune (unfrozen) for all three context lengths:
#   LC  — longcontext  (1000-step, sleek-monkey-199 ep99,   vit_depth=4)
#   MC  — medcontext   (500-step,  mae_medcontext_daily ep47, vit_depth=4)
#   SC  — shortcontext (200-step,  mae_shortcontext_daily ep9, vit_depth=12)
#
# NOTE: The SC experiment config (spectral_patch_mae_sc_valid_defog.yaml) sets
#   vit_depth=12 to match the pretrained checkpoint. Previous runs used depth=4,
#   which silently loaded only 4/12 layers. Re-run SC to get valid results.
#
# Key design choices:
#   - validity==1.0 filter only (no purity) — fog_ratio handles mixed windows
#   - Same 57 defog patients, 19/19/19 serpentine-fog-count split
#   - binary_any_fog classification target from frame-level labels
#
# Usage: bash scripts/shell/compare_context_lengths_fogr.sh [lc|mc|sc|all] [probe|finetune|both]

set -e
cd "$(dirname "$0")/../.."

CONTEXT="${1:-all}"
PHASE="${2:-both}"

LC_CKPT="checkpoints/mae/sleek-monkey-199/last.ckpt"
MC_CKPT="checkpoints/mae/mae_medcontext_daily/mae_medcontext_daily/last.ckpt"
SC_CKPT="checkpoints/mae/mae_shortcontext_daily/mae_shortcontext_daily/last.ckpt"

LC_EXP="classification/spectral_patch_mae_lc_valid_defog"
MC_EXP="classification/spectral_patch_mae_mc_valid_defog"
SC_EXP="classification/spectral_patch_mae_sc_valid_defog"

LC_SPLIT="kaggle_labeled/kfold_defog_fogcount_valid_lc"
MC_SPLIT="kaggle_labeled/kfold_defog_fogcount_valid_mc"
SC_SPLIT="kaggle_labeled/kfold_defog_fogcount_valid_sc"

# Batch sizes: LC wavelet 1000-step is ~25× memory cost of SC 200-step
LC_PROBE_BS=64
MC_PROBE_BS=256
SC_PROBE_BS=512

LC_FT_BS=32
MC_FT_BS=128
SC_FT_BS=128

run_probe() {
    local ckpt="$1" name="$2" exp="$3" split="$4" bs="$5"
    echo ""
    echo "=========================================="
    echo "PROBE: $name  (batch_size=$bs)"
    echo "=========================================="
    for fold in 0 1 2; do
        OUT_DIR="checkpoints/classification/probe_${name}_fold${fold}"
        if [ -f "${OUT_DIR}/last.ckpt" ]; then
            echo "[fold $fold] already done, skipping"
            continue
        fi
        echo "[fold $fold] training..."
        uv run python scripts/train/train_classification.py \
            experiment="${exp}" \
            "train.weights.load_from='${ckpt}'" \
            "data/splits=${split}${fold}" \
            "+train.logger.name=probe_${name}_fold${fold}" \
            "train.logger.tags=[probe,frozen,mae_2d,fogr_valid,fogcount,fold${fold}]" \
            "data.dataloader.batch_size=${bs}"
        echo "[fold $fold] done"
    done
    echo "=== Probe done: $name ==="
}

run_finetune() {
    local ckpt="$1" name="$2" exp="$3" split="$4" bs="$5"
    echo ""
    echo "=========================================="
    echo "FINETUNE: $name  (batch_size=$bs)"
    echo "=========================================="
    for fold in 0 1 2; do
        OUT_DIR="checkpoints/classification/finetune_${name}_fold${fold}"
        if [ -f "${OUT_DIR}/last.ckpt" ]; then
            echo "[fold $fold] already done, skipping"
            continue
        fi
        echo "[fold $fold] training..."
        uv run python scripts/train/train_classification.py \
            experiment="${exp}" \
            "train.weights.load_from='${ckpt}'" \
            "train.weights.freeze_backbone=false" \
            "train.weights.unfreeze_schedule.enabled=false" \
            "train/optimizer=adamw_differential_lr_1e6" \
            "data/splits=${split}${fold}" \
            "+train.logger.name=finetune_${name}_fold${fold}" \
            "train.logger.tags=[finetune,unfrozen,mae_2d,fogr_valid,fogcount,fold${fold},lr_1e6]" \
            "data.dataloader.batch_size=${bs}" \
            "train.trainer.max_epochs=50" \
            "train.scheduler.T_max=50"
        echo "[fold $fold] done"
    done
    echo "=== Finetune done: $name ==="
}

# LC
if [[ "$CONTEXT" == "lc" || "$CONTEXT" == "all" ]]; then
    [[ "$PHASE" == "probe"    || "$PHASE" == "both" ]] && run_probe    "$LC_CKPT" "lc_fogr_ep99" "$LC_EXP" "$LC_SPLIT" "$LC_PROBE_BS"
    [[ "$PHASE" == "finetune" || "$PHASE" == "both" ]] && run_finetune "$LC_CKPT" "lc_fogr_ep99" "$LC_EXP" "$LC_SPLIT" "$LC_FT_BS"
fi

# MC
if [[ "$CONTEXT" == "mc" || "$CONTEXT" == "all" ]]; then
    [[ "$PHASE" == "probe"    || "$PHASE" == "both" ]] && run_probe    "$MC_CKPT" "mc_fogr_ep47" "$MC_EXP" "$MC_SPLIT" "$MC_PROBE_BS"
    [[ "$PHASE" == "finetune" || "$PHASE" == "both" ]] && run_finetune "$MC_CKPT" "mc_fogr_ep47" "$MC_EXP" "$MC_SPLIT" "$MC_FT_BS"
fi

# SC
if [[ "$CONTEXT" == "sc" || "$CONTEXT" == "all" ]]; then
    [[ "$PHASE" == "probe"    || "$PHASE" == "both" ]] && run_probe    "$SC_CKPT" "sc_fogr_ep9"  "$SC_EXP" "$SC_SPLIT" "$SC_PROBE_BS"
    [[ "$PHASE" == "finetune" || "$PHASE" == "both" ]] && run_finetune "$SC_CKPT" "sc_fogr_ep9"  "$SC_EXP" "$SC_SPLIT" "$SC_FT_BS"
fi

echo ""
echo "=== All runs complete. FogAtHome eval commands: ==="
echo ""
echo "# LC probe"
echo "uv run python scripts/eval/eval_fogathome.py \\"
echo "  --ckpt-pattern 'checkpoints/classification/probe_lc_fogr_ep99_fold{fold}/probe_lc_fogr_ep99_fold{fold}/best.ckpt' \\"
echo "  --model-name lc_fogr_ep99_probe --n-folds 3 \\"
echo "  --dataset-path data/processed/len1000_stride200_fogathome.zarr \\"
echo "  --patch-window-s 10.0 --patch-stride-s 2.0 \\"
echo "  --threshold-method youden --bootstrap --compute-icc"
echo ""
echo "# MC probe"
echo "uv run python scripts/eval/eval_fogathome.py \\"
echo "  --ckpt-pattern 'checkpoints/classification/probe_mc_fogr_ep47_fold{fold}/probe_mc_fogr_ep47_fold{fold}/best.ckpt' \\"
echo "  --model-name mc_fogr_ep47_probe --n-folds 3 \\"
echo "  --dataset-path data/processed/len500_stride200_fogstride100_anyfog_fogathome.zarr \\"
echo "  --patch-window-s 5.0 --patch-stride-s 2.0 \\"
echo "  --threshold-method youden --bootstrap --compute-icc"
echo ""
echo "# SC probe"
echo "uv run python scripts/eval/eval_fogathome.py \\"
echo "  --ckpt-pattern 'checkpoints/classification/probe_sc_fogr_ep9_fold{fold}/probe_sc_fogr_ep9_fold{fold}/best.ckpt' \\"
echo "  --model-name sc_fogr_ep9_probe --n-folds 3 \\"
echo "  --dataset-path data/processed/len200_stride20_fogstride10_anyfog_fogathome.zarr \\"
echo "  --patch-window-s 2.0 --patch-stride-s 0.2 \\"
echo "  --threshold-method youden --bootstrap --compute-icc"

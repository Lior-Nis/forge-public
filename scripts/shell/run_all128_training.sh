#!/bin/bash
# Retrain all 9 model combos (LC/MC/SC × probe/finetune/supervised)
# using the full 128-patient DeFOG splits (kfold_defog_all128_3fold).
#
# Usage: bash scripts/shell/run_all128_training.sh [context] [phase]
#   context: lc|mc|sc|all (default: all)
#   phase:   probe|finetune|supervised|all (default: all)

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONTEXT="${1:-all}"
PHASE="${2:-all}"
SPLIT_PREFIX="kaggle_labeled/kfold_defog_all128_3fold"

LC_CKPT="checkpoints/mae/sleek-monkey-199/last.ckpt"
MC_CKPT="checkpoints/mae/mae_medcontext_daily/mae_medcontext_daily/last.ckpt"
SC_CKPT="checkpoints/mae/mae_shortcontext_vit4_daily/mae_shortcontext_vit4_daily/last.ckpt"

LC_EXP="classification/spectral_patch_mae_lc_valid_defog_soft"
MC_EXP="classification/spectral_patch_mae_mc_valid_defog_soft"
SC_EXP="classification/spectral_patch_mae_sc_valid_defog_soft"

LC_SUP_EXP="classification/supervised_lc_fogr025_defog"
MC_SUP_EXP="classification/supervised_mc_fogr025_defog"
SC_SUP_EXP="classification/supervised_sc_fogr025_defog"

run_probe() {
    local ckpt="$1" name="$2" exp="$3" bs="$4"
    echo "=== PROBE: $name ==="
    for fold in 0 1 2; do
        OUT_DIR="checkpoints/classification/soft_probe_${name}_fold${fold}"
        [ -f "${OUT_DIR}/last.ckpt" ] && echo "[fold $fold] skip" && continue
        uv run python scripts/train/train_classification.py \
            experiment="${exp}" \
            "train.weights.load_from='${ckpt}'" \
            "data/splits=${SPLIT_PREFIX}${fold}" \
            "+train.logger.name=soft_probe_${name}_fold${fold}" \
            "train.logger.tags=[probe,frozen,mae_2d,all128,fold${fold}]" \
            "data.dataloader.batch_size=${bs}"
    done
}

run_finetune() {
    local ckpt="$1" name="$2" exp="$3" bs="$4"
    echo "=== FINETUNE: $name ==="
    for fold in 0 1 2; do
        OUT_DIR="checkpoints/classification/soft_finetune_${name}_fold${fold}"
        [ -f "${OUT_DIR}/last.ckpt" ] && echo "[fold $fold] skip" && continue
        uv run python scripts/train/train_classification.py \
            experiment="${exp}" \
            "train.weights.load_from='${ckpt}'" \
            "train.weights.freeze_backbone=false" \
            "train.weights.unfreeze_schedule.enabled=false" \
            "train/optimizer=adamw_differential_lr_1e6" \
            "data/splits=${SPLIT_PREFIX}${fold}" \
            "+train.logger.name=soft_finetune_${name}_fold${fold}" \
            "train.logger.tags=[finetune,unfrozen,mae_2d,all128,fold${fold}]" \
            "data.dataloader.batch_size=${bs}" \
            "train.trainer.max_epochs=50" \
            "train.scheduler.T_max=50"
    done
}

run_supervised() {
    local name="$1" exp="$2" bs="$3"
    echo "=== SUPERVISED: $name ==="
    for fold in 0 1 2; do
        OUT_DIR="checkpoints/classification/soft_supervised_${name}_fold${fold}"
        [ -f "${OUT_DIR}/last.ckpt" ] && echo "[fold $fold] skip" && continue
        uv run python scripts/train/train_classification.py \
            experiment="${exp}" \
            "data/splits=${SPLIT_PREFIX}${fold}" \
            "+train.logger.name=soft_supervised_${name}_fold${fold}" \
            "train.logger.tags=[supervised,all128,fold${fold}]" \
            "data.dataloader.batch_size=${bs}"
    done
}

# LC
if [[ "$CONTEXT" == "lc" || "$CONTEXT" == "all" ]]; then
    [[ "$PHASE" == "probe"      || "$PHASE" == "all" ]] && run_probe      "$LC_CKPT" "lc_all128" "$LC_EXP" 64
    [[ "$PHASE" == "finetune"   || "$PHASE" == "all" ]] && run_finetune   "$LC_CKPT" "lc_all128" "$LC_EXP" 32
    [[ "$PHASE" == "supervised" || "$PHASE" == "all" ]] && run_supervised "lc_all128" "$LC_SUP_EXP" 64
fi

# MC
if [[ "$CONTEXT" == "mc" || "$CONTEXT" == "all" ]]; then
    [[ "$PHASE" == "probe"      || "$PHASE" == "all" ]] && run_probe      "$MC_CKPT" "mc_all128" "$MC_EXP" 256
    [[ "$PHASE" == "finetune"   || "$PHASE" == "all" ]] && run_finetune   "$MC_CKPT" "mc_all128" "$MC_EXP" 128
    [[ "$PHASE" == "supervised" || "$PHASE" == "all" ]] && run_supervised "mc_all128" "$MC_SUP_EXP" 256
fi

# SC
if [[ "$CONTEXT" == "sc" || "$CONTEXT" == "all" ]]; then
    [[ "$PHASE" == "probe"      || "$PHASE" == "all" ]] && run_probe      "$SC_CKPT" "sc_all128" "$SC_EXP" 512
    [[ "$PHASE" == "finetune"   || "$PHASE" == "all" ]] && run_finetune   "$SC_CKPT" "sc_all128" "$SC_EXP" 128
    [[ "$PHASE" == "supervised" || "$PHASE" == "all" ]] && run_supervised "sc_all128" "$SC_SUP_EXP" 512
fi

echo "=== All done ==="

#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Fig S? — Label-efficiency curve (supervised value of SSL pretraining).
#
# Two arms, identical MC architecture, trained on the all128 DeFOG cohort at
# increasing labeled-data budgets, then evaluated zero-shot on FogAtHome:
#   * probe   : frozen FORGE MC encoder + trainable GRU head (load pretrained, freeze)
#   * scratch : same architecture, random init, fully trainable (supervised-from-scratch)
#
# X-axis  : per-patient minutes of DeFOG labeled data kept for training, via
#           `+data.dataset.max_train_minutes=N` (train-only, stratified per patient,
#           keyed on start_frame; data/dataset/base.py::_subset_first_minutes).
#           "full" = no cap.
# Budgets : 2 5 15 30 60 full  (DeFOG per-patient minutes: median 30, p25 18, p75 84)
# CV      : the 3 all128 patient folds (kfold_defog_all128_3fold{0,1,2}).
#
# Eval/plot after training:
#   python scripts/eval/eval_label_efficiency.py   # → logs/RESULTS_label_efficiency.csv
#
# Usage: bash scripts/shell/run_label_efficiency.sh [probe|scratch|all]   (default all)
# ─────────────────────────────────────────────────────────────────────────────
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM="${1:-all}"
MC_EXP="classification/spectral_patch_mae_mc_valid_defog_soft"
MC_ENC="checkpoints/mae/mae_medcontext_daily/mae_medcontext_daily/last.ckpt"
SPLIT_PREFIX="kaggle_labeled/kfold_defog_all128_3fold"
BATCH=256
MAX_EPOCHS=30
BUDGETS=(2 5 15 30 60 full)

run_one() {
    local arm="$1" fold="$2" budget="$3"
    local name="labeleff_${arm}_mc_fold${fold}_${budget}min"
    local OUT="checkpoints/classification/${name}"
    [ -f "${OUT}/last.ckpt" ] && echo "[${name}] skip (exists)" && return

    # arm-specific weight handling
    local wargs seed_arg=""
    if [ "$arm" = "probe" ]; then        # frozen FORGE encoder + trainable head
        wargs="train.weights.load_from='${MC_ENC}' train.weights.freeze_backbone=true"
    elif [ "$arm" = "random" ]; then     # frozen RANDOM encoder + trainable head (control)
        # Distinct encoder seed per fold so the 3-fold ensemble averages over
        # THREE random inits, not one — a representative random baseline.
        wargs="train.weights.load_from=null train.weights.freeze_backbone=true"
        seed_arg="global.seed=$((100 + fold))"
    else                                 # scratch: random init, fully trainable
        wargs="train.weights.load_from=null train.weights.freeze_backbone=false"
    fi
    # budget arg ("full" → no cap)
    local budget_arg=""
    [ "$budget" != "full" ] && budget_arg="+data.dataset.max_train_minutes=${budget}"

    echo "=== ${arm} fold=${fold} budget=${budget} ==="
    uv run python scripts/train/train_classification.py \
        experiment="${MC_EXP}" \
        ${wargs} \
        ${seed_arg} \
        "data/splits=${SPLIT_PREFIX}${fold}" \
        ${budget_arg} \
        "data.dataloader.batch_size=${BATCH}" \
        "train.trainer.max_epochs=${MAX_EPOCHS}" \
        "train.scheduler.T_max=${MAX_EPOCHS}" \
        "+train.logger.name=${name}" \
        "train.logger.tags=[labeleff,${arm},mc,fold${fold},b${budget}]"
}

for arm in probe random scratch; do
    [[ "$ARM" == "$arm" || "$ARM" == "all" ]] || continue
    for fold in 0 1 2; do
        for b in "${BUDGETS[@]}"; do
            run_one "$arm" "$fold" "$b"
        done
    done
done

echo "=== Label-efficiency training done. Now: python scripts/eval/eval_label_efficiency.py ==="

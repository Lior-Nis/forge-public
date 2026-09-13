#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Fig S4 (v2) — Label-efficiency curve on the PATIENT-COUNT axis.
#
# X-axis = number of labeled DeFOG patients (all their data), the clinically
# meaningful "how many subjects must a clinic collect" question. Patients are
# chosen FOG-stratified + seeded (`+data.dataset.max_train_patients=K
# +data.dataset.train_subset_seed=S`, data/dataset/base.py::_subset_n_patients),
# so the SAME K patients are used for every arm at a given (K, seed) → fair
# probe/random/scratch comparison.
#
# Arms (identical MC architecture, differ only in the encoder):
#   probe   : frozen FORGE pretrained encoder + trainable head
#   random  : frozen RANDOM encoder + trainable head (control; per-fold init seed)
#   scratch : random init, fully trainable (supervised from scratch)
#
# Budgets : K = 2 4 8 16   (fold train sets ≈ 48 patients; the all-patients anchor
#           is REUSED from the prior full-data run: labeleff_{arm}_mc_fold{f}_fullmin)
# Seeds   : 3 subset draws (0 1 2)  → mean ± CI across seeds
# CV      : 3 all128 folds, ensembled on FogAtHome per (arm,K,seed)
#
# Eval/plot:  python scripts/eval/eval_label_efficiency_patients.py
#             python scripts/analysis/fig_label_efficiency.py   (reads the patient CSV)
#
# Usage: bash scripts/shell/run_label_efficiency_patients.sh [probe|random|scratch|all]
# ─────────────────────────────────────────────────────────────────────────────
set -e
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline   # robust against auth errors on artifact upload (don't rely on launch env)

ARM="${1:-all}"
MC_EXP="classification/spectral_patch_mae_mc_valid_defog_soft"
MC_ENC="checkpoints/mae/mae_medcontext_daily/mae_medcontext_daily/last.ckpt"
SPLIT_PREFIX="kaggle_labeled/kfold_defog_all128_3fold"
BATCH=256
MAX_EPOCHS=30
BUDGETS=(2 4 8 16)
SEEDS=(0 1 2)

run_one() {
    local arm="$1" fold="$2" k="$3" seed="$4"
    local name="labeleff_pat_${arm}_mc_fold${fold}_k${k}_s${seed}"
    local OUT="checkpoints/classification/${name}"
    [ -f "${OUT}/last.ckpt" ] && echo "[${name}] skip (exists)" && return

    local wargs seed_arg=""
    if [ "$arm" = "probe" ]; then
        wargs="train.weights.load_from='${MC_ENC}' train.weights.freeze_backbone=true"
    elif [ "$arm" = "random" ]; then
        wargs="train.weights.load_from=null train.weights.freeze_backbone=true"
        seed_arg="global.seed=$((100 + fold))"   # vary the random encoder init per fold
    else
        wargs="train.weights.load_from=null train.weights.freeze_backbone=false"
    fi

    echo "=== ${arm} fold=${fold} k=${k} seed=${seed} ==="
    uv run python scripts/train/train_classification.py \
        experiment="${MC_EXP}" \
        ${wargs} \
        ${seed_arg} \
        "data/splits=${SPLIT_PREFIX}${fold}" \
        "+data.dataset.max_train_patients=${k}" \
        "+data.dataset.train_subset_seed=${seed}" \
        "data.dataloader.batch_size=${BATCH}" \
        "train.callbacks.checkpoint.save_top_k=1" \
        "train.trainer.max_epochs=${MAX_EPOCHS}" \
        "train.scheduler.T_max=${MAX_EPOCHS}" \
        "+train.logger.name=${name}" \
        "train.logger.log_model=false" \
        "train.logger.tags=[labeleff_pat,${arm},mc,fold${fold},k${k},s${seed}]"
}

for arm in probe random scratch; do
    [[ "$ARM" == "$arm" || "$ARM" == "all" ]] || continue
    for seed in "${SEEDS[@]}"; do
        for fold in 0 1 2; do
            for k in "${BUDGETS[@]}"; do
                run_one "$arm" "$fold" "$k" "$seed"
            done
        done
    done
done

echo "=== patient-count training done. Now: python scripts/eval/eval_label_efficiency_patients.py ==="

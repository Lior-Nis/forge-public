#!/bin/bash
# Clean all-patients ENDPOINT for the patient-count label-efficiency curve.
# Replaces the reused minutes-axis `fullmin` anchor with a recipe-consistent point:
#   - max_train_patients=999 -> hook keeps ALL train patients (K=all)
#   - global.seed=42 for EVERY arm -> probe/random/scratch differ ONLY in the encoder
#     (aligned head-init + data order; the true controlled comparison)
#   - save_top_k=1, same MC experiment / splits / batch as the k-point runs
# 3 arms × 3 folds = 9 runs. Output dirs: labeleff_pat_{arm}_mc_fold{f}_kall_s0
set -e
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

MC_EXP="classification/spectral_patch_mae_mc_valid_defog_soft"
MC_ENC="checkpoints/mae/mae_medcontext_daily/mae_medcontext_daily/last.ckpt"
SPLIT_PREFIX="kaggle_labeled/kfold_defog_all128_3fold"
BATCH=256
MAX_EPOCHS=30

run_one() {
    local arm="$1" fold="$2"
    local name="labeleff_pat_${arm}_mc_fold${fold}_kall_s0"
    local OUT="checkpoints/classification/${name}"
    [ -f "${OUT}/last.ckpt" ] && echo "[${name}] skip (exists)" && return
    local wargs
    if [ "$arm" = "probe" ]; then
        wargs="train.weights.load_from='${MC_ENC}' train.weights.freeze_backbone=true"
    elif [ "$arm" = "random" ]; then
        wargs="train.weights.load_from=null train.weights.freeze_backbone=true"
    else
        wargs="train.weights.load_from=null train.weights.freeze_backbone=false"
    fi
    echo "=== ENDPOINT ${arm} fold=${fold} (K=all, seed=42) ==="
    uv run python scripts/train/train_classification.py \
        experiment="${MC_EXP}" \
        ${wargs} \
        global.seed=42 \
        "data/splits=${SPLIT_PREFIX}${fold}" \
        "+data.dataset.max_train_patients=999" \
        "+data.dataset.train_subset_seed=0" \
        "data.dataloader.batch_size=${BATCH}" \
        "train.callbacks.checkpoint.save_top_k=1" \
        "train.trainer.max_epochs=${MAX_EPOCHS}" \
        "train.scheduler.T_max=${MAX_EPOCHS}" \
        "+train.logger.name=${name}" \
        "train.logger.log_model=false" \
        "train.logger.tags=[labeleff_pat,${arm},mc,fold${fold},kall,s0]"
}

for arm in probe random scratch; do
    for fold in 0 1 2; do
        run_one "$arm" "$fold"
    done
done
echo "=== endpoint done (9 runs). Re-run eval: python scripts/eval/eval_label_efficiency_patients.py ==="

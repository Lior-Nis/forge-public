#!/bin/bash
# Across-budget LR sweep for the supervised-from-scratch arm, to make the WHOLE
# label-efficiency curve deployment-fair (not just the full-data endpoint).
# The full-data sweep was non-monotonic in LR (3e-5>3e-4>1e-4>1e-3), so we sweep
# all three lower LRs at every budget and pick the best per budget in eval. The
# existing lr=1e-3 runs (labeleff_pat_scratch_mc_fold{f}_k{K}_s{seed}) are the 4th
# LR option per budget — not re-run here.
#
# Grid: LR{3e-4,1e-4,3e-5} × K{2,4,8,16} × seed{0,1,2} × fold{0,1,2} = 108 runs.
# seed=42 fixed (aligned with probe/random); K-subset seed varies (matches the curve).
# Output: labeleff_pat_scratch_mc_fold{f}_k{K}_s{seed}_lr{tag}
set -e
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

MC_EXP="classification/spectral_patch_mae_mc_valid_defog_soft"
SPLIT_PREFIX="kaggle_labeled/kfold_defog_all128_3fold"
BATCH=256
MAX_EPOCHS=30
# 1e-4 dropped: it was dominated at full data (0.474 < 3e-4 0.510 < 3e-5 0.550).
# Candidates for best-per-budget are 1e-3 (existing patient-count runs), 3e-4, 3e-5.
declare -A LRS=( [3e-4]="3em4" [3e-5]="3em5" )
BUDGETS=(2 4 8 16)
SEEDS=(0 1 2)

for lr in "${!LRS[@]}"; do
  tag="${LRS[$lr]}"
  for seed in "${SEEDS[@]}"; do
    for fold in 0 1 2; do
      for k in "${BUDGETS[@]}"; do
        name="labeleff_pat_scratch_mc_fold${fold}_k${k}_s${seed}_lr${tag}"
        OUT="checkpoints/classification/${name}"
        [ -f "${OUT}/last.ckpt" ] && echo "[${name}] skip (exists)" && continue
        echo "=== SCRATCH-LR lr=${lr} fold=${fold} k=${k} seed=${seed} ==="
        uv run python scripts/train/train_classification.py \
            experiment="${MC_EXP}" \
            train.weights.load_from=null train.weights.freeze_backbone=false \
            global.seed=42 \
            "train.optimizer.lr=${lr}" \
            "data/splits=${SPLIT_PREFIX}${fold}" \
            "+data.dataset.max_train_patients=${k}" \
            "+data.dataset.train_subset_seed=${seed}" \
            "data.dataloader.batch_size=${BATCH}" \
            "train.callbacks.checkpoint.save_top_k=1" \
            "train.trainer.max_epochs=${MAX_EPOCHS}" \
            "train.scheduler.T_max=${MAX_EPOCHS}" \
            "+train.logger.name=${name}" \
            "train.logger.log_model=false" \
            "train.logger.tags=[scratch_lr_budget,mc,fold${fold},k${k},s${seed},lr${tag}]"
      done
    done
  done
done
echo "=== scratch LR-budget sweep done (108 runs). Eval: python scripts/eval/eval_scratch_lr_budgets.py ==="

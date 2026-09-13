#!/bin/bash
# Fair-tuning LR sweep for the supervised-from-scratch arm at FULL data (K=all).
# The scratch endpoint inherited the probe's lr=1e-3 (a frozen-backbone / linear-probe
# LR), far too high for training a 14M ViT from scratch — it collapsed. Sweep lower
# LRs so the deployment comparison (FORGE probe vs a COMPETENTLY-tuned scratch) is fair.
# 1e-3 already exists (labeleff_pat_scratch_mc_fold{f}_kall_s0). Sweep {3e-4,1e-4,3e-5}.
# 3 LRs × 3 folds = 9 runs. seed=42, K=all, save_top_k=1 (matches the clean endpoint).
set -e
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

MC_EXP="classification/spectral_patch_mae_mc_valid_defog_soft"
SPLIT_PREFIX="kaggle_labeled/kfold_defog_all128_3fold"
BATCH=256
MAX_EPOCHS=30
# LR -> tag (filesystem-safe)
declare -A LRS=( [3e-4]="3em4" [1e-4]="1em4" [3e-5]="3em5" )

for lr in "${!LRS[@]}"; do
  tag="${LRS[$lr]}"
  for fold in 0 1 2; do
    name="labeleff_pat_scratch_mc_fold${fold}_kall_lr${tag}"
    OUT="checkpoints/classification/${name}"
    [ -f "${OUT}/last.ckpt" ] && echo "[${name}] skip (exists)" && continue
    echo "=== SCRATCH-LR lr=${lr} fold=${fold} (K=all, seed=42) ==="
    uv run python scripts/train/train_classification.py \
        experiment="${MC_EXP}" \
        train.weights.load_from=null train.weights.freeze_backbone=false \
        global.seed=42 \
        "train.optimizer.lr=${lr}" \
        "data/splits=${SPLIT_PREFIX}${fold}" \
        "+data.dataset.max_train_patients=999" \
        "+data.dataset.train_subset_seed=0" \
        "data.dataloader.batch_size=${BATCH}" \
        "train.callbacks.checkpoint.save_top_k=1" \
        "train.trainer.max_epochs=${MAX_EPOCHS}" \
        "train.scheduler.T_max=${MAX_EPOCHS}" \
        "+train.logger.name=${name}" \
        "train.logger.log_model=false" \
        "train.logger.tags=[scratch_lr,mc,fold${fold},lr${tag}]"
  done
done
echo "=== scratch LR sweep done (9 runs). Eval: python scripts/eval/eval_scratch_lr_sweep.py ==="

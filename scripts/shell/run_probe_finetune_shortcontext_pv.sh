#!/bin/bash
# Probe + finetune for shortcontext (200-len) MAE backbone — Prisma-style pure training.
#
# Training data: defog_pure_valid_shortcontext view (purity==1.0 & validity==1.0)
#   — only pure-class windows (all-FOG or all-background). No boundary patches.
# Splits: kfold_defog_fogcount_200pv_3fold{0,1,2} — balanced by FOG patch count.
# Inference: sliding window + overlap averaging gives per-timestep output (eval script).
#
# Set PRETRAIN_CKPT to the best epoch from pretraining before running finetune.

set -eo pipefail
cd "$(dirname "$0")/../.."

PRETRAIN_CKPT="${1:-checkpoints/mae/mae_shortcontext_daily/mae_shortcontext_daily/last.ckpt}"
PROBE_CKPT_DIR="checkpoints/classification/shortcontext_probe_pv"
FINETUNE_CKPT_DIR="checkpoints/classification/shortcontext_finetune_pv"

echo "=== SHORTCONTEXT PROBE: 3-fold frozen backbone ==="
echo "=== Pretrain checkpoint: ${PRETRAIN_CKPT} ==="
for fold in 0 1 2; do
  echo "--- Probe fold ${fold} ---"
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/train/train_classification.py \
    experiment=classification/spectral_patch_mae_shortcontext_finetune_defog \
    "data/splits=kaggle_labeled/kfold_defog_fogcount_200pv_3fold${fold}" \
    "train.weights.load_from='${PRETRAIN_CKPT}'" \
    "train.weights.freeze_backbone=true" \
    "train.weights.unfreeze_schedule.enabled=false" \
    "train.scheduler.T_max=30" \
    "train.trainer.max_epochs=30" \
    "train/callbacks=classification" \
    "train.callbacks.checkpoint.dirpath=${PROBE_CKPT_DIR}_fold${fold}" \
    "train.logger.tags=[probe,frozen,mae_2d,shortcontext,exp026,fold${fold},pure200]" \
    "+train.logger.name=shortcontext_probe_pv_fold${fold}" \
    "+train.logger.group=probe_shortcontext_exp026_pv" || echo "WARNING: fold ${fold} exited non-zero"
  echo "--- Probe fold ${fold} done ---"
done
echo "=== PROBE DONE ==="

echo "=== SHORTCONTEXT FINETUNE: backbone LR=1e-7, head LR=1e-3 ==="
for fold in 0 1 2; do
  echo "--- Finetune fold ${fold} ---"
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/train/train_classification.py \
    experiment=classification/spectral_patch_mae_shortcontext_finetune_defog \
    "data/splits=kaggle_labeled/kfold_defog_fogcount_200pv_3fold${fold}" \
    "train.weights.load_from='${PRETRAIN_CKPT}'" \
    "train.weights.freeze_backbone=false" \
    "train.weights.unfreeze_schedule.enabled=false" \
    "train/optimizer=adamw_differential_lr_1e7" \
    "train.scheduler.T_max=50" \
    "train.trainer.max_epochs=50" \
    "train/callbacks=finetune" \
    "train.callbacks.checkpoint.dirpath=${FINETUNE_CKPT_DIR}_fold${fold}" \
    "train.logger.tags=[finetune,unfrozen,mae_2d,shortcontext,exp026,fold${fold},lr_1e7,pure200]" \
    "+train.logger.name=shortcontext_finetune_pv_fold${fold}" \
    "+train.logger.group=finetune_shortcontext_exp026_pv" || echo "WARNING: fold ${fold} exited non-zero"
  echo "--- Finetune fold ${fold} done ---"
done
echo "=== FINETUNE DONE ==="

echo "=== EVAL: Probe on FogAtHome (per-timestep) ==="
uv run python scripts/eval/eval_fogathome_segmentation.py \
  --model-name shortcontext_probe_pv_exp026 \
  --ckpt-pattern "${PROBE_CKPT_DIR}_fold{fold}/last.ckpt" \
  --n-folds 3 --batch-size 64 \
  --dataset-path "len200_stride10_fogathome.zarr" \
  --output-dir logs/eval/shortcontext_probe_pv_exp026

echo "=== EVAL: Finetune on FogAtHome (per-timestep) ==="
uv run python scripts/eval/eval_fogathome_segmentation.py \
  --model-name shortcontext_finetune_pv_exp026 \
  --ckpt-pattern "${FINETUNE_CKPT_DIR}_fold{fold}/last.ckpt" \
  --n-folds 3 --batch-size 64 \
  --dataset-path "len200_stride10_fogathome.zarr" \
  --output-dir logs/eval/shortcontext_finetune_pv_exp026

echo "=== ALL DONE ==="

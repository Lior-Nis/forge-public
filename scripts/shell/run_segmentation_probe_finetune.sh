#!/bin/bash
# Segmentation probe + finetune — 3-fold CV using #023 MAE 2D ep7 backbone.
# Pipeline: SegmentationPipeline, head outputs [B, 1000, 2] per-timestep logits.
# Downstream: kaggle_defog_pure_valid_longcontext (1000-len, purity+validity==1.0)
# Splits: kfold_defog_fogcount_3fold{0,1,2}

set -eo pipefail
cd "$(dirname "$0")/../.."

PRETRAIN_CKPT="checkpoints/ssl_comparison/mae_2d_ep7.ckpt"
PROBE_CKPT_DIR="checkpoints/classification/segmentation_probe"
FINETUNE_CKPT_DIR="checkpoints/classification/segmentation_finetune"

# ── Probe (frozen backbone) ────────────────────────────────────────────────────
echo "=== SEGMENTATION PROBE: 3-fold frozen backbone ==="
for fold in 0 1 2; do
  echo "--- Probe fold ${fold} ---"
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/train/train_classification.py \
    experiment=classification/spectral_patch_mae_segmentation_defog \
    "data/splits=kaggle_labeled/kfold_defog_fogcount_3fold${fold}" \
    "train.weights.load_from='${PRETRAIN_CKPT}'" \
    "train.weights.freeze_backbone=true" \
    "train.weights.unfreeze_schedule.enabled=false" \
    "train.scheduler.T_max=30" \
    "train.trainer.max_epochs=30" \
    "train/callbacks=classification" \
    "train.callbacks.checkpoint.dirpath=${PROBE_CKPT_DIR}_fold${fold}" \
    "train.logger.tags=[segmentation,probe,frozen,mae_2d,exp023,fold${fold}]" \
    "+train.logger.name=seg_probe_fold${fold}" \
    "+train.logger.group=segmentation_probe_exp023" || echo "WARNING: fold ${fold} exited non-zero"
  echo "--- Probe fold ${fold} done ---"
done
echo "=== PROBE DONE ==="

# ── Finetune (differential LR backbone=1e-7) ──────────────────────────────────
echo "=== SEGMENTATION FINETUNE: 3-fold differential LR backbone=1e-7, head=1e-3 ==="
for fold in 0 1 2; do
  echo "--- Finetune fold ${fold} ---"
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/train/train_classification.py \
    experiment=classification/spectral_patch_mae_segmentation_defog \
    "data/splits=kaggle_labeled/kfold_defog_fogcount_3fold${fold}" \
    "train.weights.load_from='${PRETRAIN_CKPT}'" \
    "train.weights.freeze_backbone=false" \
    "train.weights.unfreeze_schedule.enabled=false" \
    "train/optimizer=adamw_differential_lr_1e7" \
    "train.scheduler.T_max=50" \
    "train.trainer.max_epochs=50" \
    "train/callbacks=finetune" \
    "train.callbacks.checkpoint.dirpath=${FINETUNE_CKPT_DIR}_fold${fold}" \
    "train.logger.tags=[segmentation,finetune,unfrozen,mae_2d,exp023,fold${fold},lr_1e7]" \
    "+train.logger.name=seg_finetune_fold${fold}" \
    "+train.logger.group=segmentation_finetune_exp023" || echo "WARNING: fold ${fold} exited non-zero"
  echo "--- Finetune fold ${fold} done ---"
done
echo "=== ALL DONE ==="

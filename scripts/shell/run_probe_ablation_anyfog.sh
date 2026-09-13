#!/bin/bash
# Ablation: does any-fog labeling alone explain the AP improvement in exp025?
#
# Setup: same exp023 MAE backbone (sleek-monkey-199, 1000-len patches) + same 1000-len
# kaggle zarr, but training with TRUE any-fog labeling (any FOG timestep = positive)
# instead of the dominant-class binarization exp023 used.
#
# Controls for: backbone quality, sequence length, adaptive stride
# Varies: labeling strategy only
#
# Compare results to:
#   - exp023 probe (1000-len, dominant-class labeling)  → FogAtHome AP, NormAP
#   - exp025 probe (500-len, any-fog labeling)          → FogAtHome AP, NormAP

set -eo pipefail
cd "$(dirname "$0")/../.."

PRETRAIN_CKPT="checkpoints/mae/sleek-monkey-199/last.ckpt"
PROBE_CKPT_DIR="checkpoints/classification/probe_ablation_anyfog"

echo "=== ABLATION PROBE: 1000-len + true-any-fog labeling ==="
for fold in 0 1 2; do
  echo "--- Probe fold ${fold} ---"
  uv run python scripts/train/train_classification.py \
    experiment=classification/spectral_patch_mae_finetune_defog \
    "data/splits=kaggle_labeled/kfold_defog_fogcount_3fold${fold}" \
    "train.weights.load_from='${PRETRAIN_CKPT}'" \
    "train.weights.freeze_backbone=true" \
    "data.dataset.classification_strategy=true_any_fog_from_labels" \
    "train.scheduler.T_max=30" \
    "train.trainer.max_epochs=30" \
    "train/callbacks=classification" \
    "train.callbacks.checkpoint.dirpath=${PROBE_CKPT_DIR}_fold${fold}" \
    "data.dataloader.batch_size=256" \
    "train.logger.tags=[probe,frozen,mae_2d_longcontext,exp025_ablation,fold${fold},anyfog_labels]" \
    "+train.logger.name=probe_ablation_anyfog_fold${fold}" \
    "+train.logger.group=probe_ablation_anyfog_exp025" || echo "WARNING: fold ${fold} exited non-zero"
  echo "--- Probe fold ${fold} done ---"
done

echo "=== PROBE DONE ==="

echo "=== EVAL: Ablation probe on FogAtHome (1000-len, any-fog GT) ==="
uv run python scripts/eval/eval_fogathome.py \
  --model-name probe_ablation_anyfog \
  --ckpt-pattern "${PROBE_CKPT_DIR}_fold{fold}/last.ckpt" \
  --n-folds 3 \
  --batch-size 64 \
  --dataset-path "len1000_stride200_fogathome.zarr" \
  --output-dir logs/eval/probe_ablation_anyfog

echo "=== EVAL: Ablation probe on FogAtHome DailyLiving (1000-len, any-fog GT) ==="
uv run python scripts/eval/eval_fogathome_dailyliving.py \
  --model-name probe_ablation_anyfog \
  --ckpt-pattern "${PROBE_CKPT_DIR}_fold{fold}/last.ckpt" \
  --n-folds 3 \
  --batch-size 64 \
  --dataset-path "len1000_stride200_fogathome_dailyliving.zarr" \
  --output-dir logs/eval/probe_ablation_anyfog

echo "=== ALL DONE ==="

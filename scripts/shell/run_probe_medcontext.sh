#!/bin/bash
# Linear probe: 3-fold CV on medcontext defog zarr
# Checkpoint: MAE 2D pretraining (exp025), eepoch=47, val_loss=0.0152

set -e
cd "$(dirname "$0")/../.."

CKPT="checkpoints/mae/mae_medcontext_daily/mae_medcontext_daily/eepoch=47.ckpt"

for fold in 0 1 2; do
  echo "=== Probe fold ${fold} ==="
  uv run python scripts/train/train_classification.py \
    experiment=classification/spectral_patch_mae_medcontext_finetune_defog \
    "data/splits=kaggle_labeled/kfold_defog_fogcount_3fold${fold}" \
    "train.weights.load_from='${CKPT}'" \
    "data.dataloader.batch_size=256" \
    "train.logger.tags=[probe,frozen,mae_2d_medcontext,exp025,fold${fold}]" \
    "+train.logger.name=probe_mae_medcontext_fold${fold}" \
    "+train.logger.group=probe_mae_medcontext_exp025"
  echo "=== Probe fold ${fold} done (exit $?) ==="
done

echo "=== All probe folds done ==="

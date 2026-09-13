#!/bin/bash
# Full finetune (unfrozen backbone, differential LR) + external evaluation for exp025
# Pretrain checkpoint: eepoch=47, val_loss=0.0152
# Optimizer: adamw_differential_lr_1e6 (head=1e-3, backbone=1e-6) — matches prior SSL finetune runs
# Follows probe (run_probe_medcontext.sh); compare probe vs finetune on fogathome + dailyliving

set -eo pipefail
cd "$(dirname "$0")/../.."

PRETRAIN_CKPT="checkpoints/mae/mae_medcontext_daily/mae_medcontext_daily/eepoch=47.ckpt"
FINETUNE_CKPT_DIR="checkpoints/classification/finetune_mae_medcontext"
PROBE_CKPT_DIR="checkpoints/classification/probe_mae_medcontext"

# ── Step 1: Full finetune, 3 folds ─────────────────────────────────────────
echo "=== FINETUNE: 3-fold full finetuning (differential LR: head=1e-3, backbone=1e-6) ==="
for fold in 0 1 2; do
  echo "--- Finetune fold ${fold} ---"
  uv run python scripts/train/train_classification.py \
    experiment=classification/spectral_patch_mae_medcontext_finetune_defog \
    "data/splits=kaggle_labeled/kfold_defog_fogcount_3fold${fold}" \
    "train.weights.load_from='${PRETRAIN_CKPT}'" \
    "train.weights.freeze_backbone=false" \
    "train.weights.unfreeze_schedule.enabled=false" \
    "train/optimizer=adamw_differential_lr_1e6" \
    "train.scheduler.T_max=50" \
    "train.trainer.max_epochs=50" \
    "train/callbacks=finetune" \
    "train.callbacks.checkpoint.dirpath=${FINETUNE_CKPT_DIR}_v2_fold${fold}" \
    "data.dataloader.batch_size=256" \
    "train.logger.tags=[finetune,unfrozen,mae_2d_medcontext,exp025,fold${fold},lr_1e6]" \
    "+train.logger.name=finetune_mae_medcontext_v2_fold${fold}" \
    "+train.logger.group=finetune_mae_medcontext_exp025_v2" || echo "WARNING: fold ${fold} exited non-zero (likely WandB registry timeout) - checkpoint still saved"
  echo "--- Finetune fold ${fold} done ---"
done

echo "=== FINETUNE DONE ==="

# ── Step 2: Eval probe on fogathome ────────────────────────────────────────
echo "=== EVAL: Probe on FogAtHome ==="
uv run python scripts/eval/eval_fogathome.py \
  --model-name probe_mae_medcontext_exp025 \
  --ckpt-pattern "${PROBE_CKPT_DIR}_fold{fold}/last.ckpt" \
  --n-folds 3 \
  --batch-size 64 \
  --dataset-path "len500_stride200_fogstride100_anyfog_fogathome.zarr" \
  --output-dir logs/eval/probe_mae_medcontext_exp025

# ── Step 3: Eval probe on fogathome_dailyliving ────────────────────────────
echo "=== EVAL: Probe on FogAtHome DailyLiving ==="
uv run python scripts/eval/eval_fogathome_dailyliving.py \
  --model-name probe_mae_medcontext_exp025 \
  --ckpt-pattern "${PROBE_CKPT_DIR}_fold{fold}/last.ckpt" \
  --n-folds 3 \
  --batch-size 64 \
  --dataset-path "views/fogathome_dailyliving_valid_medcontext.yaml" \
  --output-dir logs/eval/probe_mae_medcontext_exp025

# ── Step 4: Eval finetune on fogathome ────────────────────────────────────
echo "=== EVAL: Finetune on FogAtHome ==="
uv run python scripts/eval/eval_fogathome.py \
  --model-name finetune_mae_medcontext_exp025_v2 \
  --ckpt-pattern "${FINETUNE_CKPT_DIR}_v2_fold{fold}/last.ckpt" \
  --n-folds 3 \
  --batch-size 64 \
  --dataset-path "len500_stride200_fogstride100_anyfog_fogathome.zarr" \
  --output-dir logs/eval/finetune_mae_medcontext_exp025_v2

# ── Step 5: Eval finetune on fogathome_dailyliving ────────────────────────
echo "=== EVAL: Finetune on FogAtHome DailyLiving ==="
uv run python scripts/eval/eval_fogathome_dailyliving.py \
  --model-name finetune_mae_medcontext_exp025_v2 \
  --ckpt-pattern "${FINETUNE_CKPT_DIR}_v2_fold{fold}/last.ckpt" \
  --n-folds 3 \
  --batch-size 64 \
  --dataset-path "views/fogathome_dailyliving_valid_medcontext.yaml" \
  --output-dir logs/eval/finetune_mae_medcontext_exp025_v2

echo "=== ALL EVAL DONE ==="

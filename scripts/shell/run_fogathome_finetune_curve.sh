#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Fig 12 — ICC vs fine-tuning-data-size curve on FogAtHome.
# Mirrors Yang et al. 2026 Fig 2 (their external-cohort fine-tuning curve), but
# for our single-lower-back-IMU FORGE MC model.
#
# Design
#   * Init  : DeFOG-trained MC model (our zero-shot model) — soft_finetune_mc_all128_fold0.
#             The 0-min point is this model's zero-shot FogAtHome ICC (already 0.909 %TF).
#   * CV    : 3 participant-level folds over the 12 FogAtHome patients
#             (configs/data/splits/fogathome_finetune_3fold{0,1,2}.yaml).
#             Fine-tune on the fold's `train` patients, eval on held-out `test`.
#   * Budget: per-patient minutes of FogAtHome training data ∈ {0, 0.5, 1, 2, 3, 5}.
#             FogAtHome valid-gait data is small (~38 min total; per-patient 1.4-5.3 min,
#             median 3.2), so budgets are per-patient minutes — not Yang's 0-90 min.
#             Resulting total train data per fold ≈ 0 / 3.5 / 7 / 13 / 18 / 20 min.
#
# Data-budget mechanism
#   `data.dataset.max_train_minutes=N` keeps, per patient, only patches within the
#   first N minutes of that patient's recording (accumulated across sessions via the
#   `start_frame` metadata). It is stratified across the fold's training patients and
#   applies to the TRAIN split only (val/test untouched). Implemented in
#   data/dataset/base.py::_subset_first_minutes. Normal shuffling is kept.
#
# After training, compute the ICC curve with:
#   python scripts/eval/eval_fogathome_finetune_curve.py
#
# Usage: bash scripts/shell/run_fogathome_finetune_curve.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SOURCE_CKPT="checkpoints/classification/soft_finetune_mc_all128_fold0/last.ckpt"
MC_EXP="classification/spectral_patch_mae_mc_valid_defog_soft"
BATCH=32
MAX_EPOCHS=10
BUDGETS_MIN=(0.5 1 2 3 5)    # per-patient minutes; 0 = zero-shot (no training; eval-only)

# NOTE: frozen-backbone PROBE adaptation (only the head moves from its zero-shot
# state). Full unfreeze collapses on 0.5-5 min of data (constant output) — the
# standard tiny-data failure. Probe is the collapse-resistant low-data recipe.

if [ ! -f "$SOURCE_CKPT" ]; then
    echo "ERROR: source checkpoint not found: $SOURCE_CKPT" >&2
    exit 1
fi

for fold in 0 1 2; do
    for mins in "${BUDGETS_MIN[@]}"; do
        name="ftcurve_mc_fold${fold}_${mins}min"
        OUT_DIR="checkpoints/classification/${name}"
        if [ -f "${OUT_DIR}/last.ckpt" ]; then
            echo "[fold ${fold} ${mins}min] skip (exists)"
            continue
        fi
        echo "=== FINETUNE fold=${fold} budget=${mins}min/patient ==="
        uv run python scripts/train/train_classification.py \
            experiment="${MC_EXP}" \
            "train.weights.load_from='${SOURCE_CKPT}'" \
            "train.weights.freeze_backbone=true" \
            "data/paths=fogathome_medcontext" \
            "data/splits=fogathome_finetune_3fold${fold}" \
            "data.dataloader.batch_size=${BATCH}" \
            "+data.dataset.max_train_minutes=${mins}" \
            "train.trainer.max_epochs=${MAX_EPOCHS}" \
            "train.scheduler.T_max=${MAX_EPOCHS}" \
            "+train.logger.name=${name}" \
            "train.logger.tags=[ftcurve,fogathome,mc,fold${fold},${mins}min,probe]"
    done
done

echo "=== Training done. Now run: python scripts/eval/eval_fogathome_finetune_curve.py ==="

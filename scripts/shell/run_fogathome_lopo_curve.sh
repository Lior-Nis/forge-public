#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Fig 12 (LOPO) — fine-tuning-data-size curve with leave-one-patient-out CV.
# Rigorous version of run_fogathome_finetune_curve.sh: trains on 10 patients
# (~up to 35 min) instead of 7, with 12 folds (each patient held out once).
# Frozen-backbone PROBE adaptation from the DeFOG MC model.
#
#   12 folds × budgets {1,2,3,5} min/patient. Pooled ICC computed across all
#   12 held-out patients by scripts/eval/eval_fogathome_lopo_curve.py.
#   0-min (zero-shot) is the source model, handled by the eval.
#
# Usage: bash scripts/shell/run_fogathome_lopo_curve.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SOURCE_CKPT="checkpoints/classification/soft_finetune_mc_all128_fold0/last.ckpt"
MC_EXP="classification/spectral_patch_mae_mc_valid_defog_soft"
BATCH=32
MAX_EPOCHS=10
LR="${1:-1e-4}"                       # fine-tuning LR (override: bash run_..._lopo_curve.sh 1e-5)
LRTAG="lr$(echo "$LR" | tr -d '.-')"  # 1e-4 -> lr1e4
PREFIX="ftcurve_lopo_${LRTAG}_mc"
BUDGETS_MIN=(1 2 3 5)

# IMPORTANT: load_strategy=backbone_and_head loads the trained head from the source
# model (the part we fine-tune). The default backbone_only would discard the head
# and train a random one from scratch — which collapses on this little data.

[ -f "$SOURCE_CKPT" ] || { echo "ERROR: missing $SOURCE_CKPT" >&2; exit 1; }

for fold in $(seq 0 11); do
    for mins in "${BUDGETS_MIN[@]}"; do
        name="${PREFIX}_fold${fold}_${mins}min"
        OUT_DIR="checkpoints/classification/${name}"
        if [ -f "${OUT_DIR}/last.ckpt" ]; then echo "[fold ${fold} ${mins}min] skip"; continue; fi
        echo "=== LOPO PROBE fold=${fold} budget=${mins}min/patient ==="
        uv run python scripts/train/train_classification.py \
            experiment="${MC_EXP}" \
            "train.weights.load_from='${SOURCE_CKPT}'" \
            "train.weights.load_strategy=backbone_and_head" \
            "train.weights.freeze_backbone=true" \
            "train.optimizer.lr=${LR}" \
            "data/paths=fogathome_medcontext" \
            "data/splits=fogathome_lopo_fold${fold}" \
            "data.dataloader.batch_size=${BATCH}" \
            "+data.dataset.max_train_minutes=${mins}" \
            "train.trainer.max_epochs=${MAX_EPOCHS}" \
            "train.scheduler.T_max=${MAX_EPOCHS}" \
            "+train.logger.name=${name}" \
            "train.logger.tags=[ftcurve,lopo,fogathome,mc,fold${fold},${mins}min,probe]"
    done
done

echo "=== Training done. Now run: python scripts/eval/eval_fogathome_lopo_curve.py ==="

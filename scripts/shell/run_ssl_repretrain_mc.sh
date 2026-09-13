#!/bin/bash
# Budget-matched SSL re-pretraining at MEDIUM context (MC, 500 timesteps).
#
# WHY: the existing SSL ablation is only fair at LC (the alternatives were never
# pretrained at any other context), and LC is FORGE's *weakest* context — so the
# context-matched comparison can't showcase 2D-MAE's real (MC) advantage. This
# script re-pretrains ALL objectives (including a fresh 2D-MAE reference) at MC on
# identical data and an identical epoch budget, so the downstream MC comparison is
# both context-matched AND budget-matched.
#
# BUDGET MATCH: every objective trains on the identical len500_stride200 daily
# zarr (~21.1M windows/epoch) for the same number of epochs (default 10) with
# limit_train_batches=null (full passes) and effective batch 640 x accumulate 4 =
# 2560. Identical data x identical epochs => identical windows-seen (~211M @ 10ep).
# This is the budget axis the prior (retracted) comparison failed on (0.4M-890M).
#
# WHY 10 EPOCHS (not 47): MAE downstream peaks very early (paper_mae Fig 8: ~ep3
# already matches Kaggle Rank 5; AP flat after) and collapse signatures (SimCLR
# ln N floor, I-JEPA ->0) appear in epoch 0-1 and persist. So 10 epochs both
# yields downstream-usable encoders AND reveals any collapse — at ~1/5 the wall
# clock. Watch losses/val_loss in WandB; a collapsing run can be killed early.
#
# IMPORTANT: a fresh 2D-MAE @ the SAME 10-epoch budget (mae2d) is included as the
# in-table reference — written to its OWN dir so it never touches the existing
# 47-epoch deployment encoder (checkpoints/mae/mae_medcontext_daily/). Do NOT
# compare the alternatives against that 47-epoch encoder (would re-break budget
# matching). Causal-MAE is excluded: at LC it scored below random-init.
#
# Usage: bash scripts/shell/run_ssl_repretrain_mc.sh [method]
#   method: mae2d|mae1d|simclr|jepa|all   (default: all)
# Env:
#   EPOCHS=10   budget (epochs of full passes over len500 daily zarr)
#
# After this completes, probe the new encoders with:
#   bash scripts/shell/run_ssl_ablation_all128.sh all mc

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

METHOD="${1:-all}"
EPOCHS="${EPOCHS:-5}"   # budget cap; MAE peaks ~ep3, collapse shows ep0-1, ~45-55min/epoch

# method -> (pretrain script, experiment config, checkpoint dir)
declare -A SCRIPT=(
  [mae2d]="scripts/train/pretrain_mae.py"
  [mae1d]="scripts/train/pretrain_mae.py"
  [simclr]="scripts/train/pretrain_simclr.py"
  [jepa]="scripts/train/pretrain_jepa.py"
)
declare -A EXP=(
  [mae2d]="pretraining/spectral_patch_mae_medcontext_daily"
  [mae1d]="pretraining/spectral_patch_mae1d_medcontext_daily"
  [simclr]="pretraining/spectral_patch_simclr_medcontext_daily"
  [jepa]="pretraining/spectral_patch_jepa_medcontext_daily"
)
# Distinct dirs; mae2d budget-N must NOT collide with the existing 47-epoch dir.
declare -A CKPT_DIR=(
  [mae2d]="checkpoints/mae/mae2d_medcontext_budget${EPOCHS}_daily"
  [mae1d]="checkpoints/mae/mae1d_medcontext_daily"
  [simclr]="checkpoints/simclr/simclr_medcontext_daily"
  [jepa]="checkpoints/jepa/jepa_medcontext_daily"
)
# Per-step batch size. Budget = samples-seen = windows x epochs, INDEPENDENT of
# batch, so a smaller batch is still budget-matched (only gradient-step count
# changes). MAE variants fit at 640; SimCLR (NxN similarity matrix) and I-JEPA
# (EMA target + predictor = ~2x encoder memory) OOM at 640 because the desktop
# eats ~8GB VRAM, so they use 256. (SimCLR was originally trained at 320.)
declare -A BATCH=(
  [mae2d]=640
  [mae1d]=640
  [simclr]=256
  [jepa]=256
)

run_pretrain() {
    local method="$1"
    local dir="${CKPT_DIR[$method]}"
    local run_name="repretrain_mc_${method}_e${EPOCHS}"
    # Checkpoints nest under the run-name subdir; match it specifically so we
    # neither false-skip on stale dummy-*/last.ckpt nor miss a finished run.
    if find "${dir}" -path "*${run_name}*" -name last.ckpt 2>/dev/null | grep -q .; then
        echo "=== SSL RE-PRETRAIN: $method @ MC  -> SKIP (done: ${dir}/${run_name}/last.ckpt) ==="
        return
    fi
    echo "=== SSL RE-PRETRAIN: $method @ MC  (budget=${EPOCHS} epochs, bs=${BATCH[$method]}, len500 daily) ==="
    # Override the checkpoint dir so every method (esp. mae2d) writes where the
    # skip-check and the probe script expect it.
    uv run python "${SCRIPT[$method]}" \
        experiment="${EXP[$method]}" \
        "train.trainer.max_epochs=${EPOCHS}" \
        "data.dataloader.batch_size=${BATCH[$method]}" \
        "train.callbacks.checkpoint.dirpath=${dir}" \
        "+train.logger.name=${run_name}" \
        "+train.logger.tags=[ssl_pretrain,repretrain_mc,budget_matched,${method},mc]"
}

for m in mae2d mae1d simclr jepa; do
    if [[ "$METHOD" == "$m" || "$METHOD" == "all" ]]; then
        run_pretrain "$m"
    fi
done

echo "=== MC re-pretraining done. Next: bash scripts/shell/run_ssl_ablation_all128.sh all mc ==="

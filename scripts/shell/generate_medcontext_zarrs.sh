#!/bin/bash
# Generate all zarr datasets for the medcontext (500-timestep) experiment.
#
# Three preprocessing changes vs longcontext:
#   1. block_len 1000 → 500
#   2. Adaptive stride: fog_stride_len=100 for FOG patches, stride=200 for background
#   3. Any-FOG labeling: patch positive if ANY frame is FOG (labeled datasets only)
#
# Run from project root. Estimated time: ~15-30 min per dataset.
# GPU not required — run while GPU is in use for other jobs.
#
# Usage: bash scripts/shell/generate_medcontext_zarrs.sh

set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Medcontext zarr generation start $(date) ==="

# 1. Kaggle daily-living unlabeled (pretraining data — fixed stride, no labels)
log "--- [1/4] Kaggle daily-living unlabeled (pretraining) ---"
uv run python -m data.process \
    paths=kaggle_daily_medcontext \
    process=kaggle_medcontext_daily
log "[1/4] done"

# 2. Kaggle defog labeled (downstream probe/finetune — adaptive stride + any-fog)
log "--- [2/4] Kaggle defog labeled (downstream) ---"
uv run python -m data.process \
    paths=kaggle_defog_medcontext \
    process=kaggle_medcontext
log "[2/4] done"

# 3. Fog@Home labeled (external eval — adaptive stride + any-fog)
log "--- [3/4] Fog@Home labeled (external eval) ---"
uv run python -m data.process \
    paths=fogathome_medcontext \
    process=kaggle_medcontext
log "[3/4] done"

# 4. Fog@Home daily-living labeled (held-out eval — adaptive stride + any-fog)
log "--- [4/4] Fog@Home daily-living labeled (held-out eval) ---"
uv run python -m data.process \
    paths=fogathome_dailyliving_medcontext \
    process=kaggle_medcontext
log "[4/4] done"

log "=== Medcontext zarr generation complete $(date) ==="
log ""
log "Next steps:"
log "  1. Run MAE 2D pretraining on the new data:"
log "     bash scripts/shell/run_mae_medcontext_pretrain.sh"
log "  2. Once first checkpoint appears, start probe loop:"
log "     bash scripts/shell/probe_loop.sh \\"
log "       --ckpt-dir checkpoints/mae/mae_medcontext_daily \\"
log "       --wandb-group probe_mae_medcontext \\"
log "       --ssl-method mae_2d_medcontext \\"
log "       --experiment classification/spectral_patch_mae_medcontext_finetune_defog \\"
log "       --n-folds 3 --interval 20"

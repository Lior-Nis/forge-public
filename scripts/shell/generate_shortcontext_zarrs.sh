#!/bin/bash
# Generate all zarr datasets for the shortcontext (200-timestep) experiment (exp026).
#
# Configuration vs longcontext (#023):
#   - block_len: 1000 → 200 (2s patches at 100 Hz)
#   - stride_len: 200 → 40 for pretraining (20% advance — same ratio as #023)
#   - stride_len: 200 → 20 for labeled downstream (10% advance)
#   - fog_stride_len: 10 for FOG regions in labeled datasets
#   - any_fog_labeling: true (same as medcontext)
#
# Downstream uses purity==1.0 & validity==1.0 filter — label noise from boundary
# patches is eliminated; any-fog == majority-vote for pure patches.
#
# GPU not required. Estimated time: ~20-40 min per dataset (more patches than medcontext).
# Run while GPU is busy with other jobs.
#
# Usage: bash scripts/shell/generate_shortcontext_zarrs.sh

set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Shortcontext zarr generation start $(date) ==="

# 1. Kaggle daily-living unlabeled (pretraining — stride=40, no labels)
log "--- [1/4] Kaggle daily-living unlabeled (pretraining, stride=40) ---"
uv run python -m data.process \
    paths=kaggle_daily_shortcontext \
    process=kaggle_shortcontext_daily
log "[1/4] done"

# 2. Kaggle defog labeled (downstream probe/finetune — stride=20, fog_stride=10, any-fog)
log "--- [2/4] Kaggle defog labeled (downstream, stride=20/10) ---"
uv run python -m data.process \
    paths=kaggle_defog_shortcontext \
    process=kaggle_shortcontext
log "[2/4] done"

# 3. Fog@Home labeled (external eval — stride=20, fog_stride=10, any-fog)
log "--- [3/4] Fog@Home labeled (external eval) ---"
uv run python -m data.process \
    paths=fogathome_shortcontext \
    process=kaggle_shortcontext
log "[3/4] done"

# 4. Fog@Home daily-living labeled (held-out eval)
log "--- [4/4] Fog@Home daily-living labeled (held-out eval) ---"
uv run python -m data.process \
    paths=fogathome_dailyliving_shortcontext \
    process=kaggle_shortcontext
log "[4/4] done"

log "=== Shortcontext zarr generation complete $(date) ==="
log ""
log "Next steps:"
log "  1. Create pure+valid views (run after zarr generation):"
log "     python scripts/data/create_shortcontext_views.py"
log "  2. Launch MAE pretraining:"
log "     bash scripts/shell/run_mae_shortcontext_pretrain.sh"

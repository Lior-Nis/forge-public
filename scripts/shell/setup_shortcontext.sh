#!/bin/bash
# Full setup for shortcontext (200-timestep) experiment (exp026).
# Step 1: Generate all 4 zarrs (CPU-only, ~1-2h total)
# Step 2: Create view YAML files from zarr stats
# After this completes, run run_mae_shortcontext_pretrain.sh when GPU is free.

set -euo pipefail
cd "$(dirname "$0")/../.."

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Shortcontext setup start ==="

# ── Step 1: Zarr generation ────────────────────────────────────────────────────

log "--- [1/4] Kaggle daily-living unlabeled (pretraining, block=200, stride=40) ---"
uv run python -m data.process \
    paths=kaggle_daily_shortcontext \
    process=kaggle_shortcontext_daily
log "[1/4] done"

log "--- [2/4] Kaggle defog labeled (downstream, block=200, stride=20, fog_stride=10) ---"
uv run python -m data.process \
    paths=kaggle_defog_shortcontext \
    process=kaggle_shortcontext
log "[2/4] done"

log "--- [3/4] Fog@Home labeled (external eval) ---"
uv run python -m data.process \
    paths=fogathome_shortcontext \
    process=kaggle_shortcontext
log "[3/4] done"

log "--- [4/4] Fog@Home daily-living labeled (held-out eval) ---"
uv run python -m data.process \
    paths=fogathome_dailyliving_shortcontext \
    process=kaggle_shortcontext
log "[4/4] done"

log "=== Zarr generation complete ==="

# ── Step 2: Create views ───────────────────────────────────────────────────────

log "--- Creating view YAML files ---"
uv run python scripts/data/create_shortcontext_views.py
log "Views created"

log ""
log "=== Setup complete. Ready to pretrain. ==="
log "Launch pretraining when GPU is free:"
log "  nohup bash scripts/shell/run_mae_shortcontext_pretrain.sh > logs/run_mae_shortcontext_pretrain.log 2>&1 &"

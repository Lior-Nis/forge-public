#!/bin/bash
# Continue shortcontext setup: generate labeled zarrs (step 1 daily already done).
set -euo pipefail
cd "$(dirname "$0")/../.."

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "--- [2/4] Kaggle defog labeled ---"
uv run python -m data.process paths=kaggle_defog_shortcontext process=kaggle_shortcontext
log "[2/4] done"

log "--- [3/4] Fog@Home labeled ---"
uv run python -m data.process paths=fogathome_shortcontext process=kaggle_shortcontext
log "[3/4] done"

log "--- [4/4] Fog@Home daily-living labeled ---"
uv run python -m data.process paths=fogathome_dailyliving_shortcontext process=kaggle_shortcontext
log "[4/4] done"

log "--- Creating views ---"
uv run python scripts/data/create_shortcontext_views.py
log "=== Shortcontext setup complete ==="

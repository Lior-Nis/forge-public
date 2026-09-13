#!/bin/bash
# Block until all 12 SSL-ablation probes finish, then run the FogAtHome eval.
# Fails fast if training dies before all checkpoints exist.
set -u
cd "$(dirname "$0")/../.."
export WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPECT=12
MAX_WAIT=$((8 * 3600))   # 8 h cap
waited=0
glob() { ls checkpoints/classification/soft_probe_ssl_{simclr,jepa,causal,mae1d}_all128_fold{0,1,2}/last.ckpt 2>/dev/null | wc -l; }

while :; do
    n=$(glob)
    echo "[watch +${waited}s] $n/$EXPECT checkpoints present"
    [ "$n" -ge "$EXPECT" ] && { echo "ALL PROBES DONE"; break; }
    if ! pgrep -f run_ssl_ablation_all128.sh >/dev/null && ! pgrep -f train_classification.py >/dev/null; then
        echo "TRAINING STOPPED with only $n/$EXPECT checkpoints — aborting eval."; exit 2
    fi
    [ "$waited" -ge "$MAX_WAIT" ] && { echo "TIMEOUT after ${waited}s ($n/$EXPECT)"; exit 3; }
    sleep 120; waited=$((waited + 120))
done

echo "=== Running FogAtHome eval for the 4 SSL probes (MC) ==="
uv run python scripts/eval/eval_comprehensive.py \
    --datasets fogathome --contexts mc \
    --models probe_ssl_simclr probe_ssl_jepa probe_ssl_causal probe_ssl_mae1d \
    --output logs/ssl_ablation_all128_eval.csv \
    --cache-dir logs/comprehensive_eval_cache 2>&1 | tail -30

echo "=== SSL ablation eval written to logs/ssl_ablation_all128_eval.csv ==="

#!/bin/bash
cd "$(dirname "$0")/../.."
export WANDB_MODE=offline PYTORCH_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg
while kill -0 3170707 2>/dev/null; do sleep 30; done
echo "[$(date)] scratch-lr sweep exited -> eval" >> logs/scratch_lr_sweep.log
uv run python scripts/eval/eval_scratch_lr_sweep.py >> logs/scratch_lr_sweep.log 2>&1
echo "[$(date)] DONE scratch-lr eval" >> logs/scratch_lr_sweep.log

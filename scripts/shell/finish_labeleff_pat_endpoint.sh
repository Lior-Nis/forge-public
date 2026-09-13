#!/bin/bash
cd "$(dirname "$0")/../.."
export WANDB_MODE=offline PYTORCH_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg
while kill -0 2827283 2>/dev/null; do sleep 30; done
echo "[$(date)] endpoint runner exited -> eval" >> logs/labeleff_pat_endpoint.log
uv run python scripts/eval/eval_label_efficiency_patients.py > logs/labeleff_patients_eval.log 2>&1
uv run python scripts/analysis/fig_label_efficiency_patients.py >> logs/labeleff_patients_eval.log 2>&1
echo "[$(date)] DONE endpoint eval+plot" >> logs/labeleff_pat_endpoint.log

#!/bin/bash
cd "$(dirname "$0")/../.."
export WANDB_MODE=offline PYTORCH_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg
# Wait for the runner process to EXIT (true completion — last.ckpt count is unreliable
# since save_last writes every epoch). Then eval + regenerate Fig S4.
while kill -0 1016786 2>/dev/null; do sleep 60; done
echo "[$(date)] runner exited -> eval" >> logs/labeleff_pat_supervisor.log
uv run python scripts/eval/eval_label_efficiency_patients.py > logs/labeleff_patients_eval.log 2>&1
uv run python scripts/analysis/fig_label_efficiency_patients.py >> logs/labeleff_patients_eval.log 2>&1
echo "[$(date)] DONE eval+plot" >> logs/labeleff_pat_supervisor.log

#!/bin/bash
# Watch the patient-count label-efficiency grid (108 runs); when complete, run the
# 3-arm eval + regenerate Fig S4. Kept in-repo (not /tmp) so it survives tmp cleanup.
cd "$(dirname "$0")/../.."
export WANDB_MODE=offline PYTORCH_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg
glob(){ ls -d checkpoints/classification/labeleff_pat_{probe,random,scratch}_mc_fold{0,1,2}_k{2,4,8,16}_s{0,1,2}/last.ckpt 2>/dev/null | wc -l; }
while :; do
  n=$(glob); echo "[watch $(date +%m-%d_%H:%M)] patient-count $n/108"
  [ "$n" -ge 108 ] && break
  if ! pgrep -f run_label_efficiency_patients.sh >/dev/null && ! pgrep -f train_classification.py >/dev/null; then
     echo "STOPPED at $n/108 — training not running. NOT auto-evaluating partial grid; exiting."
     exit 1
  fi
  sleep 300
done
echo "=== eval $(date) ==="
uv run python scripts/eval/eval_label_efficiency_patients.py > logs/labeleff_patients_eval.log 2>&1
echo "=== plot ==="
uv run python scripts/analysis/fig_label_efficiency_patients.py >> logs/labeleff_patients_eval.log 2>&1
echo "=== DONE $(date) ==="

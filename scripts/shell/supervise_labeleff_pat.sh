#!/bin/bash
# Self-healing supervisor for the patient-count label-efficiency grid (108 runs).
# - Restarts the runner if it dies with work remaining (skip-if-exists resumes).
# - Detects a STUCK state (a run that fails repeatedly → no progress across restarts)
#   and HALTS with a clear flag instead of looping forever.
# - Runs the 3-arm eval + regenerates Fig S4 once all 108 are done.
# Single supervisor (replaces the plain watcher) so there's no restart/eval race.
cd "$(dirname "$0")/../.."
export WANDB_MODE=offline PYTORCH_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg
LOG=logs/labeleff_pat_supervisor.log
glob(){ ls -d checkpoints/classification/labeleff_pat_{probe,random,scratch}_mc_fold{0,1,2}_k{2,4,8,16}_s{0,1,2}/last.ckpt 2>/dev/null | wc -l; }
runner_up(){ pgrep -f run_label_efficiency_patients.sh >/dev/null; }
say(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$LOG"; }

baseline=-1; stall=0; restarts=0
say "supervisor start ($(glob)/108)"
while :; do
  n=$(glob)
  if [ "$n" -ge 108 ]; then
     say "COMPLETE 108/108 -> eval+plot"
     uv run python scripts/eval/eval_label_efficiency_patients.py > logs/labeleff_patients_eval.log 2>&1
     uv run python scripts/analysis/fig_label_efficiency_patients.py >> logs/labeleff_patients_eval.log 2>&1
     say "DONE (figure regenerated)"
     break
  fi
  if ! runner_up; then
     if [ "$n" -gt "$baseline" ]; then
        stall=0; say "runner down, progress ($baseline->$n/108); RESTARTING"
     else
        stall=$((stall+1)); say "runner down, NO progress since last restart (stall=$stall, $n/108) — a run is failing repeatedly"
        if [ "$stall" -ge 3 ]; then
           say "HALT: stuck at $n/108 after 3 stalled restarts. Manual attention needed (see logs/labeleff_patients_train.log)."
           break
        fi
     fi
     nohup bash scripts/shell/run_label_efficiency_patients.sh all >> logs/labeleff_patients_train.log 2>&1 &
     disown
     baseline=$n; restarts=$((restarts+1))
  fi
  sleep 300
done

#!/bin/bash
# Self-healing supervisor for the overnight scratch LR-budget sweep (108 runs).
# Runs the runner in the FOREGROUND (blocking); when it returns (crash or done),
# checks progress and re-runs (skip-if-exists resumes). Halts if stuck (no progress
# across 3 consecutive runner exits). Single clean process tree — launch this via
# the harness background mechanism so it survives across turns.
cd "$(dirname "$0")/../.."
export WANDB_MODE=offline PYTORCH_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg
LOG=logs/scratch_lr_budgets_supervisor.log
glob(){ ls -d checkpoints/classification/labeleff_pat_scratch_mc_fold{0,1,2}_k{2,4,8,16}_s{0,1,2}_lr{3em4,3em5}/last.ckpt 2>/dev/null | wc -l; }
TARGET=72   # {3e-4,3e-5} × K{2,4,8,16} × seed{0,1,2} × fold{0,1,2}
say(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$LOG"; }

baseline=-1; stall=0
say "supervisor start ($(glob)/$TARGET)"
while :; do
  n=$(glob)
  if [ "$n" -ge "$TARGET" ]; then
     say "COMPLETE $TARGET/$TARGET -> eval"
     uv run python scripts/eval/eval_scratch_lr_budgets.py > logs/scratch_lr_budgets_eval.log 2>&1
     say "DONE eval (see logs/scratch_lr_budgets_eval.log)"
     break
  fi
  if [ "$n" -gt "$baseline" ]; then stall=0; else stall=$((stall+1)); fi
  if [ "$stall" -ge 3 ]; then say "HALT: stuck at $n/$TARGET (see logs/scratch_lr_budgets_train.log)"; break; fi
  baseline=$n
  say "running runner ($n/$TARGET done, stall=$stall)"
  bash scripts/shell/run_scratch_lr_budgets.sh >> logs/scratch_lr_budgets_train.log 2>&1 || say "runner exited non-zero (will re-check/resume)"
done

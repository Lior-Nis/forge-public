#!/bin/bash
# Run 5-fold cross-validation benchmark using best_fixed experiment config.
# Uses stratified splits (protocol + fog_rate) for reliable evaluation.
#
# Usage:
#   bash scripts/run_kfold_benchmark.sh                    # Run all 5 folds
#   bash scripts/run_kfold_benchmark.sh 0 2                # Run folds 0-2
#   EXTRA_ARGS="train.trainer.max_epochs=10" bash scripts/run_kfold_benchmark.sh  # Override

set -euo pipefail

FOLDS_START=${1:-0}
FOLDS_END=${2:-4}
EXTRA_ARGS="${EXTRA_ARGS:-}"

EXPERIMENT="classification/best_fixed"
DATA_PATHS="kaggle_pure_valid_no_notype_longcontext"
SPLIT_PREFIX="kaggle_labeled/kfold_stratified"
# batch_size=200 fits 16GB GPU with longcontext (block_len=1000)
BATCH_SIZE="${BATCH_SIZE:-200}"

echo "=== 5-Fold Stratified CV Benchmark ==="
echo "Experiment: ${EXPERIMENT}"
echo "Data: ${DATA_PATHS}"
echo "Folds: ${FOLDS_START} to ${FOLDS_END}"
echo "Extra args: ${EXTRA_ARGS}"
echo ""

for fold in $(seq "$FOLDS_START" "$FOLDS_END"); do
    echo "=========================================="
    echo "  FOLD ${fold} / ${FOLDS_END}"
    echo "=========================================="

    .venv/bin/python scripts/train/train_classification.py \
        experiment="${EXPERIMENT}" \
        data/paths="${DATA_PATHS}" \
        data/splits="${SPLIT_PREFIX}${fold}" \
        data.dataloader.batch_size="${BATCH_SIZE}" \
        global.experiment_name="kfold_benchmark_fold${fold}" \
        ${EXTRA_ARGS}

    echo ""
    echo "Fold ${fold} complete."
    echo ""
done

echo "=== All folds complete ==="
echo "Check WandB for runs tagged 'kfold_benchmark_fold*'"
echo "Aggregate results with: python scripts/analysis/aggregate_kfold_results.py"

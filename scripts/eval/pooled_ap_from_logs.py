"""Bootstrap confidence intervals on mean AP from 5-fold cross-validation log files.

Cannot compute pooled AP (no per-sample scores in aggregated logs), but computes
bootstrap 95% CI on the mean fold AP using the 5 fold scores.

Inputs are either:
  - 5 log files (hydra output .log files, one per fold)
  - explicit AP values passed via --ap flag
  - a WandB project + run prefix (fetches summary metrics)

Usage:
    # From per-fold AP values directly
    python scripts/pooled_ap_from_logs.py --ap 0.41 0.38 0.55 0.29 0.36

    # From hydra/stdout log files (one per fold)
    python scripts/pooled_ap_from_logs.py --logs outputs/fold0/train.log outputs/fold1/train.log ...

    # From log files matching a glob pattern
    python scripts/pooled_ap_from_logs.py --glob "outputs/kfold_defog_fogstrat*/train.log"

    # From WandB (requires wandb installed and authenticated)
    python scripts/pooled_ap_from_logs.py --wandb-project fog-classification --wandb-prefix kfold_defog_fogstrat
"""

import argparse
import glob as glob_module
import json
import logging
import re
import sys

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_ap_from_log_file(log_path: str) -> float | None:
    """Extract the final test_ap (or val_ap fallback) from a hydra log file.

    Searches for lines like:
        test_ap=0.394200
    or the JSON line:
        METRICS_JSON: {"epoch": 49, "stage": "test", ..., "test_ap": 0.394200, ...}
    """
    best_ap: float | None = None

    # Regex patterns (both structured key=value and JSON log lines)
    kv_pattern  = re.compile(r"(?:test_ap|metrics/test_ap)\s*[=:]\s*([0-9.eE+\-]+)")
    json_pattern = re.compile(r"METRICS_JSON:\s*(\{.*\})")

    try:
        with open(log_path) as f:
            for line in f:
                # Try key=value format first
                m = kv_pattern.search(line)
                if m:
                    best_ap = float(m.group(1))
                    continue

                # Try JSON line
                m = json_pattern.search(line)
                if m:
                    try:
                        data = json.loads(m.group(1))
                        if data.get("stage") == "test":
                            # Look for AP under several possible key names
                            for key in ("test_ap", "metrics/test_ap", "ap"):
                                if key in data:
                                    best_ap = float(data[key])
                                    break
                    except json.JSONDecodeError:
                        pass
    except OSError as e:
        logger.warning(f"Cannot open log file {log_path}: {e}")
        return None

    return best_ap


def fetch_ap_from_wandb(project: str, prefix: str, n_folds: int = 5,
                        metric_key: str = "metrics/test_ap") -> list[float | None]:
    """Fetch per-fold AP from WandB summary metrics."""
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        sys.exit(1)

    api = wandb.Api()
    aps: list[float | None] = []

    for fold in range(n_folds):
        name_pattern = f"{prefix}{fold}"
        matching = api.runs(
            project,
            filters={
                "config.global.experiment_name": name_pattern,
                "state": "finished",
            },
            order="-created_at",
        )
        matching = list(matching)
        if not matching:
            logger.warning(f"No finished run for fold {fold} (name pattern: {name_pattern})")
            aps.append(None)
            continue

        run = matching[0]
        val = run.summary.get(metric_key)
        if val is None:
            # Try without namespace prefix
            val = run.summary.get(metric_key.split("/")[-1])
        logger.info(f"  Fold {fold}: {run.name}  {metric_key}={val}")
        aps.append(float(val) if val is not None else None)

    return aps


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000,
                 ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Bootstrap (n_resamples) confidence interval on the mean.

    Samples with replacement from `values` (the 5 fold APs).

    Returns:
        (lower_bound, upper_bound) for the given CI level
    """
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples)
    n = len(values)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1.0 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap CI on mean fold-AP from 5-fold CV logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--ap", nargs="+", type=float, metavar="AP",
        help="Per-fold AP values directly (e.g. --ap 0.41 0.38 0.55 0.29 0.36)"
    )
    input_group.add_argument(
        "--logs", nargs="+", metavar="LOG_FILE",
        help="Log files (one per fold). Extracts test_ap from METRICS_JSON lines."
    )
    input_group.add_argument(
        "--glob", metavar="PATTERN",
        help='Glob pattern matching log files (e.g. "outputs/kfold_*/train.log")'
    )
    input_group.add_argument(
        "--wandb-project", metavar="PROJECT",
        help="WandB project name (use with --wandb-prefix)"
    )

    parser.add_argument(
        "--wandb-prefix", metavar="PREFIX", default="kfold_defog_fogstrat",
        help="WandB run name prefix, fold index appended (default: kfold_defog_fogstrat)"
    )
    parser.add_argument(
        "--wandb-metric", metavar="KEY", default="metrics/test_ap",
        help="WandB summary key for AP (default: metrics/test_ap)"
    )
    parser.add_argument(
        "--n-resamples", type=int, default=1000,
        help="Bootstrap resamples (default: 1000)"
    )
    parser.add_argument(
        "--ci", type=float, default=0.95,
        help="Confidence level (default: 0.95)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for bootstrap (default: 42)"
    )
    parser.add_argument(
        "--metric-name", default="AP",
        help="Label used in output (default: AP)"
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Collect per-fold AP values
    # -----------------------------------------------------------------------
    fold_aps: list[float | None]

    if args.ap is not None:
        fold_aps = list(args.ap)
        logger.info(f"Using {len(fold_aps)} AP values provided on command line")

    elif args.logs is not None:
        fold_aps = []
        for path in args.logs:
            ap = parse_ap_from_log_file(path)
            if ap is None:
                logger.warning(f"Could not extract AP from: {path}")
            fold_aps.append(ap)

    elif args.glob is not None:
        paths = sorted(glob_module.glob(args.glob))
        if not paths:
            logger.error(f"No files matched glob pattern: {args.glob}")
            sys.exit(1)
        logger.info(f"Found {len(paths)} log files matching '{args.glob}'")
        fold_aps = []
        for path in paths:
            ap = parse_ap_from_log_file(path)
            if ap is None:
                logger.warning(f"Could not extract AP from: {path}")
            fold_aps.append(ap)

    else:  # --wandb-project
        logger.info(f"Fetching from WandB: project={args.wandb_project}, "
                    f"prefix={args.wandb_prefix}")
        fold_aps = fetch_ap_from_wandb(
            args.wandb_project, args.wandb_prefix,
            metric_key=args.wandb_metric,
        )

    # -----------------------------------------------------------------------
    # Filter out None values and validate
    # -----------------------------------------------------------------------
    valid_pairs = [(i, ap) for i, ap in enumerate(fold_aps) if ap is not None]
    missing = [i for i, ap in enumerate(fold_aps) if ap is None]

    if missing:
        logger.warning(f"Missing AP for folds: {missing}")

    if len(valid_pairs) < 2:
        logger.error(f"Need at least 2 valid fold APs for bootstrap. Got {len(valid_pairs)}.")
        sys.exit(1)

    valid_folds = [i for i, _ in valid_pairs]
    values = np.array([ap for _, ap in valid_pairs])

    # -----------------------------------------------------------------------
    # Compute statistics
    # -----------------------------------------------------------------------
    mean_ap = float(values.mean())
    std_ap  = float(values.std(ddof=1))   # sample std (unbiased)
    sem_ap  = std_ap / np.sqrt(len(values))

    lower, upper = bootstrap_ci(values, n_resamples=args.n_resamples,
                                 ci=args.ci, seed=args.seed)

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  {args.metric_name} — {len(values)}-Fold Cross-Validation Results")
    print("=" * 60)

    print(f"\nPer-fold {args.metric_name}:")
    for fold_idx, ap in zip(valid_folds, values):
        print(f"  Fold {fold_idx}: {ap:.4f}")
    if missing:
        for fold_idx in missing:
            print(f"  Fold {fold_idx}: MISSING")

    print(f"\nSummary statistics:")
    print(f"  Mean {args.metric_name}:   {mean_ap:.4f}")
    print(f"  Std  {args.metric_name}:   {std_ap:.4f}")
    print(f"  SEM  {args.metric_name}:   {sem_ap:.4f}")
    print(f"  Range:         [{values.min():.4f}, {values.max():.4f}]")

    print(f"\nBootstrap {int(args.ci * 100)}% CI on mean {args.metric_name}:")
    print(f"  (n_resamples={args.n_resamples}, seed={args.seed})")
    print(f"  {mean_ap:.4f}  [{lower:.4f}, {upper:.4f}]")

    print()
    print("NOTE: Bootstrap CI resamples from the 5 fold scores (not per-sample).")
    print("      With only 5 folds the CI is wide by construction — this is correct.")
    print("      For tighter estimates, use pooled_ap_eval.py to compute AP on the")
    print("      combined test set (all patients, one score per sample).")
    print()


if __name__ == "__main__":
    main()

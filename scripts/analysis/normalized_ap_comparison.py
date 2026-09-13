"""
Compute normalized AP (lift and prevalence-adjusted) for all #024 SSL methods
across fogcount and fogstrat 3-fold splits.

Prevalence is computed from the zarr dataset + split configs.
Normalized AP = (AP - prevalence) / (1 - prevalence)
Lift = AP / prevalence

Usage:
    uv run python scripts/analysis/normalized_ap_comparison.py
"""

import os
import yaml
import numpy as np
import zarr
import wandb
import pandas as pd
from pathlib import Path
from collections import defaultdict

ZARR_PATH = "data/processed/len1000_stride200_kaggle.zarr"
SPLITS_DIR = "configs/data/splits/kaggle_labeled"
WANDB_ENTITY = "liornis"
WANDB_PROJECT = "fog-classification"

# Method → WandB group mapping for both splits
METHODS = {
    "mae_2d":     {"fogcount": "forge_fogcount_probe_trajectory", "fogstrat": "probe_mae_2d_fogstrat"},
    "mae_1d":     {"fogcount": "probe_mae_1d",                   "fogstrat": "probe_mae_1d_fogstrat"},
    "causal_mae": {"fogcount": "probe_causal_mae",               "fogstrat": "probe_causal_mae_fogstrat"},
    "ijepa":      {"fogcount": "probe_jepa",                     "fogstrat": "probe_ijepa_fogstrat"},
    "lejepa":     {"fogcount": "probe_lejepa",                   "fogstrat": "probe_lejepa_fogstrat"},
    "random_init":{"fogcount": "probe_random_init_vit12",        "fogstrat": "probe_random_init_fogstrat"},
}

SPLIT_PREFIXES = {
    "fogcount": "kfold_defog_fogcount_3fold",
    "fogstrat": "kfold_defog_fogstrat_3fold",
}


def compute_prevalence(split_prefix: str, fold: int, zarr_path: str) -> float:
    """Compute FOG positive rate in the test set for a given split fold."""
    split_file = Path(SPLITS_DIR) / f"{split_prefix}{fold}.yaml"
    with open(split_file) as f:
        split = yaml.safe_load(f)

    test_patients = split.get("test", [])
    if not test_patients:
        return float("nan")

    z = zarr.open(zarr_path, mode="r")
    patient_ids = z["metadata"]["patient_id"][:]
    labels = z["labels"][:]  # [N, seq_len]

    total = 0
    fog = 0
    for pid in test_patients:
        mask = patient_ids == pid
        if not mask.any():
            continue
        patient_labels = labels[mask]  # [n_samples, seq_len]
        total += patient_labels.size
        fog += (patient_labels > 0).sum()

    return fog / total if total > 0 else float("nan")


def fetch_wandb_ap(group: str, split_tag: str, n_folds: int = 3) -> dict:
    """Fetch best val_ap per fold from a WandB group."""
    api = wandb.Api()
    try:
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": group})
    except Exception as e:
        print(f"  Warning: could not fetch group '{group}': {e}")
        return {}

    best = {}
    for run in runs:
        tags = run.tags
        fold_tag = next((t for t in tags if t.startswith("fold")), None)
        if fold_tag is None:
            continue
        fold = int(fold_tag.replace("fold", ""))
        ap = run.summary.get("metrics/val_ap") or 0
        if fold not in best or ap > best[fold]:
            best[fold] = ap
    return best


def normalized_ap(ap: float, prevalence: float) -> float:
    if prevalence >= 1.0:
        return float("nan")
    return (ap - prevalence) / (1.0 - prevalence)


def lift(ap: float, prevalence: float) -> float:
    if prevalence == 0:
        return float("nan")
    return ap / prevalence


def main():
    print("Computing prevalence from zarr + splits...")
    prevalences = {}
    for split_name, prefix in SPLIT_PREFIXES.items():
        prevalences[split_name] = {}
        for fold in range(3):
            try:
                p = compute_prevalence(prefix, fold, ZARR_PATH)
                prevalences[split_name][fold] = p
                print(f"  {split_name} fold{fold}: prevalence={p:.3f}")
            except Exception as e:
                print(f"  {split_name} fold{fold}: ERROR {e}")
                prevalences[split_name][fold] = float("nan")

    print("\nFetching WandB results...")
    rows = []
    for method, groups in METHODS.items():
        for split_name, group in groups.items():
            print(f"  {method} / {split_name} (group: {group})")
            fold_aps = fetch_wandb_ap(group, split_name)
            if not fold_aps:
                print(f"    No runs found.")
                continue

            fold_rows = []
            for fold, ap in fold_aps.items():
                prev = prevalences[split_name].get(fold, float("nan"))
                row = {
                    "method": method,
                    "split": split_name,
                    "fold": fold,
                    "raw_ap": ap,
                    "prevalence": prev,
                    "lift": lift(ap, prev),
                    "normalized_ap": normalized_ap(ap, prev),
                }
                fold_rows.append(row)
                rows.append(row)

            if fold_rows:
                mean_ap = np.nanmean([r["raw_ap"] for r in fold_rows])
                mean_norm = np.nanmean([r["normalized_ap"] for r in fold_rows])
                mean_lift = np.nanmean([r["lift"] for r in fold_rows])
                mean_prev = np.nanmean([r["prevalence"] for r in fold_rows])
                print(f"    raw_ap={mean_ap:.3f}  prevalence={mean_prev:.3f}  "
                      f"lift={mean_lift:.2f}x  norm_ap={mean_norm:.3f}")

    if not rows:
        print("No data found.")
        return

    df = pd.DataFrame(rows)
    output_path = "logs/ssl_comparison/normalized_ap_results.csv"
    os.makedirs("logs/ssl_comparison", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved per-fold results to {output_path}")

    # Summary table: mean per method × split
    print("\n=== Summary: Mean normalized AP across folds ===")
    summary = df.groupby(["method", "split"]).agg(
        raw_ap=("raw_ap", "mean"),
        prevalence=("prevalence", "mean"),
        lift=("lift", "mean"),
        normalized_ap=("normalized_ap", "mean"),
        n_folds=("fold", "count"),
    ).round(3)
    print(summary.to_string())

    summary_path = "logs/ssl_comparison/normalized_ap_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()

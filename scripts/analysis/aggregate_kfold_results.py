"""Aggregate 5-fold cross-validation results from WandB.

Fetches runs matching the kfold_benchmark pattern, computes:
  - Per-fold mean +/- std (existing approach)
  - Pooled AP: concatenate test predictions across all folds → compute ONE AP
    over all 37 patients. More reliable than averaging 5 per-fold APs when
    each fold has only 7-8 test patients (high variance from patient composition).

Usage:
    python scripts/aggregate_kfold_results.py
    python scripts/aggregate_kfold_results.py --project fog-classification --prefix kfold_benchmark
    python scripts/aggregate_kfold_results.py --prefix my_experiment --csv results.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

try:
    import wandb
except ImportError:
    print("wandb not installed. Install with: pip install wandb")
    sys.exit(1)


def fetch_kfold_runs(project: str, prefix: str, n_folds: int = 5):
    """Fetch completed kfold benchmark runs from WandB."""
    api = wandb.Api()

    runs = []
    for fold in range(n_folds):
        name_pattern = f"{prefix}_fold{fold}"
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
            print(f"  WARNING: No finished run found for fold {fold} (name={name_pattern})")
            continue

        run = matching[0]  # Most recent
        print(f"  Fold {fold}: {run.name} ({run.id})")
        runs.append((fold, run))

    return runs


def extract_summary_metrics(runs):
    """Extract scalar summary metrics from each fold's run."""
    records = []
    for fold, run in runs:
        summary = run.summary
        record = {"fold": fold, "run_name": run.name, "run_id": run.id}
        for key in [
            "metrics/val_ap", "metrics/test_ap",
            "metrics/val_f1", "metrics/test_f1",
            "metrics/val_precision", "metrics/test_precision",
            "metrics/val_recall", "metrics/test_recall",
            "losses/val_loss", "losses/test_loss",
        ]:
            val = summary.get(key)
            if val is not None:
                record[key.split("/")[-1]] = val
        records.append(record)
    return pd.DataFrame(records)


def fetch_patch_predictions(run, stage: str = "test") -> pd.DataFrame | None:
    """Download the patch-level prediction table from a WandB run.

    Returns a DataFrame with columns: label, prob_class_1, patient_id, (session_id).
    Returns None if the table is not found (e.g. log_model=False or old run).
    """
    table_key = f"patch_analysis/{stage}_patch_details"
    try:
        artifact_name = f"run-{run.id}-{table_key.replace('/', '_')}:latest"
        # Tables are logged as wandb artifacts under the run
        tables = {k: v for k, v in run.summary.items() if table_key in k}
        if not tables:
            # Try fetching via logged artifacts
            for artifact in run.logged_artifacts():
                if "patch_details" in artifact.name and stage in artifact.name:
                    artifact_dir = artifact.download()
                    import os, json
                    for fname in os.listdir(artifact_dir):
                        if fname.endswith(".json"):
                            with open(os.path.join(artifact_dir, fname)) as f:
                                data = json.load(f)
                            return pd.DataFrame(data["data"], columns=data["columns"])
            return None

        # Table stored directly in summary as wandb.Table
        table_obj = list(tables.values())[0]
        if hasattr(table_obj, "get_dataframe"):
            return table_obj.get_dataframe()
        return None
    except Exception as e:
        print(f"    Could not fetch predictions for run {run.id}: {e}")
        return None


def compute_pooled_metrics(all_preds: pd.DataFrame) -> dict:
    """Compute metrics on pooled predictions across all folds."""
    y_true = all_preds["label"].values
    y_score = all_preds["prob_class_1"].values
    y_pred = (y_score >= 0.5).astype(int)

    return {
        "pooled_ap":        average_precision_score(y_true, y_score),
        "pooled_f1":        f1_score(y_true, y_pred, zero_division=0),
        "pooled_precision": precision_score(y_true, y_pred, zero_division=0),
        "pooled_recall":    recall_score(y_true, y_pred, zero_division=0),
        "n_patients":       all_preds["patient_id"].nunique() if "patient_id" in all_preds.columns else "?",
        "n_patches":        len(all_preds),
        "fog_prevalence":   y_true.mean(),
    }


def compute_per_patient_pooled(all_preds: pd.DataFrame) -> pd.DataFrame:
    """Per-patient AP from pooled predictions — shows patient-level spread."""
    if "patient_id" not in all_preds.columns:
        return pd.DataFrame()
    rows = []
    for pid, grp in all_preds.groupby("patient_id"):
        y_true = grp["label"].values
        y_score = grp["prob_class_1"].values
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            ap = float("nan")
        else:
            ap = average_precision_score(y_true, y_score)
        rows.append({
            "patient_id": pid,
            "fold": grp["fold"].iloc[0] if "fold" in grp.columns else "?",
            "n_patches": len(grp),
            "fog_pct": f"{y_true.mean()*100:.1f}%",
            "ap": ap,
        })
    return pd.DataFrame(rows).sort_values("ap", ascending=False)


def print_results(summary_df: pd.DataFrame, pooled: dict | None, per_patient: pd.DataFrame | None):
    """Print formatted results."""
    metric_cols = [c for c in summary_df.columns if c not in ("fold", "run_name", "run_id")]

    print("\n" + "=" * 70)
    print("  5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)

    print("\nPer-fold metrics (mean of patch-level scores within fold):")
    print(summary_df[["fold", "run_name"] + metric_cols].to_string(index=False, float_format="%.4f"))

    print("\nAggregated (mean +/- std across folds):")
    print("-" * 50)
    for col in metric_cols:
        values = summary_df[col].dropna()
        if len(values) > 0:
            print(f"  {col:25s}: {values.mean():.4f} +/- {values.std():.4f}"
                  f"  (range: {values.min():.4f} - {values.max():.4f})")

    if pooled:
        print("\n" + "=" * 70)
        print("  POOLED EVALUATION (all test folds combined)")
        print("=" * 70)
        print(f"  Patients: {pooled['n_patients']}  |  "
              f"Patches: {pooled['n_patches']}  |  "
              f"FoG prevalence: {pooled['fog_prevalence']*100:.1f}%")
        print(f"\n  {'Pooled AP':25s}: {pooled['pooled_ap']:.4f}")
        print(f"  {'Pooled F1':25s}: {pooled['pooled_f1']:.4f}")
        print(f"  {'Pooled Precision':25s}: {pooled['pooled_precision']:.4f}")
        print(f"  {'Pooled Recall':25s}: {pooled['pooled_recall']:.4f}")

    if per_patient is not None and not per_patient.empty:
        print("\nPer-patient AP (pooled predictions):")
        print(per_patient.to_string(index=False, float_format="%.4f"))

    print()


def main():
    parser = argparse.ArgumentParser(description="Aggregate k-fold CV results from WandB")
    parser.add_argument("--project", default="fog-classification")
    parser.add_argument("--prefix", default="kfold_benchmark")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--stage", default="test", choices=["val", "test"],
                        help="Stage to compute pooled metrics on")
    parser.add_argument("--csv", type=str, default=None, help="Save per-fold summary to CSV")
    parser.add_argument("--pooled-csv", type=str, default=None,
                        help="Save pooled per-patient breakdown to CSV")
    parser.add_argument("--no-pooled", action="store_true",
                        help="Skip downloading prediction tables (faster)")
    args = parser.parse_args()

    print(f"Fetching runs: project={args.project}, prefix='{args.prefix}'")
    runs = fetch_kfold_runs(args.project, args.prefix, args.n_folds)

    if not runs:
        print("No runs found.")
        sys.exit(1)

    summary_df = extract_summary_metrics(runs)

    pooled_metrics = None
    per_patient_df = None

    if not args.no_pooled:
        print(f"\nDownloading {args.stage} patch predictions for pooled evaluation...")
        fold_preds = []
        for fold, run in runs:
            df = fetch_patch_predictions(run, stage=args.stage)
            if df is not None and "label" in df.columns and "prob_class_1" in df.columns:
                df["fold"] = fold
                fold_preds.append(df)
                print(f"  Fold {fold}: {len(df)} patches, "
                      f"{df['patient_id'].nunique() if 'patient_id' in df.columns else '?'} patients")
            else:
                print(f"  Fold {fold}: no prediction table found")

        if fold_preds:
            all_preds = pd.concat(fold_preds, ignore_index=True)

            # Sanity check: no patient should appear in multiple folds
            if "patient_id" in all_preds.columns:
                patient_fold_counts = all_preds.groupby("patient_id")["fold"].nunique()
                leaking = patient_fold_counts[patient_fold_counts > 1]
                if not leaking.empty:
                    print(f"\n  WARNING: {len(leaking)} patients appear in multiple folds — "
                          f"CV split may be leaking: {leaking.index.tolist()}")

            pooled_metrics = compute_pooled_metrics(all_preds)
            per_patient_df = compute_per_patient_pooled(all_preds)

            if args.pooled_csv:
                per_patient_df.to_csv(args.pooled_csv, index=False)
                print(f"Per-patient pooled breakdown saved to {args.pooled_csv}")
        else:
            print("  No prediction tables found. Run with --no-pooled to skip.")
            print("  Tables are logged when log_model is not disabled and stage=test runs.")

    print_results(summary_df, pooled_metrics, per_patient_df)

    if args.csv:
        summary_df.to_csv(args.csv, index=False)
        print(f"Per-fold summary saved to {args.csv}")


if __name__ == "__main__":
    main()

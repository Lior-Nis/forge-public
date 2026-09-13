"""
Evaluate a model on FogAtHome Daily Living (external cohort).

Loads each trained classification checkpoint (fold 0–N), runs inference on all
11 FogAtHome Daily Living patients, ensembles predictions across folds, then
computes AP, F1, precision, recall, specificity, ROC-AUC, accuracy — with
optional bootstrap CIs and clinical ICC metrics (%TF, episode count, duration).

Uses the pure+valid view (purity==1 & validity==1) of the dataset.

Usage:
    uv run python scripts/eval/eval_fogathome_dailyliving.py \
        --ckpt-pattern "checkpoints/classification/probe_X_fold{fold}/last.ckpt" \
        --model-name my_model \
        --n-folds 3 \
        --output-dir logs/fogathome_dailyliving_eval

    # With bootstrap CIs and Youden threshold:
    uv run python scripts/eval/eval_fogathome_dailyliving.py \
        --ckpt-pattern "..." --model-name my_model --n-folds 3 \
        --threshold-method youden --bootstrap --output-dir logs/fogathome_dailyliving_eval

    # With ICC analysis:
    uv run python scripts/eval/eval_fogathome_dailyliving.py \
        --ckpt-pattern "..." --model-name my_model --n-folds 3 \
        --compute-icc --output-dir logs/fogathome_dailyliving_eval
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.set_float32_matmul_precision("high")

FOGATHOME_DAILYLIVING_PATIENTS = [
    "c00001", "c00002", "c00004", "c00005", "c00006", "c00007",
    "c00010", "c00011", "c00012", "c00013", "c00014",
]

logger = logging.getLogger(__name__)


def run_inference(fold: int, ckpt_path: str, output_dir: Path,
                  model_name: str, batch_size: int = 32,
                  dataset_path: str = None) -> pd.DataFrame:
    pred_csv = output_dir / f"{model_name}_fold{fold}_fogathome_dailyliving_preds.csv"
    if pred_csv.exists():
        logger.info(f"Fold {fold}: using cached {pred_csv}")
        return pd.read_csv(pred_csv)

    if dataset_path is None:
        raise ValueError(
            "--dataset-path is required. Pass the zarr/view that matches your model's "
            "window size, e.g. views/fogathome_dailyliving_valid_shortcontext.yaml"
        )

    logger.info(f"Fold {fold}: loading {ckpt_path}")

    from pipeline.classification import ClassificationPipeline
    from data.datamodule.datamodule import FOGDataModule
    from utils.paths import normalize_data_paths

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["hyper_parameters"]["config"]

    normalize_data_paths(config.data.paths)

    # Override: point to FogAtHome Daily Living view, all 11 patients as test
    from data.datamodule.config import SplitsConfig
    data_cfg = config.data.model_copy(update={
        "dataloader": config.data.dataloader.model_copy(update={"batch_size": batch_size}),
        "paths": config.data.paths.model_copy(update={
            "dataset_path": dataset_path,
        }),
        "splits": SplitsConfig(train=[], val=[], test=FOGATHOME_DAILYLIVING_PATIENTS),
    })

    data_module = FOGDataModule(data_cfg=data_cfg, task_type=config.train.pipeline_type)
    data_module.setup("test")

    model = ClassificationPipeline(config)

    # Resize patient normalizer buffer before loading (shape mismatch across cohorts)
    state_dict = ckpt["state_dict"]
    for key in ["preprocessors.preprocessors.3.normalizer.mean",
                "preprocessors.preprocessors.3.normalizer.stdev"]:
        if key in state_dict:
            saved_shape = state_dict[key].shape
            parts = key.split(".")
            mod = model
            for part in parts[:-1]:
                mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
            attr_name = parts[-1]
            current = getattr(mod, attr_name)
            if current.shape != saved_shape:
                setattr(mod, attr_name, torch.zeros(saved_shape))

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Skip stats injection — external cohort has no training fold; RevIN handles session shift
    rows = []
    test_loader = data_module.test_dataloader()

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            patch_y = batch['patch_y']
            meta = batch['metadata']

            logits = model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            labels = patch_y.numpy() if isinstance(patch_y, torch.Tensor) else np.array(patch_y)

            for i, m in enumerate(meta):
                rows.append({
                    "patient_id": m.get("patient_id", "unknown"),
                    "session_id": m.get("session_id", "unknown"),
                    "global_idx": int(m.get("global_idx", len(rows))),
                    "true_label": int(labels[i]),
                    "pred_prob_fog": float(probs[i]),
                    "fold": fold,
                })

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(pred_csv, index=False)
    logger.info(f"Fold {fold}: {len(df)} predictions, pos_rate={df['true_label'].mean():.3f}")
    return df


def compute_metrics(y_true, y_prob, threshold=None, threshold_method="f1_max"):
    ap = average_precision_score(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    prevalence = y_true.mean()
    norm_ap = (ap - prevalence) / (1.0 - prevalence) if prevalence < 1.0 else float("nan")
    ap_lift = ap / prevalence if prevalence > 0 else float("nan")

    if threshold is None:
        if threshold_method == "youden":
            fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
            youden = tpr - fpr
            threshold = float(thresholds_roc[np.argmax(youden)])
        else:  # f1_max (default)
            prec_c, rec_c, thresholds_pr = precision_recall_curve(y_true, y_prob)
            f1s = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)
            threshold = float(thresholds_pr[np.argmax(f1s[:-1])])

    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "AUC": auc, "AP": ap, "NormAP": norm_ap, "Lift": ap_lift, "Prevalence": prevalence,
        "F1": f1, "Precision": prec, "Recall": rec, "Specificity": spec,
        "Accuracy": acc, "threshold": threshold, "threshold_method": threshold_method,
    }


def bootstrap_ci(y_true, y_prob, threshold, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence intervals for AUC and F1."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        aucs.append(roc_auc_score(yt, yp))
        f1s.append(f1_score(yt, (yp >= threshold).astype(int), zero_division=0))
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    return {
        "AUC_ci_lo": float(np.quantile(aucs, lo)),
        "AUC_ci_hi": float(np.quantile(aucs, hi)),
        "F1_ci_lo": float(np.quantile(f1s, lo)),
        "F1_ci_hi": float(np.quantile(f1s, hi)),
    }


def print_results(model_name: str, pooled: pd.DataFrame, ensemble: pd.DataFrame,
                  threshold_method: str = "f1_max", run_bootstrap: bool = False,
                  generalizability_tier: str = "Tier 2"):
    print(f"\n{'='*72}")
    print(f"Model: {model_name}  |  FogAtHome Daily Living N=11 patients  |  {generalizability_tier}")
    print(f"Threshold method: {threshold_method}  |  Outcome: sample/window-level patches")
    print(f"{'='*72}")

    for fold in sorted(pooled["fold"].unique()):
        fd = pooled[pooled["fold"] == fold]
        if fd["true_label"].nunique() < 2:
            continue
        m = compute_metrics(fd["true_label"].values, fd["pred_prob_fog"].values,
                            threshold_method=threshold_method)
        print(f"  Fold {fold}: AUC={m['AUC']:.4f}  AP={m['AP']:.4f}  NormAP={m['NormAP']:.4f}  "
              f"Lift={m['Lift']:.1f}x  F1={m['F1']:.4f}  Rec={m['Recall']:.4f}  "
              f"Spec={m['Specificity']:.4f}  Acc={m['Accuracy']:.4f}")

    print(f"\n  --- Ensemble (mean prob across {pooled['fold'].nunique()} fold models) ---")
    m = compute_metrics(ensemble["true_label"].values, ensemble["mean_prob"].values,
                        threshold_method=threshold_method)
    n_patients = ensemble["patient_id"].nunique()
    print(f"  AUC={m['AUC']:.4f}  AP={m['AP']:.4f}  NormAP={m['NormAP']:.4f}  Lift={m['Lift']:.1f}x")
    print(f"  F1={m['F1']:.4f}  Prec={m['Precision']:.4f}  Rec={m['Recall']:.4f}  "
          f"Spec={m['Specificity']:.4f}  Acc={m['Accuracy']:.4f}")
    print(f"  ({len(ensemble)} patches, {n_patients} patients, "
          f"prevalence={m['Prevalence']:.4f}, threshold={m['threshold']:.3f})")

    if run_bootstrap:
        ci = bootstrap_ci(ensemble["true_label"].values, ensemble["mean_prob"].values,
                          threshold=m["threshold"])
        print(f"  95% CI  AUC=[{ci['AUC_ci_lo']:.4f}, {ci['AUC_ci_hi']:.4f}]  "
              f"F1=[{ci['F1_ci_lo']:.4f}, {ci['F1_ci_hi']:.4f}]")
        m.update(ci)

    print(f"{'='*72}\n")

    return m


# ─── Clinical ICC analysis ────────────────────────────────────────────────────

def _detect_episodes(binary: np.ndarray, gap_tolerance: int = 3) -> list[tuple[int, int]]:
    """
    Find FOG episodes in a binary patch sequence.

    Consecutive positive patches separated by ≤ gap_tolerance negative patches
    are merged into one episode.  Returns list of (start_idx, end_idx) pairs
    (inclusive, in patch index space).
    """
    if binary.sum() == 0:
        return []
    episodes = []
    in_ep = False
    ep_start = 0
    gap_count = 0
    for i, v in enumerate(binary):
        if v:
            if not in_ep:
                in_ep, ep_start, gap_count = True, i, 0
            else:
                gap_count = 0
        else:
            if in_ep:
                gap_count += 1
                if gap_count > gap_tolerance:
                    episodes.append((ep_start, i - gap_count))
                    in_ep, gap_count = False, 0
    if in_ep:
        episodes.append((ep_start, len(binary) - 1))
    return episodes


def compute_session_clinical_metrics(
    ensemble_df: pd.DataFrame,
    threshold: float,
    patch_window_s: float = 2.0,
    patch_stride_s: float = 0.2,
    gap_tolerance_patches: int = 3,
) -> pd.DataFrame:
    """
    Compute per-session clinical endpoints from ensemble predictions.

    For each (patient_id, session_id):
      - pct_tf_gt / pct_tf_pred    : % time frozen (fraction of patches)
      - n_episodes_gt / _pred      : number of FOG episodes (connected components)
      - duration_gt_s / _pred_s    : total FOG duration (seconds)
      - total_duration_s           : session duration (n_patches × stride_s)

    Returns one row per session.
    """
    rows = []
    for (pid, sid), grp in ensemble_df.groupby(["patient_id", "session_id"]):
        grp = grp.sort_values("global_idx")
        gt  = grp["true_label"].values.astype(int)
        pr  = (grp["mean_prob"].values >= threshold).astype(int)
        n   = len(grp)

        gt_eps   = _detect_episodes(gt,  gap_tolerance_patches)
        pred_eps = _detect_episodes(pr,  gap_tolerance_patches)

        def ep_duration_s(eps):
            return sum((e - s) * patch_stride_s + patch_window_s for s, e in eps)

        rows.append({
            "patient_id":       pid,
            "session_id":       sid,
            "n_patches":        n,
            "total_duration_s": n * patch_stride_s,
            "pct_tf_gt":        gt.mean() * 100,
            "pct_tf_pred":      pr.mean() * 100,
            "n_episodes_gt":    len(gt_eps),
            "n_episodes_pred":  len(pred_eps),
            "duration_gt_s":    ep_duration_s(gt_eps),
            "duration_pred_s":  ep_duration_s(pred_eps),
        })
    return pd.DataFrame(rows)


def compute_icc(session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ICC(1) and ICC(2,1) for %TF, episode count, and duration.

    Uses pingouin.intraclass_corr with raters = ['model', 'gt'].
    Returns a summary DataFrame with one row per clinical endpoint.
    """
    import pingouin as pg

    metrics = [
        ("pct_tf",       "pct_tf_gt",        "pct_tf_pred",       "% Time Frozen"),
        ("n_episodes",   "n_episodes_gt",     "n_episodes_pred",   "Episode Count"),
        ("duration_s",   "duration_gt_s",     "duration_pred_s",   "Episode Duration (s)"),
    ]

    results = []
    for key, gt_col, pred_col, label in metrics:
        n_sessions = len(session_df)
        if n_sessions < 3:
            logger.warning(f"ICC: only {n_sessions} sessions — ICC unreliable, skipping {label}")
            continue

        long = pd.concat([
            session_df[["session_id"]].assign(rater="gt",    rating=session_df[gt_col].values),
            session_df[["session_id"]].assign(rater="model", rating=session_df[pred_col].values),
        ], ignore_index=True)

        try:
            icc_df = pg.intraclass_corr(
                data=long, targets="session_id", raters="rater", ratings="rating"
            ).set_index("Type")
            icc1 = icc_df.loc["ICC1", "ICC"]
            icc2 = icc_df.loc["ICC2", "ICC"]
            ci1  = icc_df.loc["ICC1", "CI95%"]
            ci2  = icc_df.loc["ICC2", "CI95%"]
            results.append({
                "endpoint": label,
                "ICC1":     round(icc1, 3),
                "ICC1_CI":  f"[{ci1[0]:.3f}, {ci1[1]:.3f}]",
                "ICC2":     round(icc2, 3),
                "ICC2_CI":  f"[{ci2[0]:.3f}, {ci2[1]:.3f}]",
                "n_sessions": n_sessions,
                "gt_mean":   round(session_df[gt_col].mean(), 2),
                "pred_mean": round(session_df[pred_col].mean(), 2),
            })
        except Exception as exc:
            logger.warning(f"ICC failed for {label}: {exc}")

    return pd.DataFrame(results)


def print_icc_results(icc_df: pd.DataFrame, session_df: pd.DataFrame, model_name: str):
    print(f"\n{'='*65}")
    print(f"Clinical ICC Analysis: {model_name}  |  N={len(session_df)} sessions")
    print(f"{'='*65}")
    print(f"  {'Endpoint':<28}  {'ICC1':>6}  {'95% CI':>18}  {'ICC2':>6}  {'95% CI':>18}")
    for _, row in icc_df.iterrows():
        print(f"  {row['endpoint']:<28}  {row['ICC1']:>6.3f}  {row['ICC1_CI']:>18}  "
              f"{row['ICC2']:>6.3f}  {row['ICC2_CI']:>18}")
    print()
    print(f"  {'Endpoint':<28}  {'GT mean':>10}  {'Pred mean':>10}")
    for _, row in icc_df.iterrows():
        print(f"  {row['endpoint']:<28}  {row['gt_mean']:>10.2f}  {row['pred_mean']:>10.2f}")
    print(f"{'='*65}\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--ckpt-pattern", required=True,
                        help="e.g. checkpoints/classification/probe_X_fold{fold}/last.ckpt")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="logs/fogathome_dailyliving_eval")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset-path", required=True,
                        help="Zarr or view YAML to evaluate on. Must match the model's training "
                             "window size, e.g. views/fogathome_dailyliving_valid_shortcontext.yaml")
    parser.add_argument("--threshold-method", default="f1_max",
                        choices=["f1_max", "youden"],
                        help="How to select the operating-point threshold.")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Compute 95%% bootstrap CIs for AUC and F1 (n=1000 resamples).")
    parser.add_argument("--generalizability-tier", default="Tier 2",
                        help="Generalizability tier declaration (e.g. 'Tier 1', 'Tier 2').")
    parser.add_argument("--compute-icc", action="store_true",
                        help="Compute clinical ICC metrics (%TF, episode count, duration).")
    parser.add_argument("--icc-threshold", type=float, default=None,
                        help="Threshold for binarising predictions in ICC analysis. "
                             "Default: auto from threshold-method on ensemble.")
    parser.add_argument("--icc-gap-tolerance", type=int, default=3,
                        help="Max gap (patches) between positive patches within one episode.")
    parser.add_argument("--patch-window-s", type=float, default=2.0,
                        help="Patch window duration in seconds (200 steps @ 100Hz = 2.0s).")
    parser.add_argument("--patch-stride-s", type=float, default=0.2,
                        help="Stride between patches in seconds (20 steps @ 100Hz = 0.2s).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    all_dfs = []
    for fold in range(args.n_folds):
        ckpt_path = args.ckpt_pattern.format(fold=fold)
        if not Path(ckpt_path).exists():
            logger.warning(f"Missing: {ckpt_path}")
            continue
        df = run_inference(fold, ckpt_path, output_dir, args.model_name, args.batch_size,
                           dataset_path=args.dataset_path)
        all_dfs.append(df)

    if not all_dfs:
        logger.error("No predictions collected.")
        return

    pooled = pd.concat(all_dfs, ignore_index=True)
    pooled.to_csv(output_dir / f"{args.model_name}_fogathome_dailyliving_all_folds.csv", index=False)

    # Ensemble: average probabilities across fold models per patch (identified by global_idx)
    ensemble = (
        pooled.groupby(["patient_id", "session_id", "global_idx", "true_label"])
        .agg(mean_prob=("pred_prob_fog", "mean"), n_folds=("pred_prob_fog", "count"))
        .reset_index()
    )
    ensemble.to_csv(output_dir / f"{args.model_name}_fogathome_dailyliving_ensemble.csv", index=False)

    metrics = print_results(
        args.model_name, pooled, ensemble,
        threshold_method=args.threshold_method,
        run_bootstrap=args.bootstrap,
        generalizability_tier=args.generalizability_tier,
    )

    summary = {"model": args.model_name, **metrics}
    pd.DataFrame([summary]).to_csv(
        output_dir / f"{args.model_name}_fogathome_dailyliving_summary.csv", index=False
    )

    # ── Clinical ICC analysis ─────────────────────────────────────────────────
    if args.compute_icc:
        threshold = args.icc_threshold if args.icc_threshold is not None else metrics["threshold"]
        logger.info(f"ICC analysis: threshold={threshold:.3f}, "
                    f"gap_tolerance={args.icc_gap_tolerance} patches")

        session_df = compute_session_clinical_metrics(
            ensemble,
            threshold=threshold,
            patch_window_s=args.patch_window_s,
            patch_stride_s=args.patch_stride_s,
            gap_tolerance_patches=args.icc_gap_tolerance,
        )
        session_df.to_csv(
            output_dir / f"{args.model_name}_fogathome_dailyliving_session_metrics.csv", index=False
        )
        logger.info(f"Session metrics: {len(session_df)} sessions across "
                    f"{session_df['patient_id'].nunique()} patients")

        icc_df = compute_icc(session_df)
        icc_df.to_csv(
            output_dir / f"{args.model_name}_fogathome_dailyliving_icc.csv", index=False
        )
        print_icc_results(icc_df, session_df, args.model_name)


if __name__ == "__main__":
    main()

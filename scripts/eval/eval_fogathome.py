"""
Evaluate a model on FogAtHome (external cohort).

Loads each trained classification checkpoint (fold 0–N), runs inference on all
12 FogAtHome patients, ensembles predictions across folds, then computes AP, F1,
precision, recall, specificity, ROC-AUC, accuracy — with optional bootstrap CIs
and clinical ICC metrics (%TF, episode count, duration) using pingouin.

Supports two evaluation granularities:
  - Segment-level (default): metrics on patch/window predictions
  - Frame-level (--frame-level): logits averaged across overlapping patches →
    per-frame predictions evaluated against per-frame ground truth labels from zarr.

Usage:
    uv run python scripts/eval/eval_fogathome.py \
        --ckpt-pattern "checkpoints/classification/probe_X_fold{fold}/last.ckpt" \
        --model-name my_model --n-folds 3 \
        --dataset-path data/processed/len1000_stride200_fogathome.zarr \
        --patch-window-s 10.0 --patch-stride-s 2.0 \
        --threshold-method youden --bootstrap --compute-icc

    # Frame-level evaluation (soft-label models):
    uv run python scripts/eval/eval_fogathome.py \
        --ckpt-pattern "..." --model-name my_model --n-folds 3 \
        --dataset-path data/processed/len1000_stride200_fogathome.zarr \
        --patch-window-s 10.0 --patch-stride-s 2.0 \
        --frame-level --threshold-method youden --bootstrap --compute-icc
"""

import argparse
import logging
import os
from collections import defaultdict
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

FOGATHOME_PATIENTS = [
    "a00001", "a00002", "a00004", "a00006", "a00007", "a00008",
    "a00009", "a00010", "a00011", "a00012", "a00013", "a00014",
]

logger = logging.getLogger(__name__)


def _uses_fog_ratio_loss(config) -> bool:
    """Detect whether the model was trained with FogRatioLoss (soft BCE)."""
    try:
        target = config.train.loss._target_
        return "FogRatioLoss" in target
    except Exception:
        return False


def run_fogathome_inference(fold: int, ckpt_path: str, output_dir: Path,
                            model_name: str, batch_size: int = 32,
                            dataset_path: str = None) -> pd.DataFrame:
    pred_csv = output_dir / f"{model_name}_fold{fold}_fogathome_preds.csv"
    if pred_csv.exists():
        logger.info(f"Fold {fold}: using cached {pred_csv}")
        return pd.read_csv(pred_csv)

    if dataset_path is None:
        raise ValueError(
            "--dataset-path is required. Pass the zarr/view that matches your model's "
            "window size, e.g. len200_stride20_fogstride10_anyfog_fogathome.zarr"
        )

    logger.info(f"Fold {fold}: loading {ckpt_path}")

    from pipeline.classification import ClassificationPipeline
    from data.datamodule.datamodule import FOGDataModule
    from utils.paths import normalize_data_paths

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["hyper_parameters"]["config"]

    normalize_data_paths(config.data.paths)

    use_sigmoid = _uses_fog_ratio_loss(config)
    if use_sigmoid:
        logger.info(f"Fold {fold}: FogRatioLoss detected — using sigmoid (not softmax) for probs")

    from data.datamodule.config import SplitsConfig
    data_cfg = config.data.model_copy(update={
        "dataloader": config.data.dataloader.model_copy(update={"batch_size": batch_size}),
        "paths": config.data.paths.model_copy(update={
            "dataset_path": dataset_path,
        }),
        "splits": SplitsConfig(train=[], val=[], test=FOGATHOME_PATIENTS),
    })

    data_module = FOGDataModule(data_cfg=data_cfg, task_type=config.train.pipeline_type)
    data_module.setup("test")

    model = ClassificationPipeline(config)

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

    rows = []
    test_loader = data_module.test_dataloader()

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            patch_y = batch['patch_y']
            meta = batch['metadata']

            logits = model(x)

            if use_sigmoid:
                pos_logits = logits[:, 1] if logits.dim() == 2 else logits.squeeze(-1)
                probs = torch.sigmoid(pos_logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

            raw_labels = patch_y.numpy() if isinstance(patch_y, torch.Tensor) else np.array(patch_y)
            # Binarize: handles both binary long labels and float fog_ratio labels
            bin_labels = (raw_labels > 0.5).astype(int) if raw_labels.dtype.kind == 'f' else raw_labels.astype(int)

            for i, m in enumerate(meta):
                rows.append({
                    "patient_id":  m.get("patient_id", "unknown"),
                    "session_id":  m.get("session_id", "unknown"),
                    "global_idx":  int(m.get("global_idx", len(rows))),
                    "session_idx": int(m.get("session_idx", -1)),
                    "true_label":  int(bin_labels[i]),
                    "pred_prob_fog": float(probs[i]),
                    "fold": fold,
                })

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(pred_csv, index=False)
    logger.info(f"Fold {fold}: {len(df)} predictions, pos_rate={df['true_label'].mean():.3f}")
    return df


# ─── Frame-level aggregation ──────────────────────────────────────────────────

def aggregate_to_frame_level(ensemble_df: pd.DataFrame, dataset_path: str,
                              seq_len: int, stride_frames: int) -> pd.DataFrame:
    """
    Convert patch-level ensemble predictions to frame-level predictions via
    sliding-window logit averaging.

    For each absolute frame (within a session), averages the predicted probabilities
    from all overlapping patches that cover that frame. Ground-truth labels are read
    directly from the zarr `labels` array (per-frame annotations).

    Args:
        ensemble_df: patch-level ensemble with columns:
                     patient_id, session_id, global_idx, session_idx, mean_prob
        dataset_path: path to the zarr dataset (used to read per-frame labels)
        seq_len: patch length in frames (e.g. 1000 for LC, 500 for MC, 200 for SC)
        stride_frames: stride between patches in frames (e.g. 200, 100, 20)

    Returns:
        DataFrame with columns: patient_id, session_id, abs_frame, pred_prob, true_label, n_patches
    """
    import zarr

    logger.info(f"Frame aggregation: seq_len={seq_len}, stride={stride_frames} frames, "
                f"loading labels from {dataset_path}")

    z = zarr.open_group(dataset_path, mode='r')
    labels_array = z['labels'][:]  # [n_patches, seq_len] — per-frame annotations

    rows = []
    for (pid, sid), grp in ensemble_df.groupby(["patient_id", "session_id"]):
        frame_probs = defaultdict(list)
        frame_labels = {}

        for _, row in grp.iterrows():
            sidx = int(row["session_idx"])
            gidx = int(row["global_idx"])
            start_frame = sidx * stride_frames
            patch_frame_labels = labels_array[gidx]  # [seq_len]

            for f in range(seq_len):
                abs_frame = start_frame + f
                frame_probs[abs_frame].append(float(row["mean_prob"]))
                if abs_frame not in frame_labels:
                    frame_labels[abs_frame] = int(patch_frame_labels[f] > 0)

        for abs_frame in sorted(frame_probs.keys()):
            rows.append({
                "patient_id": pid,
                "session_id": sid,
                "abs_frame":  abs_frame,
                "pred_prob":  float(np.mean(frame_probs[abs_frame])),
                "true_label": frame_labels[abs_frame],
                "n_patches":  len(frame_probs[abs_frame]),
            })

    return pd.DataFrame(rows)


def compute_session_clinical_metrics_frame(
    frame_df: pd.DataFrame,
    threshold: float,
    sample_rate: float = 100.0,
    gap_tolerance_frames: int = 50,
) -> pd.DataFrame:
    """
    Compute per-session clinical endpoints from frame-level predictions.

    Episode detection operates directly on the binary frame sequence, avoiding
    the patch-fragmentation problem. Gap tolerance is in frames (default 50 = 0.5s).
    """
    rows = []
    for (pid, sid), grp in frame_df.groupby(["patient_id", "session_id"]):
        grp = grp.sort_values("abs_frame")
        gt = grp["true_label"].values.astype(int)
        pr = (grp["pred_prob"].values >= threshold).astype(int)
        n_frames = len(grp)

        gt_eps   = _detect_episodes(gt, gap_tolerance_frames)
        pred_eps = _detect_episodes(pr, gap_tolerance_frames)

        def ep_dur_s(eps):
            return sum((e - s + 1) / sample_rate for s, e in eps)

        rows.append({
            "patient_id":       pid,
            "session_id":       sid,
            "n_frames":         n_frames,
            "total_duration_s": n_frames / sample_rate,
            "pct_tf_gt":        gt.mean() * 100,
            "pct_tf_pred":      pr.mean() * 100,
            "n_episodes_gt":    len(gt_eps),
            "n_episodes_pred":  len(pred_eps),
            "duration_gt_s":    ep_dur_s(gt_eps),
            "duration_pred_s":  ep_dur_s(pred_eps),
        })
    return pd.DataFrame(rows)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, threshold=None, threshold_method="f1_max"):
    ap = average_precision_score(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    if threshold is None:
        if threshold_method == "youden":
            fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
            youden = tpr - fpr
            threshold = float(thresholds_roc[np.argmax(youden)])
        else:
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
        "AUC": auc, "AP": ap, "F1": f1,
        "Precision": prec, "Recall": rec, "Specificity": spec,
        "Accuracy": acc, "threshold": threshold,
        "threshold_method": threshold_method,
    }


def bootstrap_ci(y_true, y_prob, threshold, n_boot=1000, ci=0.95, seed=42):
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
        "F1_ci_lo":  float(np.quantile(f1s, lo)),
        "F1_ci_hi":  float(np.quantile(f1s, hi)),
    }


def print_results(model_name: str, pooled: pd.DataFrame, ensemble: pd.DataFrame,
                  threshold_method: str = "f1_max", run_bootstrap: bool = False,
                  generalizability_tier: str = "Tier 2", level: str = "segment"):
    print(f"\n{'='*72}")
    print(f"Model: {model_name}  |  FogAtHome N=12 patients  |  {generalizability_tier}")
    print(f"Threshold: {threshold_method}  |  Level: {level}")
    print(f"{'='*72}")

    prob_col = "pred_prob_fog" if "pred_prob_fog" in pooled.columns else "pred_prob"
    ens_prob_col = "mean_prob" if "mean_prob" in ensemble.columns else "pred_prob"

    for fold in sorted(pooled["fold"].unique()):
        fd = pooled[pooled["fold"] == fold]
        if fd["true_label"].nunique() < 2:
            continue
        m = compute_metrics(fd["true_label"].values, fd[prob_col].values,
                            threshold_method=threshold_method)
        print(f"  Fold {fold}: AUC={m['AUC']:.4f}  AP={m['AP']:.4f}  F1={m['F1']:.4f}  "
              f"Rec={m['Recall']:.4f}  Spec={m['Specificity']:.4f}")

    print(f"\n  --- Ensemble ({pooled['fold'].nunique()} fold models) ---")
    m = compute_metrics(ensemble["true_label"].values, ensemble[ens_prob_col].values,
                        threshold_method=threshold_method)
    n_patients = ensemble["patient_id"].nunique()
    print(f"  AUC={m['AUC']:.4f}  AP={m['AP']:.4f}  F1={m['F1']:.4f}  "
          f"Prec={m['Precision']:.4f}  Rec={m['Recall']:.4f}  "
          f"Spec={m['Specificity']:.4f}  Acc={m['Accuracy']:.4f}")
    print(f"  ({len(ensemble)} items, {n_patients} patients, "
          f"pos_rate={ensemble['true_label'].mean():.3f}, threshold={m['threshold']:.3f})")

    if run_bootstrap:
        ci = bootstrap_ci(ensemble["true_label"].values, ensemble[ens_prob_col].values,
                          threshold=m["threshold"])
        print(f"  95% CI  AUC=[{ci['AUC_ci_lo']:.4f}, {ci['AUC_ci_hi']:.4f}]  "
              f"F1=[{ci['F1_ci_lo']:.4f}, {ci['F1_ci_hi']:.4f}]")
        m.update(ci)

    print(f"{'='*72}\n")
    return m


# ─── Clinical ICC ─────────────────────────────────────────────────────────────

def _detect_episodes(binary: np.ndarray, gap_tolerance: int = 3) -> list[tuple[int, int]]:
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
    rows = []
    for (pid, sid), grp in ensemble_df.groupby(["patient_id", "session_id"]):
        grp = grp.sort_values("global_idx")
        gt  = grp["true_label"].values.astype(int)
        pr  = (grp["mean_prob"].values >= threshold).astype(int)
        n   = len(grp)

        gt_eps   = _detect_episodes(gt, gap_tolerance_patches)
        pred_eps = _detect_episodes(pr, gap_tolerance_patches)

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
    import pingouin as pg

    metrics = [
        ("pct_tf",     "pct_tf_gt",     "pct_tf_pred",     "% Time Frozen"),
        ("n_episodes", "n_episodes_gt",  "n_episodes_pred", "Episode Count"),
        ("duration_s", "duration_gt_s", "duration_pred_s", "Episode Duration (s)"),
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
                "endpoint":   label,
                "ICC1":       round(icc1, 3),
                "ICC1_CI":    f"[{ci1[0]:.3f}, {ci1[1]:.3f}]",
                "ICC2":       round(icc2, 3),
                "ICC2_CI":    f"[{ci2[0]:.3f}, {ci2[1]:.3f}]",
                "n_sessions": n_sessions,
                "gt_mean":    round(session_df[gt_col].mean(), 2),
                "pred_mean":  round(session_df[pred_col].mean(), 2),
            })
        except Exception as exc:
            logger.warning(f"ICC failed for {label}: {exc}")

    return pd.DataFrame(results)


def print_icc_results(icc_df: pd.DataFrame, session_df: pd.DataFrame, model_name: str,
                      level: str = "segment"):
    print(f"\n{'='*65}")
    print(f"Clinical ICC [{level}]: {model_name}  |  N={len(session_df)} sessions")
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--ckpt-pattern", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="logs/fogathome_eval")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--threshold-method", default="f1_max",
                        choices=["f1_max", "youden"])
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--generalizability-tier", default="Tier 2")
    parser.add_argument("--compute-icc", action="store_true")
    parser.add_argument("--icc-threshold", type=float, default=None)
    parser.add_argument("--icc-gap-tolerance", type=int, default=3,
                        help="Gap tolerance for episode detection: patches (segment) "
                             "or frames (frame-level, default 50 = 0.5s @ 100Hz).")
    parser.add_argument("--patch-window-s", type=float, default=2.0)
    parser.add_argument("--patch-stride-s", type=float, default=0.2)
    # Frame-level evaluation
    parser.add_argument("--frame-level", action="store_true",
                        help="Also compute frame-level metrics via sliding-window logit averaging.")
    parser.add_argument("--sample-rate", type=float, default=100.0,
                        help="Sensor sample rate in Hz (default 100).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # ── Inference (patch-level) ───────────────────────────────────────────────
    all_dfs = []
    for fold in range(args.n_folds):
        ckpt_path = args.ckpt_pattern.format(fold=fold)
        if not Path(ckpt_path).exists():
            logger.warning(f"Missing: {ckpt_path}")
            continue
        df = run_fogathome_inference(fold, ckpt_path, output_dir, args.model_name,
                                     args.batch_size, dataset_path=args.dataset_path)
        all_dfs.append(df)

    if not all_dfs:
        logger.error("No predictions collected.")
        return

    pooled = pd.concat(all_dfs, ignore_index=True)
    pooled.to_csv(output_dir / f"{args.model_name}_fogathome_all_folds.csv", index=False)

    ensemble = (
        pooled.groupby(["patient_id", "session_id", "global_idx", "session_idx", "true_label"])
        .agg(mean_prob=("pred_prob_fog", "mean"), n_folds=("pred_prob_fog", "count"))
        .reset_index()
    )
    ensemble.to_csv(output_dir / f"{args.model_name}_fogathome_ensemble.csv", index=False)

    # ── Segment-level metrics ─────────────────────────────────────────────────
    seg_metrics = print_results(
        args.model_name, pooled, ensemble,
        threshold_method=args.threshold_method,
        run_bootstrap=args.bootstrap,
        generalizability_tier=args.generalizability_tier,
        level="segment",
    )
    pd.DataFrame([{"model": args.model_name, "level": "segment", **seg_metrics}]).to_csv(
        output_dir / f"{args.model_name}_fogathome_summary.csv", index=False
    )

    if args.compute_icc:
        threshold = args.icc_threshold if args.icc_threshold is not None else seg_metrics["threshold"]
        session_df = compute_session_clinical_metrics(
            ensemble,
            threshold=threshold,
            patch_window_s=args.patch_window_s,
            patch_stride_s=args.patch_stride_s,
            gap_tolerance_patches=args.icc_gap_tolerance,
        )
        session_df.to_csv(
            output_dir / f"{args.model_name}_fogathome_session_metrics.csv", index=False
        )
        icc_df = compute_icc(session_df)
        icc_df.to_csv(
            output_dir / f"{args.model_name}_fogathome_icc.csv", index=False
        )
        print_icc_results(icc_df, session_df, args.model_name, level="segment")

    # ── Frame-level metrics ───────────────────────────────────────────────────
    if args.frame_level:
        seq_len = int(round(args.patch_window_s * args.sample_rate))
        stride_frames = int(round(args.patch_stride_s * args.sample_rate))
        logger.info(f"Frame-level: seq_len={seq_len}, stride={stride_frames} frames")

        # Resolve full zarr path (dataset_path may be relative to processed_dir)
        first_ckpt = args.ckpt_pattern.format(fold=0)
        _ckpt = torch.load(first_ckpt, map_location="cpu", weights_only=False)
        _cfg = _ckpt["hyper_parameters"]["config"]
        from utils.paths import normalize_data_paths
        normalize_data_paths(_cfg.data.paths)
        import os as _os
        full_zarr_path = _os.path.join(_cfg.data.paths.processed_dir, args.dataset_path)

        frame_df = aggregate_to_frame_level(
            ensemble, full_zarr_path, seq_len, stride_frames
        )
        frame_df.to_csv(
            output_dir / f"{args.model_name}_fogathome_frame_preds.csv", index=False
        )

        # Build a pseudo-pooled/ensemble for print_results compatibility
        frame_ensemble = frame_df.rename(columns={"pred_prob": "mean_prob"})
        # For per-fold display, use the ensemble (frame-level has no per-fold split)
        frame_pooled = frame_df.assign(fold=0, pred_prob_fog=frame_df["pred_prob"])

        frame_metrics = print_results(
            args.model_name, frame_pooled, frame_ensemble,
            threshold_method=args.threshold_method,
            run_bootstrap=args.bootstrap,
            generalizability_tier=args.generalizability_tier,
            level="frame",
        )
        pd.DataFrame([{"model": args.model_name, "level": "frame", **frame_metrics}]).to_csv(
            output_dir / f"{args.model_name}_fogathome_frame_summary.csv", index=False
        )

        if args.compute_icc:
            frame_threshold = args.icc_threshold if args.icc_threshold is not None else frame_metrics["threshold"]
            gap_frames = args.icc_gap_tolerance if args.icc_gap_tolerance != 3 else 50
            frame_session_df = compute_session_clinical_metrics_frame(
                frame_df,
                threshold=frame_threshold,
                sample_rate=args.sample_rate,
                gap_tolerance_frames=gap_frames,
            )
            frame_session_df.to_csv(
                output_dir / f"{args.model_name}_fogathome_frame_session_metrics.csv", index=False
            )
            frame_icc_df = compute_icc(frame_session_df)
            frame_icc_df.to_csv(
                output_dir / f"{args.model_name}_fogathome_frame_icc.csv", index=False
            )
            print_icc_results(frame_icc_df, frame_session_df, args.model_name, level="frame")


if __name__ == "__main__":
    main()

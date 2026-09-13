"""
Per-timestep FogAtHome evaluation — works for both classification and segmentation models.

For classification models ([B, 2] logits): the single patch prediction is tiled
across all block_len timesteps, then overlapping patches are averaged. This gives
a step-function prediction at 200ms resolution (stride) rather than 10ms.

For segmentation models ([B, seq_len, 2] logits): per-timestep predictions are
used directly; overlapping patches are averaged.

In both cases the final comparison is per-timestep against the zarr's GT label
array — the same metric the Kaggle competition used.

Usage:
    # Classification model (patch-level, tiled per-timestep):
    uv run python scripts/eval/eval_fogathome_segmentation.py \
        --ckpt-pattern "checkpoints/classification/probe_X_fold{fold}/last.ckpt" \
        --model-name cls_probe_exp023 --n-folds 3 \
        --output-dir logs/eval/cls_probe_exp023_timestep

    # Segmentation model:
    uv run python scripts/eval/eval_fogathome_segmentation.py \
        --ckpt-pattern "checkpoints/classification/segmentation_probe_fold{fold}/last.ckpt" \
        --model-name seg_probe_exp023 --n-folds 3 \
        --output-dir logs/eval/segmentation_probe_exp023
"""

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score,
    recall_score, precision_recall_curve,
)

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.set_float32_matmul_precision("high")

FOGATHOME_PATIENTS = [
    "a00001", "a00002", "a00004", "a00006", "a00007", "a00008",
    "a00009", "a00010", "a00011", "a00012", "a00013", "a00014",
]

logger = logging.getLogger(__name__)


def reconstruct_session_timeline(patches: list, block_len: int, stride_len: int):
    """
    Average overlapping patch predictions into a single session timeline.

    Args:
        patches: list of (session_idx, probs_1000, labels_1000) tuples
        block_len: timesteps per patch
        stride_len: stride between patch starts

    Returns:
        pred: [session_len] averaged positive-class probability
        gt:   [session_len] binary GT labels (majority across overlapping patches)
    """
    if not patches:
        return np.array([]), np.array([])

    max_idx = max(p[0] for p in patches)
    session_len = max_idx * stride_len + block_len

    pred_sum = np.zeros(session_len, dtype=np.float64)
    pred_cnt = np.zeros(session_len, dtype=np.int32)
    gt_sum   = np.zeros(session_len, dtype=np.int32)
    gt_cnt   = np.zeros(session_len, dtype=np.int32)

    for session_idx, probs, labels in patches:
        start = session_idx * stride_len
        end   = start + block_len
        pred_sum[start:end] += probs
        pred_cnt[start:end] += 1
        gt_sum[start:end]   += labels.astype(np.int32)
        gt_cnt[start:end]   += 1

    valid = pred_cnt > 0
    pred = np.where(valid, pred_sum / np.maximum(pred_cnt, 1), 0.0)
    gt   = np.where(valid, (gt_sum >= gt_cnt / 2).astype(int), -1)  # majority vote for GT

    # Keep only positions where every covering patch agrees on valid/invalid
    mask = gt >= 0
    return pred[mask], gt[mask]


def run_segmentation_inference(
    fold: int,
    ckpt_path: str,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    dataset_path: str,
) -> dict:
    """
    Run segmentation model on FogAtHome, return per-patient per-timestep results.

    Returns dict: {patient_id: (pred_probs, gt_labels)}
    """
    cache = output_dir / f"{model_name}_fold{fold}_timestep_cache.npz"
    if cache.exists():
        logger.info(f"Fold {fold}: loading cached timestep predictions from {cache}")
        d = np.load(cache, allow_pickle=True)
        return dict(d["patient_data"].item())

    logger.info(f"Fold {fold}: running per-timestep inference from {ckpt_path}")

    from pipeline.segmentation import SegmentationPipeline
    from pipeline.classification import ClassificationPipeline
    from data.datamodule.datamodule import FOGDataModule
    from data.datamodule.config import SplitsConfig
    from utils.paths import normalize_data_paths

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["hyper_parameters"]["config"]
    normalize_data_paths(config.data.paths)

    _z         = zarr.open(str(Path("data/processed") / dataset_path))
    block_len  = int(_z.attrs.get("block_len", 1000))
    stride_len = int(_z.attrs.get("stride_len", 200))

    pipeline_type = getattr(config.train, "pipeline_type", "classification")
    task_type = pipeline_type if pipeline_type in ("segmentation", "classification") else "classification"

    data_cfg = config.data.model_copy(update={
        "dataloader": config.data.dataloader.model_copy(update={"batch_size": batch_size}),
        "paths": config.data.paths.model_copy(update={"dataset_path": dataset_path}),
        "splits": SplitsConfig(train=[], val=[], test=FOGATHOME_PATIENTS),
    })

    data_module = FOGDataModule(data_cfg=data_cfg, task_type=task_type)
    data_module.setup("test")

    if pipeline_type == "segmentation":
        model = SegmentationPipeline(config)
    else:
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
            current = getattr(mod, parts[-1])
            if current.shape != saved_shape:
                setattr(mod, parts[-1], torch.zeros(saved_shape))

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    logger.info(f"Fold {fold}: pipeline_type={pipeline_type}, block_len={block_len}, stride={stride_len}")

    # Accumulate per-session patches: {session_id: [(session_idx, probs, labels)]}
    session_patches = defaultdict(list)
    session_patient = {}

    with torch.no_grad():
        for batch in data_module.test_dataloader():
            x    = batch['x'].to(device)
            y    = batch['y']   # [B, seq_len] per-timestep GT
            meta = batch['metadata']

            logits = model(x)

            # Normalise logits to per-timestep probs [B, block_len]
            if logits.dim() == 3:
                # Segmentation: [B, seq_len, 2] — take positive class
                probs = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()
            else:
                # Classification: [B, 2] — tile single score across all timesteps
                patch_prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # [B]
                probs = np.repeat(patch_prob[:, None], block_len, axis=1)        # [B, block_len]

            labels = y.numpy() if isinstance(y, torch.Tensor) else np.array(y)

            for i, m in enumerate(meta):
                sid         = m["session_id"]
                session_idx = int(m.get("session_idx", 0))
                pid         = m.get("patient_id", "unknown")

                session_patches[sid].append((session_idx, probs[i], labels[i]))
                session_patient[sid] = pid

    # Reconstruct per-patient timelines
    patient_pred  = defaultdict(list)
    patient_gt    = defaultdict(list)

    for sid, patches in session_patches.items():
        patches.sort(key=lambda p: p[0])
        pred, gt = reconstruct_session_timeline(patches, block_len, stride_len)
        if len(pred) == 0:
            continue
        pid = session_patient[sid]
        patient_pred[pid].append(pred)
        patient_gt[pid].append(gt)

    patient_data = {}
    for pid in patient_pred:
        patient_data[pid] = (
            np.concatenate(patient_pred[pid]),
            np.concatenate(patient_gt[pid]),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, patient_data=np.array(patient_data, dtype=object))
    return patient_data


def compute_metrics(y_true, y_prob, threshold=None):
    if len(np.unique(y_true)) < 2:
        return {"AP": float("nan"), "F1": 0.0, "Precision": 0.0, "Recall": 0.0,
                "Specificity": 0.0, "threshold": 0.5, "n_pos": int(y_true.sum()),
                "n_total": len(y_true)}
    ap = average_precision_score(y_true, y_prob)
    if threshold is None:
        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        threshold = float(thresholds[np.argmax(f1s[:-1])])
    y_pred = (y_prob >= threshold).astype(int)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    tn   = ((y_pred == 0) & (y_true == 0)).sum()
    fp   = ((y_pred == 1) & (y_true == 0)).sum()
    spec = tn / (tn + fp + 1e-8)
    prev = y_true.mean()
    norm_ap = (ap - prev) / (1 - prev + 1e-8)
    return {"AP": ap, "normAP": norm_ap, "prevalence": prev,
            "F1": f1, "Precision": prec, "Recall": rec, "Specificity": spec,
            "threshold": threshold, "n_pos": int(y_true.sum()), "n_total": len(y_true)}


def print_results(model_name, fold_results, ensemble):
    print(f"\n{'='*70}")
    print(f"Model: {model_name}  |  FogAtHome N=12  |  per-timestep AP")
    print(f"{'='*70}")

    for fold, m in sorted(fold_results.items()):
        print(f"  Fold {fold}: AP={m['AP']:.4f}  normAP={m['normAP']:.4f}  "
              f"F1={m['F1']:.4f}  Prec={m['Precision']:.4f}  "
              f"Rec={m['Recall']:.4f}  prev={m['prevalence']:.3f}")

    m = ensemble
    print(f"\n  --- Ensemble ({len(fold_results)} folds) ---")
    print(f"  AP={m['AP']:.4f}  normAP={m['normAP']:.4f}  F1={m['F1']:.4f}  "
          f"Prec={m['Precision']:.4f}  Rec={m['Recall']:.4f}  "
          f"prev={m['prevalence']:.3f}  n={m['n_total']:,}  threshold={m['threshold']:.3f}")

    print(f"\n  --- Kaggle competition top-5 (per-timestep defog AP) ---")
    comp = [("Rank 1", 0.357), ("Rank 2", 0.375), ("Rank 3", 0.343),
            ("Rank 4", 0.259), ("Rank 5", 0.302)]
    for name, ap in comp:
        print(f"  {name}: defog AP={ap:.3f}  (Kaggle test set, ~9% prevalence)")
    print(f"  NOTE: Kaggle metric is on Kaggle defog test set; FogAtHome is independent.")
    print(f"{'='*70}\n")
    return m


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--ckpt-pattern", required=True)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--output-dir", default="logs/eval/segmentation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset-path", default="len1000_stride200_fogathome.zarr")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fold_patient_data = {}

    for fold in range(args.n_folds):
        ckpt_path = args.ckpt_pattern.format(fold=fold)
        if not Path(ckpt_path).exists():
            logger.warning(f"Missing: {ckpt_path}")
            continue
        patient_data = run_segmentation_inference(
            fold, ckpt_path, output_dir, args.model_name,
            args.batch_size, args.dataset_path,
        )
        fold_patient_data[fold] = patient_data

    if not fold_patient_data:
        logger.error("No predictions collected.")
        return

    # Per-fold metrics (pooled across all patients in each fold)
    fold_results = {}
    for fold, patient_data in fold_patient_data.items():
        all_pred = np.concatenate([v[0] for v in patient_data.values()])
        all_gt   = np.concatenate([v[1] for v in patient_data.values()])
        keep = all_gt >= 0
        fold_results[fold] = compute_metrics(all_gt[keep].astype(int), all_pred[keep])

    # Ensemble: average pred probabilities across folds per patient
    all_patients = set().union(*[pd.keys() for pd in fold_patient_data.values()])
    ens_pred, ens_gt = [], []
    patient_rows = []

    for pid in sorted(all_patients):
        fold_preds = [fold_patient_data[f][pid][0]
                      for f in fold_patient_data if pid in fold_patient_data[f]]
        gt = fold_patient_data[list(fold_patient_data.keys())[0]][pid][1]

        min_len = min(len(p) for p in fold_preds + [gt])
        avg_pred = np.mean([p[:min_len] for p in fold_preds], axis=0)
        gt_trim  = gt[:min_len]
        keep     = gt_trim >= 0

        ens_pred.append(avg_pred[keep])
        ens_gt.append(gt_trim[keep].astype(int))

        m = compute_metrics(gt_trim[keep].astype(int), avg_pred[keep])
        patient_rows.append({"patient_id": pid, **m})

    all_ens_pred = np.concatenate(ens_pred)
    all_ens_gt   = np.concatenate(ens_gt)
    ens_metrics  = compute_metrics(all_ens_gt, all_ens_pred)

    # Save results
    pd.DataFrame(patient_rows).to_csv(
        output_dir / f"{args.model_name}_fogathome_per_patient.csv", index=False
    )
    summary = {"model": args.model_name, **ens_metrics}
    pd.DataFrame([summary]).to_csv(
        output_dir / f"{args.model_name}_fogathome_summary.csv", index=False
    )

    print_results(args.model_name, fold_results, ens_metrics)

    # Per-patient table
    print("  Per-patient ensemble AP:")
    df = pd.DataFrame(patient_rows)
    for _, row in df.iterrows():
        print(f"    {row['patient_id']}: AP={row['AP']:.3f}  normAP={row['normAP']:.3f}  "
              f"prev={row['prevalence']:.3f}")


if __name__ == "__main__":
    main()

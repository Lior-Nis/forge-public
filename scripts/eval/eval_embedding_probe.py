"""
Evaluate embedding probe GRU head on defog test folds and (optionally) FogAtHome.

For each fold: loads precomputed test embeddings + GRU head checkpoint,
runs inference, then pools predictions to compute AP.

Usage:
    # Defog pooled AP
    uv run python scripts/eval_embedding_probe.py \
        --emb-dir checkpoints/embeddings/vit12_ep6 \
        --probe-pattern "checkpoints/classification/emb_probe_vit12_ep6_fold{fold}/last.ckpt" \
        --model-name vit12_ep6 --n-folds 5

    # FogAtHome (requires fogathome embeddings precomputed separately)
    uv run python scripts/eval_embedding_probe.py \
        --emb-dir checkpoints/embeddings/vit12_ep6 \
        --probe-pattern "checkpoints/classification/emb_probe_vit12_ep6_fold{fold}/last.ckpt" \
        --model-name vit12_ep6 --n-folds 5 --fogathome
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)

KAGGLE_BENCHMARKS = [
    ("Rank 1", 0.357), ("Rank 2", 0.375), ("Rank 3", 0.343),
    ("Rank 4", 0.259), ("Rank 5", 0.302),
]
FOGATHOME_BENCHMARKS = [
    ("Rank 1", 0.644), ("Rank 2", 0.615), ("Rank 3", 0.674),
    ("Rank 4", 0.553), ("Rank 5", 0.700),
]


def load_head(ckpt_path: str, nW: int, D: int, device: str):
    from model.heads import RNNHead
    head = RNNHead(
        input_dim=D, rnn_type="GRU", rnn_layers=1, rnn_hidden_mult=1,
        num_classes=2, dropout=0.0, sequence_output=False,
        bidirectional=True, input_seq_len=nW,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["state_dict"])
    return head.eval().to(device)


@torch.no_grad()
def run_fold(fold: int, emb_path: Path, probe_path: str, device: str) -> pd.DataFrame:
    data = torch.load(emb_path, map_location="cpu", weights_only=False)
    emb = data["embeddings"]   # [N, nW, D]
    labels = data["labels"]    # [N]
    meta = data.get("metadata", [{}] * len(labels))

    nW, D = emb.shape[1], emb.shape[2]
    head = load_head(probe_path, nW, D, device)

    bs = 2048
    probs = []
    for i in range(0, len(emb), bs):
        x = emb[i:i+bs].to(device)
        logits = head(x)
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    probs = torch.cat(probs).numpy()

    rows = []
    for i, m in enumerate(meta):
        rows.append({
            "patient_id": m.get("patient_id", f"p{i}") if isinstance(m, dict) else f"p{i}",
            "true_label": int(labels[i]),
            "pred_prob_fog": float(probs[i]),
            "fold": fold,
        })
    return pd.DataFrame(rows)


def pooled_ap(df: pd.DataFrame):
    return average_precision_score(df["true_label"], df["pred_prob_fog"])


def best_f1(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    idx = np.argmax(f1s[:-1])
    return f1s[idx], thr[idx]


def print_defog_results(model_name: str, all_dfs: list):
    pooled = pd.concat(all_dfs, ignore_index=True)
    ap = pooled_ap(pooled)
    f1, thr = best_f1(pooled["true_label"].values, pooled["pred_prob_fog"].values)
    n = len(pooled)
    pos = pooled["true_label"].mean()

    print(f"\n{'='*60}")
    print(f"DEFOG POOLED AP — {model_name}")
    print(f"{'='*60}")
    for df in all_dfs:
        fold_ap = pooled_ap(df)
        print(f"  Fold {df['fold'].iloc[0]}: AP={fold_ap:.4f}  n={len(df)}  pos={df['true_label'].mean():.3f}")
    print(f"\n  Pooled AP={ap:.4f}  F1={f1:.4f}  (n={n}, pos={pos:.3f}, thr={thr:.3f})")
    print(f"\n  Kaggle benchmarks (defog test set, all patients):")
    for name, bap in KAGGLE_BENCHMARKS:
        print(f"    {name}: AP={bap:.3f}")
    print(f"{'='*60}\n")
    return ap, f1


def print_fogathome_results(model_name: str, all_dfs: list):
    # Add patch index within each fold (all folds see same 873 patches in same order)
    for df in all_dfs:
        df["patch_idx"] = range(len(df))
    pooled = pd.concat(all_dfs, ignore_index=True)
    # Ensemble: average probs per patch across folds (identified by patch_idx)
    ens = (pooled.groupby(["patch_idx", "patient_id", "true_label"])
           .agg(mean_prob=("pred_prob_fog", "mean")).reset_index())
    ap = average_precision_score(ens["true_label"], ens["mean_prob"])
    f1, thr = best_f1(ens["true_label"].values, ens["mean_prob"].values)

    print(f"\n{'='*60}")
    print(f"FOGATHOME AP — {model_name}")
    print(f"{'='*60}")
    print(f"  Ensemble AP={ap:.4f}  F1={f1:.4f}  "
          f"(n={len(ens)}, pos={ens['true_label'].mean():.3f}, thr={thr:.3f})")
    print(f"\n  FogAtHome benchmarks (competition models):")
    for name, bap in FOGATHOME_BENCHMARKS:
        print(f"    {name}: AP={bap:.3f}")
    print(f"{'='*60}\n")
    return ap, f1


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--probe-pattern", required=True,
                        help="e.g. checkpoints/classification/emb_probe_X_fold{fold}/last.ckpt")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fogathome", action="store_true",
                        help="Evaluate on FogAtHome instead of defog test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_dir = Path(args.emb_dir)
    split = "fogathome" if args.fogathome else "test"

    all_dfs = []
    for fold in range(args.n_folds):
        emb_path = emb_dir / f"fold{fold}_{split}.pt"
        probe_path = args.probe_pattern.format(fold=fold)
        if not emb_path.exists():
            logger.warning(f"Missing embeddings: {emb_path}")
            continue
        if not Path(probe_path).exists():
            logger.warning(f"Missing probe: {probe_path}")
            continue
        df = run_fold(fold, emb_path, probe_path, device)
        all_dfs.append(df)
        logger.info(f"Fold {fold}: {len(df)} samples, AP={pooled_ap(df):.4f}")

    if not all_dfs:
        logger.error("No predictions.")
        return

    if args.fogathome:
        print_fogathome_results(args.model_name, all_dfs)
    else:
        print_defog_results(args.model_name, all_dfs)


if __name__ == "__main__":
    main()

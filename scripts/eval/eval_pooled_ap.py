"""
Pooled AP evaluation across k-fold checkpoints.

Loads each trained classification checkpoint (which contains the full config),
runs test inference, collects per-sample (patient_id, true_label, pred_prob_fog),
saves to CSV, then computes pooled AP across all folds.

Usage:
    uv run python scripts/eval_pooled_ap.py \
        --model-name vit4_ep100 \
        --ckpt-pattern "checkpoints/classification/probe_vit4_ep100_fold{fold}/last.ckpt" \
        --split-prefix kaggle_labeled/kfold_defog_fogstrat \
        --n-folds 5 \
        --output-dir logs/pooled_ap
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


def run_fold_inference(fold: int, ckpt_path: str, split_prefix: str,
                       output_dir: Path, model_name: str, batch_size: int = 32) -> pd.DataFrame:
    pred_csv = output_dir / f"{model_name}_fold{fold}_preds.csv"
    if pred_csv.exists():
        logger.info(f"Fold {fold}: using cached {pred_csv}")
        return pd.read_csv(pred_csv)

    logger.info(f"Fold {fold}: loading {ckpt_path}")

    import torch
    from pipeline.classification import ClassificationPipeline
    from data.datamodule.datamodule import FOGDataModule
    from utils.paths import normalize_data_paths

    # Load checkpoint — config (with patient splits) is stored inside
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["hyper_parameters"]["config"]

    normalize_data_paths(config.data.paths)

    # Override batch size to avoid OOM when sharing GPU with other jobs
    data_cfg = config.data.model_copy(update={
        "dataloader": config.data.dataloader.model_copy(update={"batch_size": batch_size})
    })
    data_module = FOGDataModule(data_cfg=data_cfg, task_type=config.train.pipeline_type)
    data_module.setup("test")

    model = ClassificationPipeline(config)

    # The patient normalizer is initialized with shape [1, 3, 1] but the checkpoint
    # has [N_patients, 3, 1]. Resize buffers before loading so state_dict can match.
    state_dict = ckpt["state_dict"]
    for key in ["preprocessors.preprocessors.3.normalizer.mean",
                "preprocessors.preprocessors.3.normalizer.stdev"]:
        if key in state_dict:
            saved_shape = state_dict[key].shape
            param = dict(model.named_parameters()).get(key) or \
                    dict(model.named_buffers()).get(key)
            # Find and resize the actual buffer/parameter in the module
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

    # Inject normalization stats (normally done in on_train_start)
    data_module.setup("fit")
    if hasattr(model, 'preprocessors') and model.preprocessors is not None:
        try:
            stats = data_module.train_dataset.get_normalization_stats()
            model.preprocessors.inject_stats(stats)
        except Exception:
            pass  # not all preprocessors need stats injection

    rows = []
    test_loader = data_module.test_dataloader()

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            patch_y = batch['patch_y']
            meta = batch['metadata']

            logits = model(x)           # [B, num_classes]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            labels = patch_y.numpy() if isinstance(patch_y, torch.Tensor) else np.array(patch_y)

            for i, m in enumerate(meta):
                rows.append({
                    "patient_id": m.get("patient_id", "unknown"),
                    "session_id": m.get("session_id", "unknown"),
                    "true_label": int(labels[i]),
                    "pred_prob_fog": float(probs[i]),
                    "fold": fold,
                    "model": model_name,
                })

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(pred_csv, index=False)
    logger.info(f"Fold {fold}: {len(df)} predictions saved, pos_rate={df['true_label'].mean():.3f}")
    return df


def compute_and_print_results(model_name: str, all_dfs: list, output_dir: Path):
    pooled = pd.concat(all_dfs, ignore_index=True)
    pooled.to_csv(output_dir / f"{model_name}_pooled.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    fold_aps = []
    for fold in sorted(pooled["fold"].unique()):
        fold_df = pooled[pooled["fold"] == fold]
        if fold_df["true_label"].nunique() < 2:
            print(f"  Fold {fold}: only one class — skipped")
            continue
        ap = average_precision_score(fold_df["true_label"], fold_df["pred_prob_fog"])
        fold_aps.append(ap)
        print(f"  Fold {fold}: AP={ap:.4f}  n={len(fold_df)}  pos={fold_df['true_label'].sum()}")

    print(f"\n  Per-fold mean ± std: {np.mean(fold_aps):.4f} ± {np.std(fold_aps):.4f}")

    pooled_ap = average_precision_score(pooled["true_label"], pooled["pred_prob_fog"])
    n_patients = pooled["patient_id"].nunique()
    print(f"  *** Pooled AP ({len(pooled)} samples, {n_patients} patients): {pooled_ap:.4f} ***")
    print(f"      Overall positive rate: {pooled['true_label'].mean():.3f}")

    # Patient-level mean AP: each patient contributes equally regardless of size
    patient_aps = {}
    for pid in sorted(pooled["patient_id"].unique()):
        pid_df = pooled[pooled["patient_id"] == pid]
        if pid_df["true_label"].nunique() < 2:
            continue
        patient_aps[pid] = average_precision_score(pid_df["true_label"], pid_df["pred_prob_fog"])
    if patient_aps:
        vals = list(patient_aps.values())
        print(f"  *** Patient-level mean AP: {np.mean(vals):.4f} ± {np.std(vals):.4f} ({len(vals)} patients) ***")

    print(f"{'='*60}\n")
    return pooled_ap


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--ckpt-pattern", required=True,
                        help="e.g. checkpoints/classification/probe_X_fold{fold}/last.ckpt")
    parser.add_argument("--split-prefix", default="kaggle_labeled/kfold_defog_fogstrat")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="logs/pooled_ap")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    all_dfs = []
    for fold in range(args.n_folds):
        ckpt_path = args.ckpt_pattern.format(fold=fold)
        if not Path(ckpt_path).exists():
            logger.warning(f"Missing: {ckpt_path}")
            continue
        df = run_fold_inference(fold, ckpt_path, args.split_prefix, output_dir, args.model_name, args.batch_size)
        all_dfs.append(df)

    if not all_dfs:
        logger.error("No predictions collected.")
        sys.exit(1)

    compute_and_print_results(args.model_name, all_dfs, output_dir)


if __name__ == "__main__":
    main()

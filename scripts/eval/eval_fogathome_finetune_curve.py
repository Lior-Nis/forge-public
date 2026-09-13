"""
Fig 12 — compute the ICC-vs-fine-tuning-data-size curve from the checkpoints
produced by scripts/shell/run_fogathome_finetune_curve.sh.

For each (fold, budget) it runs the fine-tuned MC model on that fold's held-out
FogAtHome test patients, aggregates patch predictions to frames, and computes
ICC(%TF) and ICC(#FOG). The 0-min point uses the source DeFOG model (zero-shot).

A single fixed operating point (the DeFOG-validation PR-(1,1) threshold, ~0.26)
is used for ALL points so the curve isolates the effect of fine-tuning data, not
threshold drift — consistent with the headline ICC protocol (see compute_icc_thresholds.py).

Output: logs/RESULTS_fogathome_finetune_curve.csv  (fold, minutes, n_test, %TF ICC, #FOG ICC)

Usage:
    python scripts/eval/eval_fogathome_finetune_curve.py
"""

import argparse
import glob
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Reuse the audited inference + ICC helpers rather than re-implementing them.
_ec_spec = importlib.util.spec_from_file_location("ec", "scripts/eval/eval_comprehensive.py")
ec = importlib.util.module_from_spec(_ec_spec); _ec_spec.loader.exec_module(ec)
_icc_spec = importlib.util.spec_from_file_location("ict", "scripts/eval/compute_icc_thresholds.py")
ict = importlib.util.module_from_spec(_icc_spec); _icc_spec.loader.exec_module(ict)

CONTEXT = "mc"
ZARR_NAME = "len500_stride200_fogstride100_fogathome.zarr"
ZARR_PATH = f"data/processed/{ZARR_NAME}"
SOURCE_CKPT = "checkpoints/classification/soft_finetune_mc_all128_fold0/last.ckpt"
BUDGETS = [0, 0.5, 1, 2, 3, 5]  # per-patient minutes; 0 = zero-shot source model


def best_ckpt(fold, mins):
    """Best-val-AP checkpoint for a (fold, budget); SOURCE_CKPT for 0 min.

    Training degrades the head on tiny data (val AP decays from epoch 0), so
    `last.ckpt` is the worst — select by monitored val AP. The top-k checkpoints
    are nested as e.g. `.../eepoch=00-apmetrics/val_ap=0.839.ckpt` (the
    '/'-in-filename template artifact); pick the max val_ap across them.
    """
    if mins == 0:
        return SOURCE_CKPT if Path(SOURCE_CKPT).exists() else None
    d = f"checkpoints/classification/ftcurve_mc_fold{fold}_{mins}min"
    cks = glob.glob(f"{d}/*/val_ap=*.ckpt")
    if cks:
        return max(cks, key=lambda p: float(re.search(r"val_ap=(\d+\.\d+)", p).group(1)))
    last = f"{d}/last.ckpt"
    return last if Path(last).exists() else None


def infer(ckpt_path: str, test_patients: list[str]) -> pd.DataFrame:
    """Run a checkpoint on given FogAtHome test patients → per-patch predictions.

    Adapted from eval_comprehensive.run_inference but for an explicit checkpoint
    path + explicit patient list (the fine-tuned checkpoints are not in its CKPT map).
    """
    from pipeline.classification import ClassificationPipeline
    from data.datamodule.datamodule import FOGDataModule
    from data.datamodule.config import SplitsConfig
    from utils.paths import normalize_data_paths

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["hyper_parameters"]["config"]
    normalize_data_paths(config.data.paths)

    updates = {
        "dataloader": config.data.dataloader.model_copy(update={"batch_size": ec.BATCH_SIZE[CONTEXT]}),
        "paths": config.data.paths.model_copy(update={"dataset_path": ZARR_NAME}),
        "splits": SplitsConfig(train=test_patients, val=[], test=test_patients),
    }
    data_cfg = config.data.model_copy(update=updates)
    dm = FOGDataModule(data_cfg=data_cfg, task_type=config.train.pipeline_type)
    dm.setup("test")

    model = ClassificationPipeline(config)
    state_dict = ckpt["state_dict"]
    # Resize patient-normalizer buffers if the checkpoint's cohort size differs
    # (FogAtHome fine-tune has 13-patient stats vs a fresh 1-patient buffer).
    for key_sd in ["preprocessors.preprocessors.3.normalizer.mean",
                   "preprocessors.preprocessors.3.normalizer.stdev"]:
        if key_sd in state_dict:
            saved_shape = state_dict[key_sd].shape
            parts = key_sd.split(".")
            mod = model
            for part in parts[:-1]:
                mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
            if getattr(mod, parts[-1]).shape != saved_shape:
                setattr(mod, parts[-1], torch.zeros(saved_shape))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    rows = []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            x = batch["x"].to(device)
            probs = torch.softmax(model(x), dim=-1)[:, 1].cpu().numpy()
            for i, m in enumerate(batch["metadata"]):
                rows.append({
                    "patient_id": m.get("patient_id", ""),
                    "session_id": m.get("session_id", ""),
                    "global_idx": int(m.get("global_idx", -1)),
                    "session_idx": int(m.get("session_idx", -1)),
                    "pred_prob_fog": float(probs[i]),
                })
    del model, dm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/RESULTS_fogathome_finetune_curve.csv")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Fixed operating point. Default: DeFOG-val PR-(1,1) from the "
                         "kaggle mc/probe parquet (~0.26), matching the headline protocol.")
    args = ap.parse_args()

    # Fixed operating point (same for every curve point).
    thr = args.threshold
    if thr is None:
        kg = Path("logs/comprehensive_eval_cache/kaggle_mc_probe_frames.parquet")
        d = pd.read_parquet(kg)
        thr = ict.pr11_threshold(d["native_label"].values.astype(int), d["pred_prob_fog"].values)
    print(f"Operating point (fixed for all points): {thr:.4f}")

    records = []
    for fold in range(3):
        split = yaml.safe_load(open(f"configs/data/splits/fogathome_finetune_3fold{fold}.yaml"))
        test_patients = split["test"]
        for mins in BUDGETS:
            ckpt = best_ckpt(fold, mins)
            if ckpt is None:
                print(f"skip fold{fold} {mins}min: no checkpoint")
                continue
            preds = infer(ckpt, test_patients)
            if preds.empty:
                print(f"skip fold{fold} {mins}min: no predictions")
                continue
            # Aggregate patches → frames (with native_label) using the audited path.
            frame_cache = Path(f"logs/comprehensive_eval_cache/ftcurve_mc_fold{fold}_{mins}min_frames.parquet")
            frame_df = ec.build_frame_df(preds, ZARR_PATH, CONTEXT, frame_cache)
            if frame_df.empty or "native_label" not in frame_df.columns:
                print(f"skip fold{fold} {mins}min: frame agg failed")
                continue
            m = ict.clinical_metrics(frame_df, thr)
            records.append({
                "fold": fold, "minutes": mins, "n_test_patients": len(test_patients),
                "n_sessions": m["n_sessions"],
                "icc_tf": round(m["tf_icc"], 3), "icc_fog": round(m["fog_icc"], 3),
                "seg_f1": round(m["seg_f1"], 3),
            })
            print(f"fold{fold} {mins:>2}min: %TF ICC={m['tf_icc']:.3f}  #FOG ICC={m['fog_icc']:.3f}")

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)

    # Mean curve across folds (the figure line).
    if not df.empty:
        curve = df.groupby("minutes")[["icc_tf", "icc_fog", "seg_f1"]].mean().round(3)
        print("\nMean curve across folds:")
        print(curve.to_string())
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

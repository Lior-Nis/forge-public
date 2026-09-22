"""
Label-efficiency curve — FogAtHome AP vs DeFOG labeled-data budget, for the
FORGE MC probe (frozen pretrained encoder) vs supervised-from-scratch.

For each (arm, budget) it loads the 3 DeFOG-fold checkpoints, runs each on the
full FogAtHome cohort, averages probabilities per window (3-fold ensemble — the
same protocol as the headline numbers), and computes cls@50 AP and segment AP.

Checkpoints from the label-efficiency Hydra sweep:
    checkpoints/classification/labeleff_{arm}_mc_fold{f}_{budget}min/
Best-val-AP checkpoint is selected (not last.ckpt — on small budgets the head
degrades by the final epoch).

Output: logs/RESULTS_label_efficiency.csv  (arm, budget, cls_ap, seg_ap, prevalence)

Usage: python scripts/eval/eval_label_efficiency.py
"""
import argparse
import glob
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

_ec = importlib.util.spec_from_file_location("ec", "scripts/eval/eval_comprehensive.py")
ec = importlib.util.module_from_spec(_ec); _ec.loader.exec_module(ec)

CONTEXT = "mc"
ZARR_NAME = "len500_stride200_fogstride100_fogathome.zarr"
ZARR_PATH = f"data/processed/{ZARR_NAME}"
ARMS = ["probe", "random", "scratch"]
BUDGETS = ["2", "5", "15", "30", "60", "full"]


def best_ckpt(arm, fold, budget):
    d = f"checkpoints/classification/labeleff_{arm}_mc_fold{fold}_{budget}min"
    cks = glob.glob(f"{d}/*/val_ap=*.ckpt") + glob.glob(f"{d}/*val_ap=*.ckpt")
    if cks:
        return max(cks, key=lambda p: float(re.search(r"val_ap=(\d+\.\d+)", p).group(1)))
    last = f"{d}/last.ckpt"
    return last if Path(last).exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/RESULTS_label_efficiency.csv")
    args = ap.parse_args()

    records = []
    for arm in ARMS:
        for budget in BUDGETS:
            # 3-fold ensemble on the full FogAtHome cohort
            ens = None
            n_folds = 0
            for fold in range(3):
                ck = best_ckpt(arm, fold, budget)
                if ck is None:
                    continue
                preds = ec.infer_ckpt(ck, ZARR_NAME, CONTEXT) if hasattr(ec, "infer_ckpt") else _infer(ck)
                if preds is None or preds.empty:
                    continue
                preds = preds.set_index("global_idx")["pred_prob_fog"]
                ens = preds if ens is None else ens.add(preds, fill_value=0.0)
                n_folds += 1
            if ens is None or n_folds == 0:
                print(f"skip {arm}/{budget}: no checkpoints")
                continue
            ens = ens / n_folds
            # build_frame_df caches by filename and ignores the passed predictions
            # if the parquet exists — delete it first so we always aggregate the
            # CURRENT ensemble (otherwise a re-run silently reuses stale frames).
            fcache = Path(f"logs/comprehensive_eval_cache/labeleff_{arm}_{budget}_frames.parquet")
            fcache.unlink(missing_ok=True)
            frames = ec.build_frame_df(
                ens.rename("pred_prob_fog").reset_index(), ZARR_PATH, CONTEXT, fcache,
            )
            y = frames["native_label"].values.astype(int)
            s = frames["pred_prob_fog"].values
            seg_ap = average_precision_score(y, s)
            records.append({
                "arm": arm, "budget": budget, "n_folds": n_folds,
                "seg_ap": round(float(seg_ap), 4), "prevalence": round(float(y.mean()), 4),
            })
            print(f"{arm:8s} {budget:>4}min: seg AP={seg_ap:.4f}  (n_folds={n_folds})")

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    if not df.empty:
        print(df.pivot(index="budget", columns="arm", values="seg_ap").to_string())


def _infer(ckpt_path):
    """Fallback inference (mirrors eval_fogathome_finetune_curve.infer) if ec lacks a helper."""
    from pipeline.classification import ClassificationPipeline
    from data.datamodule.datamodule import FOGDataModule
    from data.datamodule.config import SplitsConfig
    from utils.paths import normalize_data_paths
    import yaml
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["hyper_parameters"]["config"]
    normalize_data_paths(config.data.paths)
    # all FogAtHome patients (string IDs from the split config — the zarr stores
    # patient_id int-encoded, which SplitsConfig rejects).
    sp = yaml.safe_load(open("configs/data/splits/fogathome.yaml"))
    pats = sorted({str(p) for k in ("train", "val", "test") for p in (sp.get(k) or [])})
    updates = {
        "dataloader": config.data.dataloader.model_copy(update={"batch_size": ec.BATCH_SIZE[CONTEXT]}),
        "paths": config.data.paths.model_copy(update={"dataset_path": ZARR_NAME}),
        "splits": SplitsConfig(train=pats, val=[], test=pats),
    }
    data_cfg = config.data.model_copy(update=updates)
    dm = FOGDataModule(data_cfg=data_cfg, task_type=config.train.pipeline_type)
    dm.setup("test")
    model = ClassificationPipeline(config)
    state_dict = ckpt["state_dict"]
    # Resize patient-normalizer buffers to the checkpoint's cohort size (DeFOG
    # the released models have many-patient stats vs a fresh 1-patient buffer).
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
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    rows = []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            probs = torch.softmax(model(batch["x"].to(dev)), dim=-1)[:, 1].cpu().numpy()
            for i, m in enumerate(batch["metadata"]):
                rows.append({"global_idx": int(m.get("global_idx", -1)), "pred_prob_fog": float(probs[i])})
    del model, dm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()

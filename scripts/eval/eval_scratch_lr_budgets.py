"""
Evaluate the across-budget scratch LR sweep and build the deployment-fair scratch
curve: at each budget K, pick the LR with the best FogAtHome AP (averaged over the
3 subset seeds, each a 3-fold ensemble). Compares LRs {1e-3 (existing), 3e-4, 1e-4,
3e-5}. Also reports the FORGE probe curve for reference.

Output:
  logs/RESULTS_scratch_lr_budgets.csv      (per arm/lr/k/seed seg_ap)
  logs/RESULTS_scratch_lr_budgets_best.csv (best-LR-per-budget scratch vs probe)

Usage: python scripts/eval/eval_scratch_lr_budgets.py
"""
import glob
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

_ec = importlib.util.spec_from_file_location("ec", "scripts/eval/eval_comprehensive.py")
ec = importlib.util.module_from_spec(_ec); _ec.loader.exec_module(ec)
_le = importlib.util.spec_from_file_location("le", "scripts/eval/eval_label_efficiency.py")
le = importlib.util.module_from_spec(_le); _le.loader.exec_module(le)

ZARR_PATH = "data/processed/len500_stride200_fogstride100_fogathome.zarr"
CONTEXT = "mc"
BUDGETS = [2, 4, 8, 16]
SEEDS = [0, 1, 2]
# lr label -> checkpoint dir suffix. "1e-3" = the original patient-count scratch runs
# (no lr tag in the dir name); the rest carry an _lr<tag> suffix.
# Candidates for best-per-budget. 1e-4 dropped (dominated at full data). 1e-3 = the
# original patient-count scratch runs (no _lr suffix).
LRS = {"1e-3": None, "3e-4": "3em4", "3e-5": "3em5"}


def _best(d):
    c = glob.glob(f"{d}/*/val_ap=*.ckpt")
    if c:
        return max(c, key=lambda p: float(re.search(r"val_ap=(\d+\.\d+)", p).group(1)))
    last = f"{d}/last.ckpt"
    return last if Path(last).exists() else None


def scratch_dir(fold, k, seed, lrtag):
    base = f"checkpoints/classification/labeleff_pat_scratch_mc_fold{fold}_k{k}_s{seed}"
    return base if lrtag is None else f"{base}_lr{lrtag}"


def probe_dir(fold, k, seed):
    return f"checkpoints/classification/labeleff_pat_probe_mc_fold{fold}_k{k}_s{seed}"


def ensemble_ap(dir_fn, tag):
    """3-fold ensemble seg AP for one (lr,k,seed); dir_fn(fold)->dir."""
    ens, n = None, 0
    for fold in range(3):
        ck = _best(dir_fn(fold))
        if ck is None:
            continue
        preds = le._infer(ck)
        if preds is None or preds.empty:
            continue
        preds = preds.set_index("global_idx")["pred_prob_fog"]
        ens = preds if ens is None else ens.add(preds, fill_value=0.0)
        n += 1
    if ens is None or n == 0:
        return None
    ens = ens / n
    fcache = Path(f"logs/comprehensive_eval_cache/scratchlrb_{tag}_frames.parquet")
    fcache.unlink(missing_ok=True)
    frames = ec.build_frame_df(ens.rename("pred_prob_fog").reset_index(), ZARR_PATH, CONTEXT, fcache)
    y = frames["native_label"].values.astype(int)
    return average_precision_score(y, frames["pred_prob_fog"].values)


def main():
    rows = []
    for k in BUDGETS:
        for lr, tag in LRS.items():
            for seed in SEEDS:
                ap = ensemble_ap(lambda f: scratch_dir(f, k, seed, tag), f"scratch_k{k}_{lr}_s{seed}")
                if ap is not None:
                    rows.append({"arm": "scratch", "lr": lr, "k": k, "seed": seed, "seg_ap": round(ap, 4)})
        # probe reference (existing patient-count runs)
        for seed in SEEDS:
            ap = ensemble_ap(lambda f: probe_dir(f, k, seed), f"probe_k{k}_s{seed}")
            if ap is not None:
                rows.append({"arm": "probe", "lr": "-", "k": k, "seed": seed, "seg_ap": round(ap, 4)})

    df = pd.DataFrame(rows)
    df.to_csv("logs/RESULTS_scratch_lr_budgets.csv", index=False)
    print("Wrote logs/RESULTS_scratch_lr_budgets.csv\n")

    # Best LR per budget (mean over seeds), vs probe
    best_rows = []
    for k in BUDGETS:
        sc = df[(df.arm == "scratch") & (df.k == k)]
        if not sc.empty:
            m = sc.groupby("lr")["seg_ap"].mean()
            best_lr = m.idxmax()
            best_rows.append({"k": k, "scratch_best_lr": best_lr,
                              "scratch_ap": round(m.max(), 4),
                              "scratch_ap_by_lr": {lr: round(v, 4) for lr, v in m.items()}})
        pr = df[(df.arm == "probe") & (df.k == k)]
        if best_rows and best_rows[-1]["k"] == k:
            best_rows[-1]["probe_ap"] = round(pr["seg_ap"].mean(), 4) if not pr.empty else None
    bdf = pd.DataFrame(best_rows)
    bdf.to_csv("logs/RESULTS_scratch_lr_budgets_best.csv", index=False)
    print("Best-tuned scratch per budget vs probe:")
    print(bdf.to_string(index=False))


if __name__ == "__main__":
    main()

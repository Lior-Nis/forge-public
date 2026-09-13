"""
Evaluate the supervised-from-scratch LR sweep at full data (K=all) and report the
best-tuned scratch endpoint, vs the FORGE probe (deployment-honest comparison).

For each LR it ensembles the 3 fold checkpoints on the full FogAtHome cohort and
computes segment AP. Includes the existing lr=1e-3 endpoint (the under-tuned one)
and the FORGE probe endpoint for reference.

Usage: python scripts/eval/eval_scratch_lr_sweep.py
"""
import glob
import importlib.util
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

_ec = importlib.util.spec_from_file_location("ec", "scripts/eval/eval_comprehensive.py")
ec = importlib.util.module_from_spec(_ec); _ec.loader.exec_module(ec)
_le = importlib.util.spec_from_file_location("le", "scripts/eval/eval_label_efficiency.py")
le = importlib.util.module_from_spec(_le); _le.loader.exec_module(le)

ZARR_PATH = "data/processed/len500_stride200_fogstride100_fogathome.zarr"
CONTEXT = "mc"


def _best(d):
    c = glob.glob(f"{d}/*/val_ap=*.ckpt")
    if c:
        return max(c, key=lambda p: float(re.search(r"val_ap=(\d+\.\d+)", p).group(1)))
    last = f"{d}/last.ckpt"
    return last if Path(last).exists() else None


def ensemble_ap(dir_fn, tag):
    ens, n = None, 0
    for fold in range(3):
        d = dir_fn(fold)
        ck = _best(d) if d else None
        if ck is None:
            continue
        preds = le._infer(ck)
        if preds is None or preds.empty:
            continue
        preds = preds.set_index("global_idx")["pred_prob_fog"]
        ens = preds if ens is None else ens.add(preds, fill_value=0.0)
        n += 1
    if ens is None:
        return None, 0
    ens = ens / n
    fcache = Path(f"logs/comprehensive_eval_cache/scratchlr_{tag}_frames.parquet")
    fcache.unlink(missing_ok=True)
    frames = ec.build_frame_df(ens.rename("pred_prob_fog").reset_index(), ZARR_PATH, CONTEXT, fcache)
    y = frames["native_label"].values.astype(int)
    return average_precision_score(y, frames["pred_prob_fog"].values), n


CASES = [
    ("scratch lr=1e-3 (current)", lambda f: f"checkpoints/classification/labeleff_pat_scratch_mc_fold{f}_kall_s0"),
    ("scratch lr=3e-4",           lambda f: f"checkpoints/classification/labeleff_pat_scratch_mc_fold{f}_kall_lr3em4"),
    ("scratch lr=1e-4",           lambda f: f"checkpoints/classification/labeleff_pat_scratch_mc_fold{f}_kall_lr1em4"),
    ("scratch lr=3e-5",           lambda f: f"checkpoints/classification/labeleff_pat_scratch_mc_fold{f}_kall_lr3em5"),
    ("FORGE probe (reference)",   lambda f: f"checkpoints/classification/labeleff_pat_probe_mc_fold{f}_kall_s0"),
]


def main():
    rows = []
    for name, fn in CASES:
        ap, n = ensemble_ap(fn, name.split()[0] + "_" + name.split("lr=")[-1].split()[0] if "lr=" in name else name.split()[0])
        if ap is None:
            print(f"skip {name}: no checkpoints"); continue
        rows.append({"case": name, "seg_ap": round(ap, 4), "n_folds": n})
        print(f"{name:28s}: seg AP = {ap:.4f}  (n_folds={n})")
    df = pd.DataFrame(rows)
    df.to_csv("logs/RESULTS_scratch_lr_sweep.csv", index=False)
    print("\nWrote logs/RESULTS_scratch_lr_sweep.csv")
    sc = df[df.case.str.startswith("scratch")]
    if not sc.empty:
        best = sc.loc[sc.seg_ap.idxmax()]
        print(f"\nBest-tuned scratch: {best['case']}  AP={best['seg_ap']}")


if __name__ == "__main__":
    main()

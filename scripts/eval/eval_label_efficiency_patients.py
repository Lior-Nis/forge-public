"""
Label-efficiency curve (PATIENT-COUNT axis) — FogAtHome AP vs number of labeled
DeFOG patients, for FORGE probe vs frozen-random-encoder control vs scratch.

For each (arm, K patients, seed) it ensembles the 3 DeFOG-fold checkpoints on the
full FogAtHome cohort (same protocol as the headline AP) and computes segment AP.
The all-patients anchor reuses the prior full-data checkpoints
(labeleff_{arm}_mc_fold{f}_fullmin) — at K=all, the subset seed is irrelevant.

Checkpoints from the patient-level label-efficiency Hydra sweep:
    checkpoints/classification/labeleff_pat_{arm}_mc_fold{f}_k{K}_s{seed}/

Output: logs/RESULTS_label_efficiency_patients.csv  (arm, k, seed, seg_ap, prevalence)

Usage: python scripts/eval/eval_label_efficiency_patients.py
"""
import argparse
import glob
import importlib.util
import re
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

_ec = importlib.util.spec_from_file_location("ec", "scripts/eval/eval_comprehensive.py")
ec = importlib.util.module_from_spec(_ec); _ec.loader.exec_module(ec)
# Reuse the audited inference helper from the minutes-axis eval.
_le = importlib.util.spec_from_file_location("le", "scripts/eval/eval_label_efficiency.py")
le = importlib.util.module_from_spec(_le); _le.loader.exec_module(le)

CONTEXT = "mc"
ZARR_NAME = "len500_stride200_fogstride100_fogathome.zarr"
ZARR_PATH = f"data/processed/{ZARR_NAME}"
ARMS = ["probe", "random", "scratch"]
BUDGETS = [2, 4, 8, 16]      # patient counts; "all" handled separately
SEEDS = [0, 1, 2]
N_ALL_PATIENTS = 48          # ≈ per-fold train cohort size (for the x-axis anchor)


def _best(d):
    cks = glob.glob(f"{d}/*/val_ap=*.ckpt") + glob.glob(f"{d}/*val_ap=*.ckpt")
    if cks:
        return max(cks, key=lambda p: float(re.search(r"val_ap=(\d+\.\d+)", p).group(1)))
    last = f"{d}/last.ckpt"
    return last if Path(last).exists() else None


def ckpt_patientcount(arm, fold, k, seed):
    return _best(f"checkpoints/classification/labeleff_pat_{arm}_mc_fold{fold}_k{k}_s{seed}")


def ckpt_all(arm, fold):
    # all-patients anchor = the clean recipe-consistent endpoint (K=all, seed=42,
    # save_top_k=1), trained by run_labeleff_pat_endpoint.sh.
    return _best(f"checkpoints/classification/labeleff_pat_{arm}_mc_fold{fold}_kall_s0")


def ensemble_ap(ckpt_fn, tag):
    """Average fold predictions on FogAtHome → segment AP. ckpt_fn(fold)->path."""
    ens, n = None, 0
    for fold in range(3):
        ck = ckpt_fn(fold)
        if ck is None:
            continue
        preds = le._infer(ck)
        if preds is None or preds.empty:
            continue
        preds = preds.set_index("global_idx")["pred_prob_fog"]
        ens = preds if ens is None else ens.add(preds, fill_value=0.0)
        n += 1
    if ens is None or n == 0:
        return None, None
    ens = ens / n
    fcache = Path(f"logs/comprehensive_eval_cache/labeleff_pat_{tag}_frames.parquet")
    fcache.unlink(missing_ok=True)   # always aggregate the CURRENT ensemble
    frames = ec.build_frame_df(ens.rename("pred_prob_fog").reset_index(), ZARR_PATH, CONTEXT, fcache)
    y = frames["native_label"].values.astype(int)
    return average_precision_score(y, frames["pred_prob_fog"].values), float(y.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/RESULTS_label_efficiency_patients.csv")
    args = ap.parse_args()

    records = []
    for arm in ARMS:
        for k in BUDGETS:
            for seed in SEEDS:
                seg, prev = ensemble_ap(lambda f: ckpt_patientcount(arm, f, k, seed), f"{arm}_k{k}_s{seed}")
                if seg is None:
                    print(f"skip {arm}/k{k}/s{seed}: no checkpoints"); continue
                records.append({"arm": arm, "k": k, "seed": seed,
                                "seg_ap": round(seg, 4), "prevalence": round(prev, 4)})
                print(f"{arm:8s} k={k:>2} s{seed}: seg AP={seg:.4f}")
        # all-patients anchor (reuse fullmin; seed-invariant → record once per seed for plotting symmetry)
        seg, prev = ensemble_ap(lambda f: ckpt_all(arm, f), f"{arm}_all")
        if seg is not None:
            for seed in SEEDS:
                records.append({"arm": arm, "k": N_ALL_PATIENTS, "seed": seed,
                                "seg_ap": round(seg, 4), "prevalence": round(prev, 4)})
            print(f"{arm:8s} k=all ({N_ALL_PATIENTS}): seg AP={seg:.4f}")

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    if not df.empty:
        piv = df.groupby(["k", "arm"])["seg_ap"].mean().unstack("arm")
        print(piv.to_string())


if __name__ == "__main__":
    main()

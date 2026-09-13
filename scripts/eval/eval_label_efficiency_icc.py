"""
ICC over label efficiency — clinical %TF ICC (per-session, whole-recording) on
FogAtHome vs number of labeled DeFOG patients, for FORGE probe vs best-tuned
supervised-from-scratch. Fixed operating point thr=0.26 (DeFOG-val PR-(1,1)) for
ALL points, so the curve isolates the effect of labeled data, not threshold drift
(same convention as the Fig-10 fine-tune curve).

No new training — reuses existing checkpoints. Per (arm, budget, seed): ensemble
the 3 folds on FogAtHome, build frames, compute clinical_metrics(thr=0.26).

Output: logs/RESULTS_label_efficiency_icc.csv (arm, k, seed, tf_icc, tf_icc_pat)
Usage: python scripts/eval/eval_label_efficiency_icc.py
"""
import glob, importlib.util, re
from pathlib import Path
import pandas as pd

_ec = importlib.util.spec_from_file_location("ec", "scripts/eval/eval_comprehensive.py")
ec = importlib.util.module_from_spec(_ec); _ec.loader.exec_module(ec)
_le = importlib.util.spec_from_file_location("le", "scripts/eval/eval_label_efficiency.py")
le = importlib.util.module_from_spec(_le); _le.loader.exec_module(le)
_ict = importlib.util.spec_from_file_location("ict", "scripts/eval/compute_icc_thresholds.py")
ict = importlib.util.module_from_spec(_ict); _ict.loader.exec_module(ict)

ZARR_PATH = "data/processed/len500_stride200_fogstride100_fogathome.zarr"
CONTEXT, THR = "mc", 0.26
BUDGETS = [2, 4, 8, 16]
SEEDS = [0, 1, 2]
N_ALL = 48
SCRATCH_BEST_LR = {2: "3em5", 4: "3em5", 8: "3em4", 16: "3em5"}  # from the LR-budget sweep


def _best(d):
    c = glob.glob(f"{d}/*/val_ap=*.ckpt")
    if c:
        return max(c, key=lambda p: float(re.search(r"val_ap=(\d+\.\d+)", p).group(1)))
    last = f"{d}/last.ckpt"
    return last if Path(last).exists() else None


def icc(dir_fn, tag):
    """3-fold ensemble on FogAtHome -> %TF ICC at THR. dir_fn(fold)->dir."""
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
    fcache = Path(f"logs/comprehensive_eval_cache/lei_{tag}_frames.parquet")
    fcache.unlink(missing_ok=True)
    frames = ec.build_frame_df(ens.rename("pred_prob_fog").reset_index(), ZARR_PATH, CONTEXT, fcache)
    m = ict.clinical_metrics(frames, THR)
    return m["tf_icc"], m["tf_icc_pat"]


def main():
    rows = []
    for k in BUDGETS:
        for seed in SEEDS:
            r = icc(lambda f: f"checkpoints/classification/labeleff_pat_probe_mc_fold{f}_k{k}_s{seed}", f"probe_k{k}_s{seed}")
            if r: rows.append({"arm": "probe", "k": k, "seed": seed, "tf_icc": round(r[0], 4), "tf_icc_pat": round(r[1], 4)})
            lr = SCRATCH_BEST_LR[k]
            r = icc(lambda f: f"checkpoints/classification/labeleff_pat_scratch_mc_fold{f}_k{k}_s{seed}_lr{lr}", f"scr_k{k}_s{seed}")
            if r: rows.append({"arm": "scratch", "k": k, "seed": seed, "tf_icc": round(r[0], 4), "tf_icc_pat": round(r[1], 4)})
    # all-patients endpoint (seed-invariant)
    r = icc(lambda f: f"checkpoints/classification/labeleff_pat_probe_mc_fold{f}_kall_s0", "probe_kall")
    if r: rows.append({"arm": "probe", "k": N_ALL, "seed": 0, "tf_icc": round(r[0], 4), "tf_icc_pat": round(r[1], 4)})
    r = icc(lambda f: f"checkpoints/classification/labeleff_pat_scratch_mc_fold{f}_kall_lr3em5", "scr_kall")
    if r: rows.append({"arm": "scratch", "k": N_ALL, "seed": 0, "tf_icc": round(r[0], 4), "tf_icc_pat": round(r[1], 4)})

    df = pd.DataFrame(rows)
    df.to_csv("logs/RESULTS_label_efficiency_icc.csv", index=False)
    print("Wrote logs/RESULTS_label_efficiency_icc.csv\n")
    piv = df.groupby(["k", "arm"])["tf_icc"].mean().unstack("arm")
    print("Per-session %TF ICC @ thr=0.26 (mean over seeds):")
    print(piv.to_string())


if __name__ == "__main__":
    main()

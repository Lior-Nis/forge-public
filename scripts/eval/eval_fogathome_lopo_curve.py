"""
Fig 12 (LOPO) — pooled ICC vs fine-tuning-data-size from the leave-one-patient-out
runs of run_fogathome_lopo_curve.sh.

For each budget, runs each fold's best-val-AP probe on its single held-out patient,
POOLS predictions across all 12 held-out patients (→ all 97 sessions), and computes
ICC(%TF) / ICC(#FOG). 0-min = the source DeFOG model on all patients (zero-shot).

Threshold: pooled PR-(1,1) per budget (Salomon protocol) — consistent across budgets,
isolating the effect of fine-tuning data from calibration drift.

Output: logs/RESULTS_fogathome_lopo_curve.csv
"""
import glob
import importlib.util
import re
from pathlib import Path

import pandas as pd

_e = importlib.util.spec_from_file_location("ftc", "scripts/eval/eval_fogathome_finetune_curve.py")
ftc = importlib.util.module_from_spec(_e); _e.loader.exec_module(ftc)
_i = importlib.util.spec_from_file_location("ict", "scripts/eval/compute_icc_thresholds.py")
ict = importlib.util.module_from_spec(_i); _i.loader.exec_module(ict)
_ec = importlib.util.spec_from_file_location("ec", "scripts/eval/eval_comprehensive.py")
ec = importlib.util.module_from_spec(_ec); _ec.loader.exec_module(ec)

PATIENTS = ["a00001", "a00002", "a00004", "a00006", "a00007", "a00008",
            "a00009", "a00010", "a00011", "a00012", "a00013", "a00014"]
ZARR_PATH = "data/processed/len500_stride200_fogstride100_fogathome.zarr"
BUDGETS = [0, 1, 2, 3, 5]
CACHE = Path("logs/comprehensive_eval_cache")


PREFIX = "ftcurve_lopo_lr1e4_mc"  # checkpoint name prefix (set in main from --prefix)


def lopo_best_ckpt(fold, mins):
    d = f"checkpoints/classification/{PREFIX}_fold{fold}_{mins}min"
    cks = glob.glob(f"{d}/*/val_ap=*.ckpt")
    if cks:
        return max(cks, key=lambda p: float(re.search(r"val_ap=(\d+\.\d+)", p).group(1)))
    last = f"{d}/last.ckpt"
    return last if Path(last).exists() else None


def pooled_frames(budget):
    """Concat per-fold held-out predictions → all 97 sessions of frames."""
    if budget == 0:
        preds = ftc.infer(ftc.SOURCE_CKPT, PATIENTS)
        cache = CACHE / "ftcurve_lopo_zeroshot_frames.parquet"
        return ec.build_frame_df(preds, ZARR_PATH, "mc", cache)
    parts = []
    for fold, test_pat in enumerate(PATIENTS):
        ck = lopo_best_ckpt(fold, budget)
        if ck is None:
            print(f"  · missing fold{fold} {budget}min")
            continue
        preds = ftc.infer(ck, [test_pat])
        cache = CACHE / f"{PREFIX}_fold{fold}_{budget}min_frames.parquet"
        fr = ec.build_frame_df(preds, ZARR_PATH, "mc", cache)
        if not fr.empty:
            parts.append(fr)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=PREFIX, help="checkpoint name prefix (e.g. ftcurve_lopo_lr1e5_mc)")
    args = ap.parse_args()
    globals()["PREFIX"] = args.prefix
    out = f"logs/RESULTS_{args.prefix}_curve.csv"
    records = []
    for b in BUDGETS:
        fr = pooled_frames(b)
        if fr.empty or "native_label" not in fr.columns:
            print(f"{b}min: no data"); continue
        thr = ict.pr11_threshold(fr["native_label"].values.astype(int), fr["pred_prob_fog"].values)
        m = ict.clinical_metrics(fr, thr)
        records.append({"minutes": b, "total_train_min": round(b * 10 if b else 0, 0),
                        "n_sessions": m["n_sessions"], "threshold": round(thr, 3),
                        "icc_tf": round(m["tf_icc"], 3), "icc_fog": round(m["fog_icc"], 3),
                        "seg_f1": round(m["seg_f1"], 3)})
        print(f"{b}min/patient (~{b*10 if b else 0} min total): "
              f"%TF ICC={m['tf_icc']:.3f}  #FOG ICC={m['fog_icc']:.3f}  (N={m['n_sessions']})")
    df = pd.DataFrame(records)
    df.to_csv(out, index=False)
    print("\n", df.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

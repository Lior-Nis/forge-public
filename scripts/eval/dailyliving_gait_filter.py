"""
Activity-conditioned (gait-filtered) daily-living evaluation for FORGE.

Joins the per-frame Activity annotation from the FogAtHome daily-living dataset
onto the cached MC-probe predictions and reports, for each candidate gait-code
set: kept fraction, FOG prevalence, frame-level AUC / AP / NormAP, precision &
recall at the operating threshold, and per-session / per-subject %TF ICC.

Whole-recording AUC (~0.90) is inflated by trivial FOG-vs-sedentary separation;
gait-conditioning lowers AUC (to the hard FOG-vs-walking task) but RAISES the
%TF ICC (from an uninformative 0.16 to ~0.5) by restoring between-subject spread.

The exact gait-code set is pending confirmation with the dataset authors; pass it
via --gait-codes once known. Usage:
    python scripts/eval/dailyliving_gait_filter.py --gait-codes 1 4
"""
import argparse
import os

import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import average_precision_score, roc_auc_score

RAW_DEFAULT = os.path.join(os.environ.get("FORGE_DATASETS_ROOT", os.path.expanduser("~/Datasets")), "fogathome_dailyliving/preprocesseddata")
PRED_DEFAULT = "logs/comprehensive_eval_cache/dailyliving_mc_probe_frames.parquet"
THR = 0.26  # DeFOG-validation PR-(1,1) operating point
# CONFIRMED Activity legend (Salomon, 2026-06-07):
#   0=Other  1=Walking  2=Lying  3=Sitting  4=Standing  5=Sleep  6=Non-wear
# Gait == walking == code {1}. Codes 2/3/4/5/6 are non-ambulatory and cannot
# contain true gait freezing. CANDIDATES retained only for sensitivity scans.
GAIT = [1]  # walking only
CANDIDATES = {
    "whole": None,
    "{1} walking": [1],
}


def _icc2(gt, pr):
    n = len(gt)
    if n < 3:
        return float("nan"), ""
    long = pd.concat([
        pd.DataFrame({"t": range(n), "r": "g", "v": gt}),
        pd.DataFrame({"t": range(n), "r": "m", "v": pr}),
    ])
    row = pg.intraclass_corr(data=long, targets="t", raters="r", ratings="v").set_index("Type").loc["ICC2"]
    ci = row["CI95%"]
    return float(row["ICC"]), f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW_DEFAULT, help="dir of per-session daily-living parquets with an Activity column")
    ap.add_argument("--pred", default=PRED_DEFAULT)
    ap.add_argument("--gait-codes", nargs="+", type=int, default=None,
                    help="confirmed gait codes; if set, only whole vs this set are reported")
    ap.add_argument("--thr", type=float, default=THR)
    args = ap.parse_args()

    cands = CANDIDATES if args.gait_codes is None else {"whole": None, f"gait={args.gait_codes}": args.gait_codes}

    pred = pd.read_parquet(args.pred, columns=["patient_id", "session_id", "abs_frame", "native_label", "pred_prob_fog"])
    pred["session_id"] = pred.session_id.astype(str)

    # Per-session aggregates (kept / fog / pred-positive) for each candidate, plus
    # pooled arrays for AUC/AP.
    rows, ys, ss, as_ = [], [], [], []
    for sid, g in pred.groupby("session_id", sort=False):
        g = g.sort_values("abs_frame")
        n = len(g)
        yv = g.native_label.values.astype(np.int8)
        pv = (g.pred_prob_fog.values >= args.thr).astype(np.int8)
        act = np.full(n, -1, np.int16)
        f = os.path.join(args.raw, sid + ".parquet")
        if os.path.exists(f):
            a = pd.read_parquet(f, columns=["Activity"]).Activity.values
            m = min(n, len(a))
            act[:m] = a[:m]
        ys.append(yv); ss.append(g.pred_prob_fog.values.astype(np.float32)); as_.append(act)
        rec = {"sid": sid, "pid": g.patient_id.iloc[0]}
        for nm, c in cands.items():
            mask = np.ones(n, bool) if c is None else np.isin(act, c)
            rec[nm] = (int(mask.sum()), int(yv[mask].sum()), int(pv[mask].sum()))
        rows.append(rec)
    df = pd.DataFrame(rows)
    y = np.concatenate(ys); s = np.concatenate(ss); a = np.concatenate(as_)

    print(f"frames={len(y):,} | thr={args.thr} | overall prev={y.mean()*100:.2f}% AUC={roc_auc_score(y, s):.3f}\n")
    print(f"{'gait set':14s} | kept% | prev% |  AUC  |  AP   | NormAP | P@thr R@thr | sess ICC | pat ICC [CI]")
    for nm, c in cands.items():
        mask = np.ones(len(y), bool) if c is None else np.isin(a, c)
        yy, sm = y[mask], s[mask]
        prev = yy.mean(); auc = roc_auc_score(yy, sm); apv = average_precision_score(yy, sm)
        normap = (apv - prev) / (1 - prev)
        pb = sm >= args.thr
        tp = int((pb & (yy == 1)).sum()); fp = int((pb & (yy == 0)).sum()); fn = int((~pb & (yy == 1)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        e = df[nm].apply(pd.Series); e.columns = ["kept", "fog", "pos"]; e["pid"] = df.pid
        se = e[e["kept"] > 0]
        s_icc, _ = _icc2((se["fog"] / se["kept"] * 100).values, (se["pos"] / se["kept"] * 100).values)
        p = e.groupby("pid").agg(kept=("kept", "sum"), fog=("fog", "sum"), pos=("pos", "sum"))
        p = p[p["kept"] > 0]
        p_icc, p_ci = _icc2((p["fog"] / p["kept"] * 100).values, (p["pos"] / p["kept"] * 100).values)
        print(f"{nm:14s} | {mask.mean()*100:4.1f} | {prev*100:4.2f} | {auc:.3f} | {apv:.3f} | {normap:.3f} | "
              f"{prec:.3f} {rec:.3f} | {s_icc:.3f} | {p_icc:.3f} {p_ci}")


if __name__ == "__main__":
    main()

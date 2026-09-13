"""
Patient-level bootstrap CIs for FogAtHome seg-level AP/AUC, and a paired
bootstrap test of the MAE-pretraining benefit (probe/finetune vs supervised).

Resamples the 12 FogAtHome patients with replacement (B reps), recomputing
pooled frame-level AP and AUC each rep. The paired test reuses the *same*
resampled patient set for both models, so the ΔAP/ΔAUC CI and two-sided
bootstrap p-value reflect the within-cohort paired difference.

Reads cached per-frame predictions from eval_comprehensive.py (no GPU, no
inference). Usage:
    python scripts/eval/bootstrap_significance.py --contexts mc lc sc
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

N_BOOT = 2000
SEED = 12345


def per_patient(df: pd.DataFrame):
    """Return dict patient_id -> (y_true, y_score) frame arrays."""
    out = {}
    for pid, g in df.groupby("patient_id"):
        out[pid] = (g["native_label"].values.astype(int), g["pred_prob_fog"].values)
    return out


def metric(y, s, fn):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return fn(y, s)


def point_and_ci(pp, fn, patient_order, boot_idx):
    """Point estimate (all patients pooled) + bootstrap [2.5, 97.5] CI."""
    yt = np.concatenate([pp[p][0] for p in patient_order])
    ys = np.concatenate([pp[p][1] for p in patient_order])
    point = metric(yt, ys, fn)
    vals = []
    for idx in boot_idx:
        sel = [patient_order[i] for i in idx]
        yt = np.concatenate([pp[p][0] for p in sel])
        ys = np.concatenate([pp[p][1] for p in sel])
        v = metric(yt, ys, fn)
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


def paired_test(pp_a, pp_b, fn, patient_order, boot_idx):
    """Paired bootstrap of metric(a) - metric(b). Returns (delta, lo, hi, p_two_sided)."""
    def pooled(pp, sel):
        yt = np.concatenate([pp[p][0] for p in sel])
        ys = np.concatenate([pp[p][1] for p in sel])
        return metric(yt, ys, fn)

    delta = pooled(pp_a, patient_order) - pooled(pp_b, patient_order)
    diffs = []
    for idx in boot_idx:
        sel = [patient_order[i] for i in idx]
        da, db = pooled(pp_a, sel), pooled(pp_b, sel)
        if not (np.isnan(da) or np.isnan(db)):
            diffs.append(da - db)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p: 2 x tail mass on the side of 0
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return delta, lo, hi, min(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="logs/comprehensive_eval_cache")
    ap.add_argument("--contexts", nargs="+", default=["mc", "lc", "sc"])
    ap.add_argument("--out", default="logs/RESULTS_bootstrap.md")
    args = ap.parse_args()
    cache = Path(args.cache_dir)

    rng = np.random.default_rng(SEED)
    lines = [
        "# FogAtHome seg-level bootstrap CIs & MAE-benefit paired test\n",
        f"Patient-level bootstrap, B={N_BOOT}, 12 FogAtHome patients, seed={SEED}. "
        "ΔAP/ΔAUC = (probe or finetune) − supervised, paired on the same resampled patients.\n",
        "| ctx | model | seg AP [95% CI] | seg AUC [95% CI] |",
        "|---|---|---|---|",
    ]
    paired_lines = [
        "\n| ctx | contrast | Δ seg AP [95% CI] | p | Δ seg AUC [95% CI] | p |",
        "|---|---|---|---|---|---|",
    ]

    for ctx in args.contexts:
        models = {}
        for mdl in ["probe", "finetune", "supervised"]:
            f = cache / f"fogathome_{ctx}_{mdl}_frames.parquet"
            if not f.exists():
                print(f"skip {ctx}/{mdl}: missing {f.name}")
                continue
            models[mdl] = per_patient(pd.read_parquet(f))

        if not models:
            continue
        patient_order = sorted(next(iter(models.values())).keys())
        n = len(patient_order)
        boot_idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]

        for mdl, pp in models.items():
            ap_pt, ap_lo, ap_hi = point_and_ci(pp, average_precision_score, patient_order, boot_idx)
            au_pt, au_lo, au_hi = point_and_ci(pp, roc_auc_score, patient_order, boot_idx)
            lines.append(f"| {ctx} | {mdl} | {ap_pt:.3f} [{ap_lo:.3f}, {ap_hi:.3f}] | "
                         f"{au_pt:.3f} [{au_lo:.3f}, {au_hi:.3f}] |")

        if "supervised" in models:
            for mdl in ["probe", "finetune"]:
                if mdl not in models:
                    continue
                d_ap, lo_ap, hi_ap, p_ap = paired_test(
                    models[mdl], models["supervised"], average_precision_score, patient_order, boot_idx)
                d_au, lo_au, hi_au, p_au = paired_test(
                    models[mdl], models["supervised"], roc_auc_score, patient_order, boot_idx)
                paired_lines.append(
                    f"| {ctx} | {mdl}−supervised | {d_ap:+.3f} [{lo_ap:+.3f}, {hi_ap:+.3f}] | {p_ap:.3f} | "
                    f"{d_au:+.3f} [{lo_au:+.3f}, {hi_au:+.3f}] | {p_au:.3f} |")

    report = "\n".join(lines + paired_lines) + "\n"
    Path(args.out).write_text(report)
    print(report)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

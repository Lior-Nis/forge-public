"""
Recompute ALL DailyLiving metrics on the walking+standing activity subset.

The whole-recording DailyLiving evaluation is inflated by trivial FOG-vs-sedentary
separation and has a compressed %TF range (uninformative ICC). All metrics we
report on DailyLiving are therefore conditioned on the ambulatory states in which
true gait freezing can occur: Activity ∈ {1=Walking, 4=Standing}  (legend confirmed
by Salomon, 2026-06-07: 0=Other 1=Walking 2=Lying 3=Sitting 4=Standing 5=Sleep
6=Non-wear).

For each (context, model) it joins per-frame Activity from the raw daily-living
parquets onto the cached predictions, keeps {1,4}, and recomputes frame-level
AUC / AP / NormAP / prevalence plus per-session %TF ICC at the per-model
protocol-B (DeFOG-validation) threshold.

Output: logs/RESULTS_dailyliving_walkstand.csv
  columns: context, model, AUC, AP, NormAP, prevalence, n_frames, kept_frac, tf_icc
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import average_precision_score, roc_auc_score

CACHE   = Path("logs/comprehensive_eval_cache")
RAW_DIR = os.path.join(os.environ.get("FORGE_DATASETS_ROOT", os.path.expanduser("~/Datasets")), "fogathome_dailyliving/preprocesseddata")
CONTEXTS = ["sc", "mc", "lc", "mc+sc", "lc+sc", "lc+mc", "lc+mc+sc"]
MODELS   = ["probe", "finetune", "supervised"]
KEEP     = [1, 4]                      # walking + standing
COLS     = ["session_id", "abs_frame", "pred_prob_fog", "native_label"]
OUT      = "logs/RESULTS_dailyliving_walkstand.csv"


def load_thresholds(path="logs/RESULTS_icc.md"):
    """Per-(ctx, model) protocol-B (DeFOG-validation) threshold from the ICC table."""
    thr = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("| ") or "/" not in line:
                continue
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) < 3 or "/" not in parts[0]:
                continue
            ctx, mdl = parts[0].rsplit("/", 1)
            try:
                thr[(ctx, mdl)] = float(parts[2])
            except ValueError:
                pass
    return thr


_ACT = None


def activity_cache():
    """session_id -> np.ndarray of per-frame Activity codes (read once)."""
    global _ACT
    if _ACT is None:
        _ACT = {}
        for f in os.listdir(RAW_DIR):
            if f.endswith(".parquet"):
                _ACT[f[:-8]] = pd.read_parquet(
                    os.path.join(RAW_DIR, f), columns=["Activity"]
                ).Activity.values
    return _ACT


def session_tf_icc(sess_tf):
    """ICC(2,1) between per-session true %TF and predicted %TF."""
    if len(sess_tf) < 3:
        return float("nan")
    long = pd.DataFrame({
        "session": sess_tf.index.tolist() * 2,
        "rater":   ["true"] * len(sess_tf) + ["pred"] * len(sess_tf),
        "score":   sess_tf["true"].tolist() + sess_tf["pred"].tolist(),
    })
    res = pg.intraclass_corr(data=long, targets="session", raters="rater", ratings="score")
    return float(res.loc[res["Type"] == "ICC2", "ICC"].iloc[0])


def filter_walkstand(df, act):
    """Attach Activity by per-session position (abs_frame is 0..n-1 contiguous),
    then keep only walking+standing frames."""
    df = df.sort_values(["session_id", "abs_frame"])
    keep_mask = np.zeros(len(df), bool)
    out_pos = 0
    for sid, g in df.groupby("session_id", sort=False):
        n = len(g)
        a = act.get(sid)
        col = np.full(n, -1, np.int16)
        if a is not None:
            m = min(n, len(a))
            col[:m] = a[:m]
        keep_mask[out_pos:out_pos + n] = np.isin(col, KEEP)
        out_pos += n
    return df[keep_mask]


def main():
    thr = load_thresholds()
    act = activity_cache()
    rows = []
    for ctx in CONTEXTS:
        for mdl in MODELS:
            pq = CACHE / f"dailyliving_{ctx}_{mdl}_frames.parquet"
            if not pq.exists():
                print(f"  missing {pq.name}")
                continue
            t = thr.get((ctx, mdl), 0.5)
            full = pd.read_parquet(pq, columns=COLS)
            full["session_id"] = full.session_id.astype(str)
            n_total = len(full)
            sub = filter_walkstand(full, act)
            y = sub.native_label.values.astype(np.int8)
            s = sub.pred_prob_fog.values.astype(np.float32)
            prev = float(y.mean())
            auc = roc_auc_score(y, s) if y.min() != y.max() else float("nan")
            ap = average_precision_score(y, s)
            normap = (ap - prev) / (1 - prev) if prev < 1 else float("nan")
            sub = sub.copy()
            sub["pred"] = (sub.pred_prob_fog.values >= t).astype(np.int8)
            agg = sub.groupby("session_id").agg(
                true=("native_label", "mean"), pred=("pred", "mean"))
            icc = session_tf_icc(agg)
            rows.append({
                "context": ctx, "model": mdl, "AUC": auc, "AP": ap,
                "NormAP": normap, "prevalence": prev, "n_frames": len(sub),
                "kept_frac": len(sub) / n_total, "tf_icc": icc,
            })
            print(f"  {ctx:9s}/{mdl:11s} thr={t:.2f} kept={len(sub)/n_total*100:4.1f}% "
                  f"prev={prev*100:5.2f}% AUC={auc:.3f} AP={ap:.3f} ICC={icc:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    pooled_prev = (out.prevalence * out.n_frames).sum() / out.n_frames.sum()
    print(f"\nWrote {len(out)} rows → {OUT}")
    print(f"Frame-weighted mean walking+standing prevalence ≈ {pooled_prev*100:.2f}%")


if __name__ == "__main__":
    main()

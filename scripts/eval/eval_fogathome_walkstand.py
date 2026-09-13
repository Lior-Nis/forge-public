"""
Recompute FogAtHome metrics on the walking+standing activity subset, so the
heatmaps compare FogAtHome and DailyLiving under the SAME activity filter
(Activity ∈ {1=Walking, 4=Standing}).

FogAtHome per-frame Activity comes from the dataset labels.csv (Id = session_absframe).
Frames without an Activity label are dropped (consistent with keeping only
confirmed walking+standing). For each (context, model) it recomputes frame-level
AUC / AP / NormAP / prevalence + per-session %TF ICC at the per-model protocol-B
(DeFOG-validation) threshold.

Output: logs/RESULTS_fogathome_walkstand.csv
  columns: context, model, AUC, AP, NormAP, prevalence, n_frames, kept_frac, tf_icc
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import average_precision_score, roc_auc_score

CACHE   = Path("logs/comprehensive_eval_cache")
LABELS  = os.path.join(os.environ.get("FORGE_DATASETS_ROOT", os.path.expanduser("~/Datasets")), "fogathome_dataset_forlior/fogathome_dataset/labels.csv")
CONTEXTS = ["sc", "mc", "lc", "mc+sc", "lc+sc", "lc+mc", "lc+mc+sc"]
MODELS   = ["probe", "finetune", "supervised"]
KEEP     = [1, 4]                      # walking + standing
OUT      = "logs/RESULTS_fogathome_walkstand.csv"


def load_thresholds(path="logs/RESULTS_icc.md"):
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


def gait_labels():
    gl = pd.read_csv(LABELS, usecols=["Id", "Activity"])
    gl["session_id"] = gl["Id"].str.rsplit("_", n=1).str[0]
    gl["abs_frame"] = gl["Id"].str.rsplit("_", n=1).str[1].astype(int)
    return gl[["session_id", "abs_frame", "Activity"]]


def session_tf_icc(agg):
    if len(agg) < 3:
        return float("nan")
    long = pd.DataFrame({
        "session": agg.index.tolist() * 2,
        "rater":   ["true"] * len(agg) + ["pred"] * len(agg),
        "score":   agg["true"].tolist() + agg["pred"].tolist(),
    })
    res = pg.intraclass_corr(data=long, targets="session", raters="rater", ratings="score")
    return float(res.loc[res["Type"] == "ICC2", "ICC"].iloc[0])


def main():
    thr = load_thresholds()
    gl = gait_labels()
    rows = []
    for ctx in CONTEXTS:
        for mdl in MODELS:
            pq = CACHE / f"fogathome_{ctx}_{mdl}_frames.parquet"
            if not pq.exists():
                print(f"  missing {pq.name}")
                continue
            t = thr.get((ctx, mdl), 0.5)
            df = pd.read_parquet(pq, columns=["session_id", "abs_frame", "pred_prob_fog", "native_label"])
            df["session_id"] = df.session_id.astype(str)
            n_total = len(df)
            df = df.merge(gl, on=["session_id", "abs_frame"], how="left")
            sub = df[df.Activity.isin(KEEP)]
            y = sub.native_label.values.astype(np.int8)
            s = sub.pred_prob_fog.values.astype(np.float32)
            prev = float(y.mean())
            auc = roc_auc_score(y, s) if y.min() != y.max() else float("nan")
            ap = average_precision_score(y, s)
            normap = (ap - prev) / (1 - prev) if prev < 1 else float("nan")
            sub = sub.copy()
            sub["pred"] = (sub.pred_prob_fog.values >= t).astype(np.int8)
            agg = sub.groupby("session_id").agg(true=("native_label", "mean"), pred=("pred", "mean"))
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
    pooled = (out.prevalence * out.n_frames).sum() / out.n_frames.sum()
    print(f"\nWrote {len(out)} rows → {OUT}")
    print(f"Frame-weighted mean walking+standing prevalence ≈ {pooled*100:.2f}%")


if __name__ == "__main__":
    main()

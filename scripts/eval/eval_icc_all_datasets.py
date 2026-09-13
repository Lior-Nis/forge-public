"""
Compute per-session %TF ICC for fogathome, kaggle, and dailyliving using the
per-model DeFOG-validation thresholds from RESULTS_icc.md.

Output: logs/RESULTS_icc_all_datasets.csv
  columns: context, model, dataset, tf_icc
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg

CACHE  = Path("logs/comprehensive_eval_cache")
CONTEXTS = ["sc", "mc", "lc", "mc+sc", "lc+sc", "lc+mc", "lc+mc+sc"]
MODELS   = ["probe", "finetune", "supervised"]
DATASETS = ["fogathome", "kaggle", "dailyliving"]
COLS     = ["session_id", "pred_prob_fog", "native_label"]   # skip abs_frame / fog_ratio


def load_thresholds(path="logs/RESULTS_icc.md"):
    thr = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("| ") or "/" not in line:
                continue
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) < 3 or parts[0].startswith("ctx"):
                continue
            if "/" not in parts[0]:
                continue
            ctx, mdl = parts[0].rsplit("/", 1)
            try:
                thr[(ctx, mdl)] = float(parts[2])   # thr B column
            except ValueError:
                pass
    return thr


def session_tf_icc(df: pd.DataFrame, threshold: float) -> float:
    df = df.copy()
    df["pred"] = (df["pred_prob_fog"] >= threshold).astype(np.int8)
    sess = df.groupby("session_id")[["pred", "native_label"]].mean()
    if len(sess) < 3:
        return float("nan")
    long = pd.DataFrame({
        "session": sess.index.tolist() * 2,
        "rater":   ["pred"] * len(sess) + ["true"] * len(sess),
        "score":   sess["pred"].tolist() + sess["native_label"].tolist(),
    })
    res = pg.intraclass_corr(data=long, targets="session", raters="rater", ratings="score")
    return float(res.loc[res["Type"] == "ICC2", "ICC"].iloc[0])


def main():
    thresholds = load_thresholds()
    rows = []

    for ctx in CONTEXTS:
        for mdl in MODELS:
            thr = thresholds.get((ctx, mdl))
            if thr is None:
                print(f"  no threshold for {ctx}/{mdl}, skipping")
                continue
            for ds in DATASETS:
                pq = CACHE / f"{ds}_{ctx}_{mdl}_frames.parquet"
                if not pq.exists():
                    print(f"  missing {pq.name}")
                    continue
                print(f"  {ds:12s} {ctx}/{mdl:12s} thr={thr:.2f} ...", end=" ", flush=True)
                df = pd.read_parquet(pq, columns=COLS)
                icc = session_tf_icc(df, thr)
                rows.append({"context": ctx, "model": mdl, "dataset": ds, "tf_icc": icc})
                print(f"ICC={icc:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv("logs/RESULTS_icc_all_datasets.csv", index=False)
    print(f"\nWrote {len(out)} rows → logs/RESULTS_icc_all_datasets.csv")


if __name__ == "__main__":
    main()

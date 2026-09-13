"""
Per-session #FOG-episode ICC for kaggle / fogathome / dailyliving, parallel to
eval_icc_all_datasets.py (%TF). Same basis as the fig10/10b/10c heatmaps:
full cohort, our native_label, per-(ctx,model) protocol-B (DeFOG-validation)
threshold from RESULTS_icc.md.

DailyLiving and FogAtHome are conditioned on walking+standing (Activity ∈ {1,4}):
non-{1,4} frames are masked to non-FOG before episode detection, so freezing
episodes are counted only during ambulation while the gap-merge still runs over the
intact frame sequence. Kaggle is a structured walking protocol (no free-living
activity codes) and is scored whole.

Episodes are gap-merged (0.6 s @ 100 Hz), matching compute_icc_thresholds /
Salomon / Yang. ICC(2,1) between per-session gold and predicted episode counts.

Output: logs/RESULTS_fog_icc_all_datasets.csv  (context, model, dataset, fog_icc)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg

CACHE   = Path("logs/comprehensive_eval_cache")
CONTEXTS = ["sc", "mc", "lc", "mc+sc", "lc+sc", "lc+mc", "lc+mc+sc"]
MODELS   = ["probe", "finetune", "supervised"]
DATASETS = ["kaggle", "fogathome", "dailyliving"]
WALKSTAND = {"fogathome", "dailyliving"}        # condition on Activity ∈ {1,4}
KEEP = [1, 4]
GAP = 60                                         # 0.6 s @ 100 Hz episode gap-merge

FA_LABELS = os.path.join(os.environ.get("FORGE_DATASETS_ROOT", os.path.expanduser("~/Datasets")), "fogathome_dataset_forlior/fogathome_dataset/labels.csv")
DL_RAW    = os.path.join(os.environ.get("FORGE_DATASETS_ROOT", os.path.expanduser("~/Datasets")), "fogathome_dailyliving/preprocesseddata")


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


def n_episodes(binary, gap=GAP):
    """Gap-merged FOG episode count for one session (inclusive spans, gaps < gap merged)."""
    spans = 0
    start = None
    g = 0
    for v in binary:
        if v:
            if start is None:
                start = True
            g = 0
        elif start is not None:
            g += 1
            if g > gap:
                spans += 1
                start = None
    if start is not None:
        spans += 1
    return spans


def icc2(gt, pr):
    if len(gt) < 3 or np.std(gt) < 1e-9 or np.std(pr) < 1e-9:
        return float("nan")
    long = pd.DataFrame({
        "t": list(range(len(gt))) * 2,
        "r": ["g"] * len(gt) + ["m"] * len(gt),
        "v": list(gt) + list(pr),
    })
    res = pg.intraclass_corr(data=long, targets="t", raters="r", ratings="v")
    return float(res.loc[res["Type"] == "ICC2", "ICC"].iloc[0])


_FA_ACT = None
_DL_ACT = {}


def fogathome_activity():
    global _FA_ACT
    if _FA_ACT is None:
        gl = pd.read_csv(FA_LABELS, usecols=["Id", "Activity"])
        gl["session_id"] = gl["Id"].str.rsplit("_", n=1).str[0]
        gl["abs_frame"] = gl["Id"].str.rsplit("_", n=1).str[1].astype(int)
        _FA_ACT = gl[["session_id", "abs_frame", "Activity"]]
    return _FA_ACT


def dailyliving_activity(sid):
    import os
    if sid not in _DL_ACT:
        f = os.path.join(DL_RAW, sid + ".parquet")
        _DL_ACT[sid] = (pd.read_parquet(f, columns=["Activity"]).Activity.values
                        if os.path.exists(f) else None)
    return _DL_ACT[sid]


def fog_icc(df, ds, thr):
    """Per-session gold vs predicted episode-count ICC for one (ctx,model,dataset)."""
    df = df.copy()
    df["session_id"] = df.session_id.astype(str)
    df["pred"] = (df.pred_prob_fog.values >= thr).astype(np.int8)

    if ds == "fogathome":
        df = df.merge(fogathome_activity(), on=["session_id", "abs_frame"], how="left")

    gt_counts, pr_counts = [], []
    for sid, g in df.groupby("session_id", sort=False):
        g = g.sort_values("abs_frame")
        lab = g.native_label.values.astype(np.int8).copy()
        prd = g.pred.values.astype(np.int8).copy()
        if ds in WALKSTAND:
            if ds == "fogathome":
                act = g.Activity.values
            else:  # dailyliving: positional activity (abs_frame is 0..n-1 contiguous)
                a = dailyliving_activity(sid)
                act = np.full(len(g), -1, np.int16)
                if a is not None:
                    m = min(len(g), len(a))
                    act[:m] = a[:m]
            mask = np.isin(act, KEEP)
            lab[~mask] = 0
            prd[~mask] = 0
        gt_counts.append(n_episodes(lab))
        pr_counts.append(n_episodes(prd))
    return icc2(np.array(gt_counts, float), np.array(pr_counts, float))


def main():
    thr = load_thresholds()
    rows = []
    for ds in DATASETS:
        cols = ["session_id", "abs_frame", "pred_prob_fog", "native_label"]
        for ctx in CONTEXTS:
            for mdl in MODELS:
                pq = CACHE / f"{ds}_{ctx}_{mdl}_frames.parquet"
                if not pq.exists():
                    print(f"  missing {pq.name}")
                    continue
                t = thr.get((ctx, mdl), 0.5)
                df = pd.read_parquet(pq, columns=cols)
                icc = fog_icc(df, ds, t)
                rows.append({"context": ctx, "model": mdl, "dataset": ds, "fog_icc": icc})
                print(f"  {ds:11s} {ctx:9s}/{mdl:11s} thr={t:.2f} #FOG-ICC={icc:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("logs/RESULTS_fog_icc_all_datasets.csv", index=False)
    print(f"\nWrote {len(out)} rows → logs/RESULTS_fog_icc_all_datasets.csv")


if __name__ == "__main__":
    main()

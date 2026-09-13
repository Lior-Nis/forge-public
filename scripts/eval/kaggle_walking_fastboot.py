"""Fast paired bootstrap (numpy ICC2) for walking-only FORGE vs Kaggle3."""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "eval"))
from _private_inputs import private_path  # author-only inputs; see that module

RAW = os.path.join(os.environ.get("FORGE_DATASETS_ROOT", os.path.expanduser("~/Datasets")), "fogathome_dailyliving/preprocesseddata")
KD = private_path("kaggle_pred_vectors")
CACHE = "logs/comprehensive_eval_cache"
MAP = json.load(open(KD / "_id_map.json"))
REF_PKL = "/tmp/dl_ref_mapped.pkl"
RNG = np.random.default_rng(0)
NB = 2000


def icc2_np(gt, pr):
    """ICC(2,1) absolute agreement, two raters, vectorized."""
    gt = np.asarray(gt, float); pr = np.asarray(pr, float)
    n = len(gt)
    if n < 3:
        return np.nan
    M = np.column_stack([gt, pr]); k = 2
    grand = M.mean()
    rows = M.mean(1); cols = M.mean(0)
    SSR = k * ((rows - grand) ** 2).sum()
    SSC = n * ((cols - grand) ** 2).sum()
    SST = ((M - grand) ** 2).sum()
    SSE = SST - SSR - SSC
    MSR = SSR / (n - 1); MSC = SSC / (k - 1); MSE = SSE / ((n - 1) * (k - 1))
    den = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    return (MSR - MSE) / den if den > 0 else np.nan


def episodes(b):
    b = np.asarray(b)
    return int(((b[1:] == 1) & (b[:-1] == 0)).sum() + (b[0] == 1)) if len(b) else 0


ref = pickle.load(open(REF_PKL, "rb"))
k = pd.read_csv(KD / "pred_3rd.csv", usecols=["Id"])
k["sid"] = k.Id.str.rsplit("_", n=1).str[0]; k["fr"] = k.Id.str.rsplit("_", n=1).str[1].astype(int)
KF = {s: g.fr.values for s, g in k.groupby("sid")}


def build_sessions(get):
    out = []
    for her, our in MAP.items():
        fog, act = ref[our]
        fr, sc = get(her)
        if fr is None:
            continue
        ok = fr < len(fog); fr, sc = fr[ok], sc[ok]
        m = act[fr] == 1
        if m.sum() == 0:
            continue
        y, s = fog[fr][m], sc[m]
        v = ~np.isnan(s)
        if v.sum() == 0:
            continue
        out.append((our, y[v].astype(np.int8), s[v].astype(np.float32)))
    return out


def forge_get(cfg):
    fp = pd.read_parquet(f"{CACHE}/dailyliving_{cfg}_frames.parquet",
                         columns=["session_id", "abs_frame", "pred_prob_fog"])
    fp["session_id"] = fp.session_id.astype(str)
    fp = fp[fp.session_id.isin(set(MAP.values()))]
    arrs = {}
    for sid, g in fp.groupby("session_id", sort=False):
        a = np.full(g.abs_frame.max() + 1, np.nan, np.float32)
        a[g.abs_frame.values] = g.pred_prob_fog.values
        arrs[sid] = a

    def g(her):
        our = MAP[her]; arr = arrs.get(our); fr = KF[her]
        if arr is None:
            return None, None
        sc = np.full(len(fr), np.nan, np.float32); ok = fr < len(arr); sc[ok] = arr[fr[ok]]
        return fr, sc
    return g


def kaggle_get(tag):
    d = pd.read_csv(KD / f"pred_{tag}.csv")
    d["sid"] = d.Id.str.rsplit("_", n=1).str[0]; d["fr"] = d.Id.str.rsplit("_", n=1).str[1].astype(int)
    ks = {s: (g.fr.values, g.FOG.values.astype(np.float32)) for s, g in d.groupby("sid")}

    def g(her):
        v = ks.get(her); return (v[0], v[1]) if v else (None, None)
    return g


def summ(sessions):
    """per-session (gt%TF, pr%TF, gt_ep, pr_ep) + pooled y,s using global thr."""
    y = np.concatenate([s[1] for s in sessions]); s_ = np.concatenate([s[2] for s in sessions])
    prev = y.mean(); thr = np.quantile(s_, 1 - prev)
    rows = [(yy.mean() * 100, (ss >= thr).mean() * 100, episodes(yy), episodes((ss >= thr).astype(int)),
             yy, ss) for _, yy, ss in sessions]
    return rows, prev, thr


def metrics(rows, prev):
    y = np.concatenate([r[4] for r in rows]); s = np.concatenate([r[5] for r in rows])
    ap = average_precision_score(y, s); nap = (ap - prev) / (1 - prev)
    gt = np.array([r[0] for r in rows]); pr = np.array([r[1] for r in rows])
    ge = np.array([r[2] for r in rows], float); pe = np.array([r[3] for r in rows], float)
    return nap, icc2_np(gt, pr), icc2_np(ge, pe)


pairs = [("FORGE_mc_probe", forge_get("mc_probe")),
         ("FORGE_lc+mc+sc_probe", forge_get("lc+mc+sc_probe")),
         ("Kaggle3", kaggle_get("3rd"))]
store = {name: build_sessions(g) for name, g in pairs}
# align all to common our_sids
common_ids = set.intersection(*[{s[0] for s in store[n]} for n in store])
for n in store:
    store[n] = sorted([s for s in store[n] if s[0] in common_ids], key=lambda x: x[0])
N = len(common_ids)
print(f"common walking sessions: {N}\n", flush=True)

for name in store:
    rows, prev, _ = summ(store[name])
    nap, tf, fg = metrics(rows, prev)
    print(f"{name:22s} normAP={nap:.3f} TF_ICC={tf:.3f} FOG_ICC={fg:.3f}", flush=True)

print("\n── paired Δ ICC vs Kaggle3 (2000x session bootstrap, 95% CI) ──", flush=True)
# precompute per-session 4-scalars (gt%TF, pr%TF, gt_ep, pr_ep) per model
scal = {}
for name in store:
    rows, prev, _ = summ(store[name])
    scal[name] = np.array([(r[0], r[1], r[2], r[3]) for r in rows], float)  # (N,4)
K = scal["Kaggle3"]
for name in ["FORGE_mc_probe", "FORGE_lc+mc+sc_probe"]:
    A = scal[name]
    dt, dfg = [], []
    for _ in range(NB):
        idx = RNG.integers(0, N, N)
        a, kk = A[idx], K[idx]
        dt.append(icc2_np(a[:, 0], a[:, 1]) - icc2_np(kk[:, 0], kk[:, 1]))
        dfg.append(icc2_np(a[:, 2], a[:, 3]) - icc2_np(kk[:, 2], kk[:, 3]))
    for lab, d in [("TF_ICC", dt), ("FOG_ICC", dfg)]:
        lo, hi = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5); med = np.nanmedian(d)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        print(f"  {name:20s} Δ{lab:8s}={med:+.3f} [{lo:+.3f},{hi:+.3f}] {sig}", flush=True)
print("DONE", flush=True)

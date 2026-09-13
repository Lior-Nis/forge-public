"""Paired session bootstrap on the {1,4} filter: FORGE vs Kaggle3.
normAP (200x, pooled-frame) + %TF-ICC & #FOG-ICC (2000x, scalar)."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "eval"))
from _private_inputs import private_path  # author-only inputs; see that module

KD = private_path("kaggle_pred_vectors")
KFULL = private_path("kaggle_pred_full")
CACHE = "logs/comprehensive_eval_cache"
MAP = json.load(open(KD / "_id_map.json"))
REF = pickle.load(open(KD / "_ref_gt_act.pkl", "rb"))
RNG = np.random.default_rng(0)
CODES = [1, 4]


def icc2_np(gt, pr):
    gt = np.asarray(gt, float); pr = np.asarray(pr, float); n = len(gt)
    if n < 3:
        return np.nan
    M = np.column_stack([gt, pr]); k = 2
    g = M.mean(); rows = M.mean(1); cols = M.mean(0)
    SSR = k * ((rows - g) ** 2).sum(); SSC = n * ((cols - g) ** 2).sum()
    SSE = ((M - g) ** 2).sum() - SSR - SSC
    MSR = SSR / (n - 1); MSC = SSC / (k - 1); MSE = SSE / ((n - 1) * (k - 1))
    den = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    return (MSR - MSE) / den if den > 0 else np.nan


def episodes(b):
    b = np.asarray(b)
    return int(((b[1:] == 1) & (b[:-1] == 0)).sum() + (b[0] == 1)) if len(b) else 0


def forge_scores(cfg):
    fp = pd.read_parquet(f"{CACHE}/dailyliving_{cfg}_frames.parquet",
                         columns=["session_id", "abs_frame", "pred_prob_fog"])
    fp["session_id"] = fp.session_id.astype(str)
    fp = fp[fp.session_id.isin(set(MAP.values()))]
    out = {}
    for sid, g in fp.groupby("session_id", sort=False):
        a = np.full(g.abs_frame.max() + 1, np.nan, np.float32)
        a[g.abs_frame.values] = g.pred_prob_fog.values
        out[sid] = a
    return out


def build_sessions(scores):
    """list of (our_sid, y, s) under {1,4}."""
    out = []
    for our, (fog, act) in REF.items():
        arr = scores.get(our)
        if arr is None:
            continue
        n = min(len(fog), len(arr))
        fog, act, sc = fog[:n], act[:n], arr[:n]
        m = np.isin(act, CODES) & ~np.isnan(sc)
        if m.sum() == 0:
            continue
        out.append((our, fog[m].astype(np.int8), sc[m].astype(np.float32)))
    return out


def point(sessions):
    y = np.concatenate([s[1] for s in sessions]); s = np.concatenate([s[2] for s in sessions])
    prev = y.mean(); ap = average_precision_score(y, s); nap = (ap - prev) / (1 - prev)
    thr = np.quantile(s, 1 - prev)
    gt = np.array([yy.mean() * 100 for _, yy, _ in sessions])
    pr = np.array([(ss >= thr).mean() * 100 for _, _, ss in sessions])
    ge = np.array([episodes(yy) for _, yy, _ in sessions], float)
    pe = np.array([episodes((ss >= thr).astype(int)) for _, _, ss in sessions], float)
    return dict(normAP=nap, TF_ICC=icc2_np(gt, pr), FOG_ICC=icc2_np(ge, pe))


def main():
    models = {
        "FORGE_lc+mc+sc": build_sessions(forge_scores("lc+mc+sc_probe")),
        "FORGE_mc_probe": build_sessions(forge_scores("mc_probe")),
        "Kaggle3": build_sessions(pickle.load(open(str(KFULL / "pred3.csv.scores.pkl"), "rb"))),
    }
    # align to common sessions, same order
    common = sorted(set.intersection(*[{s[0] for s in v} for v in models.values()]))
    idxmap = {name: {s[0]: s for s in v} for name, v in models.items()}
    aligned = {name: [idxmap[name][c] for c in common] for name in models}
    N = len(common)
    print(f"{{1,4}} common sessions: {N}\n", flush=True)

    for name, S in aligned.items():
        p = point(S)
        print(f"{name:16s} normAP={p['normAP']:.3f} TF_ICC={p['TF_ICC']:.3f} FOG_ICC={p['FOG_ICC']:.3f}", flush=True)

    # precompute per-session scalars (gt%TF, pr%TF, gt_ep, pr_ep) for ICC bootstrap
    scal = {}
    for name, S in aligned.items():
        y = np.concatenate([s[1] for s in S]); s_ = np.concatenate([s[2] for s in S])
        thr = np.quantile(s_, 1 - y.mean())
        scal[name] = np.array([(yy.mean() * 100, (ss >= thr).mean() * 100,
                                episodes(yy), episodes((ss >= thr).astype(int))) for _, yy, ss in S], float)

    print("\n── paired Δ vs Kaggle3 on {1,4} (95% CI) ──", flush=True)
    K = scal["Kaggle3"]
    for name in ["FORGE_lc+mc+sc", "FORGE_mc_probe"]:
        A = scal[name]
        dt, dfg = [], []
        for _ in range(2000):
            ix = RNG.integers(0, N, N)
            a, kk = A[ix], K[ix]
            dt.append(icc2_np(a[:, 0], a[:, 1]) - icc2_np(kk[:, 0], kk[:, 1]))
            dfg.append(icc2_np(a[:, 2], a[:, 3]) - icc2_np(kk[:, 2], kk[:, 3]))
        for lab, d in [("TF_ICC", dt), ("FOG_ICC", dfg)]:
            lo, hi = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5); med = np.nanmedian(d)
            sig = "SIG" if (lo > 0 or hi < 0) else "ns"
            print(f"  {name:16s} Δ{lab:8s}={med:+.3f} [{lo:+.3f},{hi:+.3f}] {sig}", flush=True)
        # normAP: lighter bootstrap with pooled frames
        SA = aligned[name]; SK = aligned["Kaggle3"]
        dn = []
        for _ in range(200):
            ix = RNG.integers(0, N, N)
            sa = [SA[i] for i in ix]; sk = [SK[i] for i in ix]
            dn.append(point(sa)["normAP"] - point(sk)["normAP"])
        lo, hi = np.nanpercentile(dn, 2.5), np.nanpercentile(dn, 97.5); med = np.nanmedian(dn)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        print(f"  {name:16s} ΔnormAP  ={med:+.3f} [{lo:+.3f},{hi:+.3f}] {sig}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

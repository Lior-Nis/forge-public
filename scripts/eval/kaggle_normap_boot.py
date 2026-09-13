"""Standalone normAP paired bootstrap on {1,4}: FORGE vs Kaggle3 (200x)."""
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
RNG = np.random.default_rng(1)
CODES = [1, 4]


def forge(cfg):
    fp = pd.read_parquet(f"{CACHE}/dailyliving_{cfg}_frames.parquet",
                         columns=["session_id", "abs_frame", "pred_prob_fog"])
    fp["session_id"] = fp.session_id.astype(str)
    fp = fp[fp.session_id.isin(set(MAP.values()))]
    o = {}
    for sid, g in fp.groupby("session_id", sort=False):
        a = np.full(g.abs_frame.max() + 1, np.nan, np.float32)
        a[g.abs_frame.values] = g.pred_prob_fog.values
        o[sid] = a
    return o


def sess(scores):
    out = []
    for our, (fog, act) in REF.items():
        arr = scores.get(our)
        if arr is None:
            continue
        n = min(len(fog), len(arr))
        m = np.isin(act[:n], CODES) & ~np.isnan(arr[:n])
        if m.sum() == 0:
            continue
        out.append((our, fog[:n][m].astype(np.int8), arr[:n][m].astype(np.float32)))
    return out


def nap(S):
    y = np.concatenate([s[1] for s in S]); s = np.concatenate([s[2] for s in S]); p = y.mean()
    return (average_precision_score(y, s) - p) / (1 - p)


M = {"FORGE_lc+mc+sc": sess(forge("lc+mc+sc_probe")),
     "FORGE_mc_probe": sess(forge("mc_probe")),
     "Kaggle3": sess(pickle.load(open(str(KFULL / "pred3.csv.scores.pkl"), "rb")))}
common = sorted(set.intersection(*[{s[0] for s in v} for v in M.values()]))
idx = {n: {s[0]: s for s in v} for n, v in M.items()}
A = {n: [idx[n][c] for c in common] for n in M}
N = len(common); K = A["Kaggle3"]
print(f"{{1,4}} sessions: {N}", flush=True)
for name in ["FORGE_lc+mc+sc", "FORGE_mc_probe"]:
    SA = A[name]; d = []
    for _ in range(200):
        ix = RNG.integers(0, N, N)
        d.append(nap([SA[i] for i in ix]) - nap([K[i] for i in ix]))
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    print(f"{name:16s} ΔnormAP={np.median(d):+.3f} [{lo:+.3f},{hi:+.3f}] "
          f"{'SIG' if (lo > 0 or hi < 0) else 'ns'}", flush=True)
print("DONE", flush=True)

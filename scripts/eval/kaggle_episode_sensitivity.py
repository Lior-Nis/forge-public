"""
#FOG episode-ICC sensitivity to the episode definition (min-duration + gap-merge),
counted on the FULL contiguous timeline, under the {1,4} evaluation threshold.
Tests whether FORGE's #FOG-ICC advantage over Kaggle3 is robust to the (unknown)
episode rule Salomon used.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "eval"))
from _private_inputs import private_path  # author-only inputs; see that module

KD = private_path("kaggle_pred_vectors")
KFULL = private_path("kaggle_pred_full")
CACHE = "logs/comprehensive_eval_cache"
MAP = json.load(open(KD / "_id_map.json"))
REF = pickle.load(open(KD / "_ref_gt_act.pkl", "rb"))
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


def count_episodes(b, min_len, gap):
    """count runs of 1 after merging runs separated by < gap and dropping runs < min_len."""
    b = np.asarray(b).astype(np.int8)
    if b.sum() == 0:
        return 0
    d = np.diff(np.concatenate([[0], b, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]  # exclusive
    # merge gaps
    if gap > 0 and len(starts) > 1:
        ms, me = [starts[0]], [ends[0]]
        for s, e in zip(starts[1:], ends[1:]):
            if s - me[-1] < gap:
                me[-1] = e
            else:
                ms.append(s); me.append(e)
        starts, ends = np.array(ms), np.array(me)
    lens = ends - starts
    return int((lens >= min_len).sum())


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


def model_threshold(scores):
    """calibration-free threshold = match {1,4} pred-positive rate to {1,4} FOG prevalence."""
    Y, S = [], []
    for our, (fog, act) in REF.items():
        arr = scores.get(our)
        if arr is None:
            continue
        n = min(len(fog), len(arr)); m = np.isin(act[:n], CODES) & ~np.isnan(arr[:n])
        Y.append(fog[:n][m]); S.append(arr[:n][m])
    y = np.concatenate(Y); s = np.concatenate(S)
    return np.quantile(s, 1 - y.mean())


MODELS = {
    "FORGE_lc+mc+sc": forge("lc+mc+sc_probe"),
    "FORGE_mc_probe": forge("mc_probe"),
    "Kaggle3": pickle.load(open(str(KFULL / "pred3.csv.scores.pkl"), "rb")),
}
THR = {n: model_threshold(s) for n, s in MODELS.items()}
common = sorted(set.intersection(*[{k for k, v in REF.items() if MODELS[n].get(k) is not None} for n in MODELS]))
print(f"sessions: {len(common)}\n")

GRID = [(ml, gp) for ml in (1, 10, 25, 50, 100) for gp in (0, 25, 100)]
print(f"{'min_len':>7} {'gap':>5} | {'FORGE_lc+mc+sc':>14} {'FORGE_mc':>9} {'Kaggle3':>8} | Δ(lc+mc+sc−K3)")
print("-" * 72)
for ml, gp in GRID:
    iccs = {}
    for name in MODELS:
        thr = THR[name]; sc = MODELS[name]
        ge, pe = [], []
        for our in common:
            fog = REF[our][0]; arr = sc[our]
            n = min(len(fog), len(arr))
            ge.append(count_episodes(fog[:n], ml, gp))
            pe.append(count_episodes((arr[:n] >= thr).astype(np.int8), ml, gp))
        iccs[name] = icc2_np(np.array(ge, float), np.array(pe, float))
    d = iccs["FORGE_lc+mc+sc"] - iccs["Kaggle3"]
    print(f"{ml:>7} {gp:>5} | {iccs['FORGE_lc+mc+sc']:>14.3f} {iccs['FORGE_mc_probe']:>9.3f} "
          f"{iccs['Kaggle3']:>8.3f} | {d:+.3f}", flush=True)
print("\nDONE")

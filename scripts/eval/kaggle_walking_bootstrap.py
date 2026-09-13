"""
Walking-only ({1}) aligned FORGE-vs-Kaggle comparison with session-level
bootstrap CIs. Emphasis on imbalance-appropriate metrics: normAP, %TF ICC,
#FOG ICC. AUC reported only as reference.

Reuses the fingerprint id-map from kaggle_pred_vector_eval.py.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import average_precision_score, roc_auc_score
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "eval"))
from _private_inputs import private_path  # author-only inputs; see that module

RAW = os.path.join(os.environ.get("FORGE_DATASETS_ROOT", os.path.expanduser("~/Datasets")), "fogathome_dailyliving/preprocesseddata")
KD = private_path("kaggle_pred_vectors")
CACHE = "logs/comprehensive_eval_cache"
MAP = json.load(open(KD / "_id_map.json"))          # her_sid -> our_sid
REF_PKL = "/tmp/dl_ref_mapped.pkl"
NBOOT = 1000
RNG = np.random.default_rng(0)

FORGE_CFGS = ["mc_probe", "mc_finetune", "lc_probe", "lc+mc_probe", "mc+sc_probe", "lc+mc+sc_probe"]


def icc2(gt, pr):
    n = len(gt)
    if n < 3 or np.std(pr) < 1e-9 or np.std(gt) < 1e-9:
        return np.nan
    long = pd.concat([pd.DataFrame({"t": range(n), "r": "g", "v": gt}),
                      pd.DataFrame({"t": range(n), "r": "m", "v": pr})])
    try:
        return float(pg.intraclass_corr(data=long, targets="t", raters="r",
                     ratings="v").set_index("Type").loc["ICC2", "ICC"])
    except Exception:
        return np.nan


def episodes(b):
    b = np.asarray(b)
    return int(((b[1:] == 1) & (b[:-1] == 0)).sum() + (b[0] == 1)) if len(b) else 0


# ── reference: our_sid -> (fog, activity) over full recording ──
def load_ref():
    if os.path.exists(REF_PKL):
        return pickle.load(open(REF_PKL, "rb"))
    ref = {}
    for her, our in MAP.items():
        d = pd.read_parquet(os.path.join(RAW, our + ".parquet"),
                            columns=["StartHesitation", "Turn", "Walking", "Activity"])
        fog = ((d.StartHesitation.values + d.Turn.values + d.Walking.values) > 0).astype(np.int8)
        ref[our] = (fog, d.Activity.values.astype(np.int16))
    pickle.dump(ref, open(REF_PKL, "wb"))
    return ref


def kaggle_frames():
    k = pd.read_csv(KD / "pred_3rd.csv", usecols=["Id"])
    k["sid"] = k.Id.str.rsplit("_", n=1).str[0]
    k["fr"] = k.Id.str.rsplit("_", n=1).str[1].astype(int)
    return {s: g.fr.values for s, g in k.groupby("sid")}


def per_session_walking(get_scores, ref, kf):
    """Return list of (our_sid, y_walk, s_walk) on walking-only frames."""
    out = []
    for her, our in MAP.items():
        fog, act = ref[our]
        frames, scores = get_scores(her)
        if frames is None:
            continue
        ok = frames < len(fog)
        frames, scores = frames[ok], scores[ok]
        a = act[frames]
        m = a == 1
        if m.sum() == 0:
            continue
        y, s = fog[frames][m], scores[m]
        v = ~np.isnan(s)
        if v.sum() == 0:
            continue
        out.append((our, y[v].astype(np.int8), s[v].astype(np.float32)))
    return out


def metrics_from_sessions(sessions):
    y = np.concatenate([s[1] for s in sessions])
    s = np.concatenate([s[2] for s in sessions])
    prev = y.mean()
    ap = average_precision_score(y, s)
    normap = (ap - prev) / (1 - prev)
    auc = roc_auc_score(y, s)
    thr = np.quantile(s, 1 - prev)  # calibration-free: match pred-positive rate to prevalence
    rows = [(yy.mean() * 100, (ss >= thr).mean() * 100, episodes(yy), episodes((ss >= thr).astype(int)))
            for _, yy, ss in sessions]
    d = pd.DataFrame(rows, columns=["gt", "pr", "gtep", "prep"])
    return dict(normAP=normap, AP=ap, AUC=auc, prev=prev,
                TF_ICC=icc2(d["gt"].values, d["pr"].values),
                FOG_ICC=icc2(d["gtep"].values.astype(float), d["prep"].values.astype(float)))


def bootstrap(sessions, nboot=NBOOT):
    """Session-level bootstrap -> CIs for normAP, TF_ICC, FOG_ICC."""
    n = len(sessions)
    keys = ["normAP", "TF_ICC", "FOG_ICC"]
    samp = {k: [] for k in keys}
    for _ in range(nboot):
        idx = RNG.integers(0, n, n)
        bs = [sessions[i] for i in idx]
        m = metrics_from_sessions(bs)
        for k in keys:
            samp[k].append(m[k])
    return {k: (np.nanpercentile(samp[k], 2.5), np.nanpercentile(samp[k], 97.5)) for k in keys}


def main():
    ref = load_ref()
    kf = kaggle_frames()
    print(f"mapped sessions: {len(MAP)}\n", flush=True)

    table = []
    sess_store = {}

    # FORGE configs
    for cfg in FORGE_CFGS:
        p = f"{CACHE}/dailyliving_{cfg}_frames.parquet"
        if not os.path.exists(p):
            continue
        fp = pd.read_parquet(p, columns=["session_id", "abs_frame", "pred_prob_fog"])
        fp["session_id"] = fp.session_id.astype(str)
        ours = set(MAP.values())
        fp = fp[fp.session_id.isin(ours)]
        arrs = {}
        for sid, g in fp.groupby("session_id", sort=False):
            a = np.full(g.abs_frame.max() + 1, np.nan, np.float32)
            a[g.abs_frame.values] = g.pred_prob_fog.values
            arrs[sid] = a

        def get(her, arrs=arrs):
            our = MAP[her]; arr = arrs.get(our); frames = kf[her]
            if arr is None:
                return None, None
            sc = np.full(len(frames), np.nan, np.float32)
            ok = frames < len(arr)
            sc[ok] = arr[frames[ok]]
            return frames, sc

        S = per_session_walking(get, ref, kf)
        sess_store["FORGE_" + cfg] = S
        m = metrics_from_sessions(S)
        table.append(["FORGE_" + cfg, m])
        print(f"FORGE_{cfg:14s} normAP={m['normAP']:.3f} TF_ICC={m['TF_ICC']:.3f} "
              f"FOG_ICC={m['FOG_ICC']:.3f} (AUC={m['AUC']:.3f})", flush=True)

    # Kaggle models
    for tag, lab in [("1st", "Kaggle1"), ("2nd", "Kaggle2"), ("3rd", "Kaggle3"),
                     ("4th", "Kaggle4"), ("5th", "Kaggle5")]:
        d = pd.read_csv(KD / f"pred_{tag}.csv")
        d["sid"] = d.Id.str.rsplit("_", n=1).str[0]
        d["fr"] = d.Id.str.rsplit("_", n=1).str[1].astype(int)
        ks = {s: (g.fr.values, g.FOG.values.astype(np.float32)) for s, g in d.groupby("sid")}

        def get(her, ks=ks):
            v = ks.get(her)
            return (v[0], v[1]) if v else (None, None)

        S = per_session_walking(get, ref, kf)
        sess_store[lab] = S
        m = metrics_from_sessions(S)
        table.append([lab, m])
        print(f"{lab:20s} normAP={m['normAP']:.3f} TF_ICC={m['TF_ICC']:.3f} "
              f"FOG_ICC={m['FOG_ICC']:.3f} (AUC={m['AUC']:.3f})", flush=True)

    # bootstrap CIs for flagship + best + Kaggle3
    print("\n── 95% bootstrap CIs (session-level, 1000x) ──", flush=True)
    for name in ["FORGE_mc_probe", "Kaggle3"]:
        ci = bootstrap(sess_store[name])
        print(f"{name:16s} normAP {ci['normAP'][0]:.3f}-{ci['normAP'][1]:.3f} | "
              f"TF_ICC {ci['TF_ICC'][0]:.3f}-{ci['TF_ICC'][1]:.3f} | "
              f"FOG_ICC {ci['FOG_ICC'][0]:.3f}-{ci['FOG_ICC'][1]:.3f}", flush=True)

    # paired bootstrap: FORGE_mc_probe minus Kaggle3 (same resampled sessions)
    A = sess_store["FORGE_mc_probe"]; B = sess_store["Kaggle3"]
    # align by our_sid
    bmap = {s[0]: s for s in B}
    common = [(a, bmap[a[0]]) for a in A if a[0] in bmap]
    n = len(common)
    diffs = {"normAP": [], "TF_ICC": [], "FOG_ICC": []}
    for _ in range(NBOOT):
        idx = RNG.integers(0, n, n)
        aa = [common[i][0] for i in idx]
        bb = [common[i][1] for i in idx]
        ma = metrics_from_sessions(aa); mb = metrics_from_sessions(bb)
        for k in diffs:
            diffs[k].append(ma[k] - mb[k])
    print("\n── paired Δ (FORGE_mc_probe − Kaggle3), 95% CI ──", flush=True)
    for k in diffs:
        lo, hi = np.nanpercentile(diffs[k], 2.5), np.nanpercentile(diffs[k], 97.5)
        med = np.nanmedian(diffs[k])
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        print(f"  Δ{k:8s} = {med:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {sig}", flush=True)

    pd.DataFrame([[t[0], t[1]["normAP"], t[1]["AP"], t[1]["TF_ICC"], t[1]["FOG_ICC"], t[1]["AUC"], t[1]["prev"]]
                  for t in table],
                 columns=["model", "normAP", "AP", "TF_ICC", "FOG_ICC", "AUC", "prev"]
                 ).to_csv(KD / "RESULTS_walking_bootstrap.csv", index=False)
    print("\nsaved RESULTS_walking_bootstrap.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()

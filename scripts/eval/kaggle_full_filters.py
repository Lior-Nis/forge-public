"""
Full-filter FORGE vs Kaggle comparison using the UN-gait-filtered Kaggle
prediction vectors (research/paper_final/data/kaggle_pred_full/pred{2,3}.csv,
every frame of every recording). Ground truth + Activity from the raw
daily-living files. Reuses the fingerprint id-map.

Now that all frames are present, {1} walking, {1,4} walk+stand, {1,3,4}, all,
and OR are ALL computable for the Kaggle models. Reports imbalance-appropriate
metrics: normAP, %TF-ICC, #FOG-ICC (AUC as reference).
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
KFULL = private_path("kaggle_pred_full")
CACHE = "logs/comprehensive_eval_cache"
MAP = json.load(open(KD / "_id_map.json"))          # her_sid -> our_sid
REF_PKL = KD / "_ref_gt_act.pkl"
FILTERS = {"{1}": [1], "{1,4}": [1, 4], "{1,3,4}": [1, 3, 4], "all": None, "OR": "or"}


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


def load_ref():
    if REF_PKL.exists():
        return pickle.load(open(REF_PKL, "rb"))
    ref = {}
    for our in set(MAP.values()):
        d = pd.read_parquet(os.path.join(RAW, our + ".parquet"),
                            columns=["StartHesitation", "Turn", "Walking", "Activity"])
        fog = ((d.StartHesitation.values + d.Turn.values + d.Walking.values) > 0).astype(np.int8)
        ref[our] = (fog, d.Activity.values.astype(np.int16))
    pickle.dump(ref, open(REF_PKL, "wb"))
    return ref


def forge_scores(cfg):
    """our_sid -> dense score array (NaN where missing)."""
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


def kaggle_full_scores(csv_path):
    """Chunked read of huge un-filtered csv -> our_sid -> dense score array (cached)."""
    cache = Path(str(csv_path) + ".scores.pkl")
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    inv = MAP  # her_sid -> our_sid
    buf = {}  # her_sid -> (frames list, scores list)
    for chunk in pd.read_csv(csv_path, usecols=["Id", "FOG"], chunksize=8_000_000):
        sid = chunk.Id.str.rsplit("_", n=1).str[0].values
        fr = chunk.Id.str.rsplit("_", n=1).str[1].astype(np.int32).values
        sc = chunk.FOG.values.astype(np.float32)
        order = np.argsort(sid, kind="stable")
        sid, fr, sc = sid[order], fr[order], sc[order]
        uniq, idx = np.unique(sid, return_index=True)
        idx = list(idx) + [len(sid)]
        for i, s in enumerate(uniq):
            if s not in inv:
                continue
            sl = slice(idx[i], idx[i + 1])
            buf.setdefault(s, ([], [])); buf[s][0].append(fr[sl]); buf[s][1].append(sc[sl])
    out = {}
    for her, (frs, scs) in buf.items():
        our = inv[her]
        fr = np.concatenate(frs); sc = np.concatenate(scs)
        n = int(fr.max()) + 1
        a = np.full(n, np.nan, np.float32); a[fr] = sc
        out[our] = a
    pickle.dump(out, open(cache, "wb"))
    return out


def evaluate(scores, ref):
    """scores: our_sid -> dense array. Returns per-filter metrics dict."""
    res = {}
    for fname, fdef in FILTERS.items():
        Y, S, sess = [], [], []
        for our, fog_act in ref.items():
            fog, act = fog_act
            arr = scores.get(our)
            if arr is None:
                continue
            n = min(len(fog), len(arr))
            fog, act, sc = fog[:n], act[:n], arr[:n]
            if fdef == "or":
                m = (act == 1) | (fog == 1)
            elif fdef is None:
                m = np.ones(n, bool)
            else:
                m = np.isin(act, fdef)
            m &= ~np.isnan(sc)
            if m.sum() == 0:
                continue
            y, s = fog[m].astype(np.int8), sc[m]
            Y.append(y); S.append(s)
            sess.append((y, s))
        y = np.concatenate(Y); s = np.concatenate(S); prev = y.mean()
        ap = average_precision_score(y, s); nap = (ap - prev) / (1 - prev)
        auc = roc_auc_score(y, s)
        thr = np.quantile(s, 1 - prev)
        rows = [(yy.mean() * 100, (ss >= thr).mean() * 100, episodes(yy), episodes((ss >= thr).astype(int)))
                for yy, ss in sess]
        d = pd.DataFrame(rows, columns=["gt", "pr", "ge", "pe"])
        res[fname] = dict(prev=prev, AUC=auc, AP=ap, normAP=nap,
                          TF_ICC=icc2(d["gt"].values, d["pr"].values),
                          FOG_ICC=icc2(d["ge"].values.astype(float), d["pe"].values.astype(float)),
                          nsess=len(d), n=len(y))
    return res


def main():
    ref = load_ref()
    print(f"mapped sessions: {len(ref)}\n", flush=True)
    models = []
    models.append(("FORGE_mc_probe", forge_scores("mc_probe")))
    models.append(("FORGE_lc+mc+sc_probe", forge_scores("lc+mc+sc_probe")))
    print("loaded FORGE", flush=True)
    for n in [1, 2, 3, 4, 5]:
        models.append((f"Kaggle{n}", kaggle_full_scores(KFULL / f"pred{n}.csv")))
        print(f"loaded Kaggle{n} full", flush=True)
    print(flush=True)

    # Intersection guard: every model must be scored on the SAME sessions, else the
    # nsess/prevalence differ across rows and the comparison is not apples-to-apples.
    # (Relevant after id-map recovery: FORGE covers all mapped sessions but the cached
    # Kaggle score pkls only hold sessions present when they were parsed.)
    common = set(ref)
    for _, sc in models:
        common &= set(sc)
    dropped = len(ref) - len(common)
    if dropped:
        print(f"intersection guard: {len(common)} sessions common to all "
              f"{len(models)} models ({dropped} dropped — not scored for every model)\n", flush=True)
    ref = {sid: ref[sid] for sid in common}

    rows = []
    for name, sc in models:
        r = evaluate(sc, ref)
        for f in FILTERS:
            x = r[f]
            rows.append([name, f, round(x["prev"] * 100, 1), round(x["normAP"], 3),
                         round(x["TF_ICC"], 3), round(x["FOG_ICC"], 3), round(x["AUC"], 3),
                         round(x["AP"], 3), x["nsess"]])
            print(f"{name:22s} {f:8s} prev={x['prev']*100:4.1f}% normAP={x['normAP']:.3f} "
                  f"TF_ICC={x['TF_ICC']:.3f} FOG_ICC={x['FOG_ICC']:.3f} (AUC={x['AUC']:.3f})", flush=True)
        print(flush=True)
    pd.DataFrame(rows, columns=["model", "filter", "prev", "normAP", "TF_ICC", "FOG_ICC",
                                "AUC", "AP", "nsess"]).to_csv(
        KFULL / "RESULTS_full_filters.csv", index=False)
    print("saved RESULTS_full_filters.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()

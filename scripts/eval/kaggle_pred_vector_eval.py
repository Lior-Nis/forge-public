"""
Apples-to-apples FORGE vs Kaggle-winner comparison on FogAtHome daily-living.

Salomon shared per-frame continuous FOG scores for the 5 Kaggle winners
(research/paper_final/data/kaggle_pred_vectors/pred_*.csv, Id="{her_sid}_{frame}",
gait-filtered = frames where walking OR FOG-label==1, original frame indices kept).

Her session IDs use a different base-36 encoding than ours, with ZERO overlap.
We recover the mapping by FINGERPRINTING each recording on its exact kept-frame
index set (md5 of the sorted int32 index array) under the walk|FOG filter — an
exact set-hash match is collision-free except for short all-walking recordings
with identical index sets (dropped as ambiguous).

Once mapped, every model is scored in ONE pipeline against our raw per-frame
FOG label + Activity code, under {walk-only} and {OR} filters, for
AUC / AP / %TF-ICC(session,subject) / #FOG-ICC.
"""
import hashlib
import json
import os
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
FORGE_PARQUET = "logs/comprehensive_eval_cache/dailyliving_mc_probe_frames.parquet"
MAP_CACHE = private_path("kaggle_pred_vectors", "_id_map.json", output=True)
THR = 0.26
KAGGLE = [("1st", "Kaggle1"), ("2nd", "Kaggle2"), ("3rd", "Kaggle3"),
          ("4th", "Kaggle4"), ("5th", "Kaggle5")]


def fp(idx):
    return hashlib.md5(np.asarray(sorted(idx), dtype=np.int32).tobytes()).hexdigest()


def icc2(gt, pr):
    n = len(gt)
    if n < 3 or np.std(pr) < 1e-9 or np.std(gt) < 1e-9:
        return float("nan")
    long = pd.concat([pd.DataFrame({"t": range(n), "r": "g", "v": gt}),
                      pd.DataFrame({"t": range(n), "r": "m", "v": pr})])
    try:
        return float(pg.intraclass_corr(data=long, targets="t", raters="r",
                     ratings="v").set_index("Type").loc["ICC2", "ICC"])
    except Exception:
        return float("nan")


def episodes(b):
    b = np.asarray(b)
    return int(((b[1:] == 1) & (b[:-1] == 0)).sum() + (b[0] == 1)) if len(b) else 0


# ── per-session kaggle frames (identical across the 5 models) ──
def kaggle_frames(model_tag="3rd"):
    d = pd.read_csv(KD / f"pred_{model_tag}.csv", usecols=["Id"])
    d["sid"] = d.Id.str.rsplit("_", n=1).str[0]
    d["fr"] = d.Id.str.rsplit("_", n=1).str[1].astype(int)
    return {sid: g.fr.values for sid, g in d.groupby("sid")}


# ── build / load her_sid -> our_sid mapping ──
def build_mapping():
    if MAP_CACHE.exists():
        return json.load(open(MAP_CACHE))
    print("building fingerprint map (one-time)...", flush=True)
    or_fp = {}
    files = [f for f in os.listdir(RAW) if f.endswith(".parquet")]
    for i, fn in enumerate(files):
        sid = fn[:-8]
        d = pd.read_parquet(os.path.join(RAW, fn),
                            columns=["StartHesitation", "Turn", "Walking", "Activity"])
        fog = (d.StartHesitation.values + d.Turn.values + d.Walking.values) > 0
        walk = d.Activity.values == 1
        oi = np.where(walk | fog)[0]
        if len(oi):
            or_fp.setdefault(fp(oi), []).append(sid)
        if i % 800 == 0:
            print(f"  ..{i}/{len(files)}", flush=True)
    kf = kaggle_frames("3rd")
    mapping = {}
    for her_sid, frames in kf.items():
        h = fp(frames)
        cand = or_fp.get(h)
        if cand and len(cand) == 1:
            mapping[her_sid] = cand[0]
    json.dump(mapping, open(MAP_CACHE, "w"))
    print(f"mapped {len(mapping)}/{len(kf)} sessions uniquely", flush=True)
    return mapping


def main():
    mp = build_mapping()
    our_sids = set(mp.values())
    print(f"using {len(mp)} mapped sessions\n", flush=True)

    # reference: our_sid -> raw FOG label + Activity
    ref = {}
    for sid in our_sids:
        d = pd.read_parquet(os.path.join(RAW, sid + ".parquet"),
                            columns=["StartHesitation", "Turn", "Walking", "Activity"])
        fog = ((d.StartHesitation.values + d.Turn.values + d.Walking.values) > 0).astype(np.int8)
        ref[sid] = (fog, d.Activity.values.astype(np.int16))

    # FORGE mc/probe scores -> per our_sid dense array
    fp_df = pd.read_parquet(FORGE_PARQUET, columns=["session_id", "abs_frame", "pred_prob_fog"])
    fp_df["session_id"] = fp_df.session_id.astype(str)
    fp_df = fp_df[fp_df.session_id.isin(our_sids)]
    forge = {}
    for sid, g in fp_df.groupby("session_id", sort=False):
        arr = np.full(g.abs_frame.max() + 1, np.nan, np.float32)
        arr[g.abs_frame.values] = g.pred_prob_fog.values
        forge[sid] = arr

    kf = kaggle_frames("3rd")  # frame index per her_sid

    def score_model(get_scores):
        """get_scores(her_sid) -> (frames, scores) in her index space, or (None,None)."""
        rows = {}
        for filt in ["walk", "OR"]:
            sess = []
            Y, S = [], []
            for her_sid, our in mp.items():
                fog, act = ref[our]
                frames, scores = get_scores(her_sid)
                if frames is None:
                    continue
                ok = frames < len(fog)
                frames, scores = frames[ok], scores[ok]
                yf, af = fog[frames], act[frames]
                m = (af == 1) if filt == "walk" else np.ones(len(frames), bool)
                if m.sum() == 0:
                    continue
                yy, ss = yf[m], scores[m]
                v = ~np.isnan(ss)
                yy, ss = yy[v], ss[v]
                if len(yy) == 0:
                    continue
                Y.append(yy); S.append(ss)
                # for ICC: %TF needs a threshold for binarized pred; kaggle scores are
                # arbitrary-scale, so use per-model median-free rank threshold = top-prev.
                sess.append((our, yy, ss))
            y = np.concatenate(Y); s = np.concatenate(S)
            auc = roc_auc_score(y, s); ap = average_precision_score(y, s); prev = y.mean()
            # %TF ICC: threshold pred at the value reproducing overall predicted-positive
            # rate = ground-truth prevalence (calibration-free, fair across arbitrary scales)
            thr = np.quantile(s, 1 - prev)
            srows = []
            for our, yy, ss in sess:
                srows.append((our.split("__")[0], yy.mean() * 100, (ss >= thr).mean() * 100,
                              episodes(yy), episodes((ss >= thr).astype(int)), len(yy)))
            # patient id: our daily-living sessions -> patient via forge parquet not loaded here;
            # use session-level ICC (subject handled in caller if needed)
            d = pd.DataFrame(srows, columns=["pid", "gt", "pr", "gtep", "prep", "kept"])
            sicc = icc2(d["gt"].values, d["pr"].values)
            fog_icc = icc2(d["gtep"].values.astype(float), d["prep"].values.astype(float))
            rows[filt] = dict(auc=auc, ap=ap, prev=prev, n=len(y), nsess=len(d),
                              tf_sessICC=sicc, fog_sessICC=fog_icc)
        return rows

    # FORGE
    def forge_get(her_sid):
        our = mp[her_sid]; arr = forge.get(our); frames = kf[her_sid]
        if arr is None:
            return None, None
        sc = np.full(len(frames), np.nan, np.float32)
        ok = frames < len(arr)
        sc[ok] = arr[frames[ok]]
        return frames, sc

    results = []
    r = score_model(forge_get)
    for filt in ["walk", "OR"]:
        x = r[filt]
        print(f"FORGE mc/probe [{filt:4s}] AUC={x['auc']:.3f} AP={x['ap']:.3f} "
              f"prev={x['prev']*100:.1f}% TF_ICC={x['tf_sessICC']:.3f} "
              f"FOG_ICC={x['fog_sessICC']:.3f} n={x['n']:,} sess={x['nsess']}", flush=True)
        results.append(["FORGE_mc_probe", filt, x['auc'], x['ap'], x['prev'],
                        x['tf_sessICC'], x['fog_sessICC'], x['nsess']])

    for tag, label in KAGGLE:
        d = pd.read_csv(KD / f"pred_{tag}.csv")
        d["sid"] = d.Id.str.rsplit("_", n=1).str[0]
        d["fr"] = d.Id.str.rsplit("_", n=1).str[1].astype(int)
        ks = {sid: (g.fr.values, g.FOG.values.astype(np.float32)) for sid, g in d.groupby("sid")}

        def kget(her_sid, ks=ks):
            v = ks.get(her_sid)
            return (v[0], v[1]) if v else (None, None)

        r = score_model(kget)
        for filt in ["walk", "OR"]:
            x = r[filt]
            print(f"{label}      [{filt:4s}] AUC={x['auc']:.3f} AP={x['ap']:.3f} "
                  f"prev={x['prev']*100:.1f}% TF_ICC={x['tf_sessICC']:.3f} "
                  f"FOG_ICC={x['fog_sessICC']:.3f} n={x['n']:,} sess={x['nsess']}", flush=True)
            results.append([label, filt, x['auc'], x['ap'], x['prev'],
                            x['tf_sessICC'], x['fog_sessICC'], x['nsess']])

    out = pd.DataFrame(results, columns=["model", "filter", "AUC", "AP", "prev",
                                         "TF_sessICC", "FOG_sessICC", "nsess"])
    out.to_csv(str(private_path("kaggle_pred_vectors", "RESULTS_aligned.csv", output=True)), index=False)
    print("\nsaved RESULTS_aligned.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()

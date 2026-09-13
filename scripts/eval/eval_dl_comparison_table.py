"""
FORGE vs Kaggle-winner daily-living comparison table (walk+stand, Activity ∈ {1,4})
with session-level cluster-bootstrap 95% CIs.

Same basis as RESULTS_full_filters.csv: Salomon-mapped sessions, her FOG label,
calibration-free prevalence-matched threshold (Kaggle scores are arbitrary-scale).
Intersection guard → every model scored on the identical session set.

Metrics per model: normAP, %TF ICC, #FOG ICC, AUC. CIs from B session-resamples
(targets/clusters resampled with replacement; metrics recomputed each draw).

ICC(2,1) is implemented in numpy (validated against pingouin) for bootstrap speed.

Output:
  research/paper_final/data/kaggle_pred_full/RESULTS_table_ci.csv
  research/paper_final/data/kaggle_pred_full/RESULTS_table_ci.md
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import average_precision_score, roc_auc_score
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "eval"))
from _private_inputs import private_path  # author-only inputs; see that module

KD = private_path("kaggle_pred_vectors")
KFULL = private_path("kaggle_pred_full")
CACHE = "logs/comprehensive_eval_cache"
MAP = json.load(open(KD / "_id_map.json"))
REF = pickle.load(open(KD / "_ref_gt_act.pkl", "rb"))
KEEP = [1, 4]
B = 2000
NBINS = 2000          # score-histogram resolution for fast bootstrap AUC/AP
RNG = np.random.default_rng(7)
MODELS = ["FORGE_lc+mc+sc_probe", "FORGE_mc_probe",
          "Kaggle1", "Kaggle2", "Kaggle3", "Kaggle4", "Kaggle5"]
FORGE_CFG = {"FORGE_lc+mc+sc_probe": "lc+mc+sc_probe", "FORGE_mc_probe": "mc_probe"}


def icc2_np(gt, pr):
    """ICC(2,1) absolute-agreement, two-way random, single measurement (numpy)."""
    n = len(gt)
    if n < 3:
        return np.nan
    X = np.column_stack([gt, pr]).astype(float)
    k = 2
    grand = X.mean()
    row = X.mean(1); col = X.mean(0)
    sst = ((X - grand) ** 2).sum()
    ssr = k * ((row - grand) ** 2).sum()
    ssc = n * ((col - grand) ** 2).sum()
    sse = sst - ssr - ssc
    if (n - 1) * (k - 1) == 0:
        return np.nan
    msr = ssr / (n - 1); msc = ssc / (k - 1); mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return np.nan if abs(denom) < 1e-12 else (msr - mse) / denom


GAP = 60  # 0.6 s @ 100 Hz — gap-merged episode definition (Salomon / Yang / fig10d)


def n_episodes(binary, gap=GAP):
    """Gap-merged FOG episode count for one session (gaps < gap frames merged)."""
    spans = 0; start = None; g = 0
    for v in binary:
        if v:
            start = True; g = 0
        elif start is not None:
            g += 1
            if g > gap:
                spans += 1; start = None
    if start is not None:
        spans += 1
    return spans


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


def kaggle_scores(n):
    return pickle.load(open(KFULL / f"pred{n}.csv.scores.pkl", "rb"))


def session_keys(scores):
    """Sids with ≥1 walk+stand frame carrying a valid score (cheap; for intersection)."""
    out = set()
    for our, (fog, act) in REF.items():
        arr = scores.get(our)
        if arr is None:
            continue
        m = min(len(fog), len(arr))
        if (np.isin(act[:m], KEEP) & ~np.isnan(arr[:m])).any():
            out.add(our)
    return out


def build_summary(scores, sids):
    """Per-session summaries aligned to `sids`, threshold fixed at the full-sample
    quantile (reproduces the point estimate). AUC/AP from {1,4}-frame score histograms;
    %TF over {1,4} frames; #FOG gap-merged on the FULL activity-masked sequence (so
    bouts split by non-ambulatory gaps are not spuriously merged)."""
    # pooled kept scores → threshold + bin grid
    kept_s, kept_y = [], []
    for sid in sids:
        fog, act = REF[sid]; arr = scores[sid]
        m = min(len(fog), len(arr))
        keep = np.isin(act[:m], KEEP) & ~np.isnan(arr[:m])
        kept_s.append(arr[:m][keep]); kept_y.append(fog[:m][keep])
    s_all = np.concatenate(kept_s); y_all = np.concatenate(kept_y)
    prev = y_all.mean()
    thr = float(np.quantile(s_all, 1 - prev))
    edges = np.linspace(s_all.min(), s_all.max(), NBINS + 1)

    n = len(sids)
    gt_tf = np.empty(n); pr_tf = np.empty(n)
    gt_ep = np.empty(n); pr_ep = np.empty(n)
    posh = np.zeros((n, NBINS)); negh = np.zeros((n, NBINS))
    for i, sid in enumerate(sids):
        fog, act = REF[sid]; arr = scores[sid]
        m = min(len(fog), len(arr))
        fog, act, sc = fog[:m], act[:m], arr[:m]
        kmask = np.isin(act, KEEP)
        keep = kmask & ~np.isnan(sc)
        yk, sk = fog[keep].astype(np.int8), sc[keep]
        pbk = (sk >= thr).astype(np.int8)
        gt_tf[i] = yk.mean() * 100; pr_tf[i] = pbk.mean() * 100
        bi = np.clip(np.searchsorted(edges, sk, side="right") - 1, 0, NBINS - 1)
        posh[i] = np.bincount(bi[yk == 1], minlength=NBINS)
        negh[i] = np.bincount(bi[yk == 0], minlength=NBINS)
        # full-length activity-masked sequences for gap-merged episode counts
        lab_full = np.where(kmask, fog, 0).astype(np.int8)
        pred_full = np.where(kmask & ~np.isnan(sc) & (sc >= thr), 1, 0).astype(np.int8)
        gt_ep[i] = n_episodes(lab_full); pr_ep[i] = n_episodes(pred_full)
    # exact point estimates (CIs come from the histogram bootstrap)
    pt = dict(auc=roc_auc_score(y_all, s_all),
              normAP=(average_precision_score(y_all, s_all) - prev) / (1 - prev),
              tf_icc=icc2_np(gt_tf, pr_tf), fog_icc=icc2_np(gt_ep, pr_ep))
    return dict(gt_tf=gt_tf, pr_tf=pr_tf, gt_ep=gt_ep, pr_ep=pr_ep,
                posh=posh, negh=negh, prev=prev, pt=pt)


def metrics_from_summary(S, sel):
    """Fast metrics for a bootstrap selection `sel` (session indices) from summary S.
    Uses per-session selection counts: histograms via matmul, ICC arrays via repeat."""
    n = S["posh"].shape[0]
    cnt = np.bincount(sel, minlength=n).astype(float)
    posh = cnt @ S["posh"]; negh = cnt @ S["negh"]
    npos, nneg = posh.sum(), negh.sum()
    if npos == 0 or nneg == 0:
        return dict(normAP=np.nan, auc=np.nan, tf_icc=np.nan, fog_icc=np.nan)
    neg_below = np.concatenate([[0], np.cumsum(negh)[:-1]])
    auc = (posh * (neg_below + 0.5 * negh)).sum() / (npos * nneg)
    tp = np.cumsum(posh[::-1]); fp = np.cumsum(negh[::-1])
    prec = tp / np.maximum(tp + fp, 1); rec = tp / npos
    ap = (np.diff(np.concatenate([[0], rec])) * prec).sum()
    prev = npos / (npos + nneg)
    icnt = cnt.astype(int)
    return dict(normAP=(ap - prev) / (1 - prev), auc=auc,
                tf_icc=icc2_np(np.repeat(S["gt_tf"], icnt), np.repeat(S["pr_tf"], icnt)),
                fog_icc=icc2_np(np.repeat(S["gt_ep"], icnt), np.repeat(S["pr_ep"], icnt)))


def main():
    keys = ["normAP", "tf_icc", "fog_icc", "auc"]
    # 1) session keys per model (cheap) → intersection
    sckeys = {}
    sc_cache = {}
    for m in MODELS:
        s = forge_scores(FORGE_CFG[m]) if m in FORGE_CFG else kaggle_scores(int(m[-1]))
        sc_cache[m] = s
        sckeys[m] = session_keys(s)
    common = set(REF)
    for m in MODELS:
        common &= sckeys[m]
    common = sorted(common)
    n = len(common)
    print(f"common walk+stand sessions: {n}", flush=True)

    # shared session-level bootstrap selections (paired across all models)
    boot = [RNG.integers(0, n, n) for _ in range(B)]
    rows, all_draws = [], {}
    for m in MODELS:
        S = build_summary(sc_cache[m], common)
        del sc_cache[m]
        fast = metrics_from_summary(S, np.arange(n))   # histogram self-check
        print(f"{m:22s} [check] AUC exact={S['pt']['auc']:.4f} hist={fast['auc']:.4f} | "
              f"normAP {S['pt']['normAP']:.4f}/{fast['normAP']:.4f}", flush=True)
        draws = {k: np.empty(B) for k in keys}
        for b, sel in enumerate(boot):
            r = metrics_from_summary(S, sel)
            for k in keys:
                draws[k][b] = r[k]
        all_draws[m] = draws
        row = {"model": m, "nsess": n}
        for k in keys:
            arr = draws[k][~np.isnan(draws[k])]
            lo, hi = np.percentile(arr, [2.5, 97.5])
            row[k] = S["pt"][k]; row[f"{k}_lo"] = lo; row[f"{k}_hi"] = hi
        rows.append(row)
        print(f"{m:22s} normAP={row['normAP']:.3f} %TF={row['tf_icc']:.3f} "
              f"#FOG={row['fog_icc']:.3f} [{row['fog_icc_lo']:.3f},{row['fog_icc_hi']:.3f}] "
              f"AUC={row['auc']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(KFULL / "RESULTS_table_ci.csv", index=False)

    # paired difference CIs: FORGE lc+mc+sc/probe − the STRONGEST Kaggle winner on
    # each metric (point estimate), using shared bootstrap draws (two-sided p = 2·min tail).
    ours = "FORGE_lc+mc+sc_probe"
    kaggle = [m for m in MODELS if m.startswith("Kaggle")]
    ptmap = {r["model"]: r for r in rows}
    paired = []
    for k in keys:
        best = max(kaggle, key=lambda m: ptmap[m][k])   # toughest competitor for this metric
        d = all_draws[ours][k] - all_draws[best][k]
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        paired.append((k, best, d.mean(), lo, hi, p))

    # markdown
    def cell(r, k):
        return f"{r[k]:.3f} [{r[f'{k}_lo']:.3f}, {r[f'{k}_hi']:.3f}]"
    disp = {"FORGE_lc+mc+sc_probe": "**FORGE lc+mc+sc/probe**", "FORGE_mc_probe": "FORGE mc/probe",
            "Kaggle1": "Kaggle1", "Kaggle2": "Kaggle2", "Kaggle3": "Kaggle3 (best winner)",
            "Kaggle4": "Kaggle4", "Kaggle5": "Kaggle5"}
    lab = {"normAP": "normAP", "tf_icc": "%TF ICC", "fog_icc": "#FOG ICC", "auc": "AUC"}
    lines = [
        f"# DailyLiving walk+stand (Activity ∈ {{1,4}}) — FORGE vs Kaggle winners",
        f"\nn={n} sessions · {B}× session-level cluster bootstrap (95% CI) · gap-merged "
        f"episodes (0.6 s) · prevalence-matched threshold · Salomon FOG label.\n",
        "| Model | normAP | %TF ICC | #FOG ICC | AUC |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {disp[r['model']]} | {cell(r,'normAP')} | {cell(r,'tf_icc')} | "
                     f"{cell(r,'fog_icc')} | {cell(r,'auc')} |")
    lines += ["\n**Paired difference — FORGE lc+mc+sc/probe − strongest Kaggle winner "
              "on each metric (shared bootstrap, two-sided p):**\n",
              "| Metric | Toughest competitor | Δ | 95% CI | p |", "|---|---|---|---|---|"]
    for k, best, mean, lo, hi, p in paired:
        lines.append(f"| {lab[k]} | {disp[best].replace(' (best winner)','')} | {mean:+.3f} "
                     f"| [{lo:+.3f}, {hi:+.3f}] | {p:.3g} |")
    (KFULL / "RESULTS_table_ci.md").write_text("\n".join(lines))
    print("\npaired Δ vs strongest Kaggle per metric:")
    for k, best, mean, lo, hi, p in paired:
        print(f"  {lab[k]:9s} vs {best:8s} Δ={mean:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3g}")
    print("\nwrote RESULTS_table_ci.csv + .md")


if __name__ == "__main__":
    main()

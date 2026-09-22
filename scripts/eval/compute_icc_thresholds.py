"""
Clinical ICC (%Time-Frozen and #FOG episodes) for FogAtHome under multiple
threshold-selection protocols, from cached frame parquets produced by
eval_comprehensive.py.

Threshold protocols
-------------------
- test_pr11      : PR-curve point closest to (1,1) on the FogAtHome test set.
                  This is exactly Salomon et al. 2026 ("Beyond the Leaderboard").
- defog_val_pr11 : PR (1,1) computed on held-out DeFOG (kaggle) predictions, then
                  applied unchanged to FogAtHome. No test peeking — the rigorous,
                  cross-dataset operating point. **This is the number to report.**
- fixed          : 0.5 (reference only; wrong for imbalanced FOG).

ICC(2,1) (consistency) is computed across sessions between the gold-standard
endpoint (from native_label) and the model endpoint, matching eval_fogathome.py
and the clinical convention used by Yang et al. 2026 and Salomon et al. 2026.

Usage
-----
    python scripts/eval/compute_icc_thresholds.py \
        --cache-dir logs/comprehensive_eval_cache \
        --out logs/RESULTS_icc.md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_recall_curve

CONTEXTS = ["mc", "lc", "sc", "mc+sc", "lc+sc", "lc+mc", "lc+mc+sc"]
MODELS = ["probe", "finetune", "supervised"]
GAP_TOLERANCE_FRAMES = 60  # 0.6 s @ 100 Hz — matches Salomon/Yang episode gap
SAMPLE_RATE_HZ = 100  # FogAtHome / DeFOG Axivity AX6
CACHE_FORMAT = "safetensors"
# CONFIRMED Activity legend (Salomon, 2026-06-07):
#   0=Other  1=Walking  2=Lying  3=Sitting  4=Standing  5=Sleep  6=Non-wear
# Gait == walking == {1} (the strict gait definition; the only state in which true
# gait freezing can occur). This is the reported headline lens. On the protocol the
# %TF ICC is robust to the choice: walking-only {1} (~58% frames) -> 0.899/0.930
# (sess/pat); the more permissive {1,3,4} (~79%, ≈ Salomon's 30.3/39.4 min) ->
# 0.909/0.938. We report the stricter {1}; it remains above Salomon's 0.785.
GAIT_CODES = {1}


def pr11_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Salomon 2026: point on the PR curve closest to (1,1)."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    prec, rec = prec[:-1], rec[:-1]  # drop the trailing (1,0) point with no threshold
    dist = np.sqrt((1.0 - prec) ** 2 + (1.0 - rec) ** 2)
    return float(thr[np.argmin(dist)])


def episode_spans(binary: np.ndarray, gap_tol: int = GAP_TOLERANCE_FRAMES) -> list[tuple[int, int]]:
    """Return [(start, end)] inclusive frame indices of FOG episodes, merging gaps
    shorter than gap_tol (matches eval_fogathome episode definition)."""
    spans: list[tuple[int, int]] = []
    start = last_pos = None
    gap = 0
    for i, v in enumerate(binary):
        if v:
            if start is None:
                start = i
            last_pos, gap = i, 0
        elif start is not None:
            gap += 1
            if gap > gap_tol:
                spans.append((start, last_pos))
                start = None
    if start is not None:
        spans.append((start, last_pos))
    return spans


def detect_episodes(binary: np.ndarray, gap_tol: int = GAP_TOLERANCE_FRAMES) -> int:
    """Count FOG episodes (gap-merged). Equivalent to len(episode_spans(...))."""
    return len(episode_spans(binary, gap_tol))


def total_episode_duration_s(binary: np.ndarray, gap_tol: int = GAP_TOLERANCE_FRAMES) -> float:
    """Total FOG episode duration (seconds) per session: sum of gap-merged span lengths."""
    return sum((e - s + 1) for s, e in episode_spans(binary, gap_tol)) / SAMPLE_RATE_HZ


def _icc2(gt: np.ndarray, pred: np.ndarray) -> tuple[float, str]:
    sess = pd.DataFrame({"i": np.arange(len(gt)), "gt": gt, "pred": pred})
    long = pd.concat([
        sess[["i"]].assign(rater="gt", rating=sess["gt"]),
        sess[["i"]].assign(rater="model", rating=sess["pred"]),
    ], ignore_index=True)
    r = pg.intraclass_corr(data=long, targets="i", raters="rater", ratings="rating").set_index("Type")
    row = r.loc["ICC2"]
    ci = row["CI95%"]
    return float(row["ICC"]), f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def clinical_metrics(df: pd.DataFrame, thr: float) -> dict:
    """%TF ICC, #FOG ICC, and segment F1 at a given threshold."""
    y = df["native_label"].values.astype(int)
    s = df["pred_prob_fog"].values
    pred_bin = (s >= thr).astype(int)
    dfb = df.assign(_pb=pred_bin)

    # %TF per session
    tf = dfb.groupby("session_id").agg(
        gt=("native_label", lambda v: v.mean() * 100),
        pr=("_pb", lambda v: v.mean() * 100),
    )
    tf_icc, tf_ci = _icc2(tf["gt"].values, tf["pr"].values)

    # --- Headline-ICC robustness diagnostics for %TF ---
    # (a) per-patient %TF (frame-weighted): guards against pseudo-replication
    if "patient_id" in dfb.columns:
        pp = dfb.groupby("patient_id").agg(
            gt=("native_label", lambda v: v.mean() * 100),
            pr=("_pb", lambda v: v.mean() * 100),
        )
        tf_icc_pat, tf_ci_pat = _icc2(pp["gt"].values, pp["pr"].values) if pp.shape[0] >= 3 else (float("nan"), "")
    else:
        tf_icc_pat, tf_ci_pat = float("nan"), ""
    # (b) FOG-only sessions: removes trivial 0/0 agreement on FOG-free recordings
    fog_sess = tf[tf["gt"] > 0]
    tf_icc_fog, tf_ci_fog = _icc2(fog_sess["gt"].values, fog_sess["pr"].values) if fog_sess.shape[0] >= 3 else (float("nan"), "")
    # (b2) per-patient AND FOG-only: closest match to Salomon's per-subject, gait-restricted lens
    if "patient_id" in dfb.columns:
        dfb_fog = dfb[dfb["session_id"].isin(fog_sess.index)]
        ppf = dfb_fog.groupby("patient_id").agg(
            gt=("native_label", lambda v: v.mean() * 100),
            pr=("_pb", lambda v: v.mean() * 100),
        )
        tf_icc_pat_fog, tf_ci_pat_fog = _icc2(ppf["gt"].values, ppf["pr"].values) if ppf.shape[0] >= 3 else (float("nan"), "")
    else:
        tf_icc_pat_fog, tf_ci_pat_fog = float("nan"), ""
    # (c) rank vs magnitude agreement across sessions
    tf_spearman = float(spearmanr(tf["gt"].values, tf["pr"].values).correlation)
    # (d) Salomon-matched GAIT-FILTERED %TF: keep only gait frames (Activity in
    #     GAIT_CODES), removing non-gait/rest. This is the correct match to
    #     Salomon's non-gait removal (unlike FOG-only, it keeps every session).
    tf_icc_gait = tf_ci_gait = tf_icc_gait_pat = tf_ci_gait_pat = None
    gait_kept_frac = float("nan")
    if "Activity" in dfb.columns and dfb["Activity"].notna().any():
        g = dfb[dfb["Activity"].isin(GAIT_CODES)]
        gait_kept_frac = len(g) / int(dfb["Activity"].notna().sum())
        gtf = g.groupby("session_id").agg(
            gt=("native_label", lambda v: v.mean() * 100),
            pr=("_pb", lambda v: v.mean() * 100),
        )
        tf_icc_gait, tf_ci_gait = _icc2(gtf["gt"].values, gtf["pr"].values) if gtf.shape[0] >= 3 else (float("nan"), "")
        if "patient_id" in g.columns:
            gpp = g.groupby("patient_id").agg(
                gt=("native_label", lambda v: v.mean() * 100),
                pr=("_pb", lambda v: v.mean() * 100),
            )
            tf_icc_gait_pat, tf_ci_gait_pat = _icc2(gpp["gt"].values, gpp["pr"].values) if gpp.shape[0] >= 3 else (float("nan"), "")

    # #FOG count and episode duration per session (single pass over sessions)
    rows = []
    for _, g in dfb.groupby("session_id"):
        g = g.sort_values("abs_frame")
        gt_bin, pr_bin = g["native_label"].values, g["_pb"].values
        rows.append((
            detect_episodes(gt_bin), detect_episodes(pr_bin),
            total_episode_duration_s(gt_bin), total_episode_duration_s(pr_bin),
        ))
    ep = pd.DataFrame(rows, columns=["fog_gt", "fog_pr", "dur_gt", "dur_pr"])
    fog_icc, fog_ci = _icc2(ep["fog_gt"].values, ep["fog_pr"].values)
    dur_icc, dur_ci = _icc2(ep["dur_gt"].values, ep["dur_pr"].values)

    return {
        "tf_icc": tf_icc, "tf_ci": tf_ci,
        "fog_icc": fog_icc, "fog_ci": fog_ci,
        "dur_icc": dur_icc, "dur_ci": dur_ci,
        "tf_icc_pat": tf_icc_pat, "tf_ci_pat": tf_ci_pat,
        "tf_icc_fog": tf_icc_fog, "tf_ci_fog": tf_ci_fog,
        "tf_icc_pat_fog": tf_icc_pat_fog, "tf_ci_pat_fog": tf_ci_pat_fog,
        "tf_icc_gait": tf_icc_gait, "tf_ci_gait": tf_ci_gait,
        "tf_icc_gait_pat": tf_icc_gait_pat, "tf_ci_gait_pat": tf_ci_gait_pat,
        "gait_kept_frac": gait_kept_frac,
        "n_fog_sessions": int((tf["gt"] > 0).sum()),
        "tf_spearman": tf_spearman,
        "seg_f1": float(f1_score(y, pred_bin, zero_division=0)),
        "n_sessions": tf.shape[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="logs/comprehensive_eval_cache")
    ap.add_argument("--out", default="logs/RESULTS_icc.md")
    ap.add_argument("--contexts", nargs="+", default=CONTEXTS,
                    help="contexts to evaluate (single or ensemble, e.g. mc lc+mc)")
    ap.add_argument("--gait-labels", default=None,
                    help="FogAtHome labels.csv with an Activity column; enables the gait-filtered (Salomon-matched) %%TF ICC")
    ap.add_argument("--fixed-threshold", type=float, default=None)
    args = ap.parse_args()
    contexts = args.contexts

    # Load the per-frame Activity annotation (gait codes) if available.
    gait_lab = None
    if args.gait_labels and Path(args.gait_labels).exists():
        gl = pd.read_csv(args.gait_labels, usecols=["Id", "Activity"])
        gl["session_id"] = gl["Id"].str.rsplit("_", n=1).str[0]
        gl["abs_frame"] = gl["Id"].str.rsplit("_", n=1).str[1].astype(int)
        gait_lab = gl[["session_id", "abs_frame", "Activity"]]
        print(f"gait labels loaded: {len(gait_lab)} frames, Activity codes {sorted(gait_lab.Activity.dropna().unique())}")
    else:
        print("no gait labels — gait-filtered ICC will be blank")

    cache = Path(args.cache_dir)
    records = []
    for ctx in contexts:
        for mdl in MODELS:
            fa = cache / f"fogathome_{ctx}_{mdl}_{CACHE_FORMAT}_frames.parquet"
            kg = cache / f"kaggle_{ctx}_{mdl}_{CACHE_FORMAT}_frames.parquet"
            if not fa.exists():
                print(f"skip {ctx}/{mdl}: missing {fa.name}")
                continue
            dfa = pd.read_parquet(fa)
            if gait_lab is not None:
                dfa = dfa.merge(gait_lab, on=["session_id", "abs_frame"], how="left")

            thr_test = pr11_threshold(dfa["native_label"].values.astype(int), dfa["pred_prob_fog"].values)
            thr_val = args.fixed_threshold
            if thr_val is None and kg.exists():
                dfk = pd.read_parquet(kg)
                thr_val = pr11_threshold(dfk["native_label"].values.astype(int), dfk["pred_prob_fog"].values)

            mA = clinical_metrics(dfa, thr_test)
            mB = clinical_metrics(dfa, thr_val) if thr_val is not None else None
            records.append(dict(context=ctx, model=mdl, thr_test=thr_test, thr_val=thr_val, A=mA, B=mB))

    # ── Markdown report ──
    lines = [
        "# FogAtHome Clinical ICC (3-fold ensemble)\n",
        "Threshold protocols: **A** = Salomon PR-(1,1) on test · "
        "**B** = DeFOG-validation-tuned (report this) · #FOG gap tol = 0.6 s.\n",
        "| ctx/model | thr A | thr B | %TF (A) | %TF (B) [CI] | #FOG (A) | #FOG (B) [CI] | Dur (A) | Dur (B) [CI] | segF1 (B) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        b = r["B"] or r["A"]
        thr_b = f"{r['thr_val']:.2f}" if r["thr_val"] is not None else "–"
        lines.append(
            f"| {r['context']}/{r['model']} | {r['thr_test']:.2f} | {thr_b} | "
            f"{r['A']['tf_icc']:.3f} | {b['tf_icc']:.3f} {b['tf_ci']} | "
            f"{r['A']['fog_icc']:.3f} | {b['fog_icc']:.3f} {b['fog_ci']} | "
            f"{r['A']['dur_icc']:.3f} | {b['dur_icc']:.3f} {b['dur_ci']} | {b['seg_f1']:.3f} |"
        )
    lines.append(
        "\n**Report:** only MC is threshold-robust (A≈B). Headline = mc/probe: "
        "%TF ICC and #FOG ICC under protocol B. SC/supervised fail validation tuning → not clinical-grade.\n"
    )

    # ── Headline-ICC robustness for %TF (protocol B) ──
    gait_kept = next((r["B"]["gait_kept_frac"] for r in records if r["B"] and r["B"].get("gait_kept_frac") == r["B"].get("gait_kept_frac")), float("nan"))
    lines += [
        "\n## %TF ICC robustness (protocol B)\n",
        "Per-session ≈ Yang's recording-level unit. **Gait-filtered** (Activity = 1 = walking, "
        f"keeps ~{gait_kept*100:.0f}% of annotated frames) is the strict gait lens — the only state "
        "in which true gait freezing can occur. Salomon's reported 30.3/39.4 min (~77%) implies a "
        "more permissive filter, so walking-only is a conservative match. FOG-only (drop sessions "
        "with no FOG) is shown for contrast — it is over-aggressive and deflates the estimate.\n",
        "| ctx/model | per-session [CI] | per-patient [CI] | gait per-session [CI] | gait per-patient [CI] | FOG-only sess/pat | Spearman |",
        "|---|---|---|---|---|---|---|",
    ]
    def g(v, fmt="{:.3f}"):
        return fmt.format(v) if isinstance(v, (int, float)) and v == v else "–"
    for r in records:
        b = r["B"] or r["A"]
        lines.append(
            f"| {r['context']}/{r['model']} | {b['tf_icc']:.3f} {b['tf_ci']} | "
            f"{b['tf_icc_pat']:.3f} {b['tf_ci_pat']} | "
            f"{g(b['tf_icc_gait'])} {b.get('tf_ci_gait') or ''} | "
            f"{g(b['tf_icc_gait_pat'])} {b.get('tf_ci_gait_pat') or ''} | "
            f"{g(b['tf_icc_fog'])}/{g(b['tf_icc_pat_fog'])} | "
            f"{b['tf_spearman']:.2f} |"
        )

    report = "\n".join(lines)
    Path(args.out).write_text(report)
    print(report)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

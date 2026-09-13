"""
Daily-living FORGE-vs-Kaggle figure (2 panels):
  A) {1,4} (walk+stand, label-independent) grouped bars on imbalance-appropriate
     metrics (normAP, %TF ICC, #FOG ICC) for all 6 models. FORGE highlighted,
     bootstrap 95% CI on the FORGE-vs-Kaggle3 leaders.
  B) Filter robustness: %TF ICC of FORGE vs Kaggle3 across OR -> {1} -> {1,4}.
     Kaggle3 collapses as standing negatives are admitted; FORGE holds.

Data: research/paper_final/data/kaggle_pred_full/RESULTS_full_filters.csv
PNG only.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "eval"))
from _private_inputs import private_path  # author-only inputs; see that module

R = pd.read_csv(str(private_path("kaggle_pred_full", "RESULTS_full_filters.csv")))
# Canonical {1,4} numbers (gap-merged 0.6 s episodes, session bootstrap) for panel A.
TCI = pd.read_csv(str(private_path("kaggle_pred_full", "RESULTS_table_ci.csv")))
TCI_COL = {"normAP": "normAP", "TF_ICC": "tf_icc", "FOG_ICC": "fog_icc"}
FIGDIR = Path("research/paper_final/figures"); FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.facecolor": "white",
})
# Unified BLUES aesthetic ------------------------------------------------------
FORGE_C  = "#08306b"   # ours / primary navy
FORGE2_C = "#2171b5"   # strong blue (2nd FORGE series)
KAG3_C   = "#4292c6"   # Kaggle 3rd (strong baseline) fill
REF_RED  = "#d62728"   # reference / chance / threshold lines + K3 highlight outline
# graded blue ramp for the 5 Kaggle baselines (3rd -> 2nd by rank order below)
KAG_RAMP = {"Kaggle3": "#4292c6", "Kaggle4": "#6baed6", "Kaggle5": "#9ecae1",
            "Kaggle1": "#c6dbef", "Kaggle2": "#deebf7"}

# Paired Δ (FORGE lc+mc+sc − toughest Kaggle competitor for that metric) on {1,4},
# gap-merged episodes, shared session bootstrap (from eval_dl_comparison_table.py).
# Competitor differs by metric: detection leader = Kaggle3, #FOG leader = Kaggle5.
DELTA_CI = {  # metric -> (competitor, Δ, lo, hi)
    "TF_ICC":  ("Kaggle3", 0.138, 0.053, 0.222),
    "FOG_ICC": ("Kaggle5", 0.185, 0.088, 0.297),
}

METRICS = [("normAP", "normalized AP"), ("TF_ICC", "%TF ICC"), ("FOG_ICC", "#FOG ICC")]
FORGE_BEST = "FORGE_lc+mc+sc_probe"
MODELS = [FORGE_BEST, "FORGE_mc_probe", "Kaggle3", "Kaggle4", "Kaggle5", "Kaggle1", "Kaggle2"]
LABELS = {FORGE_BEST: "FORGE\n(lc+mc+sc)", "FORGE_mc_probe": "FORGE\n(mc)",
          "Kaggle3": "Kaggle 3rd", "Kaggle4": "Kaggle 4th", "Kaggle5": "Kaggle 5th",
          "Kaggle1": "Kaggle 1st", "Kaggle2": "Kaggle 2nd"}


def val(model, filt, metric):
    r = R[(R.model == model) & (R["filter"] == filt)]
    return float(r[metric].iloc[0]) if len(r) else np.nan


def val14(model, metric):
    """Canonical {1,4} value (gap-merged) from the bootstrap table."""
    r = TCI[TCI.model == model]
    return float(r[TCI_COL[metric]].iloc[0]) if len(r) else np.nan


fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.7, 1]})

# ── Panel A: {1,4} grouped bars ──
nM, nMet = len(MODELS), len(METRICS)
x = np.arange(nMet); w = 0.115
for i, m in enumerate(MODELS):
    vals = [val14(m, met) for met, _ in METRICS]
    if m == FORGE_BEST:
        col = FORGE_C
    elif m == "FORGE_mc_probe":
        col = FORGE2_C
    else:
        col = KAG_RAMP[m]
    off = (i - (nM - 1) / 2) * w
    # highlight the per-metric toughest competitors (Kaggle3 detection, Kaggle5 #FOG)
    ec = REF_RED if m in ("Kaggle3", "Kaggle5") else "white"
    lw = 1.0 if m in ("Kaggle3", "Kaggle5") else 0.4
    bars = axA.bar(x + off, vals, w, label=LABELS[m], color=col,
                   edgecolor=ec, linewidth=lw, zorder=3)
    # CI whisker on FORGE leader for the two significant ICC metrics, drawn around
    # the FORGE value with the paired Δ-CI vs that metric's toughest competitor.
    if m == FORGE_BEST:
        for j, (met, _) in enumerate(METRICS):
            if met in DELTA_CI:
                comp, dlt, lo, hi = DELTA_CI[met]
                base = val14(comp, met)
                axA.errorbar(x[j] + off, vals[j],
                             yerr=[[vals[j] - (base + lo)], [(base + hi) - vals[j]]],
                             fmt="none", ecolor="0.2", elinewidth=1.3, capsize=3, zorder=4)
axA.set_xticks(x); axA.set_xticklabels([lab for _, lab in METRICS])
axA.set_ylabel("score")
axA.set_title("A.  {walk, stand} filter")
axA.legend(ncol=2, fontsize=7.5, loc="upper left", columnspacing=1.0, handlelength=1.2)
axA.set_ylim(0, max(0.85, axA.get_ylim()[1]))

# ── Panel B: filter robustness (%TF ICC across filters) ──
order = ["OR", "{1}", "{1,4}"]
order_lab = ["OR\n(walk∪FOG)", "{walk}", "{walk,stand}"]
for m, col, lab, mk, mec in [("FORGE_mc_probe", FORGE_C, "FORGE (mc) — robust", "o", "none"),
                             ("Kaggle3", KAG3_C, "Kaggle 3rd — fragile", "s", REF_RED)]:
    ys = [val(m, f, "TF_ICC") for f in order]
    axB.plot(range(len(order)), ys, marker=mk, color=col, lw=2.2, ms=8, label=lab,
             markeredgecolor=mec, markeredgewidth=1.0, zorder=3)
axB.axvspan(1.5, 2.0, color="#deebf7", alpha=0.7, zorder=0)
axB.set_xticks(range(len(order))); axB.set_xticklabels(order_lab)
axB.set_ylabel("%TF ICC"); axB.set_ylim(0, 0.55)
axB.set_title("B.  Filter robustness (%TF ICC)")
axB.legend(loc="upper center", fontsize=8.5)
axB.annotate("standing negatives\nadmitted →", xy=(1.75, 0.05), fontsize=7.5,
             ha="center", color="0.4")
axB.text(2.0, val("Kaggle3", "{1,4}", "TF_ICC") - 0.02, "Kaggle\ncollapses",
         fontsize=8, color=REF_RED, ha="right", va="top")

fig.suptitle("FogAtHome free-living — FORGE vs Kaggle winners (frame-aligned, 1,627 recordings)",
             fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = FIGDIR / "fig14_daily_living_filter_robustness.png"
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(f"saved {out}")

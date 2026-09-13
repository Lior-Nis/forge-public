"""
Fig S4 (2-panel) — deployment label efficiency: FORGE probe vs best-tuned
supervised-from-scratch across labeled-patient budgets.
  (A) FogAtHome frame-level AP (discrimination), best LR per budget.
  (B) FogAtHome per-session %TF ICC (clinical agreement) at fixed thr=0.26.
Random arm omitted (mechanism control, discussed in text). 3-fold ensemble;
line = mean over 3 subset seeds, band = seed min-max.

Sources:
  AP : logs/RESULTS_scratch_lr_budgets.csv (k 2-16) + logs/RESULTS_scratch_lr_sweep.csv (k=all)
  ICC: logs/RESULTS_label_efficiency_icc.csv
Output: research/paper_final/figures/figS4_label_efficiency.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False, "figure.facecolor": "white",
})

OUT = "research/paper_final/figures/figS4_label_efficiency.png"
N_ALL = 48
PROBE_C, SCR_C = "#08306b", "#2171b5"   # blues: ours = navy, scratch = strong blue
REF_C = "#d62728"                         # red accent — reference lines only
PROBE_L = "FORGE probe (frozen pretrained encoder)"
SCR_L = "Supervised from scratch (best-tuned LR/budget)"


def ap_curves():
    df = pd.read_csv("logs/RESULTS_scratch_lr_budgets.csv")
    budgets = sorted(df["k"].unique())
    sc = {"mean": {}, "lo": {}, "hi": {}}; pr = {}
    for k in budgets:
        s = df[(df.arm == "scratch") & (df.k == k)]
        best = s.groupby("lr")["seg_ap"].mean().idxmax()
        g = s[s.lr == best]["seg_ap"]
        sc["mean"][k], sc["lo"][k], sc["hi"][k] = g.mean(), g.min(), g.max()
        pr[k] = df[(df.arm == "probe") & (df.k == k)]["seg_ap"].mean()
    full = pd.read_csv("logs/RESULTS_scratch_lr_sweep.csv")
    sc_full = full[full.case.str.startswith("scratch")]["seg_ap"].max()
    pr_full = full[full.case.str.contains("probe")]["seg_ap"].iloc[0]
    ks = budgets + [N_ALL]
    return (ks,
            [pr[k] for k in budgets] + [pr_full],
            [sc["mean"][k] for k in budgets] + [sc_full],
            budgets, [sc["lo"][k] for k in budgets], [sc["hi"][k] for k in budgets])


def icc_curves():
    df = pd.read_csv("logs/RESULTS_label_efficiency_icc.csv")
    ks = sorted(df["k"].unique())
    out = {}
    for arm in ["probe", "scratch"]:
        g = df[df.arm == arm].groupby("k")["tf_icc"].agg(["mean", "min", "max"]).reindex(ks)
        out[arm] = g
    return ks, out


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5))

    # Panel A — AP
    ks, pr_m, sc_m, bud, sc_lo, sc_hi = ap_curves()
    axA.plot(ks, pr_m, "o-", color=PROBE_C, lw=2.2, ms=7, label=PROBE_L)
    axA.fill_between(bud, sc_lo, sc_hi, color=SCR_C, alpha=0.12)
    axA.plot(ks, sc_m, "s--", color=SCR_C, lw=2.2, ms=7, label=SCR_L)
    axA.axhline(0.31, ls=":", color=REF_C, lw=1, label="random baseline (prev 0.31)")
    axA.set_ylabel("FogAtHome frame-level AP")
    axA.set_title("(A) Discrimination (AP)")

    # Panel B — ICC
    ks2, icc = icc_curves()
    for arm, c, m, lab in [("probe", PROBE_C, "o-", PROBE_L), ("scratch", SCR_C, "s--", SCR_L)]:
        g = icc[arm]
        axB.plot(g.index, g["mean"], m, color=c, lw=2.2, ms=7, label=lab)
        axB.fill_between(g.index, g["min"], g["max"], color=c, alpha=0.12)
    axB.axhline(0.75, ls=":", color=REF_C, lw=1, label='"good agreement" (0.75)')
    axB.axhline(0.0, ls="-", color="grey", lw=0.5)
    axB.set_ylabel("FogAtHome %TF ICC (per session, thr=0.26)")
    axB.set_title("(B) Clinical agreement (ICC)")

    for ax, allks in ((axA, ks), (axB, ks2)):
        ax.set_xscale("log", base=2)
        ax.set_xticks(allks)
        ax.set_xticklabels([("all" if k == max(allks) else str(k)) for k in allks])
        ax.set_xlabel("Number of labeled DeFOG patients")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

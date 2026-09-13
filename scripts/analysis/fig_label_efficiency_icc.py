"""
Fig S4 (ICC) — clinical %TF ICC over label efficiency: per-session ICC on FogAtHome
(fixed thr=0.26) vs number of labeled DeFOG patients, FORGE probe vs best-tuned
supervised-from-scratch. Reads logs/RESULTS_label_efficiency_icc.csv.

Usage: python scripts/analysis/fig_label_efficiency_icc.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "research/paper_final/figures/figS4_label_efficiency_icc.png"
COL = {"probe": "#1f77b4", "scratch": "#d62728"}
LAB = {"probe": "FORGE probe (frozen pretrained encoder)",
       "scratch": "Supervised from scratch (best-tuned LR/budget)"}


def main():
    df = pd.read_csv(CSV := "logs/RESULTS_label_efficiency_icc.csv")
    ks = sorted(df["k"].unique())
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for arm in ["probe", "scratch"]:
        g = df[df.arm == arm]
        agg = g.groupby("k")["tf_icc"].agg(["mean", "min", "max"]).reindex(ks)
        ax.plot(agg.index, agg["mean"], "o-" if arm == "probe" else "s--",
                color=COL[arm], lw=2.2, ms=7, label=LAB[arm])
        ax.fill_between(agg.index, agg["min"], agg["max"], color=COL[arm], alpha=0.12)
    ax.axhline(0.75, ls=":", color="green", lw=1, label='"good agreement" (ICC 0.75)')
    ax.axhline(0.0, ls="-", color="grey", lw=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks); ax.set_xticklabels([("all" if k == max(ks) else str(k)) for k in ks])
    ax.set_xlabel("Number of labeled DeFOG patients")
    ax.set_ylabel("FogAtHome %TF ICC (per session, thr=0.26)")
    ax.set_title("Clinical ICC over label efficiency: pretraining vs from scratch")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

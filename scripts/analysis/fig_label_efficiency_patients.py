"""
Fig S4 (v2) — label-efficiency curve on the PATIENT-COUNT axis.
Reads logs/RESULTS_label_efficiency_patients.csv (eval_label_efficiency_patients.py):
FogAtHome segment AP vs number of labeled DeFOG patients, for FORGE probe vs
frozen-random-encoder control vs supervised-from-scratch. Line = mean over subset
seeds; band = min–max across seeds (the draw-to-draw variance).

Usage: python scripts/analysis/fig_label_efficiency_patients.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no Tk
import matplotlib.pyplot as plt
import pandas as pd

CSV = "logs/RESULTS_label_efficiency_patients.csv"
OUT = "research/paper_final/figures/figS4_label_efficiency.png"
COLORS = {"probe": "#1f77b4", "random": "#2ca02c", "scratch": "#7f7f7f"}
LABELS = {"probe": "FORGE probe (frozen pretrained encoder)",
          "random": "Frozen random encoder + trained head (control)",
          "scratch": "Supervised from scratch"}


def main():
    df = pd.read_csv(CSV)
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for arm in ["probe", "random", "scratch"]:
        g = df[df.arm == arm]
        if g.empty:
            continue
        agg = g.groupby("k")["seg_ap"].agg(["mean", "min", "max"]).sort_index()
        ax.plot(agg.index, agg["mean"], "o-", color=COLORS[arm], label=LABELS[arm], lw=2, ms=6)
        ax.fill_between(agg.index, agg["min"], agg["max"], color=COLORS[arm], alpha=0.15)
    prev = df["prevalence"].dropna().median() if "prevalence" in df else None
    if prev is not None:
        ax.axhline(prev, ls=":", color="red", lw=1, label=f"random baseline (prevalence {prev:.2f})")
    ax.set_xscale("log", base=2)
    ks = sorted(df["k"].unique())
    ax.set_xticks(ks)
    ax.set_xticklabels([("all" if k == max(ks) else str(k)) for k in ks])
    ax.set_xlabel("Number of labeled DeFOG patients (FOG-stratified)")
    ax.set_ylabel("FogAtHome segment AP (3-fold ensemble)")
    ax.set_title("Label efficiency on the patient-count axis (band = seed min–max)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)  # PNG only (project convention)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

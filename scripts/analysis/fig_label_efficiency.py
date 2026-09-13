"""
Fig S4 — label-efficiency curve. Reads logs/RESULTS_label_efficiency.csv
(produced by scripts/eval/eval_label_efficiency.py) and plots FogAtHome
segment AP vs DeFOG labeled-data budget for FORGE probe vs supervised-from-scratch.

Usage: python scripts/analysis/fig_label_efficiency.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV = "logs/RESULTS_label_efficiency.csv"
OUT = "research/paper_final/figures/figS4_label_efficiency.png"
# approximate per-patient minutes → total DeFOG train minutes is non-linear; use the
# per-patient budget directly on a log axis. "full" plotted at the cohort max (~211).
XPOS = {"2": 2, "5": 5, "15": 15, "30": 30, "60": 60, "full": 211}


def main():
    df = pd.read_csv(CSV)
    df["x"] = df["budget"].astype(str).map(XPOS)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"probe": "#1f77b4", "random": "#2ca02c", "scratch": "#7f7f7f"}
    labels = {"probe": "FORGE probe (frozen pretrained encoder)",
              "random": "Frozen random encoder + trained head (control)",
              "scratch": "Supervised from scratch"}
    for arm, g in df.groupby("arm"):
        g = g.sort_values("x")
        ax.plot(g["x"], g["seg_ap"], "o-", color=colors.get(arm, None), label=labels.get(arm, arm), lw=2, ms=7)
    prev = df["prevalence"].dropna().median() if "prevalence" in df else None
    if prev is not None:
        ax.axhline(prev, ls=":", color="red", lw=1, label=f"random baseline (prevalence {prev:.2f})")
    ax.set_xscale("log")
    ax.set_xticks(list(XPOS.values()))
    ax.set_xticklabels([k for k in XPOS])
    ax.set_xlabel("DeFOG labeled training data (per-patient minutes)")
    ax.set_ylabel("FogAtHome segment AP (3-fold ensemble)")
    ax.set_title("Label efficiency: pretraining vs. supervised-from-scratch")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)  # PNG only (project convention)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

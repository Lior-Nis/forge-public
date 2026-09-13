"""
Fig S4 (deployment) — FORGE probe vs a COMPETENTLY-TUNED supervised-from-scratch
model, across labeled-patient budgets. Scratch uses the best LR per budget (chosen
among {1e-3, 3e-4, 3e-5} by mean FogAtHome seg AP over 3 seeds). Full-data point
(K=all) comes from the dedicated full-data sweep. Random arm omitted (it was the
mechanism control, not a deployment alternative).

Reads:
  logs/RESULTS_scratch_lr_budgets.csv   (per arm/lr/k/seed; k in 2,4,8,16)
  logs/RESULTS_scratch_lr_sweep.csv     (full-data K=all: scratch LRs + probe ref)
Usage: python scripts/analysis/fig_label_efficiency_deployment.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "research/paper_final/figures/figS4_label_efficiency_deployment.png"
N_ALL = 48


def main():
    df = pd.read_csv("logs/RESULTS_scratch_lr_budgets.csv")
    budgets = sorted(df["k"].unique())  # 2,4,8,16

    # best-LR-per-budget scratch (mean + seed spread), and probe
    sc_mean, sc_lo, sc_hi, pr_mean = {}, {}, {}, {}
    for k in budgets:
        sc = df[(df.arm == "scratch") & (df.k == k)]
        best_lr = sc.groupby("lr")["seg_ap"].mean().idxmax()
        g = sc[sc.lr == best_lr]["seg_ap"]
        sc_mean[k], sc_lo[k], sc_hi[k] = g.mean(), g.min(), g.max()
        pr = df[(df.arm == "probe") & (df.k == k)]["seg_ap"]
        pr_mean[k] = pr.mean()

    # full-data point from the dedicated sweep
    full = pd.read_csv("logs/RESULTS_scratch_lr_sweep.csv")
    sc_full = full[full.case.str.startswith("scratch")]["seg_ap"].max()
    pr_full = full[full.case.str.contains("probe")]["seg_ap"].iloc[0]
    ks = budgets + [N_ALL]
    sc_m = [sc_mean[k] for k in budgets] + [sc_full]
    pr_m = [pr_mean[k] for k in budgets] + [pr_full]

    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.plot(ks, pr_m, "o-", color="#1f77b4", lw=2.2, ms=7, label="FORGE probe (frozen pretrained encoder)")
    ax.fill_between(budgets, [sc_lo[k] for k in budgets], [sc_hi[k] for k in budgets], color="#d62728", alpha=0.12)
    ax.plot(ks, sc_m, "s--", color="#d62728", lw=2.2, ms=7, label="Supervised from scratch (best-tuned LR/budget)")
    prev = 0.31
    ax.axhline(prev, ls=":", color="grey", lw=1, label=f"random baseline (prevalence {prev:.2f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in budgets] + ["all"])
    ax.set_xlabel("Number of labeled DeFOG patients")
    ax.set_ylabel("FogAtHome segment AP (3-fold ensemble)")
    ax.set_title("Deployment value of FORGE pretraining vs. training from scratch")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")
    for k, s, p in zip(ks, sc_m, pr_m):
        print(f"  k={k:>2}: scratch {s:.3f}  probe {p:.3f}  gap {p-s:+.3f}")


if __name__ == "__main__":
    main()

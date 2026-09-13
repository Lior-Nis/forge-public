#!/usr/bin/env python
"""Fig 7 — Pretraining collapse signatures (qualitative).

Three panels on independent y-scales (the point is curve SHAPE, not a head-to-head
metric comparison):
  A. SimCLR  — NT-Xent val loss pinned at the ln N uniform-similarity floor.
  B. I-JEPA  — prediction val loss ~0 (degenerate EMA-target fixed point).
  C. 2D-MAE  — healthy reconstruction val loss that actually descends (contrast).

Data: research/paper_final/data/collapse_curves/*.csv (exported from WandB).
Out:  research/paper_final/figures/fig07_ssl_collapse.{png,pdf}
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "eval"))
from _private_inputs import private_path  # author-only inputs; see that module

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / str(private_path("collapse_curves"))
OUT = ROOT / "research/paper_final/figures"
OUT.mkdir(parents=True, exist_ok=True)

FAIL_BG = "#FDECEC"   # light-red tint for the collapsed panels
OK_BG = "#EAF5EC"     # light-green tint for the healthy panel
FAIL_C = "#C0392B"
OK_C = "#2E7D32"
REF_C = "#7F8C8D"
N_BATCH = 512         # SimCLR batch for vulcan-bird-of-prey-205 -> ln N floor


def series(csv, ycol):
    df = pd.read_csv(DATA / csv)
    x = "epoch" if "epoch" in df.columns else "_step"
    s = df[[x, ycol]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(x)
    return s[x].to_numpy(), s[ycol].to_numpy()


fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

# ---- Panel A: SimCLR collapse ----
ax = axes[0]
ax.set_facecolor(FAIL_BG)
x, y = series("simclr_vulcan-bird-of-prey-205.csv", "losses/val_loss")
ax.plot(x, y, "-o", color=FAIL_C, ms=3, lw=1.8, label="SimCLR val loss")
floor = math.log(N_BATCH)
ax.axhline(floor, ls="--", color=REF_C, lw=1.3, label=f"ln N floor (N={N_BATCH}) = {floor:.2f}")
ax.set_ylim(0, max(7.0, floor + 0.8))
ax.set_title("A · SimCLR — global-pool collapse", fontsize=11, fontweight="bold")
ax.set_xlabel("epoch"); ax.set_ylabel("NT-Xent validation loss")
ax.annotate("pinned at the uniform-similarity\nfloor from the outset →\nnear-identical embeddings",
            xy=(x[len(x)//2], floor), xytext=(0.30, 0.42), textcoords="axes fraction",
            fontsize=8.5, color=FAIL_C,
            arrowprops=dict(arrowstyle="->", color=FAIL_C, lw=1))
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

# ---- Panel B: I-JEPA collapse ----
ax = axes[1]
ax.set_facecolor(FAIL_BG)
x, y = series("ijepa_amber-microwave-203.csv", "jepa/val_loss_mean")
ax.plot(x, y, "-o", color=FAIL_C, ms=3, lw=1.8, label="I-JEPA prediction val loss")
ax.axhline(0.0, ls="--", color=REF_C, lw=1.3, label="0 (degenerate target)")
ax.set_ylim(-0.0002, max(y.max() * 1.6, 0.0012))
ax.set_title("B · I-JEPA — EMA-target collapse", fontsize=11, fontweight="bold")
ax.set_xlabel("epoch"); ax.set_ylabel("latent prediction validation loss")
ax.annotate("loss ≈ 0 throughout →\nencoder matches a near-constant\nEMA target (trivial solution)",
            xy=(x[len(x)//2], y[len(y)//2]), xytext=(0.28, 0.55), textcoords="axes fraction",
            fontsize=8.5, color=FAIL_C,
            arrowprops=dict(arrowstyle="->", color=FAIL_C, lw=1))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

# ---- Panel C: MAE healthy contrast ----
ax = axes[2]
ax.set_facecolor(OK_BG)
x, y = series("mae_lc_sleek-monkey-199.csv", "losses/val_loss")
ax.plot(x, y, "-", color=OK_C, lw=1.8, label="2D-MAE recon val loss")
ax.set_title("C · 2D-MAE (FORGE) — learns (contrast)", fontsize=11, fontweight="bold")
ax.set_xlabel("epoch"); ax.set_ylabel("reconstruction validation loss")
ax.annotate(f"descends and converges\n({y[0]:.2f} → {y[-1]:.3f}),\nnot degenerate",
            xy=(x[len(x)//2], y[len(y)//2]), xytext=(0.40, 0.55), textcoords="axes fraction",
            fontsize=8.5, color=OK_C,
            arrowprops=dict(arrowstyle="->", color=OK_C, lw=1))
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

fig.suptitle("Pretraining collapse signatures (qualitative; independent y-scales, not a metric comparison)",
             fontsize=11.5, y=1.02)
fig.tight_layout()
for ext in ("png",):
    fig.savefig(OUT / f"fig07_ssl_collapse.{ext}", dpi=300, bbox_inches="tight")
print("saved:", OUT / "fig07_ssl_collapse.png")

"""
Fig 3 — FORGE dataset structure / evaluation pipeline.

Presentation-only schematic. Counts and cohort roles match
research/paper_final/draft.md and plot_specs/03_dataset_structure.md.
Explanatory prose belongs in the figure caption, not the figure itself.

Run:
  uv run python scripts/analysis/fig_dataset_pipeline_forge.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("research/paper_final/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ---- Blues-only palette -----------------------------------------------------
PRETRAIN_EC  = "#4292c6"   # pretraining box border (mid blue)
PRETRAIN_FC  = "#9ecae1"   # pretraining box fill (light-mid blue)
SUPER_EC     = "#2171b5"   # supervised box border (dark blue)
SUPER_FC     = "#deebf7"   # supervised box fill (pale blue)
HELD_EC      = "#08306b"   # held-out box border (dark navy) — most prominent
HELD_FC      = "#FFFFFF"   # held-out box fill (white)
ARM_EC       = "#4292c6"   # eval arm borders
ARM_FC       = "#deebf7"   # eval arm fill
DOMAIN_EC    = "#6baed6"   # domain-match callout border
DOMAIN_FC    = "#deebf7"   # domain-match callout fill
ARROW_C      = "#6baed6"   # arrow color
DARK         = "#08306b"   # primary text (dark navy)
MID          = "#2171b5"   # secondary text
LIGHT        = "#555555"   # tertiary text
WHITE        = "#FFFFFF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def rbox(ax, x, y, w, h, fc, ec, lw=1.2, r=0.16, ls="solid", z=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        linestyle=ls, zorder=z, clip_on=False,
    ))


def txt(ax, x, y, s, fs=8.5, color=DARK, weight="normal", style="normal", ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color,
            fontweight=weight, fontstyle=style, multialignment=ha,
            zorder=8, clip_on=False)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=1.35,
                                mutation_scale=10),
                zorder=6)


fig, ax = plt.subplots(figsize=(6.4, 6.0))
ax.set_xlim(0.4, 9.6)
ax.set_ylim(0.8, 11.2)
ax.axis("off")

# ---- Pretraining corpus -----------------------------------------------------
rbox(ax, 1.3, 9.70, 7.4, 1.30, PRETRAIN_FC, PRETRAIN_EC, lw=1.35)
txt(ax, 5, 10.60, "Kaggle — unlabeled daily-living  •  ~58k hours", 9.2, DARK, "bold")
txt(ax, 5, 10.15, "Self-supervised MAE pretraining", 8.2, MID, "bold")

arrow(ax, 5, 9.60, 5, 8.72)

# ---- Labeled cohort ---------------------------------------------------------
rbox(ax, 1.3, 7.45, 7.4, 1.10, SUPER_FC, SUPER_EC, lw=1.35)
txt(ax, 5, 8.17, "Kaggle labeled — DeFOG  •  128 patients  •  ~110 hours", 8.4, DARK, "bold")
txt(ax, 5, 7.75, "Patient-level 3-fold cross-validation", 8.0, MID)

arrow(ax, 5, 7.35, 5, 6.48)

# ---- Held-out evaluation cohort ---------------------------------------------
rbox(ax, 0.75, 2.60, 8.5, 3.60, HELD_FC, HELD_EC, lw=1.8)
txt(ax, 5, 5.90, "FogAtHome — independent evaluation cohort", 9.8, DARK, "bold")

arm_y = 3.10
rbox(ax, 1.25, arm_y, 3.55, 2.35, ARM_FC, ARM_EC, lw=1.0, r=0.12)
rbox(ax, 5.20, arm_y, 3.55, 2.35, ARM_FC, ARM_EC, lw=1.0, r=0.12)

txt(ax, 3.025, arm_y + 1.90, "Structured protocol", 8.6, MID, "bold")
txt(ax, 3.025, arm_y + 1.42, "12 patients", 8.1, DARK, "bold")
txt(ax, 3.025, arm_y + 0.95, "31% TF", 7.8, DARK)
txt(ax, 3.025, arm_y + 0.52, "~45 min total", 7.2, LIGHT)

txt(ax, 6.975, arm_y + 1.90, "Daily living", 8.6, MID, "bold")
txt(ax, 6.975, arm_y + 1.42, "11 patients", 8.1, DARK, "bold")
txt(ax, 6.975, arm_y + 0.95, "1.1% TF", 7.8, DARK)
txt(ax, 6.975, arm_y + 0.52, "~309 hours", 7.2, LIGHT)

# ---- Domain match callout ---------------------------------------------------
rbox(ax, 1.0, 1.45, 8.0, 0.72, DOMAIN_FC, DOMAIN_EC, lw=0.9, r=0.13)
txt(ax, 5, 1.82, "Identical sensor, placement & home setting", 7.8, DARK, "bold")

# ---- Legend -----------------------------------------------------------------
legend = [
    mpatches.Patch(facecolor=PRETRAIN_FC, edgecolor=PRETRAIN_EC, label="Pretraining corpus"),
    mpatches.Patch(facecolor=SUPER_FC,    edgecolor=SUPER_EC,    label="Supervised train/val"),
    mpatches.Patch(facecolor=HELD_FC,     edgecolor=HELD_EC,     label="Held-out evaluation"),
]
ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.0),
          ncol=3, frameon=False, fontsize=7.4, handlelength=1.5, columnspacing=1.3)

out = OUT / "fig03_dataset_structure.png"
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
plt.close(fig)
print(f"+ saved {out}")

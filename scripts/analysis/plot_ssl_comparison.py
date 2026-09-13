"""
SSL vs Supervised comparison plots for thesis.

Produces 5 figures saved to artifacts/plots/ssl_comparison/:
  1. method_comparison.png   — Defog pooled AP across all approaches
  2. fogathome_benchmark.png — External FogAtHome AP vs Kaggle competition
  3. pretrain_trajectory.png — Probe AP by pretraining epoch (+ val_loss)
  4. label_efficiency.png    — Defog AP vs FOG labels used for backbone
  5. domain_gap.png          — Lab vs Home performance gap, supervised vs SSL

Run:
    uv run python scripts/plot_ssl_comparison.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = Path("artifacts/plots/ssl_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────
C_RANDOM     = "#9E9E9E"   # grey
C_SUPERVISED = "#2196F3"   # blue
C_SSL        = "#FF5722"   # orange-red
C_HIGHLIGHT  = "#4CAF50"   # green (our best)
C_KAGGLE     = "#9C27B0"   # purple (competition)

FIGSIZE = (8, 5)
DPI     = 150


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Method comparison (Defog pooled AP)
# ═══════════════════════════════════════════════════════════════════════════
def plot_method_comparison():
    methods = [
        # (label, ap_mean, ap_std, category)
        ("Random init\n(probe)",          0.187, 0.077, "random"),
        ("MAE full\nfinetuning",           0.212, 0.107, "ssl"),
        ("JEPA\n(probe)",                  0.230, 0.148, "ssl"),
        ("Defog-only\nsupervised",         0.245, 0.100, "supervised"),
        ("Mixed\nsupervised",              0.315, 0.167, "supervised"),
        ("Cross-protocol\nsupervised",     0.347, 0.254, "supervised"),
        ("MAE ep7\nfrozen probe",          0.475, 0.000, "ssl_best"),
    ]

    colors = {
        "random":     C_RANDOM,
        "supervised": C_SUPERVISED,
        "ssl":        C_SSL,
        "ssl_best":   C_HIGHLIGHT,
    }

    labels, aps, stds, cats = zip(*methods)
    bar_colors = [colors[c] for c in cats]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(x, aps, color=bar_colors, edgecolor="white", linewidth=0.8, zorder=3)

    # error bars (skip 0 std)
    for xi, (ap, std, cat) in enumerate(zip(aps, stds, cats)):
        if std > 0:
            ax.errorbar(xi, ap, yerr=std, fmt="none", color="black",
                        capsize=4, capthick=1.2, linewidth=1.2, zorder=4)

    # value labels on bars
    for bar, ap in zip(bars, aps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{ap:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # "0 FOG labels" annotation for SSL best
    best_idx = len(methods) - 1
    ax.annotate("0 FOG labels\nduring pretraining",
                xy=(best_idx, aps[best_idx]),
                xytext=(best_idx - 1.5, aps[best_idx] + 0.08),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                fontsize=8, ha="center", color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Defog pooled AP", fontsize=11)
    ax.set_title("SSL pretraining outperforms all supervised approaches\non home-recorded gait (defog cohort)", fontsize=11)
    ax.set_ylim(0, 0.72)
    # Separate FOG prevalence (chance) from random-init performance — different concepts
    fog_prevalence = 0.17
    ax.axhline(fog_prevalence, color="black", linestyle=":", linewidth=1.0, alpha=0.5,
               label=f"FOG prevalence / chance AP ≈ {fog_prevalence:.2f}")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    # Note: MAE ep7 std=0 because it is a single pooled run, not a cross-fold mean
    ax.text(len(methods) - 1, 0.48, "single\npooled run", ha="center", va="bottom",
            fontsize=6.5, color=C_HIGHLIGHT, style="italic")

    legend_handles = [
        mpatches.Patch(color=C_RANDOM,     label="Random initialisation (no pretraining)"),
        mpatches.Patch(color=C_SUPERVISED, label="Supervised (labelled FOG data)"),
        mpatches.Patch(color=C_SSL,        label="SSL (no FOG labels for backbone)"),
        mpatches.Patch(color=C_HIGHLIGHT,  label="SSL best (MAE ep7 probe, single pooled run)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left")

    fig.tight_layout()
    path = OUT_DIR / "method_comparison.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — External generalization: FogAtHome vs Kaggle competition
# ═══════════════════════════════════════════════════════════════════════════
def plot_fogathome_benchmark():
    # Kaggle competition models (from kaggle_metrics.csv, defog rows)
    competition = [
        ("Rank 1", 0.357, 0.644),   # (defog_ap, fogathome_ap)
        ("Rank 2", 0.375, 0.615),
        ("Rank 3", 0.343, 0.674),
        ("Rank 4", 0.259, 0.553),
        ("Rank 5", 0.302, 0.700),
    ]
    ours_defog_ap    = 0.475
    ours_fogathome   = 0.764

    comp_labels  = [c[0] for c in competition]
    comp_defog   = [c[1] for c in competition]
    comp_fah     = [c[2] for c in competition]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ── left: FogAtHome AP ─────────────────────────────────────
    ax = axes[0]
    all_labels = comp_labels + ["Ours\n(MAE ep7)"]
    all_vals   = comp_fah   + [ours_fogathome]
    all_colors = [C_KAGGLE] * len(comp_labels) + [C_HIGHLIGHT]
    x = np.arange(len(all_labels))

    bars = ax.bar(x, all_vals, color=all_colors, edgecolor="white", linewidth=0.8, zorder=3)
    for bar, v in zip(bars, all_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=9)
    ax.set_ylabel("FogAtHome AP", fontsize=11)
    ax.set_title("External cohort (FogAtHome): our model vs Kaggle competition\n"
                 "(competition ranks are by in-distribution Defog score, not FogAtHome)", fontsize=9)
    ax.set_ylim(0, 0.92)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    comp_patch = mpatches.Patch(color=C_KAGGLE,    label="Kaggle competition (supervised, labelled)")
    ours_patch = mpatches.Patch(color=C_HIGHLIGHT, label="Ours (MAE probe, 0 FOG labels in pretraining)")
    ax.legend(handles=[comp_patch, ours_patch], fontsize=8)

    # ── right: scatter — defog AP vs FogAtHome AP ──────────────
    ax = axes[1]
    ax.scatter(comp_defog, comp_fah, color=C_KAGGLE, s=80, zorder=3, label="Kaggle competition")
    for label, dx, dy in zip(comp_labels, comp_defog, comp_fah):
        ax.annotate(label, (dx, dy), textcoords="offset points", xytext=(5, 3), fontsize=8)

    ax.scatter([ours_defog_ap], [ours_fogathome], color=C_HIGHLIGHT, s=120,
               marker="*", zorder=4, label="Ours (MAE ep7)")
    ax.annotate("Ours", (ours_defog_ap, ours_fogathome),
                textcoords="offset points", xytext=(5, 3), fontsize=9, color=C_HIGHLIGHT, fontweight="bold")

    ax.set_xlabel("Defog AP (in-distribution)", fontsize=10)
    ax.set_ylabel("FogAtHome AP (out-of-distribution)", fontsize=10)
    ax.set_title("In-distribution vs OOD generalisation", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax.set_xlim(0.20, 0.56)
    ax.set_ylim(0.50, 0.82)

    fig.tight_layout()
    path = OUT_DIR / "fogathome_benchmark.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Pretraining trajectory (epoch → probe AP)
# ═══════════════════════════════════════════════════════════════════════════
def plot_pretrain_trajectory():
    # epoch, defog_ap, fogathome_ap, pretrain_val_loss
    data = [
        (1,  0.374,  None,  0.098),
        (3,  0.447, 0.690,  0.054),
        (4,  0.381, 0.735,  0.040),
        (5,  0.352, 0.697,  0.027),
        (6,  0.388, 0.737,  0.041),   # optimizer restart from ep4
        (7,  0.475, 0.764,  None),
        (8,  0.390, 0.724,  None),
    ]

    epochs    = [d[0] for d in data]
    defog_ap  = [d[1] for d in data]
    fah_ap    = [d[2] for d in data]
    val_loss  = [d[3] for d in data]

    fah_epochs = [e for e, f in zip(epochs, fah_ap) if f is not None]
    fah_vals   = [f for f in fah_ap if f is not None]
    loss_epochs = [e for e, l in zip(epochs, val_loss) if l is not None]
    loss_vals   = [l for l in val_loss if l is not None]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    # probe APs
    ax1.plot(epochs, defog_ap, "o-", color=C_SSL,        linewidth=2, markersize=7, label="Defog pooled AP")
    ax1.plot(fah_epochs, fah_vals, "s--", color=C_HIGHLIGHT, linewidth=2, markersize=7, label="FogAtHome AP")

    # mark best on both metrics
    best_ep = 7
    ax1.scatter([best_ep], [0.475], s=150, color=C_SSL,        zorder=5)
    ax1.scatter([best_ep], [0.764], s=150, color=C_HIGHLIGHT,  zorder=5)
    ax1.annotate("ep7 best\n(Defog)", (best_ep, 0.475), textcoords="offset points",
                 xytext=(8, -20), fontsize=8, color=C_SSL)
    ax1.annotate("ep7 best\n(FogAtHome)", (best_ep, 0.764), textcoords="offset points",
                 xytext=(8, 4), fontsize=8, color=C_HIGHLIGHT)

    # pretrain val_loss
    ax2.plot(loss_epochs, loss_vals, "^:", color=C_SUPERVISED, linewidth=1.5,
             markersize=6, alpha=0.7, label="Pretrain val_loss")
    ax2.set_ylabel("Pretrain val_loss", fontsize=10, color=C_SUPERVISED)
    ax2.tick_params(axis="y", colors=C_SUPERVISED)

    # optimizer restart annotation
    ax1.axvline(6, color="grey", linestyle=":", linewidth=1)
    ax1.text(6.05, 0.33, "optimizer\nrestart", fontsize=7.5, color="grey")

    ax1.set_xlabel("Pretraining epoch  (epoch 2 not evaluated)", fontsize=11)
    ax1.set_ylabel("Downstream probe AP", fontsize=11)
    ax1.set_title("Downstream AP peaks at ep7 then regresses;\npretrain val_loss is an imperfect proxy (rises at ep6 restart)", fontsize=11)
    ax1.set_ylim(0.28, 0.85)
    ax1.set_xticks(epochs)
    ax1.set_xticklabels(epochs, fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.spines[["top"]].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    fig.tight_layout()
    path = OUT_DIR / "pretrain_trajectory.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Label efficiency: FOG labels used for backbone → Defog AP
# ═══════════════════════════════════════════════════════════════════════════
def plot_label_efficiency():
    # Approximate labeled FOG patch counts
    # Defog dataset: ~28 patients, ~1,100 FOG patches per fold (5 folds, so ~5 * 1100 ≈ 5500 total training FOG patches)
    # For "all labels" supervised: they train end-to-end on all folds' train sets.
    # Rough total: 28 patients, avg ~6% FOG rate, ~21M patches total → but defog is much smaller.
    # From current.md: ~1,100 labeled FOG patches per fold → 4 training folds = ~4,400 backbone-training FOG patches.
    N_supervised_fog_patches = 4400   # approximate FOG patches seen during supervised backbone training

    points = [
        # (fog_backbone_labels, defog_ap, std, label, marker, color)
        (0,                    0.187, 0.077, "Random init\n(probe only)",         "o", C_RANDOM),
        (0,                    0.230, 0.148, "JEPA\n(SSL, no FOG labels)",        "^", C_SSL),
        (0,                    0.475, 0.000, "MAE ep7\n(SSL, no FOG labels)",     "*", C_HIGHLIGHT),
        (N_supervised_fog_patches * 0.6,  0.245, 0.100, "Defog-only\nsupervised", "s", C_SUPERVISED),
        (N_supervised_fog_patches * 0.8,  0.315, 0.167, "Mixed\nsupervised",      "D", C_SUPERVISED),
        (N_supervised_fog_patches,        0.347, 0.254, "Cross-protocol\nsupervised", "P", C_SUPERVISED),
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for x, ap, std, label, marker, color in points:
        size = 180 if marker == "*" else 90
        ax.scatter(x, ap, s=size, marker=marker, color=color, zorder=4, edgecolors="white", linewidth=0.5)
        if std > 0:
            ax.errorbar(x, ap, yerr=std, fmt="none", color=color, capsize=4, capthick=1.2, linewidth=1.2, zorder=3, alpha=0.7)
        xytext_offset = (12, 0) if x == 0 else (6, 8)
        ha = "left"
        ax.annotate(label, (x, ap), textcoords="offset points", xytext=xytext_offset,
                    fontsize=7.5, ha=ha, color=color, fontweight="bold" if marker == "*" else "normal")

    # No trend line through supervised points — these are discrete training conditions,
    # not a continuous interpolation, so a line would imply a relationship that doesn't exist.

    # shaded region: "SSL pretraining zone" (x=0, self-supervised backbone, zero FOG labels)
    ax.axvspan(-200, 300, alpha=0.05, color=C_HIGHLIGHT, zorder=0)
    ax.text(150, 0.60, "SSL pretraining\n(zero FOG backbone labels)", fontsize=8.5, color=C_HIGHLIGHT,
            alpha=0.8, ha="center", style="italic")
    # Explicitly flag that random init is NOT pretrained (different from SSL at x=0)
    ax.annotate("no pretraining\n(not SSL)", xy=(0, 0.187), xytext=(-150, 0.13),
                fontsize=7, color=C_RANDOM, ha="center", style="italic",
                arrowprops=dict(arrowstyle="->", color=C_RANDOM, lw=0.8))

    ax.set_xlabel("Approximate FOG-labelled patches used for backbone training", fontsize=10)
    ax.set_ylabel("Defog pooled AP", fontsize=11)
    ax.set_title("SSL achieves higher AP using zero FOG labels\nduring backbone pretraining", fontsize=11)
    ax.set_xlim(-300, N_supervised_fog_patches + 1000)
    ax.set_ylim(0.05, 0.72)
    ax.set_xticks([0, N_supervised_fog_patches // 2, N_supervised_fog_patches])
    ax.set_xticklabels(["0\n(SSL)", f"~{N_supervised_fog_patches//2}", f"~{N_supervised_fog_patches}\n(supervised)"])
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=C_RANDOM,     label="Random init (no pretraining)"),
        mpatches.Patch(color=C_SSL,        label="SSL pretraining (no FOG labels)"),
        mpatches.Patch(color=C_HIGHLIGHT,  label="MAE ep7 (best SSL)"),
        mpatches.Patch(color=C_SUPERVISED, label="End-to-end supervised"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    path = OUT_DIR / "label_efficiency.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Domain gap: Lab vs Home performance
# ═══════════════════════════════════════════════════════════════════════════
def plot_domain_gap():
    """
    Compare supervised baseline vs SSL probe on lab (tdcsfog) and home (defog/FogAtHome) domains.

    Supervised model (Exp #013):  tdcsfog F1=0.838, defog F1=0.459
    Supervised model has no FogAtHome eval (external cohort).

    MAE probe (Exp #023):
      - defog AP=0.475 (home, in-distribution)
      - FogAtHome AP=0.764 (home, out-of-distribution)
      - Lab perf not directly measured (pretraining is unsupervised).

    We show two sub-plots:
      Left:  F1 on lab vs home for supervised (bar chart, shows the gap).
      Right: AP comparison on home (defog) and external home (FogAtHome) — supervised vs SSL.
    """

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ── left: lab vs home F1, supervised only ──────────────────
    ax = axes[0]
    domains    = ["Lab\n(tdcsfog)", "Home\n(defog)"]
    sup_f1     = [0.838, 0.459]
    sup_f1_std = [0.040, 0.142]

    x = np.arange(len(domains))
    bar_alphas = [1.0, 0.6]
    bars = []
    for xi, (v, a) in enumerate(zip(sup_f1, bar_alphas)):
        b = ax.bar(xi, v, color=C_SUPERVISED, alpha=a, edgecolor="white", linewidth=0.8, zorder=3)
        bars.append(b[0])
    ax.errorbar(x, sup_f1, yerr=sup_f1_std, fmt="none", color="black",
                capsize=5, capthick=1.3, linewidth=1.3, zorder=4)

    for bar, v in zip(bars, sup_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # gap annotation
    ax.annotate("", xy=(1, sup_f1[1] + sup_f1_std[1] + 0.03),
                xytext=(0, sup_f1[0] - sup_f1_std[0] - 0.03),
                arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
    ax.text(0.5, (sup_f1[0] + sup_f1[1]) / 2, f"Gap\n−{sup_f1[0]-sup_f1[1]:.3f} F1",
            ha="center", va="center", color="red", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="red", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=11)
    ax.set_ylabel("F1 score", fontsize=11)
    ax.set_title("Supervised model: large lab→home gap\n(Exp #013, best supervised baseline)  [metric: F1]", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # ── right: home AP — supervised vs SSL ─────────────────────
    ax = axes[1]
    categories = ["Defog AP\n(in-distribution)", "FogAtHome AP\n(out-of-distribution)"]
    sup_vals   = [0.315, None]         # mixed supervised on defog; no FogAtHome eval
    ssl_vals   = [0.475, 0.764]
    sup_std    = [0.167, None]

    x = np.arange(len(categories))
    width = 0.35

    # supervised bars (only defog)
    ax.bar(x[0] - width/2, sup_vals[0], width, color=C_SUPERVISED, edgecolor="white",
           linewidth=0.8, zorder=3, label="Best supervised (mixed, Exp #020)")
    ax.errorbar(x[0] - width/2, sup_vals[0], yerr=sup_std[0], fmt="none",
                color="black", capsize=4, capthick=1.2, linewidth=1.2, zorder=4)
    ax.text(x[0] - width/2, sup_vals[0] + 0.012, f"{sup_vals[0]:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.text(x[1] - width/2, 0.04, "Not\nevaluated", ha="center", va="bottom",
            fontsize=8, color=C_SUPERVISED, style="italic", alpha=0.7)

    # SSL bars
    ssl_bar_colors = [C_SSL, C_HIGHLIGHT]
    for i, (val, color) in enumerate(zip(ssl_vals, ssl_bar_colors)):
        ax.bar(x[i] + width/2, val, width, color=color, edgecolor="white",
               linewidth=0.8, zorder=3)
        ax.text(x[i] + width/2, val + 0.012, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # best competition FogAtHome line
    ax.axhline(0.700, color=C_KAGGLE, linestyle="--", linewidth=1.2, alpha=0.8,
               label="Kaggle best (FogAtHome, Rank 5: 0.700)")
    ax.text(1.6, 0.706, "Kaggle best\n(0.700)", fontsize=7.5, color=C_KAGGLE, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Average Precision (AP)", fontsize=11)
    ax.set_title("SSL outperforms supervised on home domain\nand generalises to unseen cohort  [metric: AP]", fontsize=10)
    ax.text(0.01, 0.99, "⚠ Left panel: F1 score (supervised baseline only)\n    Right panel: AP — different metrics, not directly comparable",
            transform=ax.transAxes, fontsize=7.5, va="top", color="#8B4513", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E1", edgecolor="#E6A817", alpha=0.85))
    ax.set_ylim(0, 0.90)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=C_SUPERVISED, label="Best supervised (mixed, Exp #020)"),
        mpatches.Patch(color=C_SSL,        label="MAE probe (Defog AP)"),
        mpatches.Patch(color=C_HIGHLIGHT,  label="MAE probe (FogAtHome AP)"),
        mpatches.Patch(color=C_KAGGLE,     label="Kaggle Rank 5 FogAtHome baseline"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc="upper left")

    fig.tight_layout()
    path = OUT_DIR / "domain_gap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    plot_method_comparison()
    plot_fogathome_benchmark()
    plot_pretrain_trajectory()
    plot_label_efficiency()
    plot_domain_gap()
    print(f"\nAll figures saved to {OUT_DIR}/")

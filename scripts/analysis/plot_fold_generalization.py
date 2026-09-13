"""
Train→test fold generalisation figure.

Left panel:  UMAP of all Defog patches, coloured by which fold they are the
             TEST patient in (fold 0-4). Shows whether test patients from each
             fold land in different embedding regions (bad) or are interleaved
             with training patients (good).

Right panel: Per-fold pooled AP on held-out test patients, with per-patient AP
             dots overlaid and FOG prevalence as a chance baseline.
             Annotates folds with very few FOG patches to contextualise low AP.

Run:
    uv run python scripts/plot_fold_generalization.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

EMBD_DIR = Path("logs/embeddings/vit12_ep4")
AP_DIR   = Path("logs/pooled_ap")
OUT      = Path("artifacts/plots/publication")
OUT.mkdir(parents=True, exist_ok=True)

DPI = 200

FOLD_COLORS = ["#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4"]  # 5 distinct
FOLD_LABELS = [f"Fold {k}" for k in range(5)]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


def load_fold_ap():
    rows = []
    for fold in range(5):
        df = pd.read_csv(AP_DIR / f"vit12_ep7_fold{fold}_preds.csv")
        pooled_ap = average_precision_score(df["true_label"], df["pred_prob_fog"])
        fog_rate  = df["true_label"].mean()
        n_fog_total = int(df["true_label"].sum())
        for pid, g in df.groupby("patient_id"):
            if g["true_label"].nunique() == 2:
                pat_ap  = average_precision_score(g["true_label"], g["pred_prob_fog"])
                n_fog_p = int(g["true_label"].sum())
            else:
                pat_ap  = float("nan")
                n_fog_p = int(g["true_label"].sum())
            rows.append(dict(fold=fold, patient_id=pid, pat_ap=pat_ap,
                             pooled_ap=pooled_ap, fog_rate=fog_rate,
                             n_fog=n_fog_p, n_fog_total=n_fog_total))
    return pd.DataFrame(rows)


def make_figure():
    coords = np.load(EMBD_DIR / "coords_umap.npy")
    meta   = pd.read_csv(EMBD_DIR / "metadata.csv")

    defog_mask = meta["dataset"] == "defog"
    fah_mask   = meta["dataset"] == "fogathome"

    ap_df = load_fold_ap()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [1.3, 1]})

    # ── Left: UMAP coloured by test fold ─────────────────────────────────
    ax = axes[0]

    # FogAtHome background (light grey — external reference)
    ax.scatter(coords[fah_mask, 0], coords[fah_mask, 1],
               s=4, c="#CCCCCC", alpha=0.25, linewidths=0, zorder=1,
               label="FogAtHome (external)")

    # Defog patches: colour = test fold, size/alpha = FOG vs no-FOG
    defog_meta   = meta[defog_mask].reset_index(drop=True)
    defog_coords = coords[defog_mask]

    for fold in range(5):
        fold_mask = defog_meta["fold"].values == fold
        fog_mask_f   = fold_mask & (defog_meta["true_label"].values == 1)
        nofog_mask_f = fold_mask & (defog_meta["true_label"].values == 0)
        color = FOLD_COLORS[fold]

        # No-FOG: small, transparent
        ax.scatter(defog_coords[nofog_mask_f, 0], defog_coords[nofog_mask_f, 1],
                   s=5, c=color, alpha=0.18, linewidths=0, zorder=2)
        # FOG: larger, opaque
        ax.scatter(defog_coords[fog_mask_f, 0], defog_coords[fog_mask_f, 1],
                   s=22, c=color, alpha=0.85, linewidths=0, zorder=3)

        # FOG centroid diamond for this fold's test patients
        if fog_mask_f.any():
            cx, cy = defog_coords[fog_mask_f].mean(0)
            pooled_ap = ap_df[ap_df["fold"] == fold]["pooled_ap"].iloc[0]
            ax.scatter(cx, cy, marker="D", s=160, c=color,
                       edgecolor="white", linewidth=1.5, zorder=10)
            ax.annotate(f"F{fold}  AP={pooled_ap:.2f}",
                        (cx, cy), textcoords="offset points",
                        xytext=(8, 4), fontsize=7.5, color=color, fontweight="bold")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP-1", fontsize=8, color="grey")
    ax.text(-0.01, 0.5, "UMAP-2", rotation=90, fontsize=8, color="grey",
            transform=ax.transAxes, ha="right", va="center")
    ax.set_title(
        "Embedding space: Defog patches coloured by their test fold\n"
        "Large dots = FOG patches  ·  Diamond = fold's FOG centroid\n"
        "(UMAP fit on all patches jointly; backbone: MAE vit12 ep4)",
        fontsize=9.5
    )

    # Legend: fold colours + FogAtHome
    legend_handles = [
        mpatches.Patch(color=FOLD_COLORS[k], label=f"Fold {k} test patients")
        for k in range(5)
    ] + [mpatches.Patch(color="#CCCCCC", label="FogAtHome (external, not evaluated here)")]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower left",
              frameon=True, framealpha=0.88, facecolor="white", edgecolor="grey")

    # ── Right: per-fold AP ────────────────────────────────────────────────
    ax = axes[1]

    fold_summary = ap_df.groupby("fold").first()[["pooled_ap", "fog_rate", "n_fog_total"]].reset_index()

    x = np.arange(5)
    bars = ax.bar(x, fold_summary["pooled_ap"], color=FOLD_COLORS,
                  edgecolor="white", linewidth=0.8, zorder=3, width=0.55)
    for bar, ap in zip(bars, fold_summary["pooled_ap"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{ap:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Per-patient AP dots overlaid
    for fold in range(5):
        pat_rows = ap_df[(ap_df["fold"] == fold) & ap_df["pat_ap"].notna()]
        jitter = np.random.RandomState(fold).uniform(-0.15, 0.15, len(pat_rows))
        ax.scatter(fold + jitter, pat_rows["pat_ap"],
                   s=55, c=FOLD_COLORS[fold], edgecolor="white",
                   linewidth=0.8, zorder=5, alpha=0.9)

    # FOG prevalence (chance) per fold as a scatter of horizontal ticks
    for fold, rate in enumerate(fold_summary["fog_rate"]):
        ax.plot([fold - 0.28, fold + 0.28], [rate, rate],
                color="grey", linewidth=1.2, linestyle="--", zorder=4, alpha=0.7)

    # Chance legend line
    ax.plot([], [], color="grey", linewidth=1.2, linestyle="--",
            label="FOG prevalence (chance AP per fold)")

    # Annotate folds 3 and 4 that have very few FOG patches
    for fold in [3, 4]:
        n = fold_summary.loc[fold_summary["fold"] == fold, "n_fog_total"].iloc[0]
        ax.text(fold, 0.02, f"only {n}\nFOG patches", ha="center", va="bottom",
                fontsize=7, color="grey", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {k}\n({len(ap_df[ap_df['fold']==k])} patients)"
                        for k in range(5)], fontsize=9)
    ax.set_ylabel("Pooled AP (test fold patients)", fontsize=10)
    ax.set_ylim(0, 0.80)
    ax.set_title(
        "Per-fold held-out test performance\n"
        "Dots = per-patient AP  ·  Dashed = FOG prevalence (chance)",
        fontsize=10
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25, zorder=0)

    fig.suptitle(
        "Train→test generalisation across Defog k-folds  (MAE ep7 frozen probe)\n"
        "Each fold's patients were held out from probe training; backbone pretrained without any FOG labels",
        fontsize=11, y=1.02
    )

    fig.tight_layout()
    out = OUT / "fig_fold_generalization.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_figure()

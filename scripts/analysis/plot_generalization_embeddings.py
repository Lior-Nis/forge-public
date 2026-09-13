"""
Generalization embedding plots for thesis.

Reads pre-computed CSVs from logs/embeddings/generalization/ (no GPU required).
Produces 2 figures saved to artifacts/plots/generalization/:

  1. generalization_summary.png  — 4-panel:
       A. Cross-cohort FOG centroid similarity (bar, with within-cohort baselines)
       B. Cohort PCA distance (bar — how far apart are defog vs FogAtHome distributions)
       C. NN precision@k curves (cross-cohort FOG retrieval) with FOG-prevalence baseline
       D. Per-patient FOG silhouette distribution (violin)

  2. umap_patient_vs_fog.png  — 2-panel UMAP (patient colour vs FOG label colour)
       Reads pre-computed UMAP coords from logs/embeddings/vit12_ep4/

Run:
    uv run python scripts/plot_generalization_embeddings.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

GEN_DIR  = Path("logs/embeddings/generalization")
EMBD_DIR = Path("logs/embeddings/vit12_ep4")
OUT_DIR  = Path("artifacts/plots/generalization")
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_MAE  = "#4878cf"   # blue
C_SUP  = "#d65f5f"   # red
C_BASE = "#9E9E9E"   # grey baseline
DPI    = 150

LABEL_MAE = "MAE pretrained\n(vit12 ep4 probe)"
LABEL_SUP = "Supervised\nfrom scratch"


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — 4-panel generalization summary
# ═══════════════════════════════════════════════════════════════════════════

def plot_generalization_summary():
    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    fig.suptitle(
        "MAE pretrained backbone aligns FOG representations across cohorts;\n"
        "supervised embeddings are more cohort-dependent  (vit12 ep4 embedding checkpoint)",
        fontsize=11, y=1.03
    )

    # ── A: Cross-cohort FOG centroid similarity ──────────────────────────
    ax = axes[0]
    sim = pd.read_csv(GEN_DIR / "centroid_similarity_summary.csv")

    mae_row = sim[sim["model"].str.contains("MAE")].iloc[0]
    sup_row = sim[sim["model"].str.contains("Supervised")].iloc[0]

    categories = ["Within\nDefog", "Within\nFogAtHome", "Cross-cohort\n(Defog↔FogAtHome)"]
    mae_vals = [mae_row["mean_within_defog_sim"],
                mae_row["mean_within_fogathome_sim"],
                mae_row["mean_cross_cohort_sim"]]
    sup_vals = [sup_row["mean_within_defog_sim"],
                sup_row["mean_within_fogathome_sim"],
                sup_row["mean_cross_cohort_sim"]]

    x = np.arange(len(categories))
    w = 0.32
    bars_mae = ax.bar(x - w/2, mae_vals, w, color=C_MAE, label=LABEL_MAE, zorder=3)
    bars_sup = ax.bar(x + w/2, sup_vals, w, color=C_SUP, label=LABEL_SUP, zorder=3)

    for bar, v in list(zip(bars_mae, mae_vals)) + list(zip(bars_sup, sup_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Annotate the key cross-cohort gap
    ax.annotate("", xy=(x[2] + w/2, sup_vals[2] + 0.01),
                xytext=(x[2] - w/2, mae_vals[2] - 0.01),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
    ax.text(x[2], (mae_vals[2] + sup_vals[2]) / 2, f"  Δ{mae_vals[2]-sup_vals[2]:.2f}",
            va="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("Mean cosine similarity of FOG centroids", fontsize=9)
    ax.set_title("A  FOG centroid alignment\nacross patients & cohorts", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # ── B: Cohort PCA distance ───────────────────────────────────────────
    ax = axes[1]
    pca = pd.read_csv(GEN_DIR / "cohort_pca_summary.csv")
    mae_pca = pca[pca["model"].str.contains("MAE")].iloc[0]["cohort_pca_distance"]
    sup_pca = pca[pca["model"].str.contains("Supervised")].iloc[0]["cohort_pca_distance"]

    models = [LABEL_MAE, LABEL_SUP]
    vals   = [mae_pca, sup_pca]
    colors = [C_MAE, C_SUP]
    bars = ax.barh(models, vals, color=colors, edgecolor="white", linewidth=0.8, zorder=3)

    for bar, v in zip(bars, vals):
        ax.text(v + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Euclidean distance between cohort centroids\nin top-2 PCA space  (lower = better alignment)", fontsize=8)
    ax.set_title("B  Embedding space alignment\n(defog vs FogAtHome; PCA fit on all embeddings jointly)", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 38)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

    # add "×" annotation showing the ratio
    ratio = sup_pca / mae_pca
    ax.text(sup_pca / 2, 0.5, f"{ratio:.0f}× further apart",
            ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # ── C: NN precision@k curves ─────────────────────────────────────────
    ax = axes[2]
    nn = pd.read_csv(GEN_DIR / "nn_precision_values.csv")
    fog_prevalence = 0.165   # ~defog FOG positive rate

    for label, color in [(nn["model"].unique()[0], C_MAE),
                         (nn["model"].unique()[1], C_SUP)]:
        grp = nn[nn["model"] == label]
        short = LABEL_MAE if "MAE" in label else LABEL_SUP
        ax.plot(grp["k"], grp["precision"], "o-", color=color, linewidth=1.8,
                markersize=5, label=short, zorder=3)

    ax.axhline(fog_prevalence, color=C_BASE, linestyle="--", linewidth=1.2,
               label=f"Chance baseline\n(FOG prevalence ≈{fog_prevalence:.0%})")

    ax.set_xlabel("k (nearest neighbours)", fontsize=9)
    ax.set_ylabel("FOG precision@k\n(cross-cohort retrieval)", fontsize=9)
    ax.set_title("C  FogAtHome FOG patches:\nhow often are their Defog neighbours FOG?",
                 fontsize=9, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    # Honest annotation: supervised is locally tighter (higher precision@k) but globally misaligned (panels A & B)
    ax.text(0.97, 0.08,
            "Supervised has higher local FOG precision@k\nbut worse global cohort alignment (see A & B);\nboth above chance ≈ 16.5%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color="#8B4513", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E1", edgecolor="#E6A817", alpha=0.85))

    # ── D: Per-patient silhouette violin ─────────────────────────────────
    ax = axes[3]
    sil = pd.read_csv(GEN_DIR / "per_patient_silhouette.csv")

    positions = []
    data_groups = []
    tick_labels = []
    tick_colors = []
    pos = 1
    for model_str, color, short in [
        ("MAE", C_MAE, "MAE"),
        ("Supervised", C_SUP, "Supervised"),
    ]:
        for cohort in ["defog", "fogathome"]:
            grp = sil[(sil["model"].str.contains(model_str)) &
                      (sil["cohort"] == cohort)]["silhouette"].values
            data_groups.append(grp)
            positions.append(pos)
            label = f"{short}\n({'Defog' if cohort == 'defog' else 'FogAtHome'})"
            tick_labels.append(label)
            tick_colors.append(color)
            pos += 1
        pos += 0.5   # gap between models

    vparts = ax.violinplot(data_groups, positions=positions, showmedians=True,
                           showextrema=True)
    model_colors_seq = [C_MAE, C_MAE, C_SUP, C_SUP]
    for body, color in zip(vparts["bodies"], model_colors_seq):
        body.set_facecolor(color)
        body.set_alpha(0.6)
    vparts["cmedians"].set_color("black")
    vparts["cbars"].set_color("black")
    vparts["cmins"].set_color("black")
    vparts["cmaxes"].set_color("black")

    ax.axhline(0, color=C_BASE, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(4.6, 0.005, "0 = no\nseparation", fontsize=6.5, color="grey", va="bottom", ha="right")

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    ax.set_ylabel("Per-patient FOG silhouette score\n(FOG vs no-FOG separability per patient)", fontsize=8)
    ax.set_title("D  Per-patient FOG separability\n(higher = more consistent across patients)",
                 fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = OUT_DIR / "generalization_summary.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — UMAP: patient colour vs FOG colour (side by side)
# ═══════════════════════════════════════════════════════════════════════════

def plot_umap_patient_vs_fog():
    coords_path = EMBD_DIR / "coords_umap.npy"
    meta_path   = EMBD_DIR / "metadata.csv"

    if not coords_path.exists():
        print(f"UMAP coords not found at {coords_path} — skipping Figure 2")
        return

    coords = np.load(coords_path)
    meta   = pd.read_csv(meta_path)

    assert len(coords) == len(meta), "coords/meta length mismatch"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "MAE backbone (vit12 ep4) — UMAP of 512-d patch embeddings  (UMAP fit jointly across both cohorts)\n"
        "Defog (circles) + FogAtHome (triangles)",
        fontsize=11
    )

    # ── colour palette for patients ──────────────────────────────────────
    all_patients  = meta["patient_id"].unique()
    n_patients    = len(all_patients)
    base_colors   = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    patient_color = {pid: base_colors[i % len(base_colors)]
                     for i, pid in enumerate(all_patients)}

    dataset_marker = {"defog": "o", "fogathome": "^"}
    dataset_size   = {"defog": 6,   "fogathome": 18}
    dataset_alpha  = {"defog": 0.4, "fogathome": 0.85}

    # ── Panel A: coloured by patient ─────────────────────────────────────
    ax = axes[0]
    for ds in ["defog", "fogathome"]:
        mask = meta["dataset"] == ds
        pids = meta[mask]["patient_id"].values
        colors_arr = [patient_color[p] for p in pids]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors_arr,
                   marker=dataset_marker[ds],
                   s=dataset_size[ds],
                   alpha=dataset_alpha[ds],
                   linewidths=0)

    ax.set_title("A  Coloured by patient ID\n(each colour = one patient)", fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # legend: dataset shape only (too many patients for individual legend)
    legend_handles = [
        mpatches.Patch(facecolor="grey", label=f"Defog ({(meta['dataset']=='defog').sum()} patches, "
                                                f"{meta[meta['dataset']=='defog']['patient_id'].nunique()} patients)"),
        plt.scatter([], [], marker="^", color="grey", s=40,
                    label=f"FogAtHome ({(meta['dataset']=='fogathome').sum()} patches, "
                          f"{meta[meta['dataset']=='fogathome']['patient_id'].nunique()} patients)"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc="upper right",
              framealpha=0.8, markerscale=1)

    ax.text(0.02, 0.02,
            "Patient linear probe: 22.8× chance accuracy\n→ embeddings encode patient identity\n   (patch-level; 28 Defog + 12 FogAtHome classes)",
            transform=ax.transAxes, fontsize=9.0, va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8E1", edgecolor="#E6A817", linewidth=1.5, alpha=0.95))

    # ── Panel B: coloured by FOG label ───────────────────────────────────
    ax = axes[1]
    fog_palette = {0: "#aac4e0", 1: "#d65f5f"}   # light blue = no-FOG, red = FOG
    fog_size    = {0: 5, 1: 12}
    fog_alpha   = {0: 0.3, 1: 0.8}
    fog_zorder  = {0: 1, 1: 3}
    fog_label   = {0: "No-FOG", 1: "FOG"}

    for label in [0, 1]:
        for ds in ["defog", "fogathome"]:
            mask = (meta["true_label"] == label) & (meta["dataset"] == ds)
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=fog_palette[label],
                       marker=dataset_marker[ds],
                       s=fog_size[label] if ds == "defog" else fog_size[label] * 2.5,
                       alpha=fog_alpha[label],
                       linewidths=0,
                       zorder=fog_zorder[label])

    ax.set_title("B  Coloured by FOG label\n(red = FOG, blue = no-FOG)", fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    legend_handles = [
        mpatches.Patch(color=fog_palette[1], label="FOG (Defog ○ + FogAtHome △)"),
        mpatches.Patch(color=fog_palette[0], label="No-FOG"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", framealpha=0.8)

    ax.text(0.02, 0.02,
            "FOG centroid cosine similarity across cohorts: 0.987\n"
            "→ FOG region is cohort-aligned despite patient identity being encoded\n"
            "   (computed in 512-d space; UMAP shows qualitative structure only)",
            transform=ax.transAxes, fontsize=8.5, va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor=C_MAE, linewidth=1.5, alpha=0.95))

    fig.tight_layout()
    path = OUT_DIR / "umap_patient_vs_fog.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    plot_generalization_summary()
    plot_umap_patient_vs_fog()
    print(f"\nAll figures saved to {OUT_DIR}/")

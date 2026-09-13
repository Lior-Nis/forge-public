"""
Publication-grade embedding analysis figures.

Each figure has:
  - A declarative title that states the conclusion
  - On-figure annotations that point the reader to the finding
  - Consistent colour scheme across figures
  - Minimum chartjunk

Outputs to artifacts/plots/publication/:
  fig1_cross_cohort_transfer.png  — UMAP 2×2 (model × cohort), FOG centroid alignment
  fig2_per_patient_silhouette.png — Each patient: primary vs baseline FOG silhouette
  fig3_patient_invariance.png     — Distance distributions: same-patient vs cross-patient FOG
  fig4_per_patient_ap.png         — Per-patient probe AP, sorted, probe vs finetune
  fig5_fold_centroid_similarity.png — Train→test FOG centroid alignment across k-folds

Model-interchangeable: pass --primary-embed-dir for the winning SSL model.
The baseline (supervised from scratch) is passed as --baseline-embed-dir.

Run:
    uv run python scripts/analysis/plot_publication.py \\
        --primary-embed-dir logs/embeddings/vit12_ep7 \\
        --baseline-embed-dir logs/embeddings/supervised_scratch \\
        --probe-csv logs/pooled_ap/vit12_ep7_pooled.csv \\
        --finetune-csv logs/pooled_ap/finetune_vit12_ep7_pooled.csv \\
        --primary-label "MAE 2d_patch vit12 ep7"
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity


def parse_args():
    p = argparse.ArgumentParser(description="Generate embedding analysis figures.")
    p.add_argument("--primary-embed-dir", default="logs/embeddings/vit12_ep4",
                   help="Embeddings dir for the primary (winning) SSL model.")
    p.add_argument("--baseline-embed-dir", default="logs/embeddings/supervised_scratch",
                   help="Embeddings dir for the supervised-from-scratch baseline.")
    p.add_argument("--primary-label",   default="MAE pretrained (vit12 ep4 — geometry checkpoint)",
                   help="Label for the primary model in figure titles/legends.")
    p.add_argument("--baseline-label",  default="Supervised from scratch",
                   help="Label for the baseline model.")
    p.add_argument("--probe-csv",
                   default="logs/pooled_ap/vit12_ep7_pooled.csv",
                   help="Pooled predictions CSV for the probe model (fig4).")
    p.add_argument("--finetune-csv",
                   default="logs/pooled_ap/finetune_vit12_ep7_pooled.csv",
                   help="Pooled predictions CSV for the finetuned model (fig4).")
    p.add_argument("--out-dir", default="artifacts/plots/publication",
                   help="Output directory for figures.")
    p.add_argument("--figs", nargs="*", default=["1", "2", "3", "4", "5"],
                   help="Which figures to generate (1–5). Default: all.")
    return p.parse_args()


_args = None   # populated in __main__

OUT = Path("artifacts/plots/publication")
MAE_DIR = Path("logs/embeddings/vit12_ep4")
SUP_DIR = Path("logs/embeddings/supervised_scratch")

# ─── colour scheme ────────────────────────────────────────────────────────
C_MAE   = "#2E5EAA"
C_SUP   = "#C13333"
C_FOG   = "#D62728"
C_NOFOG = "#9DC8E0"
C_BG    = "#E5E5E5"
C_MAEFG = "#1A4480"  # darker MAE for centroid emphasis
C_SUPFG = "#8B1A1A"

DPI = 200

# ─── shared style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Cross-cohort FOG transfer (UMAP 2×2)
# ═══════════════════════════════════════════════════════════════════════════

def fig1_cross_cohort_transfer():
    """
    2×2 grid:
      Rows: MAE pretrained | Supervised from scratch
      Cols: Defog (training cohort) | FogAtHome (external cohort)
    Each panel shows the same model's UMAP, filtered to one cohort, coloured by FOG.
    The FOG centroid is shown as a star. Within a row, the two stars should be in
    the same place (good transfer) or different places (bad transfer).
    """
    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.10, left=0.10, right=0.98, top=0.88, bottom=0.07)

    mae_dir = Path(_args.primary_embed_dir)  if _args else MAE_DIR
    sup_dir = Path(_args.baseline_embed_dir) if _args else SUP_DIR
    mae_lbl = _args.primary_label            if _args else "MAE pretrained\n(vit12 ep4)"
    sup_lbl = _args.baseline_label           if _args else "Supervised from scratch"

    rows = [
        (mae_lbl, mae_dir, C_MAE, C_MAEFG),
        (sup_lbl, sup_dir, C_SUP, C_SUPFG),
    ]
    cols = [
        ("Training cohort — Defog (28 patients)",   "defog"),
        ("External test cohort — FogAtHome (12 patients)", "fogathome"),
    ]

    fig.suptitle(
        "MAE-pretrained backbone aligns FOG embeddings across cohorts;\n"
        "supervised model's FOG region drifts to unseen patients\n"
        "(qualitative UMAP fit jointly per model; cosine similarity computed in 512-d space)",
        fontsize=11.5, y=0.972, x=0.54,
    )

    for r, (model_label, model_dir, model_color, fg_color) in enumerate(rows):
        coords = np.load(model_dir / "coords_umap.npy")
        meta   = pd.read_csv(model_dir / "metadata.csv")
        emb    = np.load(model_dir / "embeddings.npy")

        # Per-cohort FOG centroids (in 512-d space, what really matters)
        defog_fog_mask = (meta["dataset"] == "defog")     & (meta["true_label"] == 1)
        fah_fog_mask   = (meta["dataset"] == "fogathome") & (meta["true_label"] == 1)
        c_defog = emb[defog_fog_mask].mean(0)
        c_fah   = emb[fah_fog_mask].mean(0)
        cos_sim_512 = float(cosine_similarity(c_defog[None], c_fah[None])[0, 0])

        # Same in UMAP (for the visual centroid stars)
        cs_defog_umap = coords[defog_fog_mask].mean(0)
        cs_fah_umap   = coords[fah_fog_mask].mean(0)

        # Range to share axis limits across both cohort panels in this row
        pad = 1.0
        x_min, x_max = coords[:, 0].min() - pad, coords[:, 0].max() + pad
        y_min, y_max = coords[:, 1].min() - pad, coords[:, 1].max() + pad

        for c, (cohort_label, cohort_key) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])

            # Background — all patches faded for shape context
            ax.scatter(coords[:, 0], coords[:, 1], s=2, c=C_BG, alpha=0.35, zorder=1, linewidths=0)

            # Foreground — this cohort
            cohort_mask = meta["dataset"] == cohort_key
            sub_meta    = meta.loc[cohort_mask]
            sub_coords  = coords[cohort_mask]

            no_fog = sub_meta["true_label"].values == 0
            fog    = sub_meta["true_label"].values == 1

            ax.scatter(sub_coords[no_fog, 0], sub_coords[no_fog, 1],
                       s=8, c=C_NOFOG, alpha=0.55, edgecolor="none", zorder=2, label=f"No-FOG (n={no_fog.sum()})")
            ax.scatter(sub_coords[fog, 0], sub_coords[fog, 1],
                       s=18, c=C_FOG, alpha=0.78, edgecolor="white", linewidth=0.3, zorder=3, label=f"FOG (n={fog.sum()})")

            # Diamond centroid for THIS cohort (filled)
            this_centroid_umap = cs_defog_umap if cohort_key == "defog" else cs_fah_umap
            this_label = "Defog FOG centroid" if cohort_key == "defog" else "FogAtHome FOG centroid"
            ax.scatter(*this_centroid_umap, marker="D", s=220,
                       c=fg_color, edgecolor="white", linewidth=1.5, zorder=10,
                       label=this_label)

            # Hollow diamond for THE OTHER cohort (so the reader sees alignment/misalignment)
            other_centroid_umap = cs_fah_umap if cohort_key == "defog" else cs_defog_umap
            other_label = "FogAtHome FOG centroid" if cohort_key == "defog" else "Defog FOG centroid"
            ax.scatter(*other_centroid_umap, marker="D", s=220, facecolor="none",
                       edgecolor=fg_color, linewidth=2.0, zorder=9,
                       label=other_label)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(cohort_label, fontsize=10.5, pad=8)
            if c == 0:
                ax.set_ylabel(model_label, fontsize=11.5, fontweight="bold", labelpad=10)
            ax.set_xlabel("UMAP-1", fontsize=8, color="grey", labelpad=2)
            if c == 0:
                ax.text(-0.06, 0.5, "UMAP-2", rotation=90, fontsize=8, color="grey",
                        transform=ax.transAxes, ha="center", va="center")

            ax.legend(loc="lower left", fontsize=7.5, markerscale=0.85,
                      handletextpad=0.4, borderpad=0.3, labelspacing=0.3,
                      framealpha=0.85, facecolor="white", edgecolor="grey",
                      frameon=True)

        # Row-level annotation — the centroid alignment fact
        # Place between the two columns of this row using fig coords
        ax_right = fig.add_subplot(gs[r, 1])
        ax_left  = fig.add_subplot(gs[r, 0])
        # Don't add extra axes — just annotate on the right panel.
        # Compute axis bbox for arrow.
        # We want the conclusion in a callout box at top-right of the row.
        verdict = ("FOG centroid aligns across cohorts"
                   if cos_sim_512 > 0.9 else
                   "FOG centroid drifts across cohorts")
        # Re-grab the axis we want to annotate (right column for this row)
        # Actually we created it twice — fix below:
        plt.delaxes(ax_right)
        plt.delaxes(ax_left)

    # Re-walk rows for the verdict callouts (cleaner)
    for r, (model_label, model_dir, model_color, fg_color) in enumerate(rows):
        emb  = np.load(model_dir / "embeddings.npy")
        meta = pd.read_csv(model_dir / "metadata.csv")
        defog_fog_mask = (meta["dataset"] == "defog")     & (meta["true_label"] == 1)
        fah_fog_mask   = (meta["dataset"] == "fogathome") & (meta["true_label"] == 1)
        c_defog = emb[defog_fog_mask].mean(0)
        c_fah   = emb[fah_fog_mask].mean(0)
        sim = float(cosine_similarity(c_defog[None], c_fah[None])[0, 0])

        good = sim > 0.9
        verdict = "✓  FOG centroid aligns across cohorts" if good else "✗  FOG centroid drifts across cohorts"
        col = "#2A6F2A" if good else "#A0322A"
        # Put verdict box on the right edge of each row
        y_anchor = 0.74 if r == 0 else 0.32
        fig.text(0.99, y_anchor,
                 f"{verdict}\nCosine similarity = {sim:.3f}",
                 fontsize=9.5, ha="right", va="center", color=col, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                           edgecolor=col, linewidth=1.2))

    out = OUT / "fig1_cross_cohort_transfer.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Per-patient FOG silhouette (paired scatter)
# ═══════════════════════════════════════════════════════════════════════════

def fig2_per_patient_silhouette():
    """
    Each patient is one point.
    x = silhouette in supervised-from-scratch embeddings
    y = silhouette in MAE pretrained embeddings
    Above the diagonal → MAE separates FOG from no-FOG better for that patient.
    """
    sil = pd.read_csv("logs/embeddings/generalization/per_patient_silhouette.csv")

    pivot = sil.pivot_table(
        index=["patient_id", "cohort"],
        columns="model",
        values="silhouette",
    ).reset_index()
    mae_col = [c for c in pivot.columns if "MAE" in str(c)][0]
    sup_col = [c for c in pivot.columns if "Supervised" in str(c)][0]

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    lim = (-0.20, 0.45)

    # Equal-line region shading
    xs = np.linspace(*lim, 100)
    ax.fill_between(xs, xs, lim[1], color=C_MAE, alpha=0.06, zorder=0)
    ax.fill_between(xs, lim[0], xs, color=C_SUP, alpha=0.06, zorder=0)
    ax.plot(lim, lim, "--", color="#444444", linewidth=1.8, zorder=1, alpha=0.85)

    # Quadrant labels in the shaded regions
    ax.text(-0.13, 0.38, "MAE separates\nFOG better here", color=C_MAE,
            fontsize=10, fontweight="bold", ha="left", va="top", alpha=0.9)
    ax.text(0.38, -0.13, "Supervised separates\nFOG better here", color=C_SUP,
            fontsize=10, fontweight="bold", ha="right", va="bottom", alpha=0.9)

    cohort_styles = {
        "defog":     dict(marker="o", s=85, label="Defog (training cohort)",   facecolor="#4878cf"),
        "fogathome": dict(marker="^", s=140, label="FogAtHome (external)",     facecolor="#FF8C42"),
    }

    for cohort, style in cohort_styles.items():
        sub = pivot[pivot["cohort"] == cohort]
        ax.scatter(sub[sup_col], sub[mae_col],
                   marker=style["marker"], s=style["s"],
                   facecolor=style["facecolor"], edgecolor="white", linewidth=1.0,
                   alpha=0.92, label=style["label"], zorder=3)

    # Counts
    above = (pivot[mae_col] > pivot[sup_col]).sum()
    n     = len(pivot)
    above_def = ((pivot["cohort"] == "defog") & (pivot[mae_col] > pivot[sup_col])).sum()
    n_def     = (pivot["cohort"] == "defog").sum()
    above_fah = ((pivot["cohort"] == "fogathome") & (pivot[mae_col] > pivot[sup_col])).sum()
    n_fah     = (pivot["cohort"] == "fogathome").sum()

    summary = (
        f"MAE wins for {above}/{n} patients overall\n"
        f"  • Defog (training):     {above_def}/{n_def}\n"
        f"  • FogAtHome (external): {above_fah}/{n_fah}\n"
        f"  (patients with only one FOG/no-FOG class excluded)"
    )
    ax.text(0.022, 0.978, summary,
            transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="grey", linewidth=0.8))

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5, zorder=0)
    ax.axvline(0, color="grey", linewidth=0.5, alpha=0.5, zorder=0)

    ax.set_xlabel("Per-patient FOG silhouette score\n(Supervised from scratch)", fontsize=10)
    ax.set_ylabel("Per-patient FOG silhouette score\n(MAE pretrained)", fontsize=10)
    ax.set_title("MAE improves FOG separability for a majority of patients\n"
                 "Each point = one patient; above diagonal = MAE separates FOG better for that patient\n"
                 "(silhouette: +1 = perfectly separable within patient, 0 = indistinguishable, −1 = reversed)",
                 fontsize=10.5)
    ax.grid(alpha=0.18, zorder=0)
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor="white", framealpha=0.9, edgecolor="grey")

    out = OUT / "fig2_per_patient_silhouette.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Patient invariance: same-patient vs cross-patient FOG distances
# ═══════════════════════════════════════════════════════════════════════════

def fig3_patient_invariance():
    """
    For each model, compute pairwise cosine distance between FOG patches.
    Split into "same-patient" and "different-patient" pairs.
    A patient-invariant FOG representation has these two distributions OVERLAP:
       distance(FOG_patient_A, FOG_patient_A) ≈ distance(FOG_patient_A, FOG_patient_B)
    Patient-dependent representations have them well-separated:
       same-patient distances tiny, cross-patient huge → identity dominates.
    """
    rng = np.random.RandomState(42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    fig.suptitle(
        "Patient identity strongly drives distances in supervised embeddings;\n"
        "MAE embeddings are substantially less patient-dependent for FOG",
        fontsize=11.5, y=1.01,
    )

    mae_dir = Path(_args.primary_embed_dir)  if _args else MAE_DIR
    sup_dir = Path(_args.baseline_embed_dir) if _args else SUP_DIR
    mae_lbl = (_args.primary_label  if _args else "MAE pretrained (vit12 ep4)").replace("\n", " ")
    sup_lbl = (_args.baseline_label if _args else "Supervised from scratch").replace("\n", " ")

    for ax, (label, model_dir, color) in zip(axes, [
        (mae_lbl, mae_dir, C_MAE),
        (sup_lbl, sup_dir, C_SUP),
    ]):
        emb  = np.load(model_dir / "embeddings.npy")
        meta = pd.read_csv(model_dir / "metadata.csv")

        fog_mask = meta["true_label"].values == 1
        fog_emb  = emb[fog_mask]
        fog_pid  = meta.loc[fog_mask, "patient_id"].values

        # Subsample for speed (to ~1000 FOG patches)
        n_max = 1200
        if len(fog_emb) > n_max:
            idx = rng.choice(len(fog_emb), n_max, replace=False)
            fog_emb = fog_emb[idx]
            fog_pid = fog_pid[idx]

        D = cosine_distances(fog_emb)
        triu = np.triu(np.ones_like(D, dtype=bool), k=1)
        same_pat = (fog_pid[:, None] == fog_pid[None, :]) & triu
        diff_pat = (fog_pid[:, None] != fog_pid[None, :]) & triu

        same_d = D[same_pat]
        diff_d = D[diff_pat]

        # Compute KDE-friendly bins
        bins = np.linspace(0, max(same_d.max(), diff_d.max()) * 1.05, 60)
        ax.hist(same_d, bins=bins, density=True, color="#2A8C5F", alpha=0.55,
                label=f"Same patient (FOG↔FOG, {len(same_d):,} patch pairs)", edgecolor="white", linewidth=0.3)
        ax.hist(diff_d, bins=bins, density=True, color="#9C5BD0", alpha=0.55,
                label=f"Different patients (FOG↔FOG, {len(diff_d):,} patch pairs)", edgecolor="white", linewidth=0.3)

        # Median markers
        m_same = np.median(same_d)
        m_diff = np.median(diff_d)
        ax.axvline(m_same, color="#1A5A3A", linestyle="--", linewidth=1.5, zorder=4)
        ax.axvline(m_diff, color="#5B2C92", linestyle="--", linewidth=1.5, zorder=4)

        # Gap arrow between medians
        y_arrow = ax.get_ylim()[1] * 0.78  # tentative
        gap = m_diff - m_same
        gap_pct = (gap / m_same) * 100 if m_same > 0 else float("inf")

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Cosine distance between FOG patches")
        if ax is axes[0]:
            ax.set_ylabel("Density")

        # Conclusion text on each panel.
        # Threshold on absolute cosine gap: small absolute gap → patient-invariant,
        # large absolute gap → patient identity drives distances more than FOG state.
        if gap < 0.05:
            verdict = (f"Median gap: +{gap:.3f} (cosine units)\n"
                       f"→ FOG looks similar regardless of patient")
            verdict_color = "#2A6F2A"
        else:
            verdict = (f"Median gap: +{gap:.3f} (cosine units)\n"
                       f"→ patient identity dominates over FOG state")
            verdict_color = "#A0322A"

        ax.text(0.98, 0.95, verdict,
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
                color=verdict_color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=verdict_color, linewidth=1.2))

        ax.legend(loc="upper left", fontsize=8.8)
        ax.grid(alpha=0.18)
        # Note: x-axis scales differ between panels by design (MAE ~0–0.14, Supervised ~0–0.9)
        ax.text(0.99, 0.01, "Note: x-axis scales differ between panels",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="grey", style="italic")

    fig.tight_layout()
    out = OUT / "fig3_patient_invariance.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Per-patient probe AP (sorted, paired)
# ═══════════════════════════════════════════════════════════════════════════

def fig4_per_patient_ap():
    """
    For each defog test patient, compute AP under:
      (a) MAE probe (frozen backbone, ep7)
      (b) MAE full finetune (unfrozen backbone, ep7)
    Sort patients by MAE probe AP and show paired bars.
    The point is: probe wins, AND the per-patient distribution is a richer story
    than the pooled-AP number alone.
    """
    probe_csv   = Path(_args.probe_csv)   if _args else Path("logs/pooled_ap/vit12_ep7_pooled.csv")
    finetune_csv = Path(_args.finetune_csv) if _args else Path("logs/pooled_ap/finetune_vit12_ep7_pooled.csv")
    df_probe = pd.read_csv(probe_csv)
    df_finet = pd.read_csv(finetune_csv)

    def per_patient_ap(df):
        out = {}
        for pid, g in df.groupby("patient_id"):
            if g["true_label"].nunique() < 2:
                continue
            out[pid] = (
                average_precision_score(g["true_label"], g["pred_prob_fog"]),
                int(g["true_label"].sum()),
                len(g),
            )
        return out

    probe = per_patient_ap(df_probe)
    finet = per_patient_ap(df_finet)
    common = sorted(set(probe) & set(finet), key=lambda p: probe[p][0])

    probe_aps  = [probe[p][0] for p in common]
    finet_aps  = [finet[p][0] for p in common]
    n_fog      = [probe[p][1] for p in common]   # same patches → same n_fog
    pooled_pr  = average_precision_score(df_probe["true_label"], df_probe["pred_prob_fog"])
    pooled_ft  = average_precision_score(df_finet["true_label"], df_finet["pred_prob_fog"])
    fog_rate   = df_probe["true_label"].mean()

    fig, ax = plt.subplots(figsize=(13, 6))

    x  = np.arange(len(common))
    w  = 0.42
    bars_probe = ax.bar(x - w/2, probe_aps, w, color="#4CAF50",
                        edgecolor="white", linewidth=0.6,
                        label=f"MAE probe (frozen, pooled AP={pooled_pr:.3f})", zorder=3)
    bars_finet = ax.bar(x + w/2, finet_aps, w, color="#FF7043",
                        edgecolor="white", linewidth=0.6,
                        label=f"MAE full finetune (pooled AP={pooled_ft:.3f})", zorder=3)

    # Mean lines (included in legend)
    mean_probe = np.mean(probe_aps)
    mean_finet = np.mean(finet_aps)
    ax.axhline(mean_probe, color="#2E7D32", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"MAE probe mean AP = {mean_probe:.3f}")
    ax.axhline(mean_finet, color="#D84315", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"Full finetune mean AP = {mean_finet:.3f}")

    # Chance baseline
    ax.axhline(fog_rate, color="grey", linestyle=":", linewidth=1.2,
               label=f"Chance baseline (FOG prevalence ≈{fog_rate:.0%})")

    win_count = sum(1 for p, f in zip(probe_aps, finet_aps) if p > f)

    short_pid = [p[:6] for p in common]
    ax.set_xticks(x)
    ax.set_xticklabels(short_pid, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Defog test patients (sorted by MAE probe AP)", fontsize=10)
    ax.set_ylabel("Per-patient Average Precision (AP)", fontsize=10)
    ax.set_title(
        "Frozen MAE probe outperforms full finetuning on most defog test patients\n"
        f"({win_count}/{len(common)} patients; per-patient view exposes within-cohort heterogeneity hidden by pooled AP)",
        fontsize=11,
    )

    # Inset count of FOG patches per patient (lets reader judge reliability of per-patient AP)
    ax2 = ax.twinx()
    ax2.bar(x, n_fog, width=0.92, color="grey", alpha=0.20, zorder=1)
    ax2.set_ylabel("FOG patches per patient\n(grey bars — reliability indicator)", fontsize=8, color="grey")
    ax2.tick_params(axis="y", labelsize=8, colors="grey")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("grey")
    ax2.set_ylim(0, max(n_fog) * 4)  # squash to bottom quarter of panel

    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    out = OUT / "fig4_per_patient_ap.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Train→test FOG centroid similarity across k-folds
# ═══════════════════════════════════════════════════════════════════════════

def fig5_fold_centroid_similarity():
    """
    5 rows × 2 cols UMAP grid showing train→test fold FOG centroid alignment.
      Rows: fold 0–4 (each fold is held out as the test set once)
      Cols: MAE pretrained | Supervised from scratch

    Each panel:
      - Grey: patches from training folds (background context)
      - Blue: no-FOG patches from the held-out test fold
      - Red:  FOG patches from the held-out test fold
      - Filled diamond: FOG centroid of training-fold patients (512-d cosine)
      - Hollow diamond: FOG centroid of test-fold patients
      Cosine similarity annotated — how well do the two centroids agree?
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    mae_dir = Path(_args.primary_embed_dir)  if _args else MAE_DIR
    sup_dir = Path(_args.baseline_embed_dir) if _args else SUP_DIR
    mae_lbl = _args.primary_label            if _args else "MAE pretrained\n(vit12 ep4)"
    sup_lbl = _args.baseline_label           if _args else "Supervised from scratch"

    models = [
        (mae_lbl, mae_dir, C_MAE, C_MAEFG),
        (sup_lbl, sup_dir, C_SUP, C_SUPFG),
    ]
    n_folds = 5

    fig, axes = plt.subplots(n_folds, 2, figsize=(10, 18),
                             gridspec_kw={"hspace": 0.06, "wspace": 0.06})

    fig.suptitle(
        "Train→test FOG centroid alignment across k-folds\n"
        "Each row: one fold held out as test; diamonds show FOG centroid of train vs test patients\n"
        "(UMAP fit jointly per model; cosine similarity computed in 512-d space)",
        fontsize=11, y=1.01,
    )

    for col, (model_label, model_dir, model_color, fg_color) in enumerate(models):
        coords = np.load(model_dir / "coords_umap.npy")
        emb    = np.load(model_dir / "embeddings.npy")
        meta   = pd.read_csv(model_dir / "metadata.csv")

        defog = meta["dataset"] == "defog"

        # Shared axis limits for this model's column
        defog_coords = coords[defog.values]
        pad = 0.8
        x_min = defog_coords[:, 0].min() - pad
        x_max = defog_coords[:, 0].max() + pad
        y_min = defog_coords[:, 1].min() - pad
        y_max = defog_coords[:, 1].max() + pad

        # Column header on the first row only
        axes[0, col].set_title(model_label, fontsize=11, fontweight="bold", pad=8,
                               color=model_color)

        for row, fold in enumerate(range(n_folds)):
            ax = axes[row, col]

            train_mask  = defog & (meta["fold"] != fold)
            test_mask   = defog & (meta["fold"] == fold)
            train_nofog = train_mask & (meta["true_label"] == 0)
            train_fog   = train_mask & (meta["true_label"] == 1)
            test_nofog  = test_mask  & (meta["true_label"] == 0)
            test_fog    = test_mask  & (meta["true_label"] == 1)

            # ── colour scheme ───────────────────────────────────────────────
            # Train set: cool blue family  (light = no-FOG, saturated = FOG)
            # Test set:  warm orange-red family (light = no-FOG, saturated = FOG)
            # Same hue temperature → which set; saturation → FOG status

            # Train no-FOG — very light cool blue
            ax.scatter(coords[train_nofog, 0], coords[train_nofog, 1],
                       s=3, c="#C9D6E8", alpha=0.35, linewidths=0, zorder=1)

            # Train FOG — saturated blue
            ax.scatter(coords[train_fog, 0], coords[train_fog, 1],
                       s=9, c="#3A6DB5", alpha=0.55, linewidths=0, zorder=2)

            # Test no-FOG — light orange
            ax.scatter(coords[test_nofog, 0], coords[test_nofog, 1],
                       s=5, c="#FFE0B2", alpha=0.55, linewidths=0, zorder=3)

            # Test FOG — saturated deep orange, most prominent
            ax.scatter(coords[test_fog, 0], coords[test_fog, 1],
                       s=18, c="#E65100", alpha=0.92, edgecolor="white",
                       linewidth=0.2, zorder=4)

            # Centroids in 512-d → project to UMAP for display
            c_train_umap = coords[train_fog.values].mean(0) if train_fog.any() else None
            c_test_umap  = coords[test_fog.values].mean(0)  if test_fog.any()  else None

            # Cosine similarity in 512-d
            if train_fog.any() and test_fog.any():
                c_train_emb = emb[train_fog.values].mean(0)
                c_test_emb  = emb[test_fog.values].mean(0)
                sim = float(cos_sim(c_train_emb[None], c_test_emb[None])[0, 0])
            else:
                sim = float("nan")

            if c_train_umap is not None:
                ax.scatter(*c_train_umap, marker="D", s=70, c=fg_color,
                           edgecolor="white", linewidth=1.2, zorder=10,
                           label="Train FOG centroid")
            if c_test_umap is not None:
                ax.scatter(*c_test_umap, marker="D", s=70, facecolor="none",
                           edgecolor=fg_color, linewidth=1.8, zorder=10,
                           label="Test FOG centroid")

            # Cosine similarity annotation
            good = sim > 0.95
            sim_color  = "#2A6F2A" if good else "#A0322A"
            sim_face   = "#F0FFF0" if good else "#FFF0F0"
            ax.text(0.97, 0.97, f"sim = {sim:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, fontweight="bold", color=sim_color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=sim_face,
                              edgecolor=sim_color, linewidth=1.0))

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_yticks([])

            # Row label on the left column only
            if col == 0:
                n_test_fog = int(test_fog.sum())
                ax.set_ylabel(f"Fold {fold}\n({n_test_fog} test FOG patches)",
                              fontsize=9, labelpad=6)

    # Shared legend — bottom of figure
    legend_elements = [
        mpatches.Patch(color="#C9D6E8", label="Train — no-FOG  (light blue)"),
        mpatches.Patch(color="#3A6DB5", label="Train — FOG  (saturated blue)"),
        mpatches.Patch(color="#FFE0B2", label="Test — no-FOG  (light orange)"),
        mpatches.Patch(color="#E65100", label="Test — FOG  (saturated orange)"),
        plt.scatter([], [], marker="D", s=55, c="grey",
                    label="Train FOG centroid (filled diamond)"),
        plt.scatter([], [], marker="D", s=55, facecolor="none",
                    edgecolor="grey", linewidth=1.5,
                    label="Test FOG centroid (hollow diamond)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=8.5, frameon=True, facecolor="white",
               edgecolor="grey", bbox_to_anchor=(0.5, -0.03))

    out = OUT / "fig5_fold_centroid_similarity.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    _args = parse_args()
    OUT = Path(_args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    figs = set(_args.figs)
    if "1" in figs:
        fig1_cross_cohort_transfer()
    if "2" in figs:
        fig2_per_patient_silhouette()
    if "3" in figs:
        fig3_patient_invariance()
    if "4" in figs:
        fig4_per_patient_ap()
    if "5" in figs:
        fig5_fold_centroid_similarity()
    print(f"\nAll figures saved to {OUT}/")

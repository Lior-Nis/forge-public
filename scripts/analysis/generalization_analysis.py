"""
Generalization analysis figures for the MAE vs supervised comparison.

Analysis 1: Cross-cohort FOG centroid similarity heatmap
  - Per patient, compute mean FOG embedding (centroid)
  - Show 40×40 cosine similarity matrix (28 defog + 12 FogAtHome)
  - MAE: FogAtHome FOG centroids land near defog FOG centroids
  - Supervised: FogAtHome centroids are far from defog

Analysis 2: Nearest-neighbor FOG precision@k
  - For each FogAtHome FOG patch, find k nearest neighbors in defog space
  - Measure fraction of those neighbors that are also FOG
  - Baseline = overall defog FOG rate (~17%)
  - MAE should stay well above baseline; supervised should drop toward it

Usage:
    uv run python scripts/generalization_analysis.py \
        --model-a logs/embeddings/vit12_ep4 \
        --model-b logs/embeddings/supervised_scratch \
        --label-a "MAE probe (vit12 ep4)" \
        --label-b "Supervised from scratch" \
        --output-dir logs/embeddings/generalization
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

MODEL_COLORS = ["#4878cf", "#d65f5f"]  # blue = MAE, red = supervised


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_model(emb_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(emb_dir / "embeddings.npy")
    metadata = pd.read_csv(emb_dir / "metadata.csv")
    return embeddings, metadata


# ─── Analysis 1: FOG centroid similarity heatmap ──────────────────────────────

def compute_fog_centroids(embeddings: np.ndarray, metadata: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Returns:
        centroids: [n_patients, embed_dim]
        patient_ids: list of patient IDs in row order
        cohorts: list of "defog" or "fogathome" per patient
    """
    fog_mask = metadata["true_label"] == 1
    fog_meta = metadata[fog_mask].copy()
    fog_embs = embeddings[fog_mask]

    centroids, patient_ids, cohorts = [], [], []
    # Sort: defog first, then fogathome
    for cohort in ["defog", "fogathome"]:
        for pid in sorted(fog_meta[fog_meta["dataset"] == cohort]["patient_id"].unique()):
            pmask = (fog_meta["patient_id"] == pid).values
            centroids.append(fog_embs[pmask].mean(axis=0))
            patient_ids.append(pid)
            cohorts.append(cohort)

    return np.array(centroids), patient_ids, cohorts


def plot_centroid_heatmap(models: list[dict], output_dir: Path):
    """Side-by-side cosine similarity heatmaps, one per model."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 8))
    if n == 1:
        axes = [axes]

    fig.suptitle("FOG centroid cosine similarity across patients\n"
                 "(rows/cols sorted: defog patients first, then FogAtHome)",
                 fontsize=12)

    ref_patient_ids = None  # align both heatmaps to same patient order

    for ax, model, color in zip(axes, models, MODEL_COLORS):
        centroids, patient_ids, cohorts = compute_fog_centroids(
            model["embeddings"], model["metadata"]
        )

        if ref_patient_ids is None:
            ref_patient_ids = patient_ids

        sim = cosine_similarity(centroids)  # [n_patients, n_patients]
        n_patients = len(patient_ids)
        n_defog = sum(1 for c in cohorts if c == "defog")

        im = ax.imshow(sim, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")

        # Tick labels — short IDs
        short_ids = [p[:6] for p in patient_ids]
        ax.set_xticks(range(n_patients))
        ax.set_yticks(range(n_patients))
        ax.set_xticklabels(short_ids, rotation=90, fontsize=6)
        ax.set_yticklabels(short_ids, fontsize=6)

        # Cohort separator line
        sep = n_defog - 0.5
        ax.axhline(sep, color="black", linewidth=1.5)
        ax.axvline(sep, color="black", linewidth=1.5)

        # Cohort labels
        ax.text(n_defog / 2, -1.8, "Defog (in-domain)", ha="center",
                fontsize=8, color="#333333")
        ax.text(n_defog + (n_patients - n_defog) / 2, -1.8, "FogAtHome (external)",
                ha="center", fontsize=8, color="#333333")
        ax.text(-1.8, n_defog / 2, "Defog", ha="right", va="center",
                fontsize=8, color="#333333", rotation=90)
        ax.text(-1.8, n_defog + (n_patients - n_defog) / 2, "FogAtHome",
                ha="right", va="center", fontsize=8, color="#333333", rotation=90)

        # Key region annotations
        ax.text(n_defog / 2, n_defog + (n_patients - n_defog) / 2,
                "cross-cohort\nFOG similarity", ha="center", va="center",
                fontsize=7, color="white", alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4))

        # Mean cross-cohort similarity
        cross = sim[:n_defog, n_defog:]
        mean_cross = cross.mean()
        ax.set_title(f"{model['label']}\nmean cross-cohort FOG similarity = {mean_cross:.3f}",
                     fontsize=10, pad=12)

    plt.tight_layout()
    out_path = output_dir / "fog_centroid_similarity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved centroid heatmap → {out_path}")

    # Summary CSV
    rows = []
    for model in models:
        centroids, patient_ids, cohorts = compute_fog_centroids(
            model["embeddings"], model["metadata"]
        )
        sim = cosine_similarity(centroids)
        n_defog = sum(1 for c in cohorts if c == "defog")
        cross = sim[:n_defog, n_defog:]
        within_defog = sim[:n_defog, :n_defog]
        within_fah = sim[n_defog:, n_defog:]
        np.fill_diagonal(within_defog, np.nan)
        np.fill_diagonal(within_fah, np.nan)
        rows.append({
            "model": model["label"],
            "mean_cross_cohort_sim": float(np.nanmean(cross)),
            "mean_within_defog_sim": float(np.nanmean(within_defog)),
            "mean_within_fogathome_sim": float(np.nanmean(within_fah)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "centroid_similarity_summary.csv", index=False)
    logger.info("\nCentroid similarity summary:\n" + df.to_string(index=False))
    return df


# ─── Analysis 2: Nearest-neighbor FOG precision@k ────────────────────────────

def nn_fog_precision(
    defog_embs: np.ndarray, defog_labels: np.ndarray,
    fah_embs: np.ndarray, fah_labels: np.ndarray,
    k_values: list[int],
) -> np.ndarray:
    """
    For each FogAtHome FOG patch, find k nearest defog neighbors (cosine),
    return fraction that are FOG. Averaged across all FogAtHome FOG patches.
    Returns array of shape [len(k_values)].
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    fah_fog_mask = fah_labels == 1
    query_embs = fah_embs[fah_fog_mask]

    if len(query_embs) == 0:
        return np.zeros(len(k_values))

    # Cosine similarity: [n_query, n_defog]
    sims = cos_sim(query_embs, defog_embs)  # higher = more similar

    max_k = max(k_values)
    # Top-k indices per query (sorted descending)
    top_k_idx = np.argsort(sims, axis=1)[:, ::-1][:, :max_k]

    precisions = []
    for k in k_values:
        neighbor_labels = defog_labels[top_k_idx[:, :k]]  # [n_query, k]
        precision = neighbor_labels.mean(axis=1).mean()   # avg FOG fraction
        precisions.append(float(precision))

    return np.array(precisions)


def plot_nn_precision(models: list[dict], output_dir: Path):
    k_values = [1, 2, 5, 10, 20, 50, 100, 200]

    fig, ax = plt.subplots(figsize=(8, 5))

    for model, color in zip(models, MODEL_COLORS):
        embs = model["embeddings"]
        meta = model["metadata"]

        defog_mask = meta["dataset"] == "defog"
        fah_mask = meta["dataset"] == "fogathome"

        defog_embs = embs[defog_mask.values]
        defog_labels = meta.loc[defog_mask, "true_label"].values
        fah_embs = embs[fah_mask.values]
        fah_labels = meta.loc[fah_mask, "true_label"].values

        prec = nn_fog_precision(defog_embs, defog_labels, fah_embs, fah_labels, k_values)

        ax.plot(k_values, prec, marker="o", color=color, label=model["label"],
                linewidth=2, markersize=5)

        # Annotate k=1 and k=100
        ax.annotate(f"{prec[0]:.2f}", (k_values[0], prec[0]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=color)
        ax.annotate(f"{prec[-1]:.2f}", (k_values[-1], prec[-1]),
                    textcoords="offset points", xytext=(-20, 6),
                    fontsize=8, color=color)

    # Baseline: overall FOG rate in defog
    defog_mask = models[0]["metadata"]["dataset"] == "defog"
    baseline = models[0]["metadata"].loc[defog_mask, "true_label"].mean()
    ax.axhline(baseline, color="grey", linestyle="--", linewidth=1.2,
               label=f"Defog FOG base rate ({baseline:.2f})")

    ax.set_xscale("log")
    ax.set_xlabel("k (number of nearest neighbors)", fontsize=11)
    ax.set_ylabel("Precision@k  (fraction of neighbors that are FOG)", fontsize=11)
    ax.set_title("FogAtHome FOG patches → nearest neighbors in defog embedding space\n"
                 "Higher = model places external FOG patches near in-domain FOG patches",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "nn_fog_precision.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved NN precision plot → {out_path}")

    # Save numbers
    rows = []
    for model in models:
        embs = model["embeddings"]
        meta = model["metadata"]
        defog_mask = meta["dataset"] == "defog"
        fah_mask = meta["dataset"] == "fogathome"
        prec = nn_fog_precision(
            embs[defog_mask.values], meta.loc[defog_mask, "true_label"].values,
            embs[fah_mask.values], meta.loc[fah_mask, "true_label"].values,
            k_values,
        )
        for k, p in zip(k_values, prec):
            rows.append({"model": model["label"], "k": k, "precision": p})
    pd.DataFrame(rows).to_csv(output_dir / "nn_precision_values.csv", index=False)


# ─── Analysis 3: Per-patient FOG/non-FOG silhouette bar chart ────────────────

def plot_per_patient_silhouette(models: list[dict], output_dir: Path):
    """
    For each patient independently, compute silhouette score of FOG vs non-FOG
    within that patient's own patches. Shows whether FOG is linearly separable
    per patient, and whether that separability is consistent across cohorts.
    """
    from sklearn.metrics import silhouette_score

    fig, axes = plt.subplots(1, len(models), figsize=(14, 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    all_rows = []

    for ax, model, color in zip(axes, models, MODEL_COLORS):
        embs = model["embeddings"]
        meta = model["metadata"]

        rows = []
        for cohort in ["defog", "fogathome"]:
            cmask = meta["dataset"] == cohort
            for pid in sorted(meta.loc[cmask, "patient_id"].unique()):
                pmask = (meta["patient_id"] == pid).values
                sub_embs = embs[pmask]
                sub_labels = meta.loc[pmask, "true_label"].values
                if len(np.unique(sub_labels)) < 2 or sub_labels.sum() < 2:
                    sil = None
                else:
                    if len(sub_embs) > 500:
                        rng = np.random.default_rng(42)
                        idx = rng.choice(len(sub_embs), 500, replace=False)
                        sub_embs, sub_labels = sub_embs[idx], sub_labels[idx]
                    try:
                        sil = float(silhouette_score(sub_embs, sub_labels))
                    except Exception:
                        sil = None
                rows.append({"patient_id": pid, "cohort": cohort,
                             "silhouette": sil, "model": model["label"]})

        df = pd.DataFrame(rows)
        all_rows.extend(rows)

        defog_df = df[df["cohort"] == "defog"].dropna(subset=["silhouette"])
        fah_df = df[df["cohort"] == "fogathome"].dropna(subset=["silhouette"])

        x_defog = np.arange(len(defog_df))
        x_fah = np.arange(len(defog_df), len(defog_df) + len(fah_df))

        ax.bar(x_defog, defog_df["silhouette"], color=color, alpha=0.8, label="Defog")
        ax.bar(x_fah, fah_df["silhouette"], color=color, alpha=0.4, label="FogAtHome",
               edgecolor=color, linewidth=0.8)
        ax.axvline(len(defog_df) - 0.5, color="black", linewidth=1.2, linestyle="--")
        ax.axhline(0, color="grey", linewidth=0.7)

        mean_def = defog_df["silhouette"].mean()
        mean_fah = fah_df["silhouette"].mean()
        ax.axhline(mean_def, color=color, linewidth=1.5, linestyle="-",
                   alpha=0.9, label=f"Defog mean = {mean_def:.3f}")
        ax.axhline(mean_fah, color=color, linewidth=1.5, linestyle=":",
                   alpha=0.9, label=f"FogAtHome mean = {mean_fah:.3f}")

        ax.set_title(model["label"], fontsize=10)
        ax.set_xlabel("Patient (left = defog, right = FogAtHome)", fontsize=9)
        ax.set_ylabel("Per-patient FOG silhouette score", fontsize=9)
        ax.set_xticks([])
        ax.legend(fontsize=7, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Per-patient FOG/non-FOG separability in embedding space\n"
                 "Higher = FOG and non-FOG are more linearly separable within that patient",
                 fontsize=11)
    plt.tight_layout()
    out_path = output_dir / "per_patient_silhouette.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved per-patient silhouette → {out_path}")
    pd.DataFrame(all_rows).to_csv(output_dir / "per_patient_silhouette.csv", index=False)


# ─── Analysis 4: FOG-only UMAP colored by patient ────────────────────────────

def plot_fog_only_umap(models: list[dict], output_dir: Path, n_cpus: int = 4):
    """
    UMAP on FOG patches only, colored by patient. If FOG from different patients
    shares a common structure (MAE), patches from different patients will intermix.
    If patient-specific (supervised), they cluster by patient.
    """
    import numba
    numba.set_num_threads(n_cpus)
    import umap as umap_lib

    def _distinct_palette(n):
        base = plt.cm.tab20.colors + plt.cm.tab20b.colors
        return [base[i % len(base)] for i in range(n)]

    fig, axes = plt.subplots(1, len(models), figsize=(9 * len(models), 7))
    if len(models) == 1:
        axes = [axes]

    fig.suptitle("UMAP of FOG patches only — colored by patient\n"
                 "Intermixed = model learned shared FOG structure across patients",
                 fontsize=12)

    for ax, model in zip(axes, models):
        embs = model["embeddings"]
        meta = model["metadata"]

        fog_mask = (meta["true_label"] == 1).values
        fog_embs = embs[fog_mask]
        fog_meta = meta[fog_mask].reset_index(drop=True)

        patients = sorted(fog_meta["patient_id"].unique())
        colors = {p: c for p, c in zip(patients, _distinct_palette(len(patients)))}
        cohort_marker = {"defog": "o", "fogathome": "^"}

        coords_path = output_dir / f"fog_only_umap_{model['label'].replace(' ', '_')}.npy"
        if coords_path.exists():
            coords = np.load(coords_path)
        else:
            reducer = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                    metric="cosine", random_state=42, low_memory=True,
                                    verbose=False)
            coords = reducer.fit_transform(fog_embs)
            np.save(coords_path, coords)

        for cohort in ["defog", "fogathome"]:
            cmask = (fog_meta["dataset"] == cohort).values
            for pid in sorted(fog_meta.loc[cmask, "patient_id"].unique()):
                pmask = (fog_meta["patient_id"] == pid).values & cmask
                ax.scatter(coords[pmask, 0], coords[pmask, 1],
                           c=[colors[pid]], marker=cohort_marker[cohort],
                           s=20 if cohort == "defog" else 40,
                           alpha=0.7, linewidths=0)

        # Legend: cohort markers only (too many patients for color legend)
        ax.scatter([], [], marker="o", c="grey", s=20, label="Defog (in-domain)")
        ax.scatter([], [], marker="^", c="grey", s=40, label="FogAtHome (external)")
        ax.legend(fontsize=9, loc="lower right")
        ax.set_title(model["label"], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    out_path = output_dir / "fog_only_umap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved FOG-only UMAP → {out_path}")


# ─── Analysis 5: Cohort overlap in PCA space ─────────────────────────────────

def plot_cohort_pca(models: list[dict], output_dir: Path):
    """
    PCA of all embeddings, scatter colored by cohort (defog vs FogAtHome).
    If defog and FogAtHome overlap in PC space → domain-agnostic representations.
    If they separate → backbone encodes domain-specific features.
    Two-panel: one per model.
    """
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 6))
    if len(models) == 1:
        axes = [axes]

    COHORT_COLORS = {"defog": "#4878cf", "fogathome": "#e07b39"}
    all_rows = []

    for ax, model in zip(axes, models):
        embs = model["embeddings"]
        meta = model["metadata"]

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embs)
        var = pca.explained_variance_ratio_

        for cohort, color in COHORT_COLORS.items():
            mask = (meta["dataset"] == cohort).values
            label = f"Defog (n={mask.sum()})" if cohort == "defog" else f"FogAtHome (n={mask.sum()})"
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=8,
                       alpha=0.4, linewidths=0, label=label)

        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=10)
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=10)
        ax.set_title(model["label"], fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)

        # Compute centroid distance between cohorts in PC space as summary metric
        def_center = coords[(meta["dataset"] == "defog").values].mean(axis=0)
        fah_center = coords[(meta["dataset"] == "fogathome").values].mean(axis=0)
        dist = float(np.linalg.norm(def_center - fah_center))
        ax.set_title(f"{model['label']}\ncohort centroid distance = {dist:.2f}", fontsize=10)
        all_rows.append({"model": model["label"], "cohort_pca_distance": dist,
                         "pc1_var": float(var[0]), "pc2_var": float(var[1])})

    fig.suptitle("PCA of all embeddings — colored by cohort\n"
                 "Overlap = backbone is cohort-agnostic; separation = domain-specific features",
                 fontsize=11)
    plt.tight_layout()
    out_path = output_dir / "cohort_pca.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved cohort PCA → {out_path}")
    pd.DataFrame(all_rows).to_csv(output_dir / "cohort_pca_summary.csv", index=False)
    logger.info("\nCohort PCA summary:\n" + pd.DataFrame(all_rows).to_string(index=False))


# ─── Analysis 6: Inter-patient FOG similarity distribution ───────────────────

def plot_inter_patient_fog_similarity(models: list[dict], output_dir: Path):
    """
    Histogram of pairwise cosine similarities between FOG patches from
    DIFFERENT patients (cross-patient FOG similarity distribution).
    MAE: high inter-patient similarity → shared FOG geometry.
    Supervised: low inter-patient similarity → patient-specific FOG features.
    Compared against non-FOG inter-patient similarity as a reference.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    fig.suptitle("Cross-patient cosine similarity distribution (FOG patches only)\n"
                 "Higher = FOG patches from different patients share the same embedding direction",
                 fontsize=11)

    rng = np.random.default_rng(42)
    N_SAMPLE = 300  # patches per class to keep computation tractable

    for ax, model in zip(axes, models):
        embs = model["embeddings"]
        meta = model["metadata"]

        # Work on defog only (all held-out, consistent patient count)
        defog_mask = (meta["dataset"] == "defog").values

        for label_val, label_name, color, ls in [
            (1, "FOG", "#d65f5f", "-"),
            (0, "Non-FOG", "#4878cf", "--"),
        ]:
            class_mask = defog_mask & (meta["true_label"] == label_val).values
            class_embs = embs[class_mask]
            class_patients = meta.loc[class_mask, "patient_id"].values

            # Sample for tractability
            if len(class_embs) > N_SAMPLE:
                idx = rng.choice(len(class_embs), N_SAMPLE, replace=False)
                class_embs = class_embs[idx]
                class_patients = class_patients[idx]

            sims = cos_sim(class_embs)  # [N, N]

            # Extract only cross-patient pairs (upper triangle, different patients)
            cross_sims = []
            n = len(class_embs)
            for i in range(n):
                for j in range(i + 1, n):
                    if class_patients[i] != class_patients[j]:
                        cross_sims.append(sims[i, j])

            cross_sims = np.array(cross_sims)
            ax.hist(cross_sims, bins=60, density=True, alpha=0.5, color=color,
                    linestyle=ls, label=f"{label_name} (mean={cross_sims.mean():.3f})",
                    edgecolor="none")
            ax.axvline(cross_sims.mean(), color=color, linewidth=1.5, linestyle=ls)

        ax.set_xlabel("Cosine similarity", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(model["label"], fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = output_dir / "inter_patient_fog_similarity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved inter-patient similarity → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--output-dir", default="logs/embeddings/generalization")
    parser.add_argument("--n-cpus", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dir_a, dir_b = Path(args.model_a), Path(args.model_b)
    label_a = args.label_a or dir_a.name
    label_b = args.label_b or dir_b.name

    embs_a, meta_a = load_model(dir_a)
    embs_b, meta_b = load_model(dir_b)

    models = [
        {"embeddings": embs_a, "metadata": meta_a, "label": label_a},
        {"embeddings": embs_b, "metadata": meta_b, "label": label_b},
    ]

    logger.info("Analysis 1: FOG centroid similarity heatmap…")
    plot_centroid_heatmap(models, out_dir)

    logger.info("Analysis 2: Nearest-neighbor FOG precision@k…")
    plot_nn_precision(models, out_dir)

    logger.info("Analysis 3: Per-patient FOG silhouette…")
    plot_per_patient_silhouette(models, out_dir)

    logger.info("Analysis 4: FOG-only UMAP…")
    plot_fog_only_umap(models, out_dir, n_cpus=args.n_cpus)

    logger.info("Analysis 5: Cohort PCA…")
    plot_cohort_pca(models, out_dir)

    logger.info("Analysis 6: Inter-patient FOG similarity distribution…")
    plot_inter_patient_fog_similarity(models, out_dir)


if __name__ == "__main__":
    main()

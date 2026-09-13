"""
Cross-model embedding comparison figure.

Produces a 2×2 grid:
  rows    = model (MAE probe / supervised)
  columns = cohort (defog in-domain / FogAtHome external)

Each panel shows:
  - Grey scatter: non-FOG patches (background)
  - Red scatter:  FOG patches
  - Filled KDE contour: FOG density (where FOG concentrates in embedding space)

The figure directly shows that the supervised model creates a tight, brittle FOG
cluster that transfers poorly to FogAtHome, while MAE creates a diffuse but
consistent FOG structure across cohorts.

Also computes FOG vs non-FOG silhouette score per cell as a quantitative summary
and saves a companion bar chart.

Usage:
    uv run python scripts/compare_embeddings.py \
        --model-a logs/embeddings/vit12_ep4 \
        --model-b logs/embeddings/true-terrain-236 \
        --label-a "MAE (vit12 ep4)" \
        --label-b "Supervised (true-terrain-236)" \
        --output-dir logs/embeddings/comparison \
        --n-cpus 4
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Thread limits before numba/sklearn imports
_n_cpus = 4
for i, arg in enumerate(sys.argv):
    if arg == "--n-cpus" and i + 1 < len(sys.argv):
        _n_cpus = int(sys.argv[i + 1])
os.environ.setdefault("NUMBA_NUM_THREADS", str(_n_cpus))
os.environ.setdefault("OMP_NUM_THREADS", str(_n_cpus))
os.environ.setdefault("MKL_NUM_THREADS", str(_n_cpus))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_n_cpus))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

NON_FOG_COLOR = "#cccccc"
FOG_COLOR = "#d65f5f"
FOG_SCATTER_COLOR = "#c0392b"
COHORT_KEYS = ["defog", "fogathome"]
COHORT_LABELS = {"defog": "Defog\n(in-domain, 28 patients)",
                 "fogathome": "FogAtHome\n(external cohort, 12 patients)"}


# ─── UMAP ────────────────────────────────────────────────────────────────────

def _reduce(embeddings: np.ndarray, n_cpus: int) -> np.ndarray:
    import numba
    numba.set_num_threads(n_cpus)
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric="cosine", random_state=42, low_memory=True, verbose=False)
    return reducer.fit_transform(embeddings)


def load_or_compute_coords(emb_dir: Path, n_cpus: int) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(emb_dir / "embeddings.npy")
    metadata = pd.read_csv(emb_dir / "metadata.csv")

    coords_path = emb_dir / "coords_umap.npy"
    if coords_path.exists():
        logger.info(f"  Using cached UMAP coords from {coords_path}")
        coords = np.load(coords_path)
    else:
        logger.info(f"  Running UMAP on {len(embeddings)} patches…")
        coords = _reduce(embeddings, n_cpus)
        np.save(coords_path, coords)

    return coords, metadata


# ─── KDE contour helper ───────────────────────────────────────────────────────

def _fog_kde_contour(ax, fog_coords: np.ndarray, fill: bool = True,
                     levels: int = 5, alpha_fill: float = 0.35, alpha_line: float = 0.8):
    """Overlay a KDE density contour for FOG patches. Skips if too few points."""
    if len(fog_coords) < 10:
        return

    x, y = fog_coords[:, 0], fog_coords[:, 1]
    xmin, xmax = x.min() - 0.5, x.max() + 0.5
    ymin, ymax = y.min() - 0.5, y.max() + 0.5

    xi = np.linspace(xmin, xmax, 120)
    yi = np.linspace(ymin, ymax, 120)
    Xi, Yi = np.meshgrid(xi, yi)

    try:
        kde = gaussian_kde(np.vstack([x, y]), bw_method="scott")
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    except np.linalg.LinAlgError:
        return  # singular covariance — too few unique points

    # Filled contour in red tones
    fog_cmap = mcolors.LinearSegmentedColormap.from_list(
        "fog_density", ["#ffffff00", FOG_COLOR], N=256
    )
    if fill:
        ax.contourf(Xi, Yi, Zi, levels=levels, cmap=fog_cmap, alpha=alpha_fill)
    ax.contour(Xi, Yi, Zi, levels=levels, colors=FOG_SCATTER_COLOR,
               alpha=alpha_line, linewidths=0.7)


# ─── Silhouette score ─────────────────────────────────────────────────────────

def _silhouette(coords: np.ndarray, labels: np.ndarray) -> float | None:
    """FOG vs non-FOG silhouette on 2D UMAP coords. Returns None if only one class."""
    if len(np.unique(labels)) < 2 or len(coords) < 4:
        return None
    # Subsample for speed if large
    if len(coords) > 3000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(coords), 3000, replace=False)
        coords, labels = coords[idx], labels[idx]
    return float(silhouette_score(coords, labels, metric="euclidean"))


# ─── Main figure ─────────────────────────────────────────────────────────────

def make_comparison_figure(
    models: list[dict],   # [{"coords": ..., "meta": ..., "label": ...}, ...]
    output_dir: Path,
    title: str = "",
):
    n_models = len(models)
    n_cohorts = len(COHORT_KEYS)

    fig, axes = plt.subplots(
        n_models, n_cohorts,
        figsize=(6 * n_cohorts, 5.5 * n_models),
    )
    if n_models == 1:
        axes = axes[np.newaxis, :]

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    silhouette_records = []

    for row, model in enumerate(models):
        coords = model["coords"]
        meta = model["meta"]
        model_label = model["label"]

        for col, cohort in enumerate(COHORT_KEYS):
            ax = axes[row, col]
            mask = (meta["dataset"] == cohort).values

            if not mask.any():
                ax.set_visible(False)
                continue

            sub_coords = coords[mask]
            sub_labels = meta.loc[mask, "true_label"].values
            fog_mask = sub_labels == 1
            nonfog_mask = sub_labels == 0

            n_fog = fog_mask.sum()
            n_total = len(sub_labels)
            fog_pct = 100 * n_fog / n_total

            # Background: non-FOG grey scatter
            ax.scatter(
                sub_coords[nonfog_mask, 0], sub_coords[nonfog_mask, 1],
                c=NON_FOG_COLOR, s=8, alpha=0.4, linewidths=0, rasterized=True,
                label="No FOG",
            )

            # Foreground: FOG red scatter
            ax.scatter(
                sub_coords[fog_mask, 0], sub_coords[fog_mask, 1],
                c=FOG_SCATTER_COLOR, s=14, alpha=0.7, linewidths=0, rasterized=True,
                label="FOG",
            )

            # KDE density contour over FOG patches
            _fog_kde_contour(ax, sub_coords[fog_mask])

            # Silhouette score
            sil = _silhouette(sub_coords, sub_labels)
            sil_text = f"silhouette = {sil:.3f}" if sil is not None else ""
            silhouette_records.append({
                "model": model_label, "cohort": cohort,
                "silhouette": sil, "n_fog": int(n_fog), "n_total": int(n_total),
            })

            # Labels and annotations
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("UMAP 1", fontsize=8)
            ax.set_ylabel("UMAP 2", fontsize=8)

            cohort_title = COHORT_LABELS.get(cohort, cohort)
            ax.set_title(
                f"{cohort_title}\n{n_fog} FOG / {n_total} total ({fog_pct:.0f}%)",
                fontsize=9,
            )
            if sil_text:
                ax.text(0.02, 0.97, sil_text, transform=ax.transAxes,
                        fontsize=8, va="top", color="#333333",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

            if col == 0:
                ax.set_ylabel(f"{model_label}\n\nUMAP 2", fontsize=9)

        # Row legend (rightmost panel)
        axes[row, -1].legend(
            handles=[
                plt.scatter([], [], c=NON_FOG_COLOR, s=20, label="No FOG"),
                plt.scatter([], [], c=FOG_SCATTER_COLOR, s=20, label="FOG"),
            ],
            fontsize=8, loc="lower right", framealpha=0.8,
        )

    plt.tight_layout()
    out_path = output_dir / "fog_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison figure → {out_path}")
    return pd.DataFrame(silhouette_records)


# ─── Silhouette bar chart ─────────────────────────────────────────────────────

def make_silhouette_chart(sil_df: pd.DataFrame, output_dir: Path):
    models = sil_df["model"].unique()
    cohorts = list(COHORT_KEYS)
    cohort_short = {"defog": "Defog\n(in-domain)", "fogathome": "FogAtHome\n(external)"}

    x = np.arange(len(cohorts))
    width = 0.35
    colors = ["#4878cf", "#d65f5f"]

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [
            sil_df.loc[(sil_df["model"] == model) & (sil_df["cohort"] == c), "silhouette"].values
            for c in cohorts
        ]
        vals = [v[0] if len(v) > 0 and v[0] is not None else 0.0 for v in vals]
        bars = ax.bar(x + i * width - width / 2, vals, width * 0.9,
                      label=model, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([cohort_short.get(c, c) for c in cohorts], fontsize=10)
    ax.set_ylabel("FOG / non-FOG silhouette score", fontsize=10)
    ax.set_title("FOG cluster quality: in-domain vs external cohort", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = output_dir / "silhouette_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved silhouette chart → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True, help="Embeddings dir for model A")
    parser.add_argument("--model-b", required=True, help="Embeddings dir for model B")
    parser.add_argument("--label-a", default=None, help="Display label for model A")
    parser.add_argument("--label-b", default=None, help="Display label for model B")
    parser.add_argument("--output-dir", default="logs/embeddings/comparison")
    parser.add_argument("--n-cpus", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dir_a = Path(args.model_a)
    dir_b = Path(args.model_b)
    label_a = args.label_a or dir_a.name
    label_b = args.label_b or dir_b.name

    logger.info(f"Loading {label_a}…")
    coords_a, meta_a = load_or_compute_coords(dir_a, args.n_cpus)

    logger.info(f"Loading {label_b}…")
    coords_b, meta_b = load_or_compute_coords(dir_b, args.n_cpus)

    models = [
        {"coords": coords_a, "meta": meta_a, "label": label_a},
        {"coords": coords_b, "meta": meta_b, "label": label_b},
    ]

    logger.info("Building comparison figure…")
    sil_df = make_comparison_figure(models, out_dir)

    sil_df.to_csv(out_dir / "silhouette_scores.csv", index=False)
    logger.info("\nSilhouette scores:\n" + sil_df.to_string(index=False))

    make_silhouette_chart(sil_df, out_dir)


if __name__ == "__main__":
    main()

"""
Embedding analysis for patient generalizability paper.

Analysis 1: UMAP/t-SNE dual-coloring (patient ID vs FOG label)
  - Two side-by-side panels on the same 2D projection
  - Defog and FogAtHome patients shown with distinct marker shapes
  - Saves publication-quality figure

Analysis 2: Patient identity linear probe
  - Can a linear classifier predict patient_id from embeddings?
  - Low accuracy → embeddings don't encode patient identity → generalizable
  - 5-fold stratified CV on defog; separate eval on FogAtHome (zero-shot)
  - Reports accuracy, chance level, and confusion heatmap

Outputs saved to --output-dir:
  embedding_analysis.png       (dual t-SNE/UMAP panels)
  patient_probe_results.csv    (per-patient prediction accuracy)
  patient_probe_confusion.png  (confusion matrix heatmap)
  analysis_summary.txt         (key numbers for paper)

Usage:
    uv run python scripts/analyze_embeddings.py \
        --embeddings-dir logs/embeddings/vit12_ep4 \
        --model-name vit12_ep4
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Thread limits must be set before numba/sklearn/openblas import ────────────
# Parse --n-cpus early so the env vars are in place before heavy imports.
_n_cpus = 4  # default
for i, arg in enumerate(sys.argv):
    if arg == "--n-cpus" and i + 1 < len(sys.argv):
        _n_cpus = int(sys.argv[i + 1])
os.environ.setdefault("NUMBA_NUM_THREADS", str(_n_cpus))
os.environ.setdefault("OMP_NUM_THREADS", str(_n_cpus))
os.environ.setdefault("MKL_NUM_THREADS", str(_n_cpus))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_n_cpus))
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ─── Colour palettes ─────────────────────────────────────────────────────────

# Diverging palette for FOG vs non-FOG (colourblind-safe)
FOG_COLORS = {0: "#4878cf", 1: "#d65f5f"}   # blue = no-FOG, red = FOG
DATASET_MARKERS = {"defog": "o", "fogathome": "^"}
DATASET_ALPHA = {"defog": 0.55, "fogathome": 0.85}


def _distinct_palette(n: int) -> list[str]:
    """Return n visually distinct colours from tab20 + tab20b."""
    base = plt.cm.tab20.colors + plt.cm.tab20b.colors
    return [base[i % len(base)] for i in range(n)]


# ─── Dimensionality reduction ─────────────────────────────────────────────────

def _reduce(embeddings: np.ndarray, method: str = "umap") -> np.ndarray:
    """Project [N, D] → [N, 2]."""
    if method == "umap":
        try:
            import numba
            numba.set_num_threads(int(os.environ.get("NUMBA_NUM_THREADS", 4)))
            import umap  # noqa
            reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                                metric="cosine", random_state=42, low_memory=True,
                                verbose=False)
            return reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("umap-learn not installed, falling back to t-SNE")
            method = "tsne"

    from sklearn.manifold import TSNE
    # PCA init for reproducibility on large datasets
    from sklearn.decomposition import PCA
    init = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    reducer = TSNE(n_components=2, perplexity=min(50, max(5, len(embeddings) // 10)),
                   init=init, random_state=42, n_iter=1000)
    return reducer.fit_transform(embeddings)


# ─── Analysis 1: Dual embedding plot ─────────────────────────────────────────

def _scatter_panel(ax, coords, meta, mode, pat_colors, method_name, title_suffix=""):
    """Fill a single axes with either patient-coloured or FOG-coloured scatter."""
    if mode == "patient":
        colors = [pat_colors[p] for p in meta["patient_id"]]
        title = f"Coloured by patient ID{title_suffix}"
    else:
        colors = [FOG_COLORS[int(l)] for l in meta["true_label"]]
        title = f"Coloured by FOG label{title_suffix}"
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=14,
               alpha=0.6, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"{method_name} 1", fontsize=8)
    ax.set_ylabel(f"{method_name} 2", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_embeddings(coords: np.ndarray, metadata: pd.DataFrame,
                    model_name: str, output_dir: Path, method_name: str):
    """
    2×2 figure: rows = cohort (defog / FogAtHome), columns = colouring (patient / FOG).
    All four panels share the same UMAP/t-SNE coordinates.
    """
    datasets = [d for d in ["defog", "fogathome"] if (metadata["dataset"] == d).any()]

    # Build a single palette covering all patients (both cohorts)
    patients = sorted(metadata["patient_id"].unique())
    pat_colors = {p: c for p, c in zip(patients, _distinct_palette(len(patients)))}

    n_rows = len(datasets)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # keep 2-D indexing
    fig.suptitle(f"{model_name} — {method_name} embeddings", fontsize=13)

    row_labels = {"defog": "Defog (in-domain, n=28 patients)",
                  "fogathome": "FogAtHome (external cohort, n=12 patients)"}

    for row, dataset in enumerate(datasets):
        mask = (metadata["dataset"] == dataset).values
        sub_coords = coords[mask]
        sub_meta = metadata[mask].reset_index(drop=True)
        n_pat = sub_meta["patient_id"].nunique()
        n_pts = len(sub_meta)

        for col, mode in enumerate(["patient", "fog"]):
            ax = axes[row, col]
            _scatter_panel(ax, sub_coords, sub_meta, mode, pat_colors, method_name)

        # Row label on the left axis
        axes[row, 0].set_ylabel(
            f"{row_labels.get(dataset, dataset)}\n({n_pts} patches, {n_pat} patients)\n\n{method_name} 2",
            fontsize=8,
        )

        # Per-row patient legend (right of col-0 panel, compact)
        row_patients = sorted(sub_meta["patient_id"].unique())
        pat_handles = [mpatches.Patch(color=pat_colors[p], label=p) for p in row_patients]
        axes[row, 0].legend(
            handles=pat_handles, fontsize=6, ncol=2,
            loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
            framealpha=0.8, title="Patient", title_fontsize=7,
        )

    # FOG legend on col-1 of last row
    fog_handles = [
        mpatches.Patch(color=FOG_COLORS[0], label="No FOG"),
        mpatches.Patch(color=FOG_COLORS[1], label="FOG"),
    ]
    axes[-1, 1].legend(handles=fog_handles, fontsize=9,
                       loc="upper left", bbox_to_anchor=(1.01, 1),
                       borderaxespad=0, framealpha=0.8)

    plt.tight_layout()
    out_path = output_dir / f"embedding_{method_name.lower()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved embedding plot → {out_path}")


# ─── Analysis 2: Patient identity linear probe ────────────────────────────────

def patient_identity_probe(embeddings: np.ndarray, metadata: pd.DataFrame,
                            output_dir: Path, model_name: str) -> dict:
    """
    Train a linear classifier to predict patient_id from embeddings.
    Low accuracy = embeddings don't encode identity = generalisable.

    Returns dict of summary metrics.
    """
    results = {}

    # ── Defog: 5-fold cross-validated accuracy ──
    defog_mask = metadata["dataset"] == "defog"
    X_def = embeddings[defog_mask.values]
    y_def_raw = metadata.loc[defog_mask, "patient_id"].values
    le = LabelEncoder().fit(y_def_raw)
    y_def = le.transform(y_def_raw)
    n_patients_def = len(le.classes_)
    chance_def = 1.0 / n_patients_def

    scaler = StandardScaler()
    X_def_sc = scaler.fit_transform(X_def)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    all_true, all_pred = [], []

    for train_idx, test_idx in skf.split(X_def_sc, y_def):
        clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial",
                                 solver="lbfgs", random_state=42)
        clf.fit(X_def_sc[train_idx], y_def[train_idx])
        pred = clf.predict(X_def_sc[test_idx])
        fold_accs.append(accuracy_score(y_def[test_idx], pred))
        all_true.extend(y_def[test_idx].tolist())
        all_pred.extend(pred.tolist())

    defog_acc = float(np.mean(fold_accs))
    defog_acc_std = float(np.std(fold_accs))
    results["defog_patient_acc"] = defog_acc
    results["defog_patient_acc_std"] = defog_acc_std
    results["defog_chance"] = chance_def
    results["defog_n_patients"] = n_patients_def
    results["defog_n_patches"] = int(defog_mask.sum())

    logger.info(
        f"Defog patient probe: acc={defog_acc:.3f} ± {defog_acc_std:.3f}  "
        f"(chance={chance_def:.3f}, {n_patients_def} patients, "
        f"ratio={defog_acc/chance_def:.1f}× chance)"
    )

    # Confusion matrix (defog, pooled across CV folds)
    cm = confusion_matrix(all_true, all_pred, labels=list(range(n_patients_def)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    _plot_confusion(cm_norm, le.classes_, model_name, output_dir, "defog")

    # ── Per-patient recall on defog ──
    per_patient = []
    for i, pat in enumerate(le.classes_):
        true_i = np.array(all_true) == i
        pred_i = np.array(all_pred) == i
        recall = pred_i[true_i].mean() if true_i.sum() > 0 else 0.0
        n_patches = int(true_i.sum())
        per_patient.append({"patient_id": pat, "dataset": "defog",
                             "recall": float(recall), "n_patches": n_patches})

    # ── FogAtHome: zero-shot patient probe ──
    fah_mask = metadata["dataset"] == "fogathome"
    if fah_mask.any():
        X_fah = embeddings[fah_mask.values]
        y_fah_raw = metadata.loc[fah_mask, "patient_id"].values
        le_fah = LabelEncoder().fit(y_fah_raw)
        y_fah = le_fah.transform(y_fah_raw)
        n_patients_fah = len(le_fah.classes_)
        chance_fah = 1.0 / n_patients_fah

        X_fah_sc = scaler.transform(X_fah)  # same scaler fitted on defog

        # Train on ALL defog, test on FogAtHome — measures cross-cohort patient leakage
        # (If acc ≈ chance, backbone hasn't memorised any patient-specific signal)
        # Note: label spaces differ so we can only measure "confusion" qualitatively;
        # instead we train a fresh classifier on FogAtHome CV.
        skf_fah = StratifiedKFold(n_splits=min(5, n_patients_fah), shuffle=True, random_state=42)
        fah_accs = []
        fah_true, fah_pred = [], []
        for train_idx, test_idx in skf_fah.split(X_fah_sc, y_fah):
            clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial",
                                     solver="lbfgs", random_state=42)
            clf.fit(X_fah_sc[train_idx], y_fah[train_idx])
            pred = clf.predict(X_fah_sc[test_idx])
            fah_accs.append(accuracy_score(y_fah[test_idx], pred))
            fah_true.extend(y_fah[test_idx].tolist())
            fah_pred.extend(pred.tolist())

        fah_acc = float(np.mean(fah_accs))
        fah_acc_std = float(np.std(fah_accs))
        results["fogathome_patient_acc"] = fah_acc
        results["fogathome_patient_acc_std"] = fah_acc_std
        results["fogathome_chance"] = chance_fah
        results["fogathome_n_patients"] = n_patients_fah
        results["fogathome_n_patches"] = int(fah_mask.sum())

        logger.info(
            f"FogAtHome patient probe: acc={fah_acc:.3f} ± {fah_acc_std:.3f}  "
            f"(chance={chance_fah:.3f}, {n_patients_fah} patients, "
            f"ratio={fah_acc/chance_fah:.1f}× chance)"
        )

        cm_fah = confusion_matrix(fah_true, fah_pred, labels=list(range(n_patients_fah)))
        cm_fah_norm = cm_fah.astype(float) / (cm_fah.sum(axis=1, keepdims=True) + 1e-9)
        _plot_confusion(cm_fah_norm, le_fah.classes_, model_name, output_dir, "fogathome")

        for i, pat in enumerate(le_fah.classes_):
            true_i = np.array(fah_true) == i
            pred_i = np.array(fah_pred) == i
            recall = pred_i[true_i].mean() if true_i.sum() > 0 else 0.0
            n_patches = int(true_i.sum())
            per_patient.append({"patient_id": pat, "dataset": "fogathome",
                                 "recall": float(recall), "n_patches": n_patches})

    pd.DataFrame(per_patient).to_csv(output_dir / "patient_probe_per_patient.csv", index=False)
    return results


def _plot_confusion(cm_norm: np.ndarray, labels: np.ndarray, model_name: str,
                    output_dir: Path, suffix: str):
    n = len(labels)
    fig_size = max(8, n * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted patient")
    ax.set_ylabel("True patient")
    ax.set_title(f"{model_name} — patient identity confusion ({suffix})\n"
                 f"diagonal = recall per patient  (ideal = uniform = 1/n)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = output_dir / f"patient_probe_confusion_{suffix}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix → {out_path}")


# ─── Summary text ─────────────────────────────────────────────────────────────

def write_summary(results: dict, model_name: str, method: str, output_dir: Path):
    lines = [
        f"Embedding analysis summary — {model_name}",
        "=" * 60,
        "",
        f"Dimensionality reduction: {method}",
        "",
        "Patient identity probe (linear logistic regression on 512-d embeddings):",
        "",
        "  Defog (in-domain, 5-fold CV):",
        f"    Accuracy: {results.get('defog_patient_acc', 0):.3f} ± {results.get('defog_patient_acc_std', 0):.3f}",
        f"    Chance:   {results.get('defog_chance', 0):.3f}  ({results.get('defog_n_patients', 0)} patients)",
        f"    Ratio:    {results.get('defog_patient_acc', 0) / max(results.get('defog_chance', 1), 1e-9):.1f}× chance",
        f"    Patches:  {results.get('defog_n_patches', 0)}",
        "",
    ]
    if "fogathome_patient_acc" in results:
        lines += [
            "  FogAtHome (external cohort, 5-fold CV within cohort):",
            f"    Accuracy: {results['fogathome_patient_acc']:.3f} ± {results['fogathome_patient_acc_std']:.3f}",
            f"    Chance:   {results['fogathome_chance']:.3f}  ({results['fogathome_n_patients']} patients)",
            f"    Ratio:    {results['fogathome_patient_acc'] / max(results['fogathome_chance'], 1e-9):.1f}× chance",
            f"    Patches:  {results.get('fogathome_n_patches', 0)}",
            "",
        ]
    lines += [
        "Interpretation:",
        "  Ratio ≈ 1× chance → embeddings carry no patient identity signal",
        "  Ratio > 3×        → moderate patient-specific signal",
        "  Ratio > 10×       → strong patient-specific encoding (bad for generalization)",
        "",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    (output_dir / "analysis_summary.txt").write_text(text)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True,
                        help="Directory containing embeddings.npy and metadata.csv")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--method", default="umap", choices=["umap", "tsne"],
                        help="Dimensionality reduction method")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save figures (defaults to embeddings-dir)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip patient probe; only regenerate the embedding plot")
    parser.add_argument("--n-cpus", type=int, default=4,
                        help="Max CPU threads for UMAP/numba/sklearn (default: 4)")
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)
    out_dir = Path(args.output_dir) if args.output_dir else emb_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading embeddings from {emb_dir}")
    embeddings = np.load(emb_dir / "embeddings.npy")
    metadata = pd.read_csv(emb_dir / "metadata.csv")
    logger.info(f"  {len(embeddings)} patches, {embeddings.shape[1]}-d, "
                f"{metadata['patient_id'].nunique()} patients, "
                f"pos_rate={metadata['true_label'].mean():.3f}")

    # Analysis 1: Embedding visualisation (cache coords to avoid re-running UMAP)
    coords_path = out_dir / f"coords_{args.method}.npy"
    if coords_path.exists():
        logger.info(f"Loading cached {args.method.upper()} coords from {coords_path}")
        coords = np.load(coords_path)
    else:
        logger.info(f"Running {args.method.upper()} on {len(embeddings)} patches…")
        coords = _reduce(embeddings, method=args.method)
        np.save(coords_path, coords)
    plot_embeddings(coords, metadata, args.model_name, out_dir, args.method.upper())

    if not args.plot_only:
        # Analysis 2: Patient identity probe
        logger.info("Running patient identity probe…")
        probe_results = patient_identity_probe(embeddings, metadata, out_dir, args.model_name)
        pd.DataFrame([probe_results]).to_csv(out_dir / "patient_probe_results.csv", index=False)
        write_summary(probe_results, args.model_name, args.method.upper(), out_dir)


if __name__ == "__main__":
    main()

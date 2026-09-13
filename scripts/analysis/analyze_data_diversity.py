"""Analyze data diversity to test if raw accelerometer patches are too similar
for contrastive learning to work.

Tests:
1. Pairwise cosine similarity distribution (random pairs)
2. Within-patient vs between-patient similarity
3. PCA variance explained (how many dimensions actually matter?)
4. t-SNE visualization colored by patient
5. Nearest-neighbor uniqueness (what % of k-NN are from the same patient?)
"""

import numpy as np
import zarr
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

ZARR_PATH = Path("data/processed/len1000_stride200_kaggle_daily_unlabeled.zarr")
N_SAMPLES = 5000  # subsample for tractability
SEED = 42


def load_subsample(zarr_path, n_samples, seed):
    """Load a random subsample of patches."""
    ds = zarr.open(str(zarr_path), mode='r')
    total = ds['accs'].shape[0]
    rng = np.random.RandomState(seed)
    indices = np.sort(rng.choice(total, size=min(n_samples, total), replace=False))

    acc = ds['accs'][indices]  # [N, T, C]
    acc = acc.transpose(0, 2, 1)  # -> [N, C, T]
    metadata = {k: np.array(ds['metadata'][k][indices]) for k in ds['metadata'].keys()}
    return acc, metadata, indices


def analyze_similarity(acc):
    """Analyze pairwise similarity distribution."""
    N, C, T = acc.shape
    flat = acc.reshape(N, -1)  # [N, C*T]

    # Subsample pairs for speed
    n_pairs = min(50000, N * (N - 1) // 2)
    rng = np.random.RandomState(42)
    idx1 = rng.randint(0, N, n_pairs)
    idx2 = rng.randint(0, N, n_pairs)
    # Avoid self-pairs
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    # Cosine similarity
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    flat_normed = flat / norms
    sims = np.sum(flat_normed[idx1] * flat_normed[idx2], axis=1)

    return sims


def analyze_patient_similarity(acc, metadata):
    """Compare within-patient vs between-patient similarity."""
    N, C, T = acc.shape
    flat = acc.reshape(N, -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    flat_normed = flat / norms

    patients = metadata.get('patient_id', metadata.get('patient', None))
    if patients is None:
        print("No patient_id in metadata, skipping patient analysis")
        return None, None

    if hasattr(patients, '__array__'):
        patients = np.array(patients)

    rng = np.random.RandomState(42)
    n_pairs = 20000

    within_sims = []
    between_sims = []

    unique_patients = np.unique(patients)
    patient_indices = {p: np.where(patients == p)[0] for p in unique_patients}

    # Within-patient pairs
    for p in unique_patients:
        pidx = patient_indices[p]
        if len(pidx) < 2:
            continue
        pairs = rng.choice(pidx, size=(min(500, len(pidx)), 2), replace=True)
        mask = pairs[:, 0] != pairs[:, 1]
        pairs = pairs[mask]
        if len(pairs) > 0:
            s = np.sum(flat_normed[pairs[:, 0]] * flat_normed[pairs[:, 1]], axis=1)
            within_sims.extend(s.tolist())

    # Between-patient pairs
    idx1 = rng.randint(0, N, n_pairs)
    idx2 = rng.randint(0, N, n_pairs)
    mask = (idx1 != idx2) & (patients[idx1] != patients[idx2])
    idx1, idx2 = idx1[mask], idx2[mask]
    s = np.sum(flat_normed[idx1] * flat_normed[idx2], axis=1)
    between_sims = s.tolist()

    return np.array(within_sims), np.array(between_sims)


def analyze_pca(acc):
    """PCA analysis — how many dimensions explain the variance?"""
    N, C, T = acc.shape
    flat = acc.reshape(N, -1)

    # Standardize
    flat = (flat - flat.mean(axis=0)) / (flat.std(axis=0) + 1e-8)

    n_components = min(100, N, flat.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(flat)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dims_90 = np.searchsorted(cumvar, 0.90) + 1
    dims_95 = np.searchsorted(cumvar, 0.95) + 1
    dims_99 = np.searchsorted(cumvar, 0.99) + 1

    return pca, cumvar, dims_90, dims_95, dims_99


def analyze_knn_patient_overlap(acc, metadata, k=10):
    """What fraction of k nearest neighbors are from the same patient?"""
    N, C, T = acc.shape
    flat = acc.reshape(N, -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    flat_normed = flat / norms

    patients = metadata.get('patient_id', metadata.get('patient', None))
    if patients is None:
        return None

    if hasattr(patients, '__array__'):
        patients = np.array(patients)

    # Subsample queries for speed
    rng = np.random.RandomState(42)
    n_queries = min(1000, N)
    query_idx = rng.choice(N, n_queries, replace=False)

    # Compute similarities for queries
    query_vecs = flat_normed[query_idx]  # [n_queries, D]
    sim_matrix = query_vecs @ flat_normed.T  # [n_queries, N]

    same_patient_fracs = []
    for i, qi in enumerate(query_idx):
        sims = sim_matrix[i]
        sims[qi] = -np.inf  # exclude self
        top_k = np.argsort(sims)[-k:]
        same = np.sum(patients[top_k] == patients[qi])
        same_patient_fracs.append(same / k)

    return np.array(same_patient_fracs)


def main():
    print(f"Loading {N_SAMPLES} samples from {ZARR_PATH}...")
    acc, metadata, indices = load_subsample(ZARR_PATH, N_SAMPLES, SEED)
    print(f"Loaded: {acc.shape} (N, C, T)")
    print(f"Metadata keys: {list(metadata.keys())}")

    # 1. Pairwise similarity
    print("\n=== 1. Pairwise Cosine Similarity ===")
    sims = analyze_similarity(acc)
    print(f"Mean: {sims.mean():.4f}")
    print(f"Std:  {sims.std():.4f}")
    print(f"Median: {np.median(sims):.4f}")
    print(f"P10/P90: {np.percentile(sims, 10):.4f} / {np.percentile(sims, 90):.4f}")
    print(f"% > 0.9: {(sims > 0.9).mean()*100:.1f}%")
    print(f"% > 0.95: {(sims > 0.95).mean()*100:.1f}%")
    print(f"% > 0.99: {(sims > 0.99).mean()*100:.1f}%")

    # 2. Within vs between patient
    print("\n=== 2. Within vs Between Patient Similarity ===")
    within, between = analyze_patient_similarity(acc, metadata)
    if within is not None and len(within) > 0:
        print(f"Within-patient:  mean={within.mean():.4f}, std={within.std():.4f}")
        print(f"Between-patient: mean={between.mean():.4f}, std={between.std():.4f}")
        print(f"Separation: {within.mean() - between.mean():.4f}")
        print(f"Within > 0.95: {(within > 0.95).mean()*100:.1f}%")
        print(f"Between > 0.95: {(between > 0.95).mean()*100:.1f}%")

    # 3. PCA
    print("\n=== 3. PCA Variance Analysis ===")
    pca, cumvar, d90, d95, d99 = analyze_pca(acc)
    print(f"Dims for 90% variance: {d90} / {acc.shape[1]*acc.shape[2]}")
    print(f"Dims for 95% variance: {d95}")
    print(f"Dims for 99% variance: {d99}")
    print(f"First 10 components explain: {cumvar[9]*100:.1f}%")

    # 4. kNN patient overlap
    print("\n=== 4. k-NN Patient Overlap (k=10) ===")
    overlaps = analyze_knn_patient_overlap(acc, metadata, k=10)
    if overlaps is not None:
        print(f"Mean same-patient fraction in k=10 NN: {overlaps.mean():.4f}")
        print(f"Std: {overlaps.std():.4f}")

        patients = metadata.get('patient_id', metadata.get('patient', None))
        if hasattr(patients, '__array__'):
            patients = np.array(patients)
        n_patients = len(np.unique(patients))
        random_baseline = 1.0 / n_patients
        print(f"Random baseline (1/{n_patients}): {random_baseline:.4f}")
        print(f"Ratio vs random: {overlaps.mean() / random_baseline:.1f}x")

    # 5. Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Similarity histogram
    axes[0, 0].hist(sims, bins=100, alpha=0.7, density=True)
    axes[0, 0].axvline(sims.mean(), color='r', linestyle='--', label=f'mean={sims.mean():.3f}')
    axes[0, 0].set_title('Pairwise Cosine Similarity (Random Pairs)')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].legend()

    # Within vs between
    if within is not None and len(within) > 0:
        axes[0, 1].hist(within, bins=80, alpha=0.6, density=True, label=f'Within (mean={within.mean():.3f})')
        axes[0, 1].hist(between, bins=80, alpha=0.6, density=True, label=f'Between (mean={between.mean():.3f})')
        axes[0, 1].set_title('Within vs Between Patient Similarity')
        axes[0, 1].legend()

    # PCA variance
    axes[1, 0].plot(range(1, len(cumvar)+1), cumvar, 'b-')
    axes[1, 0].axhline(0.90, color='r', linestyle='--', alpha=0.5, label=f'90% @ {d90} dims')
    axes[1, 0].axhline(0.95, color='g', linestyle='--', alpha=0.5, label=f'95% @ {d95} dims')
    axes[1, 0].set_title('PCA Cumulative Variance')
    axes[1, 0].set_xlabel('Components')
    axes[1, 0].set_ylabel('Cumulative Variance Explained')
    axes[1, 0].legend()

    # kNN overlap histogram
    if overlaps is not None:
        axes[1, 1].hist(overlaps, bins=50, alpha=0.7, density=True)
        axes[1, 1].axvline(overlaps.mean(), color='r', linestyle='--', label=f'mean={overlaps.mean():.3f}')
        axes[1, 1].axvline(random_baseline, color='g', linestyle='--', label=f'random={random_baseline:.3f}')
        axes[1, 1].set_title('k-NN Same-Patient Fraction (k=10)')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('artifacts/data_diversity_analysis.png', dpi=150)
    print(f"\nPlot saved to artifacts/data_diversity_analysis.png")


if __name__ == "__main__":
    main()

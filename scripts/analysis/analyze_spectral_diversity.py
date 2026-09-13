"""
Analyze intrinsic dimensionality of spectral representations.

Tests whether wavelet/SpectralPatchEncoder features escape the ~3D
intrinsic dimensionality problem that killed SimCLR on raw accelerometer data.

Analyzes PCA effective rank at three levels:
  Level 0 — Raw signal (flattened [C, T]) — baseline, known ~3D
  Level 1 — Spectrogram (flattened [C, H, W]) — does wavelet help?
  Level 2 — Patch embeddings before transformer [N_tokens, D]
  Level 3 — Transformer token output [N_tokens, D]

For levels 2–3, reports rank at two granularities:
  Per-sample: mean-pool tokens → [N, D] — what global pooling sees (SimCLR v1/v2)
  Per-patch:  all tokens flat → [N*N_tokens, D] — what patch-level loss sees

The gap between per-patch and per-sample rank answers the key question:
is there local spectral structure that global aggregation discards?
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Repo root on path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.transforms import WaveletTransform
from model.backbones import SpectralPatchEncoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ZARR_PATH = ROOT / "data/processed/len1000_stride200_kaggle_daily_unlabeled.zarr"
N_SAMPLES = 3000        # how many patches to load
BATCH_SIZE = 64         # GPU batch size for model forward pass
N_PCA_COMPONENTS = 100  # max PCA components to fit
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = ROOT / "artifacts"

# Match the spectral_patch_simclr_daily experiment config
WAVELET_KWARGS = dict(
    wavelet="morl",
    max_freq=50.0,
    n_scales=100,
    highlight_transient=False,
    transient_weight=0.0,
    interpolate=False,
    img_size=224,   # unused when interpolate=False
)

ENCODER_KWARGS = dict(
    freq_patch=20,
    time_patch=50,
    embed_dim=512,
    num_heads=8,
    vit_depth=4,
    dropout=0.0,  # deterministic
    use_alibi=True,
    input_channels=3,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_subsample(zarr_path: Path, n_samples: int, seed: int):
    ds = zarr.open(str(zarr_path), mode="r")
    total = ds["accs"].shape[0]
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(total, size=min(n_samples, total), replace=False))
    acc = np.array(ds["accs"][idx])      # [N, T, C]
    acc = acc.transpose(0, 2, 1)         # -> [N, C, T]
    meta = {}
    if "metadata" in ds:
        for k in ds["metadata"].keys():
            meta[k] = np.array(ds["metadata"][k][idx])
    print(f"Loaded {acc.shape[0]} samples, shape {acc.shape}  (N, C, T)")
    return acc, meta


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------
def effective_rank(explained_variance_ratio: np.ndarray) -> dict:
    """Compute dims needed for 90/95/99% variance."""
    cumvar = np.cumsum(explained_variance_ratio)
    return {
        "dims_90": int(np.searchsorted(cumvar, 0.90) + 1),
        "dims_95": int(np.searchsorted(cumvar, 0.95) + 1),
        "dims_99": int(np.searchsorted(cumvar, 0.99) + 1),
        "top10_pct": float(cumvar[min(9, len(cumvar) - 1)] * 100),
        "cumvar": cumvar,
    }


def run_pca(matrix: np.ndarray, n_components: int, label: str) -> dict:
    """Standardize and fit PCA; return effective rank stats."""
    N, D = matrix.shape
    n_components = min(n_components, N, D)
    # Standardize column-wise
    mu = matrix.mean(axis=0)
    std = matrix.std(axis=0) + 1e-8
    mat = (matrix - mu) / std

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=SEED)
    pca.fit(mat)
    stats = effective_rank(pca.explained_variance_ratio_)
    print(
        f"  [{label}]  N={N}, D={D}  |  "
        f"dims@90%={stats['dims_90']}  "
        f"dims@95%={stats['dims_95']}  "
        f"dims@99%={stats['dims_99']}  |  "
        f"top-10 explain {stats['top10_pct']:.1f}%"
    )
    return stats


# ---------------------------------------------------------------------------
# GPU forward pass — collect representations
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_representations(acc_np: np.ndarray, wavelet: WaveletTransform, encoder: SpectralPatchEncoder):
    """
    Returns:
        spectro_flat:     [N, C*H*W]      raw spectrogram (downsampled for PCA)
        patch_emb_sample: [N, D]          patch_embed output, mean-pooled over tokens
        patch_emb_patch:  [N*N_tok, D]    patch_embed output, all tokens
        trans_sample:     [N, D]          transformer output, mean-pooled
        trans_patch:      [N*N_tok, D]    transformer output, all tokens
    """
    N = acc_np.shape[0]
    spectro_list = []
    pemb_sample_list, pemb_patch_list = [], []
    trans_sample_list, trans_patch_list = [], []

    wavelet.to(DEVICE).eval()
    encoder.to(DEVICE).eval()

    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        x = torch.tensor(acc_np[start:end], dtype=torch.float32, device=DEVICE)

        # --- Level 1: Spectrogram ---
        spec = wavelet(x)                     # [B, C, H, W]
        # Downsample spatially for PCA tractability: global avg over time+freq
        # We store mean over time → [B, C, H] to capture frequency structure
        spectro_list.append(spec.mean(dim=3).cpu().numpy().reshape(end - start, -1))

        # --- Level 2: Patch embeddings (before transformer) ---
        tokens_raw = encoder.patch_embed(spec)      # [B, D, nH, nW]
        B, D, nH, nW = tokens_raw.shape
        N_tok = nH * nW
        tokens_flat = tokens_raw.permute(0, 2, 3, 1).reshape(B, N_tok, D)
        tokens_flat = encoder.norm(tokens_flat)

        pemb_sample_list.append(tokens_flat.mean(dim=1).cpu().numpy())
        pemb_patch_list.append(tokens_flat.reshape(B * N_tok, D).cpu().numpy())

        # --- Level 3: Transformer output ---
        alibi = encoder._get_alibi_bias(N_tok, DEVICE)
        if alibi is not None:
            n_heads = encoder.transformer.layers[0].self_attn.num_heads
            attn_mask = alibi.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n_heads, N_tok, N_tok)
        else:
            attn_mask = None

        tokens_out = encoder.transformer(tokens_flat, mask=attn_mask)  # [B, N_tok, D]
        trans_sample_list.append(tokens_out.mean(dim=1).cpu().numpy())
        trans_patch_list.append(tokens_out.reshape(B * N_tok, D).cpu().numpy())

        if (start // BATCH_SIZE) % 5 == 0:
            print(f"  processed {end}/{N} samples...", flush=True)

    return (
        np.concatenate(spectro_list, axis=0),
        np.concatenate(pemb_sample_list, axis=0),
        np.concatenate(pemb_patch_list, axis=0),
        np.concatenate(trans_sample_list, axis=0),
        np.concatenate(trans_patch_list, axis=0),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_cumvar(stats_dict: dict, out_path: Path):
    fig, axes = plt.subplots(1, len(stats_dict), figsize=(5 * len(stats_dict), 4))
    if len(stats_dict) == 1:
        axes = [axes]

    for ax, (label, stats) in zip(axes, stats_dict.items()):
        cumvar = stats["cumvar"]
        ax.plot(range(1, len(cumvar) + 1), cumvar, "b-", lw=1.5)
        ax.axhline(0.90, color="r", ls="--", alpha=0.6, label=f"90% @ {stats['dims_90']}")
        ax.axhline(0.95, color="g", ls="--", alpha=0.6, label=f"95% @ {stats['dims_95']}")
        ax.axhline(0.99, color="orange", ls="--", alpha=0.6, label=f"99% @ {stats['dims_99']}")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("PCA components")
        ax.set_ylabel("Cumulative variance")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    print(f"\nPlot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Spectral diversity / effective rank analysis")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--zarr", type=Path, default=ZARR_PATH)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    # --- Load raw acc ---
    acc_np, meta = load_subsample(args.zarr, args.n_samples, SEED)
    N = acc_np.shape[0]

    # --- Level 0: Raw signal PCA (baseline) ---
    print("\n=== Level 0: Raw signal ===")
    raw_flat = acc_np.reshape(N, -1)   # [N, C*T]
    raw_stats = run_pca(raw_flat, N_PCA_COMPONENTS, "raw [N, C*T]")

    # --- Build models ---
    print("\nBuilding WaveletTransform + SpectralPatchEncoder ...")
    wavelet = WaveletTransform(**WAVELET_KWARGS)
    encoder = SpectralPatchEncoder(**ENCODER_KWARGS)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder params: {n_params/1e6:.1f}M  (random init, no pretraining)")

    # --- Collect representations ---
    print(f"\nRunning forward pass on {N} samples (batch={BATCH_SIZE}) ...")
    spectro_flat, pemb_sample, pemb_patch, trans_sample, trans_patch = collect_representations(
        acc_np, wavelet, encoder
    )
    N_tok = pemb_patch.shape[0] // N
    print(f"\nTokens per sample: {N_tok}  |  patch embedding shape: {pemb_patch.shape}")

    # --- Level 1: Spectrogram (mean over time) ---
    print("\n=== Level 1: Spectrogram (mean over time dim) ===")
    spec_stats = run_pca(spectro_flat, N_PCA_COMPONENTS, "spectrogram [N, C*H]")

    # --- Level 2: Patch embeddings ---
    print("\n=== Level 2: Patch embeddings (before transformer) ===")
    pe_sample_stats = run_pca(pemb_sample, N_PCA_COMPONENTS, "patch_emb per-sample (mean-pool)")
    pe_patch_stats  = run_pca(pemb_patch,  N_PCA_COMPONENTS, "patch_emb per-patch  (all tokens)")

    # --- Level 3: Transformer output ---
    print("\n=== Level 3: Transformer output ===")
    tr_sample_stats = run_pca(trans_sample, N_PCA_COMPONENTS, "transformer per-sample (mean-pool)")
    tr_patch_stats  = run_pca(trans_patch,  N_PCA_COMPONENTS, "transformer per-patch  (all tokens)")

    # --- Summary ---
    print("\n" + "="*70)
    print("SUMMARY — dims needed to explain 95% variance")
    print("="*70)
    rows = [
        ("Raw signal (baseline)",              "global",  raw_stats),
        ("Spectrogram mean-over-time",          "global",  spec_stats),
        ("Patch embed — mean-pooled (global)",  "global",  pe_sample_stats),
        ("Patch embed — per-patch token",       "patch",   pe_patch_stats),
        ("Transformer — mean-pooled (global)",  "global",  tr_sample_stats),
        ("Transformer — per-patch token",       "patch",   tr_patch_stats),
    ]
    for label, granularity, stats in rows:
        verdict = ""
        if granularity == "global" and stats["dims_95"] <= 5:
            verdict = "  ← COLLAPSE RISK (trivial task for SimCLR)"
        elif granularity == "patch" and stats["dims_95"] >= 20:
            verdict = "  ← GOOD (patch-level loss may work)"
        print(f"  {label:<45}  dims@95% = {stats['dims_95']:>4}{verdict}")

    print()
    patch_global_ratio = tr_patch_stats["dims_95"] / max(tr_sample_stats["dims_95"], 1)
    print(f"Patch-level / global rank ratio (transformer): {patch_global_ratio:.1f}x")
    if patch_global_ratio >= 3:
        print("→ Patch tokens carry substantially more diversity than global mean.")
        print("  Patch-level contrastive loss is MOTIVATED.")
    else:
        print("→ Patch tokens add little over global mean. Patch-level loss unlikely to help.")

    # --- Plot ---
    all_stats = {
        "Raw signal": raw_stats,
        "Spectrogram (mean-t)": spec_stats,
        "PatchEmbed global": pe_sample_stats,
        "PatchEmbed per-patch": pe_patch_stats,
        "Transformer global": tr_sample_stats,
        "Transformer per-patch": tr_patch_stats,
    }
    plot_cumvar(all_stats, OUT_DIR / "spectral_diversity_analysis.png")


if __name__ == "__main__":
    main()

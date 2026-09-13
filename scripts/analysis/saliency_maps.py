"""
Input-gradient saliency analysis: supervised vs FORGE.

Tests the shortcut hypothesis: a supervised SpectralPatchEncoder focuses on the
0.5–3 Hz gait oscillation (patient-baseline shortcut), while FORGE focuses on
the 3–8 Hz freeze band (true physiological FOG signature).

Method:
  For each model and each labelled window:
    1. Apply preprocessors → wavelet transform (detached, no grad)
    2. Set requires_grad on the spectrogram
    3. Forward: backbone → freq-mean pool → BiGRU head → FOG logit
    4. Backprop: saliency = |d(FOG logit)/d(spectrogram)|, averaged over channels
  Average saliency over N FOG windows and N non-FOG windows separately.

Outputs (--output-dir):
  saliency_comparison.png   — supervised vs FORGE side-by-side on mean FOG spectrogram
  saliency_fog_vs_nonfog.png — FOG vs non-FOG saliency per model
  saliency_data.npz         — raw arrays for downstream analysis

Usage:
  uv run python scripts/analysis/saliency_maps.py \\
    --supervised-ckpt checkpoints/classification/supervised_scratch_fold0/last.ckpt \\
    --forge-ckpt checkpoints/classification/finetune_vit12_ep7_direct_fogcount0/last.ckpt \\
    --n-fog 200 --n-nonfog 100 \\
    --output-dir logs/saliency \\
    --device cpu

Runtime: ~30 min on CPU, ~1 min on GPU.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frequency axis
# ---------------------------------------------------------------------------

def get_freq_axis(transform) -> np.ndarray:
    """Recover Hz frequencies for each row of the [C, 100, 1000] spectrogram."""
    if hasattr(transform, "scales"):
        scales = transform.scales.cpu().float().numpy()
        fc = 0.8125  # Morlet center frequency
        fs = 100.0
        freqs = fc * fs / scales  # same order as rows in the output
        return freqs
    # Fallback from known WaveletTransform(n_scales=100, max_freq=50) config
    n = 100
    loco = np.logspace(np.log10(0.5), np.log10(3.0), int(n * 0.25))
    fog  = np.logspace(np.log10(3.0), np.log10(8.0), int(n * 0.625))
    high = np.logspace(np.log10(8.0), np.log10(50.0), n - int(n * 0.25) - int(n * 0.625))
    return np.concatenate([loco, fog, high])

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_datamodule(ckpt_path: str, batch_size: int):
    from pipeline.classification import ClassificationPipeline
    from data.datamodule.datamodule import FOGDataModule
    from utils.paths import normalize_data_paths

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["hyper_parameters"]["config"]
    normalize_data_paths(config.data.paths)

    data_cfg = config.data.model_copy(update={
        "dataloader": config.data.dataloader.model_copy(
            update={"batch_size": batch_size, "num_workers": 0}
        )
    })
    dm = FOGDataModule(data_cfg=data_cfg, task_type=config.train.pipeline_type)

    model = ClassificationPipeline(config)

    # Resize patient-normalizer buffers to match saved shapes
    state_dict = ckpt["state_dict"]
    for key in [
        "preprocessors.preprocessors.3.normalizer.mean",
        "preprocessors.preprocessors.3.normalizer.stdev",
    ]:
        if key in state_dict:
            saved_shape = state_dict[key].shape
            parts = key.split(".")
            mod = model
            for p in parts[:-1]:
                mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
            setattr(mod, parts[-1], torch.zeros(saved_shape))

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Inject per-patient normalisation stats (training-fold only, no leakage)
    dm.setup("fit")
    if model.preprocessors is not None:
        try:
            stats = dm.train_dataset.get_normalization_stats()
            model.preprocessors.inject_stats(stats)
        except Exception as e:
            logger.warning(f"Could not inject normalisation stats: {e}")

    dm.setup("test")
    return model, dm

# ---------------------------------------------------------------------------
# Saliency computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _to_spectrogram(model, x_raw: torch.Tensor) -> torch.Tensor:
    """Preprocessors + wavelet transform; returns detached [B, C, H, W]."""
    x = model.preprocessors(x_raw) if model.preprocessors is not None else x_raw
    return model.transform(x).detach()


def _saliency_from_spec(model, x_spec: torch.Tensor) -> torch.Tensor:
    """
    Gradient of FOG logit w.r.t. spectrogram x_spec.
    Returns |grad|, mean-pooled over channels: [B, H, W].
    """
    x = x_spec.clone().requires_grad_(True)

    tokens = model.backbone(x)  # [B, N, D]
    if hasattr(model.backbone, "_last_nH"):
        nH = model.backbone._last_nH
        nW = model.backbone._last_nW
        B, N, D = tokens.shape
        tokens = tokens.view(B, nH, nW, D).mean(dim=1)  # [B, nW, D]

    logit = model.head(tokens)  # [B, 2] or [B, 1]
    # FOG score: class-1 logit for 2-class output, raw logit otherwise
    fog_score = logit[:, 1].sum() if (logit.ndim == 2 and logit.shape[1] >= 2) else logit.sum()
    fog_score.backward()

    return x.grad.abs().mean(dim=1).detach()  # [B, H, W]


def collect_saliency(
    model,
    dm,
    n_fog: int,
    n_nonfog: int,
    device: torch.device,
) -> dict:
    """
    Iterate the test loader and accumulate saliency maps.

    Returns dict with keys:
      fog_sal      [H, W] mean saliency for FOG windows
      nonfog_sal   [H, W] mean saliency for non-FOG windows
      fog_spec     [H, W] mean spectrogram (mean over B, C) for FOG windows
      n_fog        actual number of FOG windows collected
      n_nonfog     actual number of non-FOG windows collected
    """
    model = model.to(device)
    loader = dm.test_dataloader()

    fog_sal_acc   = None
    nonfog_sal_acc = None
    fog_spec_acc  = None
    fog_count = 0
    nonfog_count = 0

    for batch in loader:
        if fog_count >= n_fog and nonfog_count >= n_nonfog:
            break

        x_raw = batch["acc"].to(device)

        labels = batch.get("label", batch.get("labels", None))
        if labels is None:
            logger.warning("Batch has no label key — skipping")
            continue
        labels = labels.to(device)
        if labels.ndim > 1:          # sequence labels → window label by majority
            labels = labels.float().mean(dim=-1).round().long()

        with torch.no_grad():
            x_spec = _to_spectrogram(model, x_raw)  # [B, C, H, W]

        fog_idx   = (labels == 1).nonzero(as_tuple=True)[0]
        nonfog_idx = (labels == 0).nonzero(as_tuple=True)[0]

        def _accumulate(idx_tensor, sal_acc, count, limit):
            if len(idx_tensor) == 0 or count >= limit:
                return sal_acc, count
            need = min(len(idx_tensor), limit - count)
            idx = idx_tensor[:need]
            sal = _saliency_from_spec(model, x_spec[idx])   # [k, H, W]
            sal_sum = sal.sum(dim=0).cpu()
            return (sal_acc + sal_sum if sal_acc is not None else sal_sum), count + need

        fog_sal_acc, fog_count     = _accumulate(fog_idx,   fog_sal_acc,   fog_count,   n_fog)
        nonfog_sal_acc, nonfog_count = _accumulate(nonfog_idx, nonfog_sal_acc, nonfog_count, n_nonfog)

        # Accumulate mean FOG spectrogram (averaged over C)
        if len(fog_idx) > 0 and fog_count <= n_fog:
            spec_sum = x_spec[fog_idx].mean(dim=1).sum(dim=0).cpu()  # [H, W]
            fog_spec_acc = fog_spec_acc + spec_sum if fog_spec_acc is not None else spec_sum

    if fog_count == 0:
        raise RuntimeError("No FOG windows found in test loader.")
    if nonfog_count == 0:
        raise RuntimeError("No non-FOG windows found in test loader.")

    logger.info(f"  Collected {fog_count} FOG, {nonfog_count} non-FOG windows")

    return {
        "fog_sal":   (fog_sal_acc   / fog_count).numpy(),
        "nonfog_sal": (nonfog_sal_acc / nonfog_count).numpy(),
        "fog_spec":  (fog_spec_acc  / fog_count).numpy(),
        "n_fog":     fog_count,
        "n_nonfog":  nonfog_count,
    }

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

GAIT_BAND  = (0.5, 3.0)   # locomotor oscillation — patient-baseline shortcut
FREEZE_BAND = (3.0, 8.0)  # FOG freeze band — target physiological signature
FS = 100.0
WIN_SEC = 10.0             # window length in seconds


def _band_spans(ax, freqs: np.ndarray, alpha: float = 0.08):
    """Draw horizontal shading for gait and freeze bands."""
    gait_rows  = np.where((freqs >= GAIT_BAND[0])  & (freqs <= GAIT_BAND[1]))[0]
    freeze_rows = np.where((freqs >= FREEZE_BAND[0]) & (freqs <= FREEZE_BAND[1]))[0]
    n = len(freqs)
    if len(gait_rows):
        ax.axhspan(gait_rows[0] / n, gait_rows[-1] / n,
                   color="#4C9BE8", alpha=alpha, transform=ax.transAxes)
    if len(freeze_rows):
        ax.axhspan(freeze_rows[0] / n, freeze_rows[-1] / n,
                   color="#E84C4C", alpha=alpha, transform=ax.transAxes)


def _yticks_hz(ax, freqs: np.ndarray, targets=(0.5, 1, 2, 3, 5, 8, 10, 20, 50)):
    """Place y-ticks at meaningful Hz values."""
    ticks, labels = [], []
    for hz in targets:
        idx = np.argmin(np.abs(freqs - hz))
        ticks.append(idx)
        labels.append(f"{hz:g}")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)


def _xticks_sec(ax, n_time: int, targets=(0, 2, 4, 6, 8, 10)):
    ticks  = [int(t / WIN_SEC * n_time) for t in targets]
    labels = [str(t) for t in targets]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7)


def _normalise(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


def plot_comparison(
    sup: dict,
    forge: dict,
    freqs: np.ndarray,
    output_path: Path,
):
    """3-column figure: mean FOG spectrogram | supervised saliency | FORGE saliency."""
    saliency_cmap = LinearSegmentedColormap.from_list(
        "sal", ["#1a1a2e", "#e94560", "#f5a623", "#ffff66"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # --- column 0: mean FOG spectrogram (use supervised model's spec as reference) ---
    spec = _normalise(sup["fog_spec"])
    axes[0].imshow(spec, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Mean FOG spectrogram\n(reference)", fontsize=9)

    # --- columns 1 & 2: saliency maps overlaid on spectrogram ---
    for col, (data, title) in enumerate(
        [(sup, "Supervised (from scratch)"), (forge, "FORGE (MAE-pretrained)")], start=1
    ):
        sal = _normalise(data["fog_sal"])
        spec_bg = _normalise(data["fog_spec"])
        # Blend: spectrogram as lightness, saliency as colour
        axes[col].imshow(spec_bg, aspect="auto", origin="lower", cmap="Greys_r", alpha=0.35)
        im = axes[col].imshow(sal, aspect="auto", origin="lower", cmap=saliency_cmap, alpha=0.85)
        axes[col].set_title(title, fontsize=9)
        plt.colorbar(im, ax=axes[col], fraction=0.03, label="Saliency (normalised)")

    # --- annotations and axis labels ---
    n_h, n_w = sup["fog_spec"].shape
    for ax in axes:
        _band_spans(ax, freqs)
        _yticks_hz(ax, freqs)
        _xticks_sec(ax, n_w)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Frequency (Hz)", fontsize=8)

    # Band legend
    patches = [
        mpatches.Patch(color="#4C9BE8", alpha=0.4, label="0.5–3 Hz  gait oscillation"),
        mpatches.Patch(color="#E84C4C", alpha=0.4, label="3–8 Hz  freeze band"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        f"Input-gradient saliency on Morlet CWT spectrogram\n"
        f"(supervised n={sup['n_fog']} FOG; FORGE n={forge['n_fog']} FOG)",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_fog_vs_nonfog(data: dict, label: str, freqs: np.ndarray, output_path: Path):
    """2-panel figure: FOG saliency vs non-FOG saliency for one model."""
    saliency_cmap = LinearSegmentedColormap.from_list(
        "sal", ["#1a1a2e", "#e94560", "#f5a623", "#ffff66"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, sal_key, title in [
        (axes[0], "fog_sal",    f"FOG windows (n={data['n_fog']})"),
        (axes[1], "nonfog_sal", f"Non-FOG windows (n={data['n_nonfog']})"),
    ]:
        sal = _normalise(data[sal_key])
        im = ax.imshow(sal, aspect="auto", origin="lower", cmap=saliency_cmap)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.03, label="Saliency (normalised)")
        _band_spans(ax, freqs)
        _yticks_hz(ax, freqs)
        _xticks_sec(ax, sal.shape[1])
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Frequency (Hz)", fontsize=8)

    patches = [
        mpatches.Patch(color="#4C9BE8", alpha=0.4, label="0.5–3 Hz  gait oscillation"),
        mpatches.Patch(color="#E84C4C", alpha=0.4, label="3–8 Hz  freeze band"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{label} — FOG vs non-FOG saliency", fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--supervised-ckpt", required=True,
                   help="Supervised-from-scratch checkpoint (.ckpt)")
    p.add_argument("--forge-ckpt", required=True,
                   help="FORGE finetuned checkpoint (.ckpt)")
    p.add_argument("--n-fog",    type=int, default=200,
                   help="Max FOG windows per model (default 200)")
    p.add_argument("--n-nonfog", type=int, default=100,
                   help="Max non-FOG windows per model (default 100)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size for inference (default 4; use 16+ on GPU)")
    p.add_argument("--output-dir", default="logs/saliency")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parents[2]))

    results = {}
    for name, ckpt_path in [("supervised", args.supervised_ckpt),
                              ("forge",      args.forge_ckpt)]:
        logger.info(f"\n--- {name.upper()} ---")
        model, dm = load_model_and_datamodule(ckpt_path, args.batch_size)
        freqs = get_freq_axis(model.transform)
        logger.info(f"  Freq axis: {freqs[0]:.2f}–{freqs[-1]:.2f} Hz ({len(freqs)} rows)")

        results[name] = collect_saliency(model, dm, args.n_fog, args.n_nonfog, device)
        results[name]["freqs"] = freqs

        # Per-model FOG vs non-FOG figure
        plot_fog_vs_nonfog(
            results[name],
            label=name.capitalize(),
            freqs=freqs,
            output_path=output_dir / f"saliency_fog_vs_nonfog_{name}.png",
        )

        # Move model back to CPU to free device memory
        model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Main comparison figure
    plot_comparison(
        sup=results["supervised"],
        forge=results["forge"],
        freqs=results["supervised"]["freqs"],
        output_path=output_dir / "saliency_comparison.png",
    )

    # Save raw arrays for downstream analysis (tables, statistics)
    np.savez(
        output_dir / "saliency_data.npz",
        supervised_fog_sal   = results["supervised"]["fog_sal"],
        supervised_nonfog_sal= results["supervised"]["nonfog_sal"],
        supervised_fog_spec  = results["supervised"]["fog_spec"],
        forge_fog_sal        = results["forge"]["fog_sal"],
        forge_nonfog_sal     = results["forge"]["nonfog_sal"],
        forge_fog_spec       = results["forge"]["fog_spec"],
        freqs                = results["supervised"]["freqs"],
    )
    logger.info(f"\nAll outputs in: {output_dir}")

    # Print quick band-energy summary (for paper numbers)
    for name in ("supervised", "forge"):
        freqs = results[name]["freqs"]
        sal = results[name]["fog_sal"]
        gait_mask   = (freqs >= 0.5) & (freqs <= 3.0)
        freeze_mask = (freqs >= 3.0) & (freqs <= 8.0)
        total = sal.sum() + 1e-12
        gait_pct   = sal[gait_mask].sum() / total * 100
        freeze_pct = sal[freeze_mask].sum() / total * 100
        logger.info(
            f"{name:>12}: gait-band={gait_pct:.1f}%  freeze-band={freeze_pct:.1f}%  "
            f"ratio={freeze_pct/gait_pct:.2f}"
        )


if __name__ == "__main__":
    main()

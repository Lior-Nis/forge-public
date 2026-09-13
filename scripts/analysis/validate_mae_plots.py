"""Validate MAE reconstruction visualization for both temporal and frequency masking."""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "artifacts/mae_plot_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_fake_spectrogram(B=1, C=3, H=100, W=200):
    """Create a fake spectrogram with visible structure."""
    t = torch.linspace(0, 1, W).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,W]
    f = torch.linspace(0, 1, H).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1,1,H,1]
    # Create a pattern: horizontal bands + vertical modulation
    spec = torch.sin(2 * np.pi * 5 * f) * torch.cos(2 * np.pi * 3 * t)
    spec = spec.expand(B, C, H, W) + 0.1 * torch.randn(B, C, H, W)
    return spec


def make_fake_raw(B=1, C=3, H=1, W=200):
    """Create a fake raw signal."""
    t = torch.linspace(0, 4 * np.pi, W)
    signals = []
    for c in range(C):
        sig = torch.sin(t * (c + 1)) + 0.1 * torch.randn(W)
        signals.append(sig)
    raw = torch.stack(signals).unsqueeze(0).unsqueeze(2)  # [1, C, 1, W]
    return raw.expand(B, C, H, W)


def create_temporal_mask(W, patch_size, mask_ratio, B=1):
    """Create temporal patch mask [B, num_patches]."""
    num_patches = W // patch_size
    num_masked = max(1, int(num_patches * mask_ratio))
    mask = torch.zeros(B, num_patches, dtype=torch.bool)
    for b in range(B):
        indices = torch.randperm(num_patches)[:num_masked]
        mask[b, indices] = True
    return mask


def create_frequency_mask(H, mask_ratio, B=1):
    """Create frequency band mask [B, H]."""
    num_masked = max(1, int(H * mask_ratio))
    mask = torch.zeros(B, H, dtype=torch.bool)
    for b in range(B):
        indices = torch.randperm(H)[:num_masked]
        mask[b, indices] = True
    return mask


def test_temporal_raw(patch_size=10, mask_ratio=0.4):
    """Test temporal masking with raw signal (H=1)."""
    print("=== Temporal masking + raw signal (H=1) ===")
    W = 200
    original = make_fake_raw(B=1, C=3, H=1, W=W)
    mask = create_temporal_mask(W, patch_size, mask_ratio, B=1)

    # Simulate reconstruction: copy original, add noise to masked patches
    reconstructed = original.clone()
    for p in range(mask.shape[1]):
        if mask[0, p]:
            start = p * patch_size
            end = (p + 1) * patch_size
            reconstructed[0, :, :, start:end] += 0.3 * torch.randn_like(
                reconstructed[0, :, :, start:end]
            )

    sample = {
        "original": original,
        "reconstructed": reconstructed,
        "mask": mask,
        "loss": 0.123,
    }
    return sample, "temporal_raw"


def test_temporal_spectrogram(patch_size=10, mask_ratio=0.4):
    """Test temporal masking with spectrogram (H>1)."""
    print("=== Temporal masking + spectrogram (H=100) ===")
    W = 200
    H = 100
    original = make_fake_spectrogram(B=1, C=3, H=H, W=W)
    mask = create_temporal_mask(W, patch_size, mask_ratio, B=1)

    reconstructed = original.clone()
    for p in range(mask.shape[1]):
        if mask[0, p]:
            start = p * patch_size
            end = (p + 1) * patch_size
            reconstructed[0, :, :, start:end] += 0.5 * torch.randn_like(
                reconstructed[0, :, :, start:end]
            )

    sample = {
        "original": original,
        "reconstructed": reconstructed,
        "mask": mask,
        "loss": 0.456,
    }
    return sample, "temporal_spectrogram"


def test_frequency_spectrogram(mask_ratio=0.4):
    """Test frequency masking with spectrogram (H>1)."""
    print("=== Frequency masking + spectrogram (H=100) ===")
    W = 200
    H = 100
    original = make_fake_spectrogram(B=1, C=3, H=H, W=W)
    mask = create_frequency_mask(H, mask_ratio, B=1)

    # Simulate reconstruction: add noise to masked frequency bands
    reconstructed = original.clone()
    for h in range(H):
        if mask[0, h]:
            reconstructed[0, :, h, :] += 0.5 * torch.randn_like(
                reconstructed[0, :, h, :]
            )

    sample = {
        "original": original,
        "reconstructed": reconstructed,
        "mask": mask,
        "loss": 0.789,
    }
    return sample, "frequency_spectrogram"


def render_reconstruction_plot(samples, patch_size, mask_mode, title_suffix=""):
    """
    Replicate the logic from MAELoggingManager._create_reconstruction_visualization
    to validate it produces correct plots.
    """
    n_samples = min(3, len(samples))

    first_orig = samples[0]["original"][0].numpy()
    if first_orig.ndim == 2:
        first_orig = first_orig[:, np.newaxis, :]
    C_0, H_0, _ = first_orig.shape

    if H_0 == 1:
        # Raw signal branch
        channel_labels = ["AccV", "AccML", "AccAP"] if C_0 == 3 else [f"Ch {i}" for i in range(C_0)]
        n_cols = C_0 + 1
        fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, sample in enumerate(samples[:n_samples]):
            original = sample["original"][0].numpy()
            reconstructed = sample["reconstructed"][0].numpy()
            mask = sample["mask"][0].numpy() if sample["mask"] is not None else None
            loss = sample.get("loss")

            if original.ndim == 2:
                original = original[:, np.newaxis, :]
                reconstructed = reconstructed[:, np.newaxis, :]

            C = original.shape[0]
            orig_2d = original[:, 0, :]
            recon_2d = reconstructed[:, 0, :]
            error_2d = np.abs(orig_2d - recon_2d)
            mean_error = error_2d.mean(axis=0)

            for c in range(C):
                ax = axes[i, c]
                ymin = min(orig_2d[c].min(), recon_2d[c].min())
                ymax = max(orig_2d[c].max(), recon_2d[c].max())
                padding = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
                ax.set_ylim(ymin - padding, ymax + padding)
                ax.plot(orig_2d[c], '--', color='gray', alpha=0.5, label='original')
                ax.plot(recon_2d[c], label='reconstructed')
                if i == 0:
                    ax.set_title(channel_labels[c] if c < len(channel_labels) else f"Ch {c}")
                if c == 0:
                    ylabel = f"Sample {i+1}\nLoss={loss:.5f}" if loss is not None else f"Sample {i+1}"
                    ax.set_ylabel(ylabel)

                # THIS IS THE MASK RENDERING — temporal patches shaded red
                if mask is not None:
                    for p, is_masked in enumerate(mask):
                        if is_masked:
                            ax.axvspan(p * patch_size, (p + 1) * patch_size, alpha=0.15, color='red', lw=0)

            ax_err = axes[i, C]
            ax_err.plot(mean_error, color='black')
            ax_err.grid(alpha=0.3)
            if i == 0:
                ax_err.set_title("|Error| (mean ch)")
            if mask is not None:
                for p, is_masked in enumerate(mask):
                    if is_masked:
                        ax_err.axvspan(p * patch_size, (p + 1) * patch_size, alpha=0.15, color='red', lw=0)

    else:
        # Spectrogram branch
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, sample in enumerate(samples[:n_samples]):
            original = sample["original"][0].numpy()
            reconstructed = sample["reconstructed"][0].numpy()
            mask = sample["mask"][0].numpy() if sample["mask"] is not None else None
            loss = sample.get("loss")

            orig_img = original.mean(axis=0)
            recon_img = reconstructed.mean(axis=0)
            error_img = np.abs(original - reconstructed).mean(axis=0)

            vmin = min(orig_img.min(), recon_img.min())
            vmax = max(orig_img.max(), recon_img.max())

            title0 = f"Original — Loss={loss:.5f}" if loss is not None else f"Original (Sample {i+1})"
            im0 = axes[i, 0].imshow(orig_img, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(title0)
            plt.colorbar(im0, ax=axes[i, 0])

            im1 = axes[i, 1].imshow(recon_img, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f'Reconstructed (Sample {i+1})')
            plt.colorbar(im1, ax=axes[i, 1])

            im2 = axes[i, 2].imshow(error_img, aspect='auto', cmap='Reds')
            axes[i, 2].set_title(f'|Error| (Sample {i+1})')
            plt.colorbar(im2, ax=axes[i, 2])

            # CURRENT CODE: always draws vertical lines regardless of mask type
            # This is WRONG for frequency masks — should draw horizontal lines
            if mask is not None:
                for ax_col in [axes[i, 0], axes[i, 1], axes[i, 2]]:
                    for p, is_masked in enumerate(mask):
                        if is_masked:
                            if mask_mode == "frequency":
                                # frequency mask: mask[p] means freq band p is masked
                                # draw horizontal lines at band boundaries
                                ax_col.axhspan(p - 0.5, p + 0.5, alpha=0.2, color='red', lw=0)
                            else:
                                # temporal mask: mask[p] means patch p is masked
                                ax_col.axvline(p * patch_size, color='white', alpha=0.4, lw=0.5)
                                ax_col.axvline((p + 1) * patch_size, color='white', alpha=0.4, lw=0.5)

    plt.suptitle(f'MAE Reconstructions — {mask_mode} masking {title_suffix}')
    plt.tight_layout()
    return fig


def main():
    print(f"Saving validation plots to {OUTPUT_DIR}/\n")

    # Test 1: Temporal masking + raw signal
    sample, name = test_temporal_raw()
    fig = render_reconstruction_plot([sample], patch_size=10, mask_mode="temporal", title_suffix="(raw H=1)")
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Test 2: Temporal masking + spectrogram
    sample, name = test_temporal_spectrogram()
    fig = render_reconstruction_plot([sample], patch_size=10, mask_mode="temporal", title_suffix="(spectrogram H=100)")
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Test 3: Frequency masking + spectrogram
    sample, name = test_frequency_spectrogram()
    fig = render_reconstruction_plot([sample], patch_size=10, mask_mode="frequency", title_suffix="(spectrogram H=100)")
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Test 4: Current buggy code — frequency mask rendered as temporal (vertical lines)
    sample, _ = test_frequency_spectrogram()
    fig = render_reconstruction_plot([sample], patch_size=10, mask_mode="temporal", title_suffix="(BUGGY: freq mask as temporal)")
    path = os.path.join(OUTPUT_DIR, "frequency_spectrogram_BUGGY.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path} (shows what current code does wrong)")

    print(f"\nDone! Inspect plots in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

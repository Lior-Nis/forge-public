"""
fig_masking_forge.py -- Fig 6: 1D temporal vs 2D spectral-temporal masking (FORGE).

Three stacked full-width rows on a REAL DeFOG Morlet-CWT window:

    (1) Input                       -> full CWT spectrogram + 2D patch grid
    (2) 1D Temporal Masking         -> whole time-columns greyed out (all freqs)
    (3) 2D Spectral-Temporal Mask   -> individual 2D patches greyed out

Key message: 1D temporal masking lets the encoder interpolate trivially through
the bandlimited gait signal, while FORGE's 2D patch masking forces the encoder to
predict spectral context from temporal context and vice versa (freeze band
3-8 Hz vs locomotor band 0.5-3 Hz).

FORGE patch geometry: 20 frequency bins x 50 timesteps per patch. On a 10 s,
1000-timestep window with 100 CWT scales this gives a 5 (freq) x 20 (time)
patch grid. The 1D row masks whole 50-timestep columns; the 2D row masks
individual 20x50 patches. Both mask ~50%.

Real window: DeFOG FOG-positive idx=20051 (same window as fig_spectral_signature.py),
loaded from the Kaggle zarr -- no synthetic data.

Output: research/paper_final/figures/fig06_masking_mechanism.png (PNG, 300 dpi)

Run with: .venv/bin/python scripts/analysis/fig_masking_forge.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pywt

# -- Paths -----------------------------------------------------------------------
THIS = Path(__file__).resolve()
REPO = THIS.parents[2]
OUT = REPO / "research" / "paper_final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
ZARR = REPO / "data" / "processed" / "len1000_stride200_kaggle.zarr"

# Same real FOG-positive DeFOG window used by fig_spectral_signature.py
FOG_IDX = 20051
FOG_CH = 0

# -- Style (matches paper_mae fig_masking_mechanism.py) --------------------------
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# -- Signal / spectrogram parameters ---------------------------------------------
FS = 100.0          # Hz
DURATION = 10.0     # s (1000 timesteps)

# CWT matches the project's WaveletTransform (morl, 0.3-50 Hz, 100 scales),
# but we crop the displayed axis to the FOG-relevant band.
CWT_FMIN, CWT_FMAX = 0.3, 50.0
N_SCALES = 100
PLOT_FMIN, PLOT_FMAX = 0.5, 12.0   # displayed frequency range (log axis)

# FORGE 2D patch geometry: 20 frequency bins x 50 timesteps per patch.
# 100 displayed scales / 20 = 5 freq patches; 1000 timesteps / 50 = 20 time patches.
PATCH_FREQ_BINS = 20
PATCH_TIMESTEPS = 50
N_FREQ_PATCHES = 5    # over the displayed band
N_TIME_PATCHES = 20
N_PATCHES = N_FREQ_PATCHES * N_TIME_PATCHES
MASK_RATIO = 0.5

# Colors (BLUES aesthetic): masked patches → navy, visible → pale spectrogram
MASK_COLOR = "#2b2b2b"
MASK_ALPHA = 0.88
GRID_COLOR = "#08306b"
GRID_LW = 0.7

# Band annotations
FREEZE_BAND = (3.0, 8.0)      # Hz
LOCOMOTOR_BAND = (0.5, 3.0)   # Hz
FREEZE_COLOR = "#08306b"
LOCO_COLOR = "#2171b5"


# -- Data ------------------------------------------------------------------------
def load_window(idx, ch):
    """Load one real DeFOG window (single channel, mean-removed) from the zarr.

    Reuses the proven chunk-read path from fig_spectral_signature.py rather than
    scanning the full metadata arrays (slow on chunked stores).
    """
    from numcodecs.blosc import decompress

    chunk_size = 256
    chunk_idx = idx // chunk_size
    offset = idx % chunk_size
    chunk_path = ZARR / "accs" / "c" / str(chunk_idx) / "0" / "0"
    raw_bytes = decompress(chunk_path.read_bytes())
    raw = np.frombuffer(raw_bytes, dtype="<f4").reshape(chunk_size, 1000, 3)[
        offset
    ].astype(np.float64)
    sig = raw - raw.mean(axis=0, keepdims=True)
    return sig[:, ch]


def compute_cwt(signal, fs=FS, n_scales=N_SCALES, fmin=CWT_FMIN, fmax=CWT_FMAX):
    """Morlet CWT matching WaveletTransform (morl, 0.3-50 Hz, 100 scales)."""
    cf = pywt.central_frequency("morl")
    dt = 1.0 / fs
    scales = np.geomspace(cf / (fmax * dt), cf / (fmin * dt), n_scales)
    coefs, freqs = pywt.cwt(
        signal.astype(np.float64), scales, "morl", sampling_period=dt
    )
    power = np.abs(coefs) ** 2
    order = np.argsort(freqs)
    return power[order], freqs[order]


# -- Masking ---------------------------------------------------------------------
def make_1d_temporal_mask(seed=42, mask_ratio=MASK_RATIO):
    """Mask whole 50-timestep columns across all frequency bins."""
    rng = np.random.default_rng(seed)
    cols = rng.choice(
        N_TIME_PATCHES, size=int(N_TIME_PATCHES * mask_ratio), replace=False
    )
    mask = np.zeros((N_FREQ_PATCHES, N_TIME_PATCHES), dtype=bool)
    mask[:, cols] = True
    return mask


def make_2d_patch_mask(seed=42, mask_ratio=MASK_RATIO):
    """Mask individual 20x50 spectral-temporal patches."""
    rng = np.random.default_rng(seed)
    flat = rng.choice(N_PATCHES, size=int(N_PATCHES * mask_ratio), replace=False)
    mask = np.zeros(N_PATCHES, dtype=bool)
    mask[flat] = True
    return mask.reshape(N_FREQ_PATCHES, N_TIME_PATCHES)


# -- Drawing primitives ----------------------------------------------------------
def _freq_edges():
    return np.geomspace(PLOT_FMIN, PLOT_FMAX, N_FREQ_PATCHES + 1)


def _time_edges():
    return np.linspace(0, DURATION, N_TIME_PATCHES + 1)


def _setup_ax(ax, log_p, freqs, times, vmin, vmax, show_xlabel):
    fmask = (freqs >= PLOT_FMIN) & (freqs <= PLOT_FMAX)
    ax.pcolormesh(
        times,
        freqs[fmask],
        log_p[fmask],
        vmin=vmin,
        vmax=vmax,
        cmap="Blues",
        shading="gouraud",
        rasterized=True,
    )
    ax.set_yscale("log")
    ax.set_ylim(PLOT_FMIN, PLOT_FMAX)
    ax.set_xlim(0, DURATION)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.1f}")
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([0.5, 1, 3, 5, 8, 12])
    ax.set_ylabel("Freq (Hz)", fontsize=9)
    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
    else:
        ax.tick_params(labelbottom=False)


def _draw_grid(ax):
    for t in _time_edges():
        ax.axvline(t, color=GRID_COLOR, lw=GRID_LW, zorder=3, alpha=0.45)
    for f in _freq_edges():
        ax.axhline(f, color=GRID_COLOR, lw=GRID_LW, zorder=3, alpha=0.45)


def _draw_mask(ax, mask_2d):
    fe = _freq_edges()
    te = _time_edges()
    for fi in range(N_FREQ_PATCHES):
        for ti in range(N_TIME_PATCHES):
            if mask_2d[fi, ti]:
                ax.fill_between(
                    [te[ti], te[ti + 1]],
                    fe[fi],
                    fe[fi + 1],
                    color=MASK_COLOR,
                    alpha=MASK_ALPHA,
                    zorder=4,
                    linewidth=0,
                )


def _band_brackets(ax):
    """Subtle freeze (3-8 Hz) / locomotor (0.5-3 Hz) brackets on the right edge."""
    x = 1.012  # axes fraction
    trans = ax.get_yaxis_transform()

    def bracket(y0, y1, color, label):
        ax.plot(
            [x, x], [y0, y1], color=color, lw=1.6, clip_on=False,
            transform=trans, solid_capstyle="round", zorder=6,
        )
        for yy in (y0, y1):
            ax.plot(
                [x, x + 0.012], [yy, yy], color=color, lw=1.6,
                clip_on=False, transform=trans, zorder=6,
            )
        ax.text(
            x + 0.055, np.sqrt(y0 * y1), label, color=color, fontsize=6.5,
            rotation=90, ha="center", va="center", clip_on=False,
            transform=trans, fontweight="bold",
        )

    bracket(*LOCOMOTOR_BAND, LOCO_COLOR, "0.5-3 Hz\nlocomotor")
    bracket(*FREEZE_BAND, FREEZE_COLOR, "3-8 Hz\nfreeze")


# -- Main ------------------------------------------------------------------------
def main():
    print("=== fig_masking_forge.py ===")
    print(f"  zarr: {ZARR}")
    if not ZARR.exists():
        sys.exit(f"ERROR: zarr not found at {ZARR}")

    sig = load_window(FOG_IDX, FOG_CH)
    print(f"  Loaded REAL DeFOG window idx={FOG_IDX} ch={FOG_CH} "
          f"len={len(sig)} ({len(sig)/FS:.0f}s)")

    print("  Computing Morlet CWT (morl, 0.3-50 Hz, 100 scales)...")
    power, freqs = compute_cwt(sig)
    times = np.linspace(0, DURATION, power.shape[1])

    # Smooth + log to suppress Morlet ringing before display.
    from scipy.ndimage import gaussian_filter

    log_p = np.log1p(gaussian_filter(power.astype(np.float64), sigma=(2.0, 4.0)))
    fmask = (freqs >= PLOT_FMIN) & (freqs <= PLOT_FMAX)
    vmin = float(np.percentile(log_p[fmask], 2))
    vmax = float(np.percentile(log_p[fmask], 99))

    mask_1d = make_1d_temporal_mask(seed=42)
    mask_2d = make_2d_patch_mask(seed=42)
    print(f"  1D mask: {mask_1d.sum()}/{N_PATCHES} "
          f"({100*mask_1d.sum()/N_PATCHES:.0f}%)  "
          f"2D mask: {mask_2d.sum()}/{N_PATCHES} "
          f"({100*mask_2d.sum()/N_PATCHES:.0f}%)")

    # -- 3 stacked full-width rows -----------------------------------------------
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9, 7.6),
        gridspec_kw={
            "hspace": 0.34,
            "left": 0.06,
            "right": 0.93,
            "top": 0.97,
            "bottom": 0.07,
        },
    )
    ax_a, ax_b, ax_c = axes

    _setup_ax(ax_a, log_p, freqs, times, vmin, vmax, show_xlabel=False)
    _draw_grid(ax_a)
    _band_brackets(ax_a)
    ax_a.set_title("Input", fontsize=10, fontweight="bold", color="#222222", loc="left")

    _setup_ax(ax_b, log_p, freqs, times, vmin, vmax, show_xlabel=False)
    _draw_grid(ax_b)
    _draw_mask(ax_b, mask_1d)
    ax_b.set_title(
        "1D Temporal Masking", fontsize=10, fontweight="bold", color="#222222", loc="left"
    )

    _setup_ax(ax_c, log_p, freqs, times, vmin, vmax, show_xlabel=True)
    _draw_grid(ax_c)
    _draw_mask(ax_c, mask_2d)
    ax_c.set_title(
        "2D Spectral-Temporal Masking",
        fontsize=10,
        fontweight="bold",
        color="#222222",
        loc="left",
    )

    png = OUT / "fig06_masking_mechanism.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    print(f"+ saved {png}")


if __name__ == "__main__":
    main()

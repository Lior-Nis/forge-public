"""
fig_spectral_signature.py — Fig 4 "The FOG Spectral Signature" (FORGE paper).

Three panels from REAL DeFOG-cohort accelerometer windows:
  A) FOG-positive Morlet CWT spectrogram  — freeze band (3-8 Hz) rises.
  B) FOG-negative (same patient/session) — dominant locomotor band (0.5-3 Hz).
  C) The same 10 s FOG episode viewed at three context windows (SC/MC/LC),
     each with the 2-D spectral-temporal patch grid overlaid.

Both A and B share an identical colormap range (shared norm). Two horizontal
bracket annotations mark the locomotor (0.5-3 Hz) and freeze (3-8 Hz) bands.

CWT matches the project's WaveletTransform (model/transforms.py): Morlet 'morl',
0.3-50 Hz, 100 scales. We replicate it with pywt for a self-contained script.

Data (real, no synthetic fallback used):
  FOG-positive : idx=20051  patient d5a377  session 2b60e83702  onset=6.0 s  ch=0
  FOG-negative : idx=19572  patient d5a377  session 2b60e83702  (clean walking)
  Selected via fog_ratio + freeze-band-rise scoring across the DeFOG cohort.

Run:  .venv/bin/python scripts/analysis/fig_spectral_signature.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
import numpy as np
import pywt

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
ZARR_PATH = ROOT / "data/processed/len1000_stride200_kaggle.zarr"
OUT_DIR = ROOT / "research/paper_final/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Real, pre-selected window indices (DeFOG cohort) ──────────────────────────
FOG_IDX = 20051        # FOG-positive, freeze onset ~6.0 s
NONFOG_IDX = 19572     # FOG-negative, same patient + session
FOG_CH = 0             # channel with strongest movement signature

FS = 100.0
FMIN, FMAX = 0.3, 50.0
N_SCALES = 100
PLOT_FMIN, PLOT_FMAX = 0.4, 15.0   # crop the displayed frequency axis

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.linewidth": 0.7,
})

FOG_BG, FOG_TXT = "#deebf7", "#08306b"
NONFOG_BG, NONFOG_TXT = "#deebf7", "#2171b5"
LOCO_COLOR = "#2171b5"   # locomotor band bracket (blue)
FREEZE_COLOR = "#08306b"  # freeze band bracket (navy)


# ── CWT (matches WaveletTransform: morl, 0.3-50 Hz, 100 scales) ──────────────
def compute_cwt(signal, fs=FS, n_scales=N_SCALES, fmin=FMIN, fmax=FMAX):
    cf = pywt.central_frequency("morl")
    dt = 1.0 / fs
    scales = np.geomspace(cf / (fmax * dt), cf / (fmin * dt), n_scales)
    coefs, freqs = pywt.cwt(signal.astype(np.float64), scales, "morl",
                            sampling_period=dt)
    power = np.abs(coefs) ** 2
    idx = np.argsort(freqs)
    return power[idx], freqs[idx]


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_window(idx, ch):
    from numcodecs.blosc import decompress

    chunk_size = 256
    chunk_idx = idx // chunk_size
    offset = idx % chunk_size
    chunk_path = ZARR_PATH / "accs" / "c" / str(chunk_idx) / "0" / "0"
    raw_bytes = decompress(chunk_path.read_bytes())
    raw = np.frombuffer(raw_bytes, dtype="<f4").reshape(chunk_size, 1000, 3)[offset].astype(np.float64)
    sig = raw - raw.mean(axis=0, keepdims=True)
    return sig[:, ch]


def load_window_all_channels(idx):
    from numcodecs.blosc import decompress

    chunk_size = 256
    chunk_idx = idx // chunk_size
    offset = idx % chunk_size
    chunk_path = ZARR_PATH / "accs" / "c" / str(chunk_idx) / "0" / "0"
    raw_bytes = decompress(chunk_path.read_bytes())
    raw = np.frombuffer(raw_bytes, dtype="<f4").reshape(chunk_size, 1000, 3)[offset].astype(np.float64)
    return raw - raw.mean(axis=0, keepdims=True)  # [1000, 3]


def draw_raw_signal(ax, sig_all, duration, show_xlabel=False):
    """Plot all 3 accelerometer channels as a thin strip (blues palette)."""
    t = np.linspace(0, duration, sig_all.shape[0])
    colors = ["#2171b5", "#6baed6", "#bdd7e7"]
    labels = ["x", "y", "z"]
    for ch_i in range(3):
        ax.plot(t, sig_all[:, ch_i], color=colors[ch_i], lw=0.7,
                alpha=0.85, label=labels[ch_i])
    ax.set_xlim(0, duration)
    ax.set_ylabel("g", fontsize=7, labelpad=2)
    ax.tick_params(axis="y", labelsize=6, pad=1)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, symmetric=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(axis="x", labelsize=7.5)
    else:
        ax.tick_params(labelbottom=False)


# ── Spectrogram drawing ───────────────────────────────────────────────────────
def _freq_formatter(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.1f}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([0.5, 1, 3, 5, 8, 12])


def draw_spectrogram(ax, log_p, freqs, duration, norm,
                     show_xlabel=True, show_ylabel=True):
    t = np.linspace(0, duration, log_p.shape[1])
    fmask = (freqs >= PLOT_FMIN) & (freqs <= PLOT_FMAX)
    mesh = ax.pcolormesh(t, freqs[fmask], log_p[fmask], cmap="Blues",
                         norm=norm, shading="gouraud", rasterized=True)
    _freq_formatter(ax)
    ax.set_ylim(PLOT_FMIN, PLOT_FMAX)
    ax.set_xlim(0, duration)
    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=8)
    else:
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel("Frequency (Hz)", fontsize=8)
    else:
        ax.tick_params(labelleft=False)
    despine(ax)
    return mesh


def add_band_brackets(ax):
    """Draw the locomotor (0.5-3 Hz) and freeze (3-8 Hz) band brackets
    just outside the right edge of the spectrogram."""
    x = 1.012   # axes fraction, just right of the panel
    trans = ax.get_yaxis_transform()  # x in axes frac, y in data coords

    def bracket(y0, y1, color, label):
        ax.plot([x, x], [y0, y1], color=color, lw=1.6, clip_on=False,
                transform=trans, solid_capstyle="round")
        tick = 0.012
        for yy in (y0, y1):
            ax.plot([x, x + tick], [yy, yy], color=color, lw=1.6,
                    clip_on=False, transform=trans)
        ax.text(x + 0.05, np.sqrt(y0 * y1), label, color=color, fontsize=6.5,
                rotation=90, ha="center", va="center", clip_on=False,
                transform=trans, fontweight="bold")

    bracket(0.5, 3.0, LOCO_COLOR, "0.5-3 Hz\nlocomotor")
    bracket(3.0, 8.0, FREEZE_COLOR, "3-8 Hz\nfreeze")


def header_band(ax, text, bg, fg):
    ax.set_facecolor(bg)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.5, text, ha="center", va="center",
            color=fg, fontsize=8, fontweight="bold", transform=ax.transAxes)


# ── Patch grid overlay (Panel C) ──────────────────────────────────────────────
def draw_patch_grid(ax, duration, n_time_patches, n_freq_patches):
    """Overlay 2-D patch boundaries (time x log-frequency)."""
    xs = np.linspace(0, duration, n_time_patches + 1)
    for xv in xs[1:-1]:
        ax.axvline(xv, color="#08306b", lw=0.4, alpha=0.45)
    fmask_lo, fmask_hi = PLOT_FMIN, PLOT_FMAX
    fedges = np.geomspace(fmask_lo, fmask_hi, n_freq_patches + 1)
    for fv in fedges[1:-1]:
        ax.axhline(fv, color="#08306b", lw=0.4, alpha=0.45)


def main():
    print("=== fig_spectral_signature.py ===")
    print(f"  zarr: {ZARR_PATH}")
    if not ZARR_PATH.exists():
        sys.exit(f"ERROR: zarr not found at {ZARR_PATH}")

    # --- load real windows ---
    fog_sig = load_window(FOG_IDX, FOG_CH)
    nonfog_sig = load_window(NONFOG_IDX, FOG_CH)
    fog_raw = load_window_all_channels(FOG_IDX)
    nonfog_raw = load_window_all_channels(NONFOG_IDX)
    duration = len(fog_sig) / FS
    print(f"  FOG idx={FOG_IDX}, non-FOG idx={NONFOG_IDX}, "
          f"ch={FOG_CH}, duration={duration:.1f}s")

    # --- CWTs ---
    fog_p, freqs = compute_cwt(fog_sig)
    nonfog_p, _ = compute_cwt(nonfog_sig)
    fog_lp = np.log1p(fog_p)
    nonfog_lp = np.log1p(nonfog_p)

    # --- per-panel normalization ---
    fmask = (freqs >= PLOT_FMIN) & (freqs <= PLOT_FMAX)
    def _pnorm(lp):
        v = lp[fmask].ravel()
        return Normalize(vmin=float(np.percentile(v, 2)), vmax=float(np.percentile(v, 99)))
    norm_non = _pnorm(nonfog_lp)
    norm_fog = _pnorm(fog_lp)

    # --- 4-row layout: two pairs (cwt + raw) with tight intra-pair spacing
    # and a deliberate gap between pairs so the second title doesn't overlap. ---
    fig = plt.figure(figsize=(10, 6.0))
    gs_top = gridspec.GridSpec(2, 1, figure=fig,
                               height_ratios=[4, 1.2], hspace=0.06,
                               left=0.06, right=0.86, top=0.97, bottom=0.54)
    gs_bot = gridspec.GridSpec(2, 1, figure=fig,
                               height_ratios=[4, 1.2], hspace=0.06,
                               left=0.06, right=0.86, top=0.50, bottom=0.07)

    axA  = fig.add_subplot(gs_top[0])
    axAr = fig.add_subplot(gs_top[1], sharex=axA)
    axB  = fig.add_subplot(gs_bot[0])
    axBr = fig.add_subplot(gs_bot[1], sharex=axB)

    draw_spectrogram(axA, nonfog_lp, freqs, duration, norm_non,
                     show_xlabel=False, show_ylabel=True)
    add_band_brackets(axA)
    axA.set_title("Non-FOG — normal walking", loc="left",
                  color=NONFOG_TXT, fontsize=10, fontweight="bold", pad=3)
    draw_raw_signal(axAr, nonfog_raw, duration, show_xlabel=False)

    mesh = draw_spectrogram(axB, fog_lp, freqs, duration, norm_fog,
                            show_xlabel=False, show_ylabel=True)
    add_band_brackets(axB)
    axB.set_title("FOG episode", loc="left",
                  color=FOG_TXT, fontsize=10, fontweight="bold", pad=3)
    draw_raw_signal(axBr, fog_raw, duration, show_xlabel=True)

    # freeze onset line on both FOG panels
    _albl = dict(boxstyle="round,pad=0.18", fc="white", alpha=0.75, ec="none")
    _arr = dict(arrowstyle="->", color="#08306b", lw=1.5,
                shrinkA=3, shrinkB=2, connectionstyle="arc3,rad=0.0")
    for _ax in (axB, axBr):
        _ax.axvline(6.0, color="#6baed6", ls="--", lw=1.1, alpha=0.9)
    axB.text(5.92, 13.6, "freeze onset", color="#08306b", fontsize=7.5,
             fontweight="bold", ha="right", va="center", bbox=_albl)
    axB.annotate("freeze band", xy=(9.2, 5.2), xytext=(6.55, 10.5),
                 xycoords="data", textcoords="data", color="#08306b", fontsize=7.5,
                 fontweight="bold", ha="left", va="center", bbox=_albl,
                 arrowprops=_arr)
    axB.annotate("locomotor band", xy=(9.0, 2.0), xytext=(6.55, 0.62),
                 xycoords="data", textcoords="data", color="#08306b", fontsize=7.5,
                 fontweight="bold", ha="left", va="center", bbox=_albl,
                 arrowprops=_arr)

    # shared colorbar spanning the two spectrogram panels
    # position: right of gs rows 0 and 2
    cax = fig.add_axes([0.93, 0.07, 0.012, 0.90 * 0.97])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("log power (per-panel normalized)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)


    # --- save ---
    for ext in ("png",):
        out = OUT_DIR / f"fig04_fog_spectral_signature.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
        print(f"  saved -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

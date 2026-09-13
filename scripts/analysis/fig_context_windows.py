"""
fig_context_windows.py — Fig 4b "Multi-scale context windows" (FORGE paper).

Single LC (10 s) CWT spectrogram with SC (2 s) and MC (5 s) sub-windows shown
as nested brackets. SC and MC are trailing slices of the same LC window, so they
are literally contained within it. Patch grids for each context overlaid over
their respective extents. Compact single-panel design saves figure space.

Run:  uv run python scripts/analysis/fig_context_windows.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pywt

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
ZARR_PATH = ROOT / "data/processed/len1000_stride200_kaggle.zarr"
OUT_DIR = ROOT / "research/paper_final/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOG_IDX, FOG_CH = 20051, 0
FS = 100.0
FMIN, FMAX = 0.3, 50.0
N_SCALES = 100
PLOT_FMIN, PLOT_FMAX = 0.4, 15.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 0.7,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# Blues palette — SC darkest (finest), LC lightest (coarsest)
SC_C  = "#08306b"   # dark navy
MC_C  = "#2171b5"   # mid blue
LC_C  = "#6baed6"   # light blue

# name, ctx_s, n_time_tokens, n_freq_tokens, patch_time_frames, color
CONTEXTS = [
    ("LC", 10.0, 20, 5, 50, LC_C),
    ("MC",  5.0, 10, 5, 50, MC_C),
    ("SC",  2.0, 20, 5, 10, SC_C),
]


def compute_cwt(signal):
    cf = pywt.central_frequency("morl")
    dt = 1.0 / FS
    scales = np.geomspace(cf / (FMAX * dt), cf / (FMIN * dt), N_SCALES)
    coefs, freqs = pywt.cwt(signal.astype(np.float64), scales, "morl", sampling_period=dt)
    power = np.abs(coefs) ** 2
    idx = np.argsort(freqs)
    return power[idx], freqs[idx]


def load_window(idx, ch):
    from numcodecs.blosc import decompress
    chunk_size = 256
    chunk_idx, offset = idx // chunk_size, idx % chunk_size
    chunk_path = ZARR_PATH / "accs" / "c" / str(chunk_idx) / "0" / "0"
    raw = np.frombuffer(decompress(chunk_path.read_bytes()), dtype="<f4").reshape(
        chunk_size, 1000, 3)[offset].astype(np.float64)
    return (raw - raw.mean(axis=0, keepdims=True))[:, ch]


def draw_bracket(ax, x0, x1, y, color, label, label_side="left"):
    """Draw a horizontal bracket in data-x / axes-y coordinates."""
    trans = ax.get_xaxis_transform()   # x=data, y=axes fraction
    tick = 0.025
    # horizontal bar
    ax.plot([x0, x1], [y, y], color=color, lw=1.6, clip_on=False,
            transform=trans, solid_capstyle="round")
    # end ticks
    for xv in (x0, x1):
        ax.plot([xv, xv], [y - tick, y + tick], color=color, lw=1.4,
                clip_on=False, transform=trans)
    # label
    lx = x0 if label_side == "left" else x1
    ha = "left" if label_side == "left" else "right"
    ax.text(lx, y + 0.04, label, transform=trans, color=color,
            fontsize=8.5, fontweight="bold", ha=ha, va="bottom", clip_on=False)


def main():
    print("=== fig_context_windows.py ===")
    if not ZARR_PATH.exists():
        sys.exit(f"ERROR: zarr not found at {ZARR_PATH}")

    fog_sig = load_window(FOG_IDX, FOG_CH)
    total_s = len(fog_sig) / FS   # 10 s

    # Compute LC CWT (full 10 s) — shared spectrogram for all contexts
    lp, freqs = compute_cwt(fog_sig)
    lp = np.log1p(lp)
    fm = (freqs >= PLOT_FMIN) & (freqs <= PLOT_FMAX)
    norm = Normalize(vmin=float(np.percentile(lp[fm], 2)),
                     vmax=float(np.percentile(lp[fm], 99)))

    # Layout: bracket strip (top) + spectrogram
    fig = plt.figure(figsize=(9, 3.2))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.32, 1],
                           hspace=0.0, left=0.07, right=0.88, top=0.97, bottom=0.12)
    ax_bk = fig.add_subplot(gs[0])   # bracket strip
    ax    = fig.add_subplot(gs[1])   # spectrogram

    # ── Spectrogram ────────────────────────────────────────────────────────────
    t = np.linspace(0, total_s, lp.shape[1])
    mesh = ax.pcolormesh(t, freqs[fm], lp[fm], cmap="Blues",
                         norm=norm, shading="gouraud", rasterized=True)
    ax.set_yscale("log")
    ax.set_ylim(PLOT_FMIN, PLOT_FMAX)
    ax.set_xlim(0, total_s)
    ax.set_yticks([0.5, 1, 3, 8])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.1f}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("Hz", fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ── Patch grids (each context left-aligned from t=0) ──────────────────────
    for name, ctx_s, n_tt, n_ft, ptf, color in CONTEXTS:
        t_end = ctx_s
        alpha = 0.50 if name == "SC" else 0.35 if name == "MC" else 0.22
        # vertical patch boundaries (time)
        for xv in np.linspace(0, t_end, n_tt + 1)[1:-1]:
            ax.axvline(xv, color=color, lw=0.5, alpha=alpha, zorder=4)
        # horizontal patch boundaries (frequency)
        for fv in np.geomspace(PLOT_FMIN, PLOT_FMAX, n_ft + 1)[1:-1]:
            ax.plot([0, t_end], [fv, fv], color=color,
                    lw=0.5, alpha=alpha, zorder=4)
        # right edge boundary for SC/MC
        if name in ("SC", "MC"):
            ax.axvline(t_end, color=color, lw=1.0, alpha=0.6,
                       ls="--", zorder=5)

    # ── Bracket strip ──────────────────────────────────────────────────────────
    ax_bk.set_xlim(0, total_s)
    ax_bk.set_ylim(0, 1)
    ax_bk.axis("off")

    # three bracket rows, sharing x-axis with spectrogram
    # re-use ax.get_xaxis_transform for consistent alignment
    bracket_ys = [0.78, 0.45, 0.12]   # LC, MC, SC (top → bottom in strip)
    for (name, ctx_s, n_tt, n_ft, ptf, color), by in zip(CONTEXTS, bracket_ys):
        t_end = ctx_s
        trans = ax_bk.transData
        # bracket bar from t=0 to t=ctx_s
        ax_bk.annotate("", xy=(t_end, by), xytext=(0, by),
                        xycoords=trans, textcoords=trans,
                        arrowprops=dict(arrowstyle="<->", color=color, lw=1.5,
                                        mutation_scale=6))
        if name == "LC":
            ax_bk.text(ctx_s, by - 0.28,
                       f"{name}  ·  {ctx_s:.0f} s",
                       color=color, fontsize=8.5, fontweight="bold",
                       ha="right", va="top", clip_on=False, transform=trans)
        else:
            ax_bk.text(ctx_s + 0.15, by,
                       f"{name}  ·  {ctx_s:.0f} s",
                       color=color, fontsize=8.5, fontweight="bold",
                       ha="left", va="center", clip_on=False, transform=trans)

    # ── Colorbar ───────────────────────────────────────────────────────────────
    cax = fig.add_axes([0.90, 0.12, 0.013, 0.80])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("log power", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)

    out = OUT_DIR / "fig04b_context_windows.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()

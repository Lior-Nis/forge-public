"""
Fig 5 — FORGE SpectralPatchEncoder architecture diagram.

Clean left-to-right pipeline that forks at the encoder, modelled on the
paper_mae architecture figure (research/paper_mae/figures/fig_architecture.png):

  3-axis acc. -> window -> Morlet CWT -> spectral-temporal patches -> SpectralPatchEncoder
      then the encoder FORKS:
        up   (orange) : MAE decoder -> MSE on masked patches   [Pretraining (unlabelled)]
        down (teal)   : BiGRU head  -> per-window FOG logit     [Downstream probe]

Presentation-only schematic; no data loading. Architecture details verified
against research/paper_final/draft.md §3.3.

Run:
  .venv/bin/python scripts/analysis/fig_architecture_forge.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ── Paths ────────────────────────────────────────────────────────────────────
OUT = Path("research/paper_final/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Colours (unified BLUES aesthetic) ────────────────────────────────────────
# Pretraining branch = light-mid blue; downstream branch = mid/strong blue;
# encoder block = navy; spectrograms/patch emphasis use the blue family.
ORANGE_BG = "#9ecae1"   # pretraining branch (light-mid blue)
ORANGE_DRK = "#2171b5"  # pretraining branch edges/arrows (strong blue)
TEAL = "#6baed6"        # downstream branch (light blue)
TEAL_DARK = "#08306b"   # downstream branch edges/arrows (navy)
DARK_SLATE = "#08306b"  # encoder block (navy)
WHITE = "#FFFFFF"
GRAY_SHAPE = "#888888"
LIGHT_GRAY = "#deebf7"  # pale blue window panels
ARROW_CLR = "#08306b"   # arrows/connectors (navy)
RED_ACCENT = "#d62728"  # sparing accent

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# ── Figure setup ─────────────────────────────────────────────────────────────
FIG_W, FIG_H = 13, 3.8
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0.0, 15.5)
ax.set_ylim(1.0, 4.7)
ax.axis("off")


# ── Helpers ──────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, fc, ec="#333333", lw=1.0, ls="solid", r=0.18, zorder=3):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color=ARROW_CLR, lw=1.3, zorder=6):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=10),
        zorder=zorder,
    )


def block_label(ax, x, y, text, color="#333333", fs=7.5, bold=False):
    ax.text(
        x,
        y,
        text,
        fontsize=fs,
        color=color,
        ha="center",
        va="center",
        fontweight="bold" if bold else "normal",
        multialignment="center",
    )


def stage_caption(ax, x, y, text, fs=6.5):
    ax.text(x, y, text, fontsize=fs, color="#333333", ha="center", va="top", multialignment="center")


# ── Layout constants ─────────────────────────────────────────────────────────
CY = 2.7
BH = 1.0
BY = CY - BH / 2
ARW = 0.35

N_BATCH = 3
DX_3D = 0.10
DY_3D = 0.08
MX3, MY3 = 0.03, 0.03


def pw3d(W):
    return W - 2 * MX3 - (N_BATCH - 1) * DX_3D


def ph3d():
    return BH - 2 * MY3 - (N_BATCH - 1) * DY_3D


# Stage x positions (left edge)
X_WIN = 0.25; W_WIN = 1.55     # Recording (long waveform + 2s window highlight)
X1 = X_WIN + W_WIN + ARW; W1 = 1.35      # Window (2/5/10 s)
X2 = X1 + W1 + ARW; W2 = 1.45            # Morlet CWT
X3 = X2 + W2 + ARW; W3 = 1.75            # Spectral-temporal patches
X4 = X3 + W3 + ARW; W4 = 2.70            # SpectralPatchEncoder (largest block)
FORK_X = X4 + W4 + 0.30

# Branch geometry (block size ~ param count: BiGRU 3.8M medium, decoder small)
BH_GRU = 0.72; BW_GRU = 1.95
BH_DEC = 0.56; BW_DEC = 1.75
LOSS_W = 1.65; LOSS_H = 0.50

BRANCH_X = FORK_X + 0.55
LOSS_X = BRANCH_X + max(BW_GRU, BW_DEC) + 0.34

UP_Y = CY + 1.45
DN_Y = CY - 1.45


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0: Recording — long waveform with a window highlighted
# ══════════════════════════════════════════════════════════════════════════════
WIN_FC = "#deebf7"
WIN_EC = "#2171b5"
rbox(ax, X_WIN, BY, W_WIN, BH, fc=WIN_FC, ec=WIN_EC, lw=1.0)

rng_rec = np.random.RandomState(77)
t_rec = np.linspace(0, 1, 200)
sig_rec = (
    0.12 * np.sin(2 * np.pi * 3.0 * t_rec)
    + 0.07 * np.sin(2 * np.pi * 7.1 * t_rec + 0.9)
    + rng_rec.randn(200) * 0.025
)
wx_rec = np.linspace(X_WIN + 0.06, X_WIN + W_WIN - 0.06, 200)
wy_rec = CY + sig_rec
ax.plot(wx_rec, wy_rec, color=WIN_EC, lw=0.65, zorder=5, clip_on=False)

hl_lo = int(200 * 0.40); hl_hi = int(200 * 0.66)
hl_x0 = wx_rec[hl_lo]; hl_x1 = wx_rec[hl_hi]
ax.add_patch(
    mpatches.Rectangle(
        (hl_x0, BY + 0.06),
        hl_x1 - hl_x0,
        BH - 0.12,
        facecolor="#9ecae1",
        alpha=0.50,
        edgecolor="#2171b5",
        linewidth=0.8,
        zorder=6,
        clip_on=False,
    )
)
ax.text(
    (hl_x0 + hl_x1) / 2,
    BY + BH + 0.08,
    "window",
    fontsize=5.5,
    color="#08306b",
    ha="center",
    va="bottom",
    clip_on=False,
    zorder=7,
)

block_label(ax, X_WIN + W_WIN / 2, BY - 0.20, "Recording", fs=9, bold=True)
arrow(ax, X_WIN + W_WIN, CY, X1, CY)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Window — 3-D stacked accelerometer panels
# ══════════════════════════════════════════════════════════════════════════════
wave_colours = ["#08306b", "#2171b5", "#6baed6"]
t = np.linspace(0, 1, 90)
PH = ph3d()

PW1 = pw3d(W1)
PX0_1 = X1 + MX3
PY0_1 = BY + MY3

for i in range(N_BATCH - 1, -1, -1):
    px = PX0_1 + i * DX_3D
    py = PY0_1 + i * DY_3D
    zz = 3 + (N_BATCH - 1 - i)
    rect = mpatches.FancyBboxPatch(
        (px, py),
        PW1,
        PH,
        boxstyle="round,pad=0,rounding_size=0.04",
        facecolor=LIGHT_GRAY,
        edgecolor="#999999",
        linewidth=0.65,
        zorder=zz,
        clip_on=False,
    )
    ax.add_patch(rect)
    rng_i = np.random.RandomState(i * 17 + 3)
    for ch, c in enumerate(wave_colours):
        ch_off = (1 - ch) * PH * 0.27
        amp = PH * 0.055
        sig = amp * np.sin(2 * np.pi * (2.5 + ch * 0.7) * t + i * 1.1) + rng_i.randn(90) * amp * 0.35
        wx = np.linspace(px + 0.05, px + PW1 - 0.05, 90)
        wy = np.clip(py + PH / 2 + ch_off + sig, py + 0.01, py + PH - 0.01)
        ax.plot(wx, wy, color=c, lw=0.6, zorder=zz + 0.5, clip_on=False)

block_label(ax, X1 + W1 / 2, BY - 0.20, "Window", fs=9, bold=True)
arrow(ax, X1 + W1, CY, X2, CY)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Morlet CWT — 3-D stacked spectrogram panels
# ══════════════════════════════════════════════════════════════════════════════
PW2 = pw3d(W2)
PX0_2 = X2 + MX3
PY0_2 = BY + MY3

for i in range(N_BATCH - 1, -1, -1):
    px = PX0_2 + i * DX_3D
    py = PY0_2 + i * DY_3D
    zz = 3 + (N_BATCH - 1 - i)
    rect = mpatches.FancyBboxPatch(
        (px, py),
        PW2,
        PH,
        boxstyle="round,pad=0,rounding_size=0.04",
        facecolor="#deebf7",
        edgecolor="#9ecae1",
        linewidth=0.65,
        zorder=zz,
        clip_on=False,
    )
    ax.add_patch(rect)
    rng_s = np.random.RandomState(i * 7 + 42)
    spec = rng_s.rand(12, 28)
    spec = np.cumsum(spec * 0.25, axis=0)
    spec[2:6, :] *= 1.6
    spec /= spec.max()
    ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="Blues",
        extent=[px + 0.04, px + PW2 - 0.04, py + 0.02, py + PH - 0.02],
        zorder=zz + 0.5,
        clip_on=False,
    )

block_label(ax, X2 + W2 / 2, BY - 0.20, "Morlet CWT", fs=9, bold=True)
arrow(ax, X2 + W2, CY, X3, CY)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: Spectral-temporal patch tokens — 3-D stacked patch-grid panels
# ══════════════════════════════════════════════════════════════════════════════
PW3 = pw3d(W3)
PX0_3 = X3 + MX3
PY0_3 = BY + MY3
PCOLS, PROWS = 7, 4

for i in range(N_BATCH - 1, -1, -1):
    px = PX0_3 + i * DX_3D
    py = PY0_3 + i * DY_3D
    zz = 3 + (N_BATCH - 1 - i)
    rect = mpatches.FancyBboxPatch(
        (px, py),
        PW3,
        PH,
        boxstyle="round,pad=0,rounding_size=0.04",
        facecolor="#deebf7",
        edgecolor="#9ecae1",
        linewidth=0.65,
        zorder=zz,
        clip_on=False,
    )
    ax.add_patch(rect)
    rng_p = np.random.RandomState(i * 11 + 5)
    cpw = (PW3 - 0.06) / PCOLS
    cph = (PH - 0.06) / PROWS
    for row in range(PROWS):
        for col in range(PCOLS):
            fc_p = "#c6dbef" if rng_p.rand() < 0.45 else "#2171b5"
            pr = mpatches.Rectangle(
                (px + 0.03 + col * cpw, py + 0.03 + row * cph),
                cpw * 0.85,
                cph * 0.80,
                facecolor=fc_p,
                edgecolor="#9ecae1",
                linewidth=0.2,
                zorder=zz + 0.5,
                clip_on=False,
            )
            ax.add_patch(pr)

block_label(ax, X3 + W3 / 2, BY - 0.20, "Patch tokens", fs=9, bold=True)
arrow(ax, X3 + W3, CY, X4, CY)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4: SpectralPatchEncoder (largest block)
# ══════════════════════════════════════════════════════════════════════════════
rbox(ax, X4 - 0.10, BY - 0.20, W4 + 0.20, BH + 0.40, fc="#deebf7", ec=RED_ACCENT, lw=1.3, ls="dashed", r=0.22, zorder=2)
rbox(ax, X4, BY, W4, BH, fc=DARK_SLATE, ec="#06224d", lw=1.2)
block_label(ax, X4 + W4 / 2, CY, "FORGE Encoder", fs=11, color=WHITE, bold=True)
ax.text(
    X4 + W4 / 2,
    BY + BH + 0.26,
    "[frozen at probe]",
    fontsize=5.8,
    color=RED_ACCENT,
    ha="center",
    va="bottom",
    style="italic",
)


# ══════════════════════════════════════════════════════════════════════════════
# Fork point
# ══════════════════════════════════════════════════════════════════════════════
FORK_Y = CY
ax.plot(FORK_X, FORK_Y, "o", color="#444444", ms=4.5, zorder=7)

# ── Branch UP: MAE pretraining (orange) ─────────────────────────────────────
arrow(ax, FORK_X, FORK_Y, FORK_X, UP_Y, color=ORANGE_DRK)
arrow(ax, FORK_X, UP_Y, BRANCH_X, UP_Y, color=ORANGE_DRK)

DEC_Y = UP_Y - BH_DEC / 2
rbox(ax, BRANCH_X, DEC_Y, BW_DEC, BH_DEC, fc=ORANGE_BG, ec=ORANGE_DRK, lw=1.1)
block_label(ax, BRANCH_X + BW_DEC / 2, UP_Y, "MAE decoder", fs=9, color="#08306b", bold=True)

ax.text(FORK_X + 0.12, (CY + UP_Y) / 2, "MAE pretraining",
        fontsize=9, color="#08306b", ha="left", va="center", fontweight="bold")
arrow(ax, BRANCH_X + BW_DEC, UP_Y, LOSS_X, UP_Y, color=ORANGE_DRK)
rbox(ax, LOSS_X, UP_Y - LOSS_H / 2, LOSS_W, LOSS_H, fc="#deebf7", ec=ORANGE_DRK, lw=0.9)
block_label(ax, LOSS_X + LOSS_W / 2, UP_Y + 0.06, "MSE on", fs=8.5, color="#08306b")
block_label(ax, LOSS_X + LOSS_W / 2, UP_Y - 0.10, "masked patches", fs=8.5, color="#08306b")


# ── Branch DOWN: Downstream probe (teal) ────────────────────────────────────
arrow(ax, FORK_X, FORK_Y, FORK_X, DN_Y, color=TEAL_DARK)
arrow(ax, FORK_X, DN_Y, BRANCH_X, DN_Y, color=TEAL_DARK)

GRU_Y = DN_Y - BH_GRU / 2
rbox(ax, BRANCH_X, GRU_Y, BW_GRU, BH_GRU, fc="#c6dbef", ec=TEAL_DARK, lw=1.1)
block_label(ax, BRANCH_X + BW_GRU / 2, DN_Y, "BiGRU head", fs=9, color="#08306b", bold=True)

arrow(ax, BRANCH_X + BW_GRU, DN_Y, LOSS_X, DN_Y, color=TEAL_DARK)
rbox(ax, LOSS_X, DN_Y - LOSS_H / 2, LOSS_W, LOSS_H, fc="#deebf7", ec=TEAL_DARK, lw=0.9)
block_label(ax, LOSS_X + LOSS_W / 2, DN_Y + 0.06, "per-window", fs=8.5, color="#08306b")
block_label(ax, LOSS_X + LOSS_W / 2, DN_Y - 0.10, "FOG logit", fs=8.5, color="#08306b")

ax.text(FORK_X + 0.12, (CY + DN_Y) / 2, "Downstream probe",
        fontsize=9, color="#08306b", ha="left", va="center", fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════════
# Context note (unobtrusive, lower-left under the main pipeline)
# ══════════════════════════════════════════════════════════════════════════════


# ── Save ─────────────────────────────────────────────────────────────────────
out = OUT / "fig05_architecture.png"
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
plt.close(fig)
print(f"+ saved {out}")

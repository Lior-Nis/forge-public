"""
Fig 13 — Sensor accessibility schematic (pure illustration, NO data).

Side-by-side deployment comparison:
  LEFT  = Yang et al. 2026 — 5 lower-limb IMUs (pelvis, both tibia/ankle, both feet),
          structured clinical assessment.
  RIGHT = FORGE — 1 lower-back IMU, concealed under clothing, deployment-practical
          home sensor geometry.

Each panel annotates setup burden / clothing compatibility / 24-7 suitability /
clinical use case. The central block shows the clinical-ICC contrast: FORGE
0.909 zero-shot exceeds Yang 0.732 (after 50 min local fine-tune), at different
deployment contexts.

See research/paper_final/plot_specs/13_sensor_accessibility.md and draft.md §5
("Sensor design as a clinical choice") and §1 ("The single-sensor argument").
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

FIGDIR = Path("research/paper_final/figures")

# Unified BLUES aesthetic: Yang in mid-blue, FORGE (ours) highlighted in red accent.
YANG_BLUE = "#4292c6"   # mid blue — Yang et al. sensors
FORGE_TEAL = "#08306b"  # dark navy — FORGE sensor (highlighted vs Yang's mid-blue)
BODY_GRAY = "#6baed6"   # light blue silhouette outline
BODY_FILL = "#deebf7"   # pale blue body fill

plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def draw_body(ax, cx, color=BODY_GRAY, fill=BODY_FILL):
    """Simple gray humanoid silhouette centered at horizontal position cx.

    Coordinate convention (axis 0..10 vertical): feet ~1, back/pelvis ~5,
    shoulders ~7.6, head ~9. Returns key anchor y-positions as a dict.
    """
    lw = 1.6
    # Head
    ax.add_patch(Circle((cx, 9.0), 0.55, facecolor=fill, edgecolor=color, lw=lw, zorder=2))
    # Torso (rounded box)
    torso = FancyBboxPatch(
        (cx - 0.85, 4.7), 1.7, 3.55,
        boxstyle="round,pad=0.02,rounding_size=0.5",
        facecolor=fill, edgecolor=color, lw=lw, zorder=2,
    )
    ax.add_patch(torso)
    # Arms (simple lines along the torso)
    for sx in (-1, 1):
        ax.plot([cx + sx * 0.85, cx + sx * 1.15], [7.9, 5.4],
                color=color, lw=lw, solid_capstyle="round", zorder=1)
    # Legs: two tapered limbs from pelvis (~4.7) down to feet (~1.0)
    hip_y = 4.7
    for sx in (-1, 1):
        hip_x = cx + sx * 0.42
        knee_x = cx + sx * 0.5
        ankle_x = cx + sx * 0.42
        ax.plot([hip_x, knee_x], [hip_y, 2.7], color=color, lw=lw + 1,
                solid_capstyle="round", zorder=1)
        ax.plot([knee_x, ankle_x], [2.7, 1.15], color=color, lw=lw + 1,
                solid_capstyle="round", zorder=1)
        # Foot
        foot = MplPath(
            [(ankle_x - sx * 0.05, 1.15), (ankle_x + sx * 0.55, 1.0),
             (ankle_x + sx * 0.55, 0.78), (ankle_x - sx * 0.18, 0.78),
             (ankle_x - sx * 0.05, 1.15)],
            [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY],
        )
        ax.add_patch(PathPatch(foot, facecolor=fill, edgecolor=color, lw=lw, zorder=1))

    return {
        "pelvis": (cx, 4.85),
        "tibia_l": (cx - 0.5, 3.3), "tibia_r": (cx + 0.5, 3.3),
        "ankle_l": (cx - 0.42, 1.35), "ankle_r": (cx + 0.42, 1.35),
        "foot_l": (cx - 0.42, 0.95), "foot_r": (cx + 0.42, 0.95),
        "lower_back": (cx, 5.55),
    }


def sensor(ax, xy, color, r=0.2, dashed=False):
    ec = "white"
    ax.add_patch(Circle(xy, r, facecolor=color, edgecolor=ec, lw=1.4, zorder=5))
    # subtle ring
    ax.add_patch(Circle(xy, r + 0.07, facecolor="none",
                        edgecolor=color, lw=1.0, alpha=0.5, zorder=4))


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.5, 6.2))

    for ax in (axL, axR):
        ax.set_xlim(0, 6)
        ax.set_ylim(-0.2, 10.8)
        ax.axis("off")
        ax.set_aspect("equal")

    # ---------------- LEFT: Yang et al. — 5 lower-limb IMUs ----------------
    cxL = 3.0
    aL = draw_body(axL, cxL, color=BODY_GRAY)
    for key in ("pelvis", "tibia_l", "tibia_r", "foot_l", "foot_r"):
        sensor(axL, aL[key], YANG_BLUE)
    # callout labels for the 5 sensors
    axL.annotate("pelvis", aL["pelvis"], xytext=(cxL + 1.55, 5.6), fontsize=8.5,
                 color=YANG_BLUE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=YANG_BLUE, lw=0.9))
    axL.annotate("tibia / ankle (L)", aL["tibia_l"], xytext=(cxL - 3.2, 3.3),
                 fontsize=8.5, color=YANG_BLUE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=YANG_BLUE, lw=0.9,
                                 shrinkB=4))
    axL.annotate("tibia / ankle (R)", aL["tibia_r"], xytext=(cxL + 1.3, 2.85),
                 fontsize=8.5, color=YANG_BLUE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=YANG_BLUE, lw=0.9,
                                 shrinkB=4))
    axL.annotate("foot (L)", aL["foot_l"], xytext=(cxL - 2.65, 0.95),
                 fontsize=8.5, color=YANG_BLUE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=YANG_BLUE, lw=0.9,
                                 shrinkB=4))
    axL.annotate("foot (R)", aL["foot_r"], xytext=(cxL + 1.45, 0.7),
                 fontsize=8.5, color=YANG_BLUE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=YANG_BLUE, lw=0.9,
                                 shrinkB=4))

    axL.text(cxL, 10.3, "Yang et al. 2026", ha="center", va="center",
             fontsize=11, fontweight="bold")
    axL.add_patch(FancyBboxPatch((cxL - 1.7, 0.05), 3.4, 0.55,
                  boxstyle="round,pad=0.02,rounding_size=0.12",
                  facecolor=YANG_BLUE, edgecolor="none", alpha=0.92, zorder=6))
    axL.text(cxL, 0.32, "5 IMUs · lower limbs", ha="center", va="center",
             fontsize=10, fontweight="bold", color="white", zorder=7)

    # ---------------- RIGHT: FORGE — 1 lower-back IMU ----------------
    cxR = 3.0
    aR = draw_body(axR, cxR, color=BODY_GRAY)
    # dashed "concealed under clothing" region over the torso/lower back
    axR.add_patch(FancyBboxPatch((cxR - 1.05, 4.55), 2.1, 1.9,
                  boxstyle="round,pad=0.02,rounding_size=0.25",
                  facecolor="none", edgecolor=FORGE_TEAL, lw=1.3, ls=(0, (4, 3)),
                  alpha=0.85, zorder=3))
    sensor(axR, aR["lower_back"], FORGE_TEAL, r=0.22)
    axR.annotate("lower back\n(under clothing)", aR["lower_back"],
                 xytext=(cxR + 0.95, 6.15), fontsize=8.0, color=FORGE_TEAL,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=FORGE_TEAL, lw=0.9))

    axR.text(cxR, 10.3, "FORGE", ha="center", va="center",
             fontsize=11, fontweight="bold")
    axR.add_patch(FancyBboxPatch((cxR - 1.7, 0.05), 3.4, 0.55,
                  boxstyle="round,pad=0.02,rounding_size=0.12",
                  facecolor=FORGE_TEAL, edgecolor="none", alpha=0.95, zorder=6))
    axR.text(cxR, 0.32, "1 IMU · lower back", ha="center", va="center",
             fontsize=10, fontweight="bold", color="white", zorder=7)

    # ---------------- Minimal central connective note ----------------
    # Deployment-tradeoff details and the ICC contrast live in the caption; the
    # figure stays a clean schematic.
    fig.text(0.5, 0.52, "vs", ha="center", va="center",
             fontsize=12, fontweight="bold", color="0.55")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02, wspace=0.06)

    out = FIGDIR / "fig13_sensor_accessibility.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

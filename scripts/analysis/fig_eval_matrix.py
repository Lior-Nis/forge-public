"""
Fig 2 — Evaluation design matrix.

A visual matrix comparing prior FOG papers against FORGE on six evaluation
criteria. Blues-only palette; variable column widths (supplementary columns
narrow); no partial-legend entry (no partial cells exist in the table).
Caption and footnote text belong in the paper caption, not the figure.

Output (300 dpi, bbox_inches='tight'):
  research/paper_final/figures/fig02_eval_design_matrix.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

FIGDIR = Path("research/paper_final/figures")

# ---- Blues-only color palette -----------------------------------------------
MET         = "#2171b5"   # criterion met — dark blue, white symbol
NOT_MET     = "#deebf7"   # criterion not met — very pale blue, dark symbol
NOT_MET_TXT = "#1a5276"   # dark navy ✗ in pale cells
FLAG_BG     = "#4292c6"   # flagged numeric (medium blue) — white text
NA          = "#bdc8d4"   # not applicable — muted blue-gray, dark text
NEUTRAL     = "#deebf7"   # numeric / metric cells — pale blue, dark text
NEUTRAL_TXT = "#08306b"   # dark navy for text in pale cells
HILITE      = "#9ecae1"   # FORGE row highlight — medium-light blue
HEADER_BG   = "#08306b"   # column headers — dark navy
WHITE       = "#FFFFFF"

# ---- Column definitions: (header text, display width) -----------------------
# Criteria columns (1–4) carry the main comparison; supplementary
# columns (5–6) kept minimal width to avoid dead whitespace.
COLS = [
    ("Home-recorded\ntraining",              1.20),
    ("Patient-level\nsplit",                 1.20),
    ("External\ncohort",                     1.20),
    ("Naturalistic\ntest",                   1.20),
    ("N ext.\npatients",                     0.72),
    ("Metric",                               0.95),
]

COL_LABELS = [c[0] for c in COLS]
COL_WIDTHS = [c[1] for c in COLS]

# ---- Row data ---------------------------------------------------------------
# Cell codes: "Y"=criterion met  "N"=not met  "A"=N/A  str+"!"=flagged numeric
ROWS = [
    ("Al-Adhaileh et al. 2025",
     ["N", "N", "N", "N", "A", "AUC"]),
    ("Shaban 2024",
     ["N", "Y", "N", "N", "3", "Accuracy"]),
    ("Sigcha et al. 2024",
     ["N", "Y", "Y", "N", "—", "AUC"]),
    ("Borzì et al. 2025",
     ["N", "Y", "Y", "N", "—", "FP rate"]),
    ("Rodríguez-Martín et al. 2017",
     ["N", "Y", "N", "N", "A", "Sens./Spec."]),
    ("Salomon et al. 2024 Kaggle",
     ["Y", "Y", "N", "N", "A", "AP / F1"]),
    ("Salomon et al. 2026 (no retrain)",
     ["A", "Y", "Y", "N", "12", "F1"]),
    ("FORGE (ours)",
     ["Y", "Y", "Y", "Y", "12", "ICC / AP"]),
]

FORGE_IDX = len(ROWS) - 1

SYMBOL = {"Y": "✓", "N": "✗", "A": "N/A"}
SYM_BG = {"Y": MET,     "N": NOT_MET, "A": NA}
SYM_TC = {"Y": WHITE,   "N": NOT_MET_TXT, "A": NEUTRAL_TXT}


def draw_cell(ax, x, y, w, h, value, fontsize=12):
    """Render one matrix cell as a colored rectangle + centered text."""
    is_flag = isinstance(value, str) and value.endswith("!")
    if value in SYMBOL:
        bg, txt, tc = SYM_BG[value], SYMBOL[value], SYM_TC[value]
        fw = "bold"
    else:
        clean = value[:-1] if is_flag else value
        bg  = FLAG_BG if is_flag else NEUTRAL
        txt = clean
        tc  = WHITE if is_flag else NEUTRAL_TXT
        fw  = "bold" if is_flag else "normal"
        fontsize = fontsize - 2

    ax.add_patch(Rectangle((x, y), w, h, facecolor=bg, edgecolor=WHITE,
                           linewidth=1.5, zorder=2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
            color=tc, fontsize=fontsize, fontweight=fw, zorder=3)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    n_rows   = len(ROWS)
    label_w  = 3.0
    cell_h   = 0.62
    header_h = 1.05

    col_xs  = [label_w + sum(COL_WIDTHS[:j]) for j in range(len(COLS))]
    grid_w  = label_w + sum(COL_WIDTHS)
    grid_h  = n_rows * cell_h + header_h

    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    ax.set_xlim(-0.1, grid_w + 0.1)
    ax.set_ylim(-0.55, grid_h + 0.10)
    ax.axis("off")

    # ---- Header row ---------------------------------------------------------
    hy = n_rows * cell_h
    ax.add_patch(Rectangle((0, hy), label_w, header_h, facecolor=HEADER_BG,
                           edgecolor=WHITE, linewidth=1.5, zorder=2))
    ax.text(label_w / 2, hy + header_h / 2, "Study", ha="center", va="center",
            color=WHITE, fontsize=12, fontweight="bold", zorder=3)

    for col, cx, cw in zip(COL_LABELS, col_xs, COL_WIDTHS):
        ax.add_patch(Rectangle((cx, hy), cw, header_h, facecolor=HEADER_BG,
                               edgecolor=WHITE, linewidth=1.5, zorder=2))
        ax.text(cx + cw / 2, hy + header_h / 2, col, ha="center",
                va="center", color=WHITE, fontsize=9.5, rotation=0,
                zorder=3, linespacing=1.1)

    # ---- Data rows (drawn top → bottom) ------------------------------------
    for i, (label, vals) in enumerate(ROWS):
        ry = (n_rows - 1 - i) * cell_h
        is_forge = (i == FORGE_IDX)

        lbl_bg = HILITE if is_forge else WHITE
        ax.add_patch(Rectangle((0, ry), label_w, cell_h, facecolor=lbl_bg,
                               edgecolor="#BDC3C7", linewidth=1.0, zorder=2))
        ax.text(0.12, ry + cell_h / 2, label, ha="left", va="center",
                color="#1B2631", fontsize=9.0,
                fontweight="bold" if is_forge else "normal", zorder=3)

        for v, cx, cw in zip(vals, col_xs, COL_WIDTHS):
            if is_forge:
                ax.add_patch(Rectangle((cx, ry), cw, cell_h,
                                       facecolor=HILITE, edgecolor="none",
                                       zorder=1))
            draw_cell(ax, cx, ry, cw, cell_h, v)

    # ---- FORGE row highlight border -----------------------------------------
    fy = (n_rows - 1 - FORGE_IDX) * cell_h
    ax.add_patch(FancyBboxPatch(
        (0.02, fy + 0.02), grid_w - 0.04, cell_h - 0.04,
        boxstyle="square,pad=0", facecolor="none", edgecolor=HEADER_BG,
        linewidth=2.0, alpha=0.7, zorder=4))

    # ---- Legend (no "partial" entry — zero partial cells in the table) ------
    legend_items = [
        (MET,     "✓", WHITE,        "criterion met"),
        (NOT_MET, "✗", NOT_MET_TXT,  "not met"),
        (NA,      "N/A", NEUTRAL_TXT, "not applicable"),
    ]
    lx, ly, sw = 0.0, -0.44, 0.32
    for color, sym, sym_tc, desc in legend_items:
        ax.add_patch(Rectangle((lx, ly), sw, sw, facecolor=color,
                               edgecolor=WHITE, linewidth=1.0, zorder=2))
        ax.text(lx + sw / 2, ly + sw / 2, sym, ha="center", va="center",
                color=sym_tc, fontsize=9.5, fontweight="bold", zorder=3)
        ax.text(lx + sw + 0.12, ly + sw / 2, desc, ha="left", va="center",
                color=NEUTRAL_TXT, fontsize=9.0, zorder=3)
        lx += sw + 0.12 + 0.105 * len(desc) + 0.50

    fig.tight_layout()
    png = FIGDIR / "fig02_eval_design_matrix.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()

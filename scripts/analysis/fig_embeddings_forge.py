"""
fig_embeddings_comparison.py — FORGE vs Supervised: FOG clustering comparison.

1×2 panel, both coloured by FOG state:
  Left:  FORGE embeddings  → FOG clusters tightly
  Right: Supervised from scratch → FOG scattered

Narrative: FORGE pretrained features cluster FOG across patients; a model
trained purely on labels overfits to patient-specific patterns.

Run: uv run python research/paper_mae/scripts_new/fig_embeddings_comparison.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT        = Path(__file__).resolve().parents[2]   # repo root (acc_base)
EMBD_DIR    = ROOT / "logs/embeddings"
OUT         = ROOT / "research/paper_final/figures"
OUT.mkdir(parents=True, exist_ok=True)

# Unified BLUES aesthetic: by-FOG colouring -> 2 blues
FOG_CLR    = "#08306b"   # FOG = primary navy
NONFOG_CLR = "#9ecae1"   # non-FOG = light blue
TITLE_FORGE = "#08306b"  # ours / primary navy
TITLE_SUP   = "#4292c6"  # mid blue

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":        8,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})


def load_embeddings(name):
    path = EMBD_DIR / name
    emb  = np.load(path / "embeddings.npy")
    meta = pd.read_csv(path / "metadata.csv")
    return emb, meta


def compute_umap(emb, seed=42):
    import umap as umap_lib
    reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1,
                             metric="cosine", random_state=seed)
    return reducer.fit_transform(emb)


def _clean_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_fog_panel(ax, coords, meta, title, title_color, bg_color):
    is_fog = meta["true_label"].values == 1
    rng    = np.random.default_rng(0)

    bg_idx = np.where(~is_fog)[0]
    bg_sel = rng.choice(bg_idx, int(len(bg_idx) * 0.12), replace=False)
    ax.scatter(coords[bg_sel, 0], coords[bg_sel, 1],
               c=NONFOG_CLR, s=3, alpha=0.20, linewidths=0, rasterized=True)

    fg_idx = np.where(is_fog)[0]
    ax.scatter(coords[fg_idx, 0], coords[fg_idx, 1],
               c=FOG_CLR, s=6, alpha=0.80, linewidths=0, rasterized=True)

    ax.legend(handles=[
        mpatches.Patch(color=FOG_CLR,    label="FOG"),
        mpatches.Patch(color=NONFOG_CLR, label="non-FOG"),
    ], loc="lower right", fontsize=7, frameon=True, framealpha=0.9,
              edgecolor="none", handlelength=0.8)
    _clean_ax(ax)
    ax.set_facecolor(bg_color)
    ax.set_title(title, fontsize=8, fontweight="bold", color=title_color, pad=5)


def main():
    print("Loading embeddings...")
    emb_forge, meta_forge = load_embeddings("vit12_ep4")
    emb_sup,   meta_sup   = load_embeddings("supervised_scratch")

    mask_f = meta_forge["dataset"].isin(["defog", "fogathome"])
    mask_s = meta_sup["dataset"].isin(["defog", "fogathome"])
    emb_forge, meta_forge = emb_forge[mask_f], meta_forge[mask_f].reset_index(drop=True)
    emb_sup,   meta_sup   = emb_sup[mask_s],   meta_sup[mask_s].reset_index(drop=True)

    print(f"  FORGE:      {emb_forge.shape}")
    print(f"  Supervised: {emb_sup.shape}")

    print("Computing UMAP (FORGE)...")
    coords_forge = compute_umap(emb_forge)
    print("Computing UMAP (Supervised)...")
    coords_sup   = compute_umap(emb_sup)

    fig, (ax_forge, ax_sup) = plt.subplots(
        1, 2, figsize=(8.5, 3.9),
        gridspec_kw={"wspace": 0.06, "left": 0.02, "right": 0.98,
                     "top": 0.90, "bottom": 0.02}
    )

    plot_fog_panel(ax_forge, coords_forge, meta_forge,
                   "FORGE — FOG state", TITLE_FORGE, "#f7fbff")
    plot_fog_panel(ax_sup,   coords_sup,   meta_sup,
                   "Supervised from scratch — FOG state", TITLE_SUP, "#deebf7")

    fig.suptitle("FORGE clusters FOG across patients · Supervised scatters FOG with patient structure",
                 fontsize=9, y=0.97)

    out_path = OUT / "figS1_embeddings.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

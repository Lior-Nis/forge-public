#!/usr/bin/env python
"""Fig S1 — Embedding geometry: the patient shortcut (per-model UMAP).

One row, two panels; each model is projected with its OWN UMAP (visualization only):
  FORGE | supervised, coloured by PATIENT identity (gray background + the 6
  highest-window-count patients highlighted in shared colors). Shows the shortcut:
  supervised forms same-color patient islands; FORGE intermixes patients. Annotated
  with the patient-identity silhouette on the displayed 2-D UMAP coords (higher =
  more patient-clustered; supervised −0.168 > FORGE −0.394). Absolute values are
  negative (40 patients) — read the COMPARISON, not the magnitude.

The cross-cohort FOG-centroid cosine (FORGE's FOG representation is cohort-invariant,
supervised's is not) is reported in the caption line, not as a separate scatter — a
FOG/non-FOG colouring of a cohort-imbalanced per-model UMAP does not convey it, and
a dedicated visual exists at logs/embeddings/generalization/fog_centroid_similarity.png.

Data: logs/embeddings/{vit12_ep4 (FORGE), supervised_scratch}/{coords_umap.npy, metadata.csv}
Out:  research/paper_final/figures/figS1_embedding_geometry.png  (PNG only)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white",
})

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "logs/embeddings"
GEN = EMB / "generalization"
OUT = ROOT / "research/paper_final/figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [("vit12_ep4", "FORGE (2D-MAE pretrained)", "MAE probe (vit12 ep4)"),
          ("supervised_scratch", "Supervised from scratch", "Supervised from scratch")]
N_HL = 6                       # number of highlighted patients
GRAY = "#c7ccd1"
HL_CMAP = plt.cm.tab10

# cross-cohort FOG-centroid cosine (for the caption line)
cent_cos = pd.read_csv(GEN / "centroid_similarity_summary.csv").set_index("model")["mean_cross_cohort_sim"]


def load(tag):
    co = np.load(EMB / tag / "coords_umap.npy")
    md = pd.read_csv(EMB / tag / "metadata.csv")
    return co, md


def patient_silhouette(co, md):
    """Patient-identity silhouette on the DISPLAYED 2-D UMAP coords (full data, no
    sampling -> deterministic), so the annotation describes the picture the reader
    sees. NOT the per_patient_silhouette.csv (that is within-patient FOG)."""
    pid = LabelEncoder().fit_transform(md["patient_id"])
    return float(silhouette_score(co, pid))


# highlighted patients: top-N by window count (most visible); shared across panels
_, md0 = load(MODELS[0][0])
hl_patients = md0["patient_id"].value_counts().head(N_HL).index.tolist()
hl_color = {p: HL_CMAP(i) for i, p in enumerate(hl_patients)}

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), constrained_layout=True)

for col, (tag, title, _key) in enumerate(MODELS):
    co, md = load(tag)
    is_hl = md["patient_id"].isin(hl_patients).to_numpy()
    ax = axes[col]
    ax.scatter(co[~is_hl, 0], co[~is_hl, 1], c=GRAY, s=3, alpha=0.25, linewidths=0, zorder=1)
    for p in hl_patients:
        m = (md["patient_id"] == p).to_numpy()
        ax.scatter(co[m, 0], co[m, 1], c=[hl_color[p]], s=10, alpha=0.85, linewidths=0, zorder=3)
    s = patient_silhouette(co, md)
    ax.text(0.03, 0.97, f"patient silhouette (UMAP) = {s:+.3f}", transform=ax.transAxes,
            fontsize=9, va="top", ha="left", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

axes[0].set_ylabel("coloured by patient identity", fontsize=10, fontweight="bold")

# highlighted-patient legend (to the right of the right panel)
hl_handles = [Line2D([0], [0], marker="o", ls="", mfc=hl_color[p], mec="none", ms=6,
                     label=f"patient {p}") for p in hl_patients]
hl_handles.append(Line2D([0], [0], marker="o", ls="", mfc=GRAY, mec="none", ms=6,
                         label="other patients"))
axes[1].legend(handles=hl_handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
               fontsize=7.5, frameon=False, title="highlighted", title_fontsize=8)

fcos, scos = cent_cos.get("MAE probe (vit12 ep4)"), cent_cos.get("Supervised from scratch")

fig.savefig(OUT / "figS1_embedding_geometry.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("saved:", OUT / "figS1_embedding_geometry.png")
print(f"FOG-centroid cos (caption): FORGE={fcos:.3f}  supervised={scos:.3f}")
print(f"highlighted patients: {hl_patients}")

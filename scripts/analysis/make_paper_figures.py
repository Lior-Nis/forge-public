"""
Generate the publication/thesis figure set for the FORGE paper from the
evaluation outputs. Each figure is saved as PNG (300 dpi) to
research/paper_final/figures/.

Figures
  1 headline_icc          — ICC(%TF) & ICC(#FOG): FORGE vs Yang 2026 + inter-rater band
  2 context_dataset_ap    — AP heatmap, context × model, per dataset
  3 strategy_ap           — pretraining benefit: probe/finetune/supervised AP per context
  4 threshold_robustness  — %TF ICC under fixed-0.5 / test-PR11 / DeFOG-val (transfer test)
  5 pr_roc_fogathome      — PR & ROC curves on FogAtHome (best models)
  6 icc_scatter           — per-session %TF predicted vs ground truth (the 0.909 figure)
  7 finetune_curve        — ICC vs fine-tuning minutes (Fig 12; if data present)
  8 auc_by_dataset        — AUC across datasets incl. rare-event DailyLiving

Usage: python scripts/analysis/make_paper_figures.py
"""

import os
import importlib.util
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

# ── reuse audited ICC/threshold helpers ──
_spec = importlib.util.spec_from_file_location("ict", "scripts/eval/compute_icc_thresholds.py")
ict = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ict)

FIGDIR = Path("research/paper_final/figures")
CACHE = Path("logs/comprehensive_eval_cache")
CSV = "logs/comprehensive_eval.csv"
# FogAtHome AND DailyLiving metrics are reported on the same walking+standing
# activity subset (Activity ∈ {1,4}) so the heatmaps compare them apples-to-apples;
# see scripts/eval/eval_{fogathome,dailyliving}_walkstand.py. Whole-recording numbers
# are inflated by trivial FOG-vs-sedentary separation. Kaggle is a structured walking
# protocol (no free-living activity codes) and stays on the whole recording.
WALKSTAND = {
    "fogathome":   "logs/RESULTS_fogathome_walkstand.csv",
    "dailyliving": "logs/RESULTS_dailyliving_walkstand.csv",
}
# Per-frame Activity annotation → enables the walking-only (gait-matched) %TF ICC.
# Legend (Salomon, confirmed 2026-06-07): 1=walking is the only gait state.
GAIT_LABELS = os.path.join(os.environ.get("FORGE_DATASETS_ROOT", os.path.expanduser("~/Datasets")), "fogathome_dataset_forlior/fogathome_dataset/labels.csv")
FIGDIR.mkdir(parents=True, exist_ok=True)

# ── style ──
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.55,
    "legend.frameon": False, "figure.facecolor": "white",
})
C = {
    # Unified sequential-blues palette (ours = darkest navy → baselines lighter).
    "probe": "#08306b",       # ours / primary — darkest navy
    "finetune": "#2171b5",    # strong blue
    "supervised": "#6baed6",  # light blue
    "forge": "#08306b",       # FORGE = ours
    "yang": "#4292c6",        # mid blue (reference baseline)
    "salomon": "#9ecae1",     # light-mid blue (reference baseline)
    "ref": "#d62728",         # red accent — reference / chance / threshold lines only
    "band": "#deebf7",
    "dark": "#2C3E50",
}

# ── Yang et al. 2026 (npj PD) reference numbers ──
YANG = {
    "tf":  {"zero": (0.562, 0.141), "ft50": (0.732, 0.138), "indist": 0.886},
    "fog": {"zero": (0.298, 0.270), "ft50": (0.623, 0.214), "indist": 0.573},
}
# Reported human inter-rater ICC bands for video FOG scoring (verified against the
# cited sources, 2026-06-09): %TF spans Scully 2024 (0.50) to Kondo 2022 (0.89);
# #FOG only reported by Scully 2024 (0.63, 95% CI 0.43-0.85); Duration spans Scully
# (0.78) to Kondo (0.99). Replaces the earlier unsourced 0.73-0.99 / 0.39-0.95 band,
# which conflated outcomes (Kondo's 0.99 is duration, not %TF) and a third study
# (Morris 2012, %TF 0.73) that neither cited paper's own data supports.
RATER = {"tf": (0.50, 0.89), "fog": (0.43, 0.85), "dur": (0.78, 0.99)}
# Salomon 2026 best transferred Kaggle model (1 lower-back IMU, same FogAtHome
# cohort, gait-filtered, per-subject) — fixed literature reference (ICC_HANDOFF §2).
SALOMON = {"tf": (0.785, (0.28, 0.94)), "fog": (0.656, (0.12, 0.89)), "dur": (0.950, (0.83, 0.99))}


def save(fig, name):
    for ext in ("png",):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  ✓ {name}")


def load_csv():
    df = pd.read_csv(CSV)
    df["model"] = df.model_type
    df["threshold_pct"] = df.threshold_pct.astype(str)
    return _apply_dl_walkstand(df)


_WS_CACHE = {}


def walkstand(dataset):
    """Walking+standing metrics for a dataset (context, model, AUC, AP, NormAP,
    prevalence, tf_icc), cached. None if not yet computed."""
    if dataset not in _WS_CACHE:
        p = Path(WALKSTAND.get(dataset, ""))
        _WS_CACHE[dataset] = pd.read_csv(p) if p.exists() else None
    return _WS_CACHE[dataset]


def _apply_dl_walkstand(df):
    """Override whole-recording frame-level (segmentation) AUC/AP/NormAP/prevalence
    with the walking+standing values for every dataset that has them."""
    for dataset in WALKSTAND:
        ws = walkstand(dataset)
        if ws is None:
            continue
        wmap = ws.set_index(["context", "model"])
        mask = (df.dataset == dataset) & (df.level == "segmentation")
        keys = list(zip(df.loc[mask, "context"], df.loc[mask, "model"]))
        for col in ["AUC", "AP", "NormAP", "prevalence"]:
            df.loc[mask, col] = [wmap.loc[k, col] if k in wmap.index else np.nan
                                 for k in keys]
    return df


_GAIT_LAB = None


def _gait_labels():
    """Per-frame Activity annotation (session_id, abs_frame, Activity), cached."""
    global _GAIT_LAB
    if _GAIT_LAB is None and Path(GAIT_LABELS).exists():
        gl = pd.read_csv(GAIT_LABELS, usecols=["Id", "Activity"])
        gl["session_id"] = gl["Id"].str.rsplit("_", n=1).str[0]
        gl["abs_frame"] = gl["Id"].str.rsplit("_", n=1).str[1].astype(int)
        _GAIT_LAB = gl[["session_id", "abs_frame", "Activity"]]
    return _GAIT_LAB


def fa_frames(ctx, model):
    p = CACHE / f"fogathome_{ctx}_{model}_frames.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    gl = _gait_labels()
    if gl is not None and "Activity" not in df.columns and {"session_id", "abs_frame"} <= set(df.columns):
        df = df.merge(gl, on=["session_id", "abs_frame"], how="left")
    return df


def defog_threshold(ctx, model):
    p = CACHE / f"kaggle_{ctx}_{model}_frames.parquet"
    if not p.exists():
        return 0.5
    d = pd.read_parquet(p)
    return ict.pr11_threshold(d["native_label"].values.astype(int), d["pred_prob_fog"].values)


# ─────────────────────────────────────────────────────────────────────────────
def fig_headline_icc():
    """Forest-style comparison of ICC(%TF) and ICC(#FOG)."""
    df = fa_frames("mc", "probe")
    thr = defog_threshold("mc", "probe")
    m = ict.clinical_metrics(df, thr)
    ours = {"tf": m["tf_icc"], "fog": m["fog_icc"]}

    labels = ["Yang zero-shot\n(6 ext. cohorts, 5 IMU)", "Yang +50 min fine-tune\n(5 IMU)",
              "FORGE mc/probe\nzero-shot, 1 IMU (ours)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, ep, title in [(axes[0], "tf", "ICC(% Time Frozen)"),
                          (axes[1], "fog", "ICC(# FOG Episodes)")]:
        y = YANG[ep]; rater = RATER[ep]
        ax.axvspan(rater[0], rater[1], color=C["band"], alpha=0.35, zorder=0,
                   label=f"inter-rater ({rater[0]}–{rater[1]})")
        # NB: Yang in-distribution ceiling (0.886/0.573) deliberately NOT plotted
        # — ICC panels must not imply FORGE beats Yang's in-distribution ceiling
        # (ICC_HANDOFF §0.1). Yang values are 5-IMU reference, not a beaten baseline.
        rows = [(y["zero"][0], y["zero"][1], C["yang"]),
                (y["ft50"][0], y["ft50"][1], C["yang"]),
                (ours[ep], 0, C["forge"])]
        ypos = np.arange(len(rows))[::-1]
        for yp, (val, err, col) in zip(ypos, rows):
            ax.errorbar(val, yp, xerr=(err if err else None), fmt="o", ms=7.5,
                        color=col, capsize=3, lw=1.6, zorder=3)
            ax.text(val + (err or 0) + 0.025, yp, f"{val:.3f}", ha="left", va="center",
                    fontsize=8.5, fontweight="bold", color=col)
        ax.set_ylim(-0.6, 2.6); ax.set_xlim(0, 1.12)
        ax.set_xlabel("ICC"); ax.set_title(title)
        ax.axvline(0.75, color=C["ref"], ls=":", lw=1.1)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower left", fontsize=7.5)
    axes[0].set_yticks([2, 1, 0]); axes[0].set_yticklabels(labels, fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig01_headline_icc_fogathome")


# ─────────────────────────────────────────────────────────────────────────────
def _context_heatmaps(df, metric, fname, suptitle, shared):
    """Three dataset heatmaps (context × training strategy), Blues colormap, with a
    SINGLE shared color scale + one colorbar so cell colors mean the same thing
    across every panel.
      shared=(vmin,vmax) -> fixed bounds.
      shared="auto"      -> bounds = global min/max across all three matrices."""
    datasets = ["kaggle", "fogathome", "dailyliving"]
    ctx_order = ["sc", "mc", "lc", "mc+sc", "lc+sc", "lc+mc", "lc+mc+sc"]
    models = ["probe", "finetune", "supervised"]

    # Build every matrix first so the color scale can span all three.
    mats = {}
    for d in datasets:
        sub = df[(df.dataset == d) & (df.level == "segmentation")]
        mats[d] = (sub.pivot_table(index="context", columns="model", values=metric)
                      .reindex(index=ctx_order, columns=models))
    if shared == "auto":
        allv = np.concatenate([m.values.ravel() for m in mats.values()])
        vmin, vmax = float(np.nanmin(allv)), float(np.nanmax(allv))
    else:
        vmin, vmax = shared

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    last_im = None
    for ax, d in zip(axes, datasets):
        mat = mats[d]
        last_im = ax.imshow(mat.values, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=25, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(ctx_order))); ax.set_yticklabels(ctx_order, fontsize=8)
        if d in WALKSTAND:
            dprev = df[(df.dataset == d) & (df.level == "segmentation")]["prevalence"].mean() * 100
            ax.set_title(f"{d} · walk+stand\n(frame {metric} · {dprev:.0f}% prev)", fontsize=9)
        else:
            ax.set_title(f"{d}\n(frame {metric})", fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if not np.isnan(v):
                    norm = (v - vmin) / (vmax - vmin + 1e-9)
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if norm > 0.45 else "#1B2631", fontsize=7.5)
        ax.grid(False)
    fig.subplots_adjust(left=0.06, right=0.87, top=0.97, bottom=0.18, wspace=0.32)
    cax = fig.add_axes([0.895, 0.18, 0.013, 0.79])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label(metric, fontsize=9, rotation=90, labelpad=6)
    save(fig, fname)


def fig_context_dataset_ap(df):
    # Fig 10  — AUC heatmaps (single shared scale = global min/max across datasets)
    _context_heatmaps(
        df, "AUC", "fig10_context_interaction",
        "AUC across context × training strategy × dataset (single-context + ensembles)",
        shared="auto")
    # Fig 10b — AP heatmaps (single shared scale = global min/max; DailyLiving is a
    # rare-event task so its cells read pale against the structured sets — by design)
    _context_heatmaps(
        df, "AP", "fig10b_context_interaction_ap",
        "AP across context × training strategy × dataset  (single shared scale across datasets)",
        shared="auto")


def _icc_grid(df_icc, valcol, endpoint, fname):
    """Three dataset ICC heatmaps (context × training strategy), fixed 0–1 scale
    (the natural ICC range). df_icc has columns context, model, dataset, <valcol>.
    fogathome/dailyliving panels are labelled walk+stand."""
    ctx_order = ["sc", "mc", "lc", "mc+sc", "lc+sc", "lc+mc", "lc+mc+sc"]
    models    = ["probe", "finetune", "supervised"]
    datasets  = ["kaggle", "fogathome", "dailyliving"]
    vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    last_im = None
    for ax, ds in zip(axes, datasets):
        sub = df_icc[df_icc.dataset == ds]
        mat = (sub.pivot_table(index="context", columns="model", values=valcol)
                  .reindex(index=ctx_order, columns=models))
        last_im = ax.imshow(mat.values, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(ctx_order)))
        ax.set_yticklabels(ctx_order, fontsize=8)
        title = (f"{ds} · walk+stand  ({endpoint} ICC)" if ds in WALKSTAND
                 else f"{ds}  ({endpoint} ICC)")
        ax.set_title(title, fontsize=9.5)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if not np.isnan(v):
                    norm = (np.clip(v, vmin, vmax) - vmin) / (vmax - vmin + 1e-9)
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if norm > 0.45 else "#1B2631", fontsize=7.5)
        ax.grid(False)

    fig.subplots_adjust(left=0.06, right=0.87, top=0.97, bottom=0.18, wspace=0.32)
    cax = fig.add_axes([0.895, 0.18, 0.013, 0.79])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("ICC", fontsize=9, rotation=90, labelpad=6)
    save(fig, fname)


def fig_icc_heatmaps():
    """Fig 10c — %TF ICC heatmaps, context × training strategy, protocol-B threshold.
    FogAtHome + DailyLiving on the walking+standing subset."""
    df_icc = pd.read_csv("logs/RESULTS_icc_all_datasets.csv")
    for dataset in WALKSTAND:
        ws = walkstand(dataset)
        if ws is None:
            continue
        df_icc = df_icc[df_icc.dataset != dataset]
        dl = ws[["context", "model", "tf_icc"]].copy()
        dl["dataset"] = dataset
        df_icc = pd.concat([df_icc, dl], ignore_index=True)
    _icc_grid(df_icc, "tf_icc", "%TF", "fig10c_icc_heatmaps")


def fig_fog_icc_heatmaps():
    """Fig 10d — #FOG-episode ICC heatmaps, same basis as fig10c. Episodes are
    gap-merged (0.6 s); FogAtHome + DailyLiving conditioned on walking+standing
    (computed in eval_fog_icc_all_datasets.py)."""
    p = Path("logs/RESULTS_fog_icc_all_datasets.csv")
    if not p.exists():
        print("  · skip fig10d (RESULTS_fog_icc_all_datasets.csv missing)")
        return
    _icc_grid(pd.read_csv(p), "fog_icc", "#FOG", "fig10d_fog_icc_heatmaps")


# ─────────────────────────────────────────────────────────────────────────────
def fig_strategy_ap(df):
    """Pretraining benefit: AP per context, grouped by strategy, FogAtHome."""
    sub = df[(df.dataset == "fogathome") & (df.level == "segmentation")]
    ctxs = ["sc", "mc", "lc"]; models = ["supervised", "probe", "finetune"]
    fig, ax = plt.subplots(figsize=(7, 5))
    w = 0.25; x = np.arange(len(ctxs))
    for i, mdl in enumerate(models):
        vals = [sub[(sub.context == c) & (sub.model == mdl)]["AP"].mean() for c in ctxs]
        ax.bar(x + (i - 1) * w, vals, w, label=mdl, color=C[mdl], edgecolor="white", linewidth=0.7)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels([c.upper() for c in ctxs])
    ax.set_ylabel("Average Precision (frame-level)")
    ax.set_xlabel("Context length"); ax.set_ylim(0, 0.95)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0),
              fontsize=8, handlelength=1.4, columnspacing=1.2)
    fig.tight_layout()
    save(fig, "fig08_probe_vs_finetune")


# ─────────────────────────────────────────────────────────────────────────────
def fig_threshold_robustness():
    """%TF ICC under fixed-0.5, test-PR11, DeFOG-val — shows MC transfers, SC/sup don't."""
    rows = []
    for ctx in ["mc", "lc", "sc"]:
        for mdl in ["probe", "finetune", "supervised"]:
            fa = fa_frames(ctx, mdl)
            if fa is None:
                continue
            y = fa["native_label"].values.astype(int); s = fa["pred_prob_fog"].values
            thr_test = ict.pr11_threshold(y, s)
            thr_val = defog_threshold(ctx, mdl)
            rows.append({
                "name": f"{ctx}/{mdl}",
                "fixed0.5": ict.clinical_metrics(fa, 0.5)["tf_icc"],
                "test-PR11": ict.clinical_metrics(fa, thr_test)["tf_icc"],
                "DeFOG-val": ict.clinical_metrics(fa, thr_val)["tf_icc"],
            })
    d = pd.DataFrame(rows).set_index("name")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(d)); w = 0.27
    for i, (col, color) in enumerate([("fixed0.5", "#bdbdbd"), ("test-PR11", C["probe"]),
                                      ("DeFOG-val", C["finetune"])]):
        ax.bar(x + (i - 1) * w, d[col].values, w, label=col, color=color,
               edgecolor="white", linewidth=0.5)
    ax.axhspan(*RATER["tf"], color=C["band"], alpha=0.35, zorder=0, label="inter-rater")
    ax.axhline(0.75, color=C["ref"], ls=":", lw=1.1)
    ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=30, ha="right", fontsize=7.5)
    for sep in (2.5, 5.5):  # separate mc / lc / sc context groups
        ax.axvline(sep, color="0.85", lw=0.8, zorder=0)
    ax.set_ylabel("ICC(% Time Frozen)"); ax.set_ylim(-0.05, 1.0)
    ax.set_title("Threshold robustness: MC operating point transfers (test ≈ DeFOG-val)", fontsize=10)
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    save(fig, "figS4_threshold_robustness")


# ─────────────────────────────────────────────────────────────────────────────
def fig_pr_roc():
    fig, (axp, axr) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    combos = [("mc", "probe", C["probe"]), ("mc", "finetune", C["finetune"]),
              ("mc", "supervised", C["supervised"])]
    for ctx, mdl, col in combos:
        fa = fa_frames(ctx, mdl)
        if fa is None:
            continue
        y = fa["native_label"].values.astype(int); s = fa["pred_prob_fog"].values
        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        axp.plot(rec, prec, color=col, lw=2, label=f"{ctx}/{mdl} (AP={ap:.3f})")
        fpr, tpr, _ = roc_curve(y, s); auc = roc_auc_score(y, s)
        axr.plot(fpr, tpr, color=col, lw=2, label=f"{ctx}/{mdl} (AUC={auc:.3f})")
    prev = y.mean()
    axp.axhline(prev, color=C["ref"], ls="--", lw=1, label=f"chance ({prev:.2f})")
    axp.set_xlabel("Recall"); axp.set_ylabel("Precision")
    axp.set_title("Precision–Recall", fontsize=10)
    axp.legend(fontsize=8, loc="upper right", framealpha=0.85, handlelength=1.5)
    # Salomon 2026 reports only a summary AUC for its best transferred model
    # (no per-frame scores are available), so we annotate the scalar rather than
    # fabricate an operating point or draw a curve we don't have.
    axr.plot([0, 1], [0, 1], color="0.7", ls="--", lw=1)
    axr.text(0.04, 0.96,
             "Salomon 2026 (ref.): AUC = 0.803",
             transform=axr.transAxes, fontsize=7.5, style="italic", color=C["dark"],
             ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
    axr.set_xlabel("False positive rate"); axr.set_ylabel("True positive rate")
    axr.set_title("ROC", fontsize=10)
    axr.legend(fontsize=7.5, loc="lower right", handlelength=1.5)
    fig.suptitle("FogAtHome (held-out external cohort) — frame-level FOG discrimination",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig09_pr_roc_fogathome_framelevel")


# ─────────────────────────────────────────────────────────────────────────────
def _ci_pair(s):
    """Parse a '[lo, hi]' CI string from clinical_metrics into (lo, hi) floats."""
    lo, hi = (float(x) for x in s.strip().strip("[]").split(","))
    return lo, hi


def fig_clinical_icc_comparison():
    """Fig 11 — clinical ICC comparison (ICC_HANDOFF §6): grouped bars over
    %TF / #FOG / Duration for {FORGE mc/probe (1 IMU), Salomon 2026 best (1 IMU),
    Yang +50-min fine-tune (5 IMU)}, with human inter-rater bands shaded.
    FORGE %TF is gait-matched (walking-only, Activity=1) for the Salomon
    comparison; CIs overlap. Yang has no Duration. No in-distribution bar."""
    m = ict.clinical_metrics(fa_frames("mc", "probe"), defog_threshold("mc", "probe"))
    # Salomon removes non-gait → match with the walking-only %TF ICC (0.899),
    # falling back to whole-recording only if the activity annotation is absent.
    tf_icc = m["tf_icc_gait"] if m.get("tf_icc_gait") is not None else m["tf_icc"]
    tf_ci = m["tf_ci_gait"] if m.get("tf_icc_gait") is not None else m["tf_ci"]
    forge = {"tf": (tf_icc, _ci_pair(tf_ci)),
             "fog": (m["fog_icc"], _ci_pair(m["fog_ci"])),
             "dur": (m["dur_icc"], _ci_pair(m["dur_ci"]))}
    eps = [("tf", "% Time Frozen\n(gait-matched)"), ("fog", "# FOG Episodes"), ("dur", "Episode Duration")]
    x = np.arange(len(eps)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))

    # human inter-rater range: shaded rectangle behind each endpoint group (%TF, #FOG only)
    for xi, (key, _) in zip(x, eps):
        rater = RATER.get(key)
        if rater:
            ax.add_patch(plt.Rectangle((xi - 0.42, rater[0]), 0.84, rater[1] - rater[0],
                         color=C["band"], alpha=0.5, zorder=0))

    def asym(val, ci):
        return [[max(0.0, val - ci[0])], [max(0.0, ci[1] - val)]]

    series = [("FORGE mc/probe (1 IMU, zero-shot)", C["forge"], "forge"),
              ("Salomon 2026 best (1 IMU)", C["salomon"], "salomon"),
              ("Yang +50-min ft (5 IMU)", C["yang"], "yang")]
    for i, (label, col, src) in enumerate(series):
        offs = (i - 1) * w; first = True
        for xi, (key, _) in zip(x, eps):
            if src == "forge":
                val, ci = forge[key]; err = asym(val, ci)
            elif src == "salomon":
                if key not in SALOMON:
                    continue
                val, ci = SALOMON[key]; err = asym(val, ci)
            else:  # Yang fine-tune — no Duration reported
                if key == "dur":
                    continue
                val, sd = YANG[key]["ft50"]; err = [[sd], [sd]]
            ax.bar(xi + offs, val, w, color=col, edgecolor="white", lw=0.6, zorder=2,
                   label=label if first else None)
            ax.errorbar(xi + offs, val, yerr=err, fmt="none", ecolor="#333333",
                        elinewidth=1, capsize=2.5, zorder=3)
            ax.text(xi + offs, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
            first = False

    ax.set_xticks(x); ax.set_xticklabels([e[1] for e in eps], fontsize=9)
    ax.set_ylabel("ICC(2,1)"); ax.set_ylim(0, 1.05)
    handles, labs = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=C["band"], alpha=0.5)); labs.append("inter-rater range")
    ax.legend(handles, labs, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig11_clinical_icc")


def fig_icc_scatter():
    """Fig 11b (supplementary) — per-session %TF predicted vs ground truth for
    mc/probe; the ICC=0.909 agreement visual. (Fig 11 itself is the comparison
    bars above, per ICC_HANDOFF §6.)"""
    fa = fa_frames("mc", "probe")
    thr = defog_threshold("mc", "probe")
    pb = (fa["pred_prob_fog"].values >= thr).astype(int)
    s = fa.assign(_pb=pb).groupby("session_id").agg(
        gt=("native_label", lambda v: v.mean() * 100),
        pr=("_pb", lambda v: v.mean() * 100)).reset_index()
    m = ict.clinical_metrics(fa, thr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(s["gt"], s["pr"], s=28, color=C["probe"], alpha=0.62, edgecolor="white", lw=0.35)
    lim = max(s["gt"].max(), s["pr"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="0.5", ls="--", lw=1, label="identity")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Ground-truth % time frozen"); ax.set_ylabel("Predicted % time frozen")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    save(fig, "fig11b_icc_scatter")


# ─────────────────────────────────────────────────────────────────────────────
def fig_finetune_curve():
    """LR=1e-5 LOPO fine-tuning curve (canonical). Pooled over 12 held-out patients
    (97 sessions). Shows the inverted-U: gentle local fine-tuning improves over
    zero-shot, peaking ~2 min/patient, then overfits."""
    f = Path("logs/RESULTS_finetune_curve_FINAL.csv")
    if not f.exists():
        print("  · skip finetune_curve (RESULTS_finetune_curve_FINAL.csv missing)")
        return
    d = pd.read_csv(f).sort_values("minutes")
    x = d["minutes"].values
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10, 4.2))
    # Left: clinical ICC
    for ep, col, lab in [("icc_tf", C["forge"], "ICC(%TF)"), ("icc_fog", C["finetune"], "ICC(#FOG)")]:
        axl.plot(x, d[ep], "-o", color=col, lw=2, label=lab)
    axl.axhline(d["icc_tf"].iloc[0], color=C["forge"], ls=":", lw=1, alpha=0.6)
    axl.annotate(f"zero-shot {d['icc_tf'].iloc[0]:.3f}", (0, d["icc_tf"].iloc[0]),
                 textcoords="offset points", xytext=(6, -12), fontsize=8, color=C["forge"])
    pk = d.loc[d["icc_tf"].idxmax()]
    axl.set_ylim(0.40, 0.83)
    axl.annotate(f"peak {pk['icc_tf']:.3f}  (+{pk['icc_tf']-d['icc_tf'].iloc[0]:.3f})",
                 xy=(pk["minutes"], pk["icc_tf"]), xytext=(2.55, 0.808), textcoords="data",
                 ha="left", va="center", fontsize=8.5, fontweight="bold", color=C["forge"],
                 arrowprops=dict(arrowstyle="->", color=C["forge"], lw=0.9))
    axl.set_xlabel("Fine-tuning data per patient (min)"); axl.set_ylabel("ICC")
    axl.set_title("Clinical agreement"); axl.legend(fontsize=9, loc="lower left")
    # Right: threshold-free AP / AUC
    for ep, col, lab in [("auc", "#4393c3", "AUC"), ("ap", C["probe"], "AP")]:
        axr.plot(x, d[ep], "-o", color=col, lw=2, label=lab)
    axr.axhline(d["auc"].iloc[0], color="#4393c3", ls=":", lw=1, alpha=0.6)
    axr.set_xlabel("Fine-tuning data per patient (min)"); axr.set_ylabel("score")
    axr.set_title("Detection: AP & AUC (threshold-free)"); axr.legend(fontsize=9, loc="center")
    fig.tight_layout()
    save(fig, "fig12_finetune_curve")


def main():
    df = load_csv()
    print("Generating figures →", FIGDIR)
    fig_headline_icc()
    fig_context_dataset_ap(df)
    fig_icc_heatmaps()
    fig_fog_icc_heatmaps()
    fig_strategy_ap(df)
    fig_threshold_robustness()
    fig_pr_roc()
    fig_clinical_icc_comparison()
    fig_icc_scatter()
    fig_finetune_curve()
    print("Done.")


if __name__ == "__main__":
    main()

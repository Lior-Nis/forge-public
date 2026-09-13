"""Single source of truth for the FORGE release set (30 weight files).

The release is weights-only: each artifact is a `.safetensors` tensor set, and
the model it belongs to is rebuilt from this repository's Hydra configs. Every
entry therefore records the `experiment` config that defines the architecture
and, for classification heads, the `splits` config of the fold they were
trained on.

Reported metrics are the manuscript's released-detector numbers (see _RESULTS).
"""
from __future__ import annotations
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

HF_WEIGHTS_REPO = "Liornis/forge-fog"
HF_DATASET_REPO = "Liornis/fog-dataset"

# --- Encoders (the released FORGE backbones) ---------------------------------
_ENCODERS = {
    "lc": {
        "name": "encoder_lc_mae",
        "hf_path": "encoders/lc.safetensors",
        "local": "release/forge-fog/encoders/lc.safetensors",
        "experiment": "pretraining/spectral_patch_mae_daily",
        "window_frames": 1000, "vit_depth": 4, "params": 14147072,
        "role": "FORGE backbone (long context, 10s)",
    },
    "mc": {
        "name": "encoder_mc_mae",
        "hf_path": "encoders/mc.safetensors",
        "local": "release/forge-fog/encoders/mc.safetensors",
        "experiment": "pretraining/spectral_patch_mae_medcontext_daily",
        "window_frames": 500, "vit_depth": 4, "params": 14147072,
        "role": "FORGE backbone (medium context, 5s)",
    },
    "sc": {
        "name": "encoder_sc_mae",
        "hf_path": "encoders/sc.safetensors",
        "local": "release/forge-fog/encoders/sc.safetensors",
        "experiment": "pretraining/spectral_patch_mae_shortcontext_daily",
        "window_frames": 200, "vit_depth": 4, "params": 12918272,
        "role": "FORGE backbone (short context, 2s)",
    },
}

# --- Downstream classification (27 = 3 phases x 3 contexts x 3 folds) --------
# The experiment config that defines each model's architecture. probe and
# finetune share the SSL-initialised config (they differ only in optimiser and
# whether the backbone is frozen, neither of which affects inference);
# supervised-from-scratch has its own.
_SSL_CONFIGS = {
    "lc": "classification/spectral_patch_mae_lc_valid_defog_soft",
    "mc": "classification/spectral_patch_mae_mc_valid_defog_soft",
    "sc": "classification/spectral_patch_mae_sc_valid_defog_soft",
}
_SUPERVISED_CONFIGS = {
    "lc": "classification/supervised_lc_fogr025_defog",
    "mc": "classification/supervised_mc_fogr025_defog",
    "sc": "classification/supervised_sc_fogr025_defog",
}

def _experiment_for(phase: str, ctx: str) -> str:
    return _SUPERVISED_CONFIGS[ctx] if phase == "supervised" else _SSL_CONFIGS[ctx]


def _splits_for(ctx: str, fold: int) -> str:
    """DeFOG participant-level CV fold the head was trained on (57 participants)."""
    return f"kaggle_labeled/kfold_defog_fogcount_valid_{ctx}{fold}"


def _classification_set() -> list[dict]:
    out = []
    for phase in ("probe", "finetune", "supervised"):
        for ctx in ("lc", "mc", "sc"):
            for fold in (0, 1, 2):
                out.append({
                    "name": f"{ctx}_{phase}_fold{fold}",
                    "context": ctx, "phase": phase, "fold": fold, "seed": 42,
                    "hf_path": f"classification/{ctx}_{phase}_fold{fold}.safetensors",
                    "local": (f"release/forge-fog/classification/"
                              f"{ctx}_{phase}_fold{fold}.safetensors"),
                    "experiment": _experiment_for(phase, ctx),
                    "splits": _splits_for(ctx, fold),
                    "headline": (ctx == "mc" and phase == "probe"),
                })
    return out

RELEASE_SET = {"encoders": _ENCODERS, "classification": _classification_set()}

# --- Manuscript results -----------------------------------------------------
# All values below come from the manuscript and are the numbers of record. The
# `table` field names where each one is reported, and the wording of `basis`
# follows the manuscript's own definitions.
#
# The manuscript reports TWO model sets, and they must not be read as one:
#
#   `mc_probe_ensemble`  The RELEASED DETECTOR (Table 3). Nine BiGRU heads over
#                        one shared frozen MC (5 s) encoder: three participant
#                        folds x three seeds (42, 43, 44). Within a seed, the
#                        three fold scores are averaged at window level; the
#                        three seed-specific frame scores are then averaged with
#                        equal weight. Applied to every external cohort at the
#                        unchanged DeFOG-derived operating point of 0.35.
#
#   `mc_matched_arms`    The CONTROLLED COMPARISON (Table 2), which is the
#                        abstract's headline effect. Two arms differing ONLY in
#                        encoder initialisation, under matched downstream
#                        training: self-supervised initialisation (joint
#                        encoder-classifier optimisation, i.e. the fine-tune
#                        arm) versus supervised from scratch. Its numbers are
#                        LOWER than the released detector's on the same cohort
#                        (e.g. FogAtHome-provoking AUROC 0.861 vs 0.887) because
#                        it is a different, deliberately constrained model set —
#                        not a worse estimate of the same thing.
#
# AUROC and AP are duration-weighted pooled-frame endpoints. ICC(%TF) is
# participant-level agreement. Frame-level metrics use only annotator-verified
# frames (FogAtHome-provoking: 236,474); trailing unannotated samples are
# excluded rather than treated as confirmed non-FOG.
#
# `script` is present only where this repo's pipeline reproduces the number.
# Those scripts evaluate a seed-42 three-fold ensemble over the full window grid,
# so they land within ~0.02 of the manuscript value rather than on it exactly --
# reproduce.py checks them in that ballpark (see RESULT_CHECKS there).
_RELEASED = "mc_probe_ensemble"
_MATCHED = "mc_matched_arms"

_RESULTS = [
    # == Released detector, Table 3 ==========================================
    # -- FogAtHome-provoking (12 participants, cross-study / cross-cohort) ----
    {"id": "fogathome_icc_tf", "model": _RELEASED, "cohort": "fogathome",
     "metric": "ICC(%TF)", "value": 0.899, "ci": [0.700, 0.970],
     "basis": "participant-level %TF at the fixed DeFOG-derived operating point "
              "of 0.35; annotator-verified frames",
     "table": "Table 3", "figure": "Fig. 4a",
     "script": ("source .venv/bin/activate && python scripts/eval/compute_icc_thresholds.py "
                "--out logs/RESULTS_icc.md")},
    {"id": "fogathome_auroc", "model": _RELEASED, "cohort": "fogathome",
     "metric": "AUROC", "value": 0.887, "ci": [0.830, 0.922],
     "basis": "duration-weighted pooled-frame; annotator-verified frames",
     "table": "Table 3",
     "script": ("source .venv/bin/activate && python scripts/eval/eval_comprehensive.py "
                "--datasets fogathome --contexts mc --models probe "
                "--output logs/comprehensive_eval.csv "
                "--cache-dir logs/comprehensive_eval_cache")},
    {"id": "fogathome_ap", "model": _RELEASED, "cohort": "fogathome",
     "metric": "AP", "value": 0.804, "ci": [0.573, 0.902],
     "basis": "duration-weighted pooled-frame; annotator-verified frames; NormAP 0.695",
     "table": "Table 3"},

    # -- tDCS-FOG (71 participants, cross-protocol / acquisition) -------------
    {"id": "tdcs_auroc", "model": _RELEASED, "cohort": "tdcs_fog",
     "metric": "AUROC", "value": 0.917, "ci": [0.863, 0.950],
     "basis": "duration-weighted pooled-frame, 6,578,808 frames resampled 128 -> 100 Hz",
     "table": "Table 3"},
    {"id": "tdcs_ap", "model": _RELEASED, "cohort": "tdcs_fog",
     "metric": "AP", "value": 0.812, "ci": [0.550, 0.923],
     "basis": "duration-weighted pooled-frame; NormAP 0.702 [0.403, 0.847]",
     "table": "Table 3"},
    {"id": "tdcs_icc_tf", "model": _RELEASED, "cohort": "tdcs_fog",
     "metric": "ICC(%TF)", "value": 0.876, "ci": [0.810, 0.920],
     "basis": "participant-level %TF at the fixed 0.35 operating point; "
              "sens 0.717 / spec 0.850",
     "table": "Table 3"},

    # -- Stanford (7 participants; site / device / treatment-state shift) -----
    # Negative evidence: discrimination survives, the fixed threshold does not.
    {"id": "stanford_auroc", "model": _RELEASED, "cohort": "stanford",
     "metric": "AUROC", "value": 0.734, "ci": [0.624, 0.846],
     "basis": "duration-weighted pooled-frame, lumbar APDM channel resampled 128 -> 100 Hz",
     "table": "Table 3"},
    {"id": "stanford_ap", "model": _RELEASED, "cohort": "stanford",
     "metric": "AP", "value": 0.400, "basis": "duration-weighted pooled-frame",
     "table": "Table 3"},
    {"id": "stanford_icc_tf", "model": _RELEASED, "cohort": "stanford",
     "metric": "ICC(%TF)", "value": -0.119, "ci": [-0.920, 0.670],
     "basis": "participant-level at the unchanged 0.35 operating point; the oracle-rule "
              "threshold is 0.18 (ICC 0.327). Operating point does not transfer.",
     "table": "Table 3"},

    # -- FogAtHome daily living (11 participants, 301.8 h unscripted) ---------
    # Gait-conditioned: label-independent walking-and-standing domain, NOT
    # whole-recording %TF (whole-recording ICC is 0.153).
    {"id": "dailyliving_auroc", "model": _RELEASED, "cohort": "dailyliving",
     "metric": "AUROC", "value": 0.803, "ci": [0.737, 0.877],
     "basis": "duration-weighted pooled-frame within the prespecified "
              "label-independent walking-and-standing periods (58.18 h)",
     "table": "Table 3", "figure": "Fig. 5a"},
    {"id": "dailyliving_ap", "model": _RELEASED, "cohort": "dailyliving",
     "metric": "AP", "value": 0.105,
     "basis": "eligible domain, prevalence 2.92%; NormAP 0.078",
     "table": "Table 3"},
    {"id": "dailyliving_icc_tf", "model": _RELEASED, "cohort": "dailyliving",
     "metric": "ICC(%TF)", "value": 0.656, "ci": [-0.129, 0.872],
     "basis": "participant-level eligible-domain %TF at the 0.35 operating point; "
              "MAE 2.38 pp",
     "table": "Table 3", "figure": "Supplementary Fig. S2b"},

    # == Controlled comparison, Table 2 ======================================
    # Same architecture, trainable parameters, DeFOG folds, optimiser and
    # downstream procedure in both arms; only encoder initialisation differs.
    # ICC(%TF) is secondary here: each arm uses its OWN independently selected
    # DeFOG validation operating point, not the released detector's 0.35.
    {"id": "matched_fogathome_auroc", "model": _MATCHED, "cohort": "fogathome",
     "metric": "AUROC", "value": 0.861,
     "arms": {"self_supervised": 0.861, "supervised_from_scratch": 0.752},
     "difference": 0.109, "difference_ci": [0.029, 0.182], "p_value": 0.0070,
     "basis": "duration-weighted pooled-frame; paired participant-clustered "
              "bootstrap contrast",
     "table": "Table 2", "figure": "Fig. 3a",
     "script": ("source .venv/bin/activate && python scripts/eval/eval_comprehensive.py "
                "--datasets fogathome --contexts mc --models finetune supervised "
                "--output logs/comprehensive_eval.csv "
                "--cache-dir logs/comprehensive_eval_cache")},
    {"id": "matched_fogathome_ap", "model": _MATCHED, "cohort": "fogathome",
     "metric": "AP", "value": 0.784,
     "arms": {"self_supervised": 0.784, "supervised_from_scratch": 0.592},
     "difference": 0.192, "difference_ci": [0.064, 0.338], "p_value": 0.0005,
     "basis": "duration-weighted pooled-frame; paired participant-clustered "
              "bootstrap contrast",
     "table": "Table 2", "figure": "Fig. 3a",
     "script": ("source .venv/bin/activate && python scripts/eval/eval_comprehensive.py "
                "--datasets fogathome --contexts mc --models finetune supervised "
                "--output logs/comprehensive_eval.csv "
                "--cache-dir logs/comprehensive_eval_cache")},
    {"id": "matched_fogathome_icc_tf", "model": _MATCHED, "cohort": "fogathome",
     "metric": "ICC(%TF)", "value": 0.873,
     "arms": {"self_supervised": 0.873, "supervised_from_scratch": 0.002},
     "difference": 0.871, "difference_ci": [0.783, 0.932], "p_value": 0.0005,
     "basis": "secondary endpoint; each arm at its own independently selected "
              "DeFOG validation operating point",
     "table": "Table 2",
     "script": ("source .venv/bin/activate && python scripts/eval/compute_icc_thresholds.py "
                "--out logs/RESULTS_icc.md")},
    {"id": "matched_tdcs_auroc", "model": _MATCHED, "cohort": "tdcs_fog",
     "metric": "AUROC", "value": 0.8692,
     "arms": {"self_supervised": 0.8692, "supervised_from_scratch": 0.6568},
     "difference": 0.2123, "difference_ci": [0.126, 0.280], "p_value": 0.0005,
     "basis": "duration-weighted pooled-frame; paired participant-clustered "
              "bootstrap contrast",
     "table": "Table 2", "figure": "Fig. 3a"},
    {"id": "matched_tdcs_ap", "model": _MATCHED, "cohort": "tdcs_fog",
     "metric": "AP", "value": 0.8110,
     "arms": {"self_supervised": 0.8110, "supervised_from_scratch": 0.5007},
     "difference": 0.3103, "difference_ci": [0.133, 0.397], "p_value": 0.0005,
     "basis": "duration-weighted pooled-frame; paired participant-clustered bootstrap "
              "contrast. NormAP 0.7004 [0.407, 0.841] self-supervised vs 0.2084 "
              "[0.075, 0.360] supervised from scratch",
     "table": "Table 2", "figure": "Fig. 3a"},
    {"id": "matched_tdcs_icc_tf", "model": _MATCHED, "cohort": "tdcs_fog",
     "metric": "ICC(%TF)", "value": 0.669,
     "arms": {"self_supervised": 0.669, "supervised_from_scratch": 0.053},
     "difference": 0.616, "difference_ci": [0.424, 0.738], "p_value": 0.0005,
     "basis": "secondary endpoint; each arm at its own independently selected "
              "DeFOG validation operating point",
     "table": "Table 2"},

    # == In-distribution reference, Supplementary Table S7 ====================
    {"id": "defog_window_ap", "model": "mc_probe_3fold_seed42", "cohort": "kaggle",
     "model_members": ["mc_probe_fold0", "mc_probe_fold1", "mc_probe_fold2"],
     "metric": "AP", "value": 0.730,
     "basis": "window-level AP on the DeFOG validation folds, single-seed arm "
              "(seed 42); fine-tune 0.708 and supervised-from-scratch 0.492 at the "
              "same 5 s context",
     "table": "Supplementary Table S7",
     "script": ("source .venv/bin/activate && python scripts/eval/eval_comprehensive.py "
                "--datasets kaggle --contexts mc --models probe "
                "--output logs/comprehensive_eval.csv "
                "--cache-dir logs/comprehensive_eval_cache")},
]

# Verification status of each reported number, relative to THIS repository.
#
#   one_click  reproduce.py runs it and checks the produced value (see
#              RESULT_CHECKS in reproduce.py -- kept in sync by a test).
#   scripted   an in-repo script reproduces it, but it is outside the one-click
#              flow because it needs an extra pass over the released data.
#   recorded   manuscript number with NO in-repo pipeline: the cohort is not part
#              of the public data release, so nothing here can recompute it.
#
# Every `recorded` result must carry a `note` saying why. A number the repo
# cannot recompute must never be presented as if it had been verified.
_VERIFICATION = {
    "fogathome_icc_tf":   ("one_click", None),
    "fogathome_auroc":    ("one_click", None),
    "fogathome_ap":       ("one_click", None),
    "defog_window_ap":    ("one_click", None),
    "dailyliving_auroc":  ("scripted",  None),
    "dailyliving_ap":     ("scripted",  None),
    "dailyliving_icc_tf": ("scripted",  None),
    # Controlled comparison (Table 2). On FogAtHome-provoking both arms are in the
    # public release -- the self-supervised arm is `finetune` (joint
    # encoder-classifier optimisation), the comparator is `supervised`. The
    # release ships seed 42 only, so a recomputation reproduces the direction and
    # lands near, not on, the manuscript's multi-seed values.
    "matched_fogathome_auroc":  ("scripted", None),
    "matched_fogathome_ap":     ("scripted", None),
    "matched_fogathome_icc_tf": ("scripted", None),
    "matched_tdcs_auroc":  ("recorded", "tDCS-FOG is not part of the public data "
                                        "release; no in-repo pipeline recomputes it."),
    "matched_tdcs_ap":     ("recorded", "tDCS-FOG is not part of the public data "
                                        "release; no in-repo pipeline recomputes it."),
    "matched_tdcs_icc_tf": ("recorded", "tDCS-FOG is not part of the public data "
                                        "release; no in-repo pipeline recomputes it."),
    "tdcs_auroc":    ("recorded", "tDCS-FOG is not part of the public data release; "
                                  "no in-repo pipeline recomputes it."),
    "tdcs_ap":       ("recorded", "tDCS-FOG is not part of the public data release; "
                                  "no in-repo pipeline recomputes it."),
    "tdcs_icc_tf":   ("recorded", "tDCS-FOG is not part of the public data release; "
                                  "no in-repo pipeline recomputes it."),
    "stanford_auroc":  ("recorded", "Stanford cohort is not part of the public data "
                                    "release; no in-repo pipeline recomputes it."),
    "stanford_ap":     ("recorded", "Stanford cohort is not part of the public data "
                                    "release; no in-repo pipeline recomputes it."),
    "stanford_icc_tf": ("recorded", "Stanford cohort is not part of the public data "
                                    "release; no in-repo pipeline recomputes it."),
}

# The daily-living numbers are scored on the gait-conditioned eligible domain, not
# on whole recordings, so they need this pass rather than the plain eval sweep.
# Emits AUC / AP / ICC(%TF) per (context, model). Needs the dailyliving frame
# parquets and the ICC threshold table first, so the full chain is given here --
# run cold it fails on the missing logs/RESULTS_icc.md.
_SCRIPTED_CMD = (
    "source .venv/bin/activate && "
    "python scripts/eval/eval_comprehensive.py --datasets dailyliving --contexts mc "
    "--models probe --output logs/comprehensive_eval.csv "
    "--cache-dir logs/comprehensive_eval_cache && "
    "python scripts/eval/compute_icc_thresholds.py "
    "--cache-dir logs/comprehensive_eval_cache --out logs/RESULTS_icc.md && "
    "python scripts/eval/eval_dailyliving_walkstand.py")


_DATASETS = {
    "fogathome":   {"hf_subdir": "fogathome",             "role": "external_structured",   "prevalence": 0.356, "n_patients": 12},
    "dailyliving": {"hf_subdir": "fogathome_dailyliving", "role": "external_naturalistic", "prevalence": 0.0109, "prevalence_eligible": 0.0292, "n_patients": 11},
    "kaggle":      {"hf_subdir": "kaggle_labeled",        "role": "in_distribution",       "n_patients": 57},
}

# Cohort labels exactly as the manuscript names them, so a result's `cohort` key
# can be traced to the paper without guessing. `in_public_release` says whether
# this repository's data release contains the cohort at all.
_COHORTS = {
    "kaggle": {
        "label": "DeFOG", "n_participants": 57, "in_public_release": True,
        "role": "supervised development + in-distribution reference",
        "note": "27.0 h structured at-home FOG protocol; source of the operating point.",
    },
    "fogathome": {
        "label": "FogAtHome-provoking", "n_participants": 12, "in_public_release": True,
        "shift": "Cross-study / cross-cohort",
    },
    "tdcs_fog": {
        "label": "tDCS-FOG", "n_participants": 71, "in_public_release": False,
        "shift": "Cross-protocol / acquisition",
    },
    "stanford": {
        "label": "Stanford", "n_participants": 7, "in_public_release": False,
        "shift": "Cross-site / device / treatment state",
    },
    "dailyliving": {
        "label": "FogAtHome daily living", "n_participants": 11, "in_public_release": True,
        "shift": "Structured supervised development -> naturalistic labeled evaluation",
        "note": "Metrics are computed within prespecified label-independent "
                "walking-and-standing periods.",
    },
}

# The two model sets the manuscript reports. Keep these labels in sync with the
# `model` field of each result.
_MODEL_SETS = {
    _RELEASED: {
        "label": "Released detector",
        "table": "Table 3",
        "definition": "Nine BiGRU heads over one shared frozen MC (5 s) encoder: "
                      "3 participant folds x 3 seeds (42, 43, 44). Within a seed the "
                      "three fold scores are averaged at window level; the three "
                      "seed-specific frame scores are then averaged with equal weight. "
                      "Applied to every external cohort at the unchanged DeFOG-derived "
                      "operating point of 0.35.",
    },
    _MATCHED: {
        "label": "Controlled comparison: self-supervised vs supervised from scratch",
        "table": "Table 2",
        "definition": "Two arms differing only in encoder initialisation, under matched "
                      "downstream training (same architecture, trainable parameters, "
                      "DeFOG folds and optimiser). The self-supervised arm is joint "
                      "encoder-classifier optimisation (the fine-tune arm). This is the "
                      "abstract's headline effect and is a DIFFERENT model set from the "
                      "released detector, so its values on the same cohort are lower.",
    },
    "mc_probe_3fold_seed42": {
        "label": "Seed-42 three-fold MC probe (in-distribution reference)",
        "table": "Supplementary Table S7",
        "definition": "The three released seed-42 MC probe heads, averaged across folds. "
                      "This is the model set this repository's evaluation scripts run.",
    },
}

def _results_with_verification() -> list[dict]:
    """Stamp each result with how (or whether) this repo can recompute it.

    Fails loudly on a result that has no verification status: a new number must
    declare whether it is reproducible here, rather than defaulting to silence.
    """
    out = []
    for r in _RESULTS:
        try:
            status, note = _VERIFICATION[r["id"]]
        except KeyError:
            raise KeyError(
                f"result {r['id']!r} has no _VERIFICATION entry -- declare it as "
                "one_click / scripted / recorded before releasing it") from None
        entry = {**r, "verification": status}
        if status == "recorded":
            assert note, f"{r['id']}: a `recorded` result must say why"
            entry["verification_note"] = note
        if status == "scripted" and not entry.get("script"):
            entry["script"] = _SCRIPTED_CMD
        out.append(entry)
    return out


def build_manifest() -> dict:
    return {
        "meta": {
            "paper": ("Self-supervised learning improves cross-cohort freezing-of-gait "
                      "detection from a single lower-back accelerometer"),
            "authors": ("Lior Nisimov, Amit Salomon, Eran Gazit, Talia Herman, "
                        "Lior Rokach, Jeffrey M. Hausdorff, Nathaniel Shimoni"),
            "headline_model": "mc_probe_ensemble",
            "released_seed": 42,
            "manuscript_seeds": [42, 43, 44],
            "headline_model_note": (
                "The manuscript's released detector is nine BiGRU heads (3 participant "
                "folds x 3 seeds) over one shared frozen MC encoder. This repo ships the "
                "seed-42 heads (mc_probe_fold{0,1,2}.safetensors), which rebuild the "
                "seed-42 three-fold ensemble -- the configuration this repo's eval "
                "scripts run. Each head file already contains its encoder, so "
                "encoders/*.safetensors are needed only to train new heads. Values under "
                "`results` are the nine-head numbers; the seed-42 subset lands within "
                "~0.02 of them."
            ),
            "weights_format": (
                "safetensors, weights only. Each file holds trained tensors and "
                "data-derived normalisation buffers -- no training configuration, no "
                "optimiser state, no file paths. Rebuild a model from the `experiment` "
                "config named on its entry: utils.released_weights.load_released_model()."
            ),
            "threshold_protocol": "defog_val_pr11",
            "license": "MIT",
            "hf_weights_repo": HF_WEIGHTS_REPO,
            "hf_dataset_repo": HF_DATASET_REPO,
        },
        "encoders": RELEASE_SET["encoders"],
        "classification": RELEASE_SET["classification"],
        "datasets": _DATASETS,
        "cohorts": _COHORTS,
        "model_sets": _MODEL_SETS,
        "results": _results_with_verification(),
    }

def validate_manifest(m: dict, repo_root: Path) -> list[str]:
    missing = []
    for enc in m["encoders"].values():
        if not (repo_root / enc["local"]).is_file():
            missing.append(enc["local"])
    for c in m["classification"]:
        if not (repo_root / c["local"]).is_file():
            missing.append(c["local"])
    return missing

def write_manifest(path: Path, validate: bool = True) -> dict:
    m = build_manifest()
    if validate:
        missing = validate_manifest(m, REPO_ROOT)
        if missing:
            raise FileNotFoundError(f"Release checkpoints missing: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(m, sort_keys=False))
    return m

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-validate", action="store_true",
                    help="write the manifest without checking that every `local` "
                         "checkpoint exists (for machines that only hold the metadata)")
    args = ap.parse_args()

    out = REPO_ROOT / "release" / "manifest.yaml"
    write_manifest(out, validate=not args.no_validate)
    print(f"Wrote {out} (30 artifacts"
          f"{'' if args.no_validate else ', validated'})")

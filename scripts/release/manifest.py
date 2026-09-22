"""Single source of truth for the FORGE release set.

Derived from the authoritative CKPT dict in scripts/eval/eval_comprehensive.py.
Reported metrics are the manuscript's released-detector numbers (see _RESULTS).
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

HF_WEIGHTS_REPO = "Liornis/forge-fog"
HF_DATASET_REPO = "Liornis/fog-dataset"
HF_WEIGHTS_REVISION = "154dbdf09a570e50eb256d218729f796519883bc"
HF_DATASET_REVISION = "f24fab4a5c1c88e376813fa9b0ec7bf28b1b32d3"

# --- Encoders (the released FORGE backbones) ---------------------------------
_ENCODERS = {
    "lc": {
        "name": "encoder_lc_mae",
        "hf_path": "encoders/lc.safetensors",
        "local": "release/forge-fog/encoders/lc.safetensors",
        "window_frames": 1000, "vit_depth": 4, "params": 14147072,
        "role": "FORGE backbone (long context, 10s)",
    },
    "mc": {
        "name": "encoder_mc_mae",
        "hf_path": "encoders/mc.safetensors",
        "local": "release/forge-fog/encoders/mc.safetensors",
        "window_frames": 500, "vit_depth": 4, "params": 14147072,
        "role": "FORGE backbone (medium context, 5s)",
    },
    "sc": {
        "name": "encoder_sc_mae",
        "hf_path": "encoders/sc.safetensors",
        "local": "release/forge-fog/encoders/sc.safetensors",
        "window_frames": 200, "vit_depth": 4, "params": 12918272,
        "role": "FORGE backbone (short context, 2s)",
    },
}

# --- Downstream classification (27 = 3 phases x 3 contexts x 3 folds) --------
_CONFIGS = {
    "lc": "classification/spectral_patch_mae_lc_valid_defog_soft",
    "mc": "classification/spectral_patch_mae_mc_valid_defog_soft",
    "sc": "classification/spectral_patch_mae_sc_valid_defog_soft",
}

def _classification_set() -> list[dict]:
    out = []
    for phase in ("probe", "finetune", "supervised"):
        for ctx in ("lc", "mc", "sc"):
            for fold in (0, 1, 2):
                out.append({
                    "name": f"{ctx}_{phase}_fold{fold}",
                    "context": ctx, "phase": phase, "fold": fold, "seed": 42,
                    "hf_path": f"classification/{ctx}_{phase}_fold{fold}.safetensors",
                    "local": f"release/forge-fog/classification/{ctx}_{phase}_fold{fold}.safetensors",
                    "experiment": (_CONFIGS[ctx] if phase != "supervised"
                                   else f"classification/supervised_{ctx}_fogr025_defog"),
                    "splits": f"kaggle_labeled/kfold_defog_fogcount_valid_{ctx}{fold}",
                    "headline": (ctx == "mc" and phase == "probe"),
                })
    for seed in (43, 44):
        for fold in (0, 1, 2):
            name = f"mc_probe_s{seed}_fold{fold}"
            out.append({
                "name": name, "context": "mc", "phase": "probe",
                "fold": fold, "seed": seed,
                "hf_path": f"classification/{name}.safetensors",
                "local": f"release/forge-fog/classification/{name}.safetensors",
                "experiment": _CONFIGS["mc"],
                "splits": f"kaggle_labeled/kfold_defog_fogcount_valid_mc{fold}",
                "headline": True,
            })
    return out

RELEASE_SET = {"encoders": _ENCODERS, "classification": _classification_set()}

# --- Released-detector results (manuscript, Table 2 "external ladder") ------
# All values below come from the manuscript and are the numbers of record.
#
# Released detector = nine-head MC frozen-probe ensemble (3 participant folds x
# 3 seeds), transferred DeFOG operating point 0.35. Frame-level metrics use only
# annotator-verified frames; the public FogAtHome build retains 227,500 frames
# covered by validity==1 windows. Trailing padding is excluded rather than
# treated as confirmed non-FOG.
#
# Every external result is reproduced by the public nine-head pipeline.
_RELEASED = "mc_probe_ensemble"

_RESULTS = [
    # -- FogAtHome-provoking (12 participants, cross-study shift) --------------
    {"id": "fogathome_icc_tf", "model": _RELEASED, "cohort": "fogathome",
     "metric": "ICC(%TF)", "value": 0.899, "ci": [0.700, 0.970],
     "basis": "participant-level, 227,500 public-pipeline frames, threshold 0.35",
     "figure": "fig03",
     "script": "./reproduce.sh --datasets fogathome"},
    {"id": "fogathome_auroc", "model": _RELEASED, "cohort": "fogathome",
     "metric": "AUROC", "value": 0.887, "ci": [0.830, 0.922],
     "basis": "frame-level, 227,500 public-pipeline frames",
     "script": "./reproduce.sh --datasets fogathome"},
    {"id": "fogathome_ap", "model": _RELEASED, "cohort": "fogathome",
     "metric": "AP", "value": 0.804, "ci": [0.573, 0.902],
     "basis": "frame-level, 227,500 public-pipeline frames; NormAP 0.695",
     "script": "./reproduce.sh --datasets fogathome"},

    # -- tDCS-FOG (71 participants, principal external evaluation) ------------
    {"id": "tdcs_auroc", "model": _RELEASED, "cohort": "tdcs_fog",
     "metric": "AUROC", "value": 0.917,
     "basis": "frame-level, 7,220,623 frames from all 71 participants, resampled 128 -> 100 Hz",
     "script": "./reproduce.sh --datasets tdcsfog"},
    {"id": "tdcs_ap", "model": _RELEASED, "cohort": "tdcs_fog",
     "metric": "AP", "value": 0.866, "ci": [0.661, 0.946],
     "basis": "frame-level, same all-71 nine-head assembly as AUROC and ICC(%TF)",
     "script": "./reproduce.sh --datasets tdcsfog"},
    {"id": "tdcs_icc_tf", "model": _RELEASED, "cohort": "tdcs_fog",
     "metric": "ICC(%TF)", "value": 0.876,
     "basis": "participant-level, threshold 0.35; sens 0.717 / spec 0.850",
     "script": "./reproduce.sh --datasets tdcsfog"},

    # -- Stanford (7 participants; site + device + treatment-state shift) -----
    # Negative evidence: discrimination survives, the fixed threshold does not.
    {"id": "stanford_auroc", "model": _RELEASED, "cohort": "stanford",
     "metric": "AUROC", "value": 0.734,
     "basis": "frame-level, lumbar APDM channel resampled 128 -> 100 Hz",
     "script": "./reproduce.sh --datasets stanford"},
    {"id": "stanford_ap", "model": _RELEASED, "cohort": "stanford",
     "metric": "AP", "value": 0.400, "basis": "frame-level",
     "script": "./reproduce.sh --datasets stanford"},
    {"id": "stanford_icc_tf", "model": _RELEASED, "cohort": "stanford",
     "metric": "ICC(%TF)", "value": -0.119,
     "basis": "participant-level at the unchanged 0.35 threshold; the oracle-rule "
              "threshold is 0.18 (ICC 0.327). Operating point does not transfer.",
     "script": "./reproduce.sh --datasets stanford"},

    # -- FogAtHome daily living (11 participants, 301.8 h unscripted) ---------
    # Gait-conditioned: label-independent walking-and-standing domain, NOT
    # whole-recording %TF (whole-recording ICC is 0.153).
    {"id": "dailyliving_auroc", "model": _RELEASED, "cohort": "dailyliving",
     "metric": "AUROC", "value": 0.803, "ci": [0.737, 0.877],
     "basis": "frame-level within the walking-and-standing eligible domain (58.18 h)",
     "figure": "fig05", "script": "./reproduce.sh --datasets dailyliving"},
    {"id": "dailyliving_ap", "model": _RELEASED, "cohort": "dailyliving",
     "metric": "AP", "value": 0.105,
     "basis": "eligible domain, prevalence 2.92%; NormAP 0.078",
     "script": "./reproduce.sh --datasets dailyliving"},
    {"id": "dailyliving_icc_tf", "model": _RELEASED, "cohort": "dailyliving",
     "metric": "ICC(%TF)", "value": 0.656, "ci": [-0.129, 0.872],
     "basis": "participant-level eligible-domain %TF, threshold 0.35; MAE 2.38 pp",
     "script": "./reproduce.sh --datasets dailyliving"},

    # -- In-distribution reference (DeFOG held-out folds) ---------------------
    {"id": "defog_window_ap", "model": "mc_probe_3fold_seed42", "cohort": "kaggle",
     "model_members": ["mc_probe_fold0", "mc_probe_fold1", "mc_probe_fold2"],
     "metric": "AP", "value": 0.730,
     "basis": "window-level AP on held-out DeFOG folds, seed 42 (Supp. Table S8)"},
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
    "defog_window_ap":    ("recorded", "Internal DeFOG validation is intentionally "
                                        "excluded from public external reproduction."),
    "dailyliving_auroc":  ("one_click", None),
    "dailyliving_ap":     ("one_click", None),
    "dailyliving_icc_tf": ("one_click", None),
    "tdcs_auroc":         ("one_click", None),
    "tdcs_ap":            ("one_click", None),
    "tdcs_icc_tf":        ("one_click", None),
    "stanford_auroc":     ("one_click", None),
    "stanford_ap":        ("one_click", None),
    "stanford_icc_tf":    ("one_click", None),
}


_DATASETS = {
    "fogathome":   {"hf_subdir": "fogathome",             "role": "external_structured",   "prevalence": 0.356, "n_patients": 12},
    "dailyliving": {"hf_subdir": "fogathome_dailyliving", "activity_sidecar": "fogathome_dailyliving/activity.parquet", "role": "external_naturalistic", "prevalence": 0.0109, "prevalence_eligible": 0.0292, "n_patients": 11},
    "tdcs_fog":    {"hf_subdir": "kaggle_labeled/tdcsfog", "role": "external_provoked", "sampling_rate_hz": 128, "evaluation_rate_hz": 100, "n_patients": 71},
    "stanford":    {"source": "https://github.com/stanfordnmbl/imu-fog-detection", "revision": "e95687842801ca3565463486a50d562fff44d182", "role": "external_cross_device", "n_patients": 7},
    "kaggle":      {"hf_subdir": "kaggle_labeled",        "role": "in_distribution",       "n_patients": 128},
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
        out.append(entry)
    return out


def build_manifest() -> dict:
    return {
        "meta": {
            "paper": ("FORGE - Self-supervised learning improves cross-cohort "
                      "freezing-of-gait detection from a single lower-back accelerometer"),
            "headline_model": "mc_probe_ensemble",
            "released_seed": 42,
            "manuscript_seeds": [42, 43, 44],
            "headline_model_note": (
                "The manuscript's released detector is nine BiGRU heads (3 participant "
                "folds x 3 seeds) over one shared frozen MC encoder. The public release "
                "contains all nine safetensors heads. External evaluation averages all "
                "nine; the DeFOG reference remains the seed-42 out-of-fold result."
            ),
            "threshold_protocol": "defog_val_pr11",
            "license": "MIT",
            "hf_weights_repo": HF_WEIGHTS_REPO,
            "hf_dataset_repo": HF_DATASET_REPO,
            "hf_weights_revision": HF_WEIGHTS_REVISION,
            "hf_dataset_revision": HF_DATASET_REVISION,
        },
        "encoders": RELEASE_SET["encoders"],
        "classification": RELEASE_SET["classification"],
        "datasets": _DATASETS,
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
    manifest = write_manifest(out, validate=not args.no_validate)
    total = len(manifest["encoders"]) + len(manifest["classification"])
    print(f"Wrote {out} ({total} artifacts"
          f"{'' if args.no_validate else ', validated'})")

---
name: onboard-repo
description: Use to set up the FORGE repo on a fresh machine — download released weights and datasets from HuggingFace, build the evaluation zarr views, and smoke-test reproducibility. Run this before reproduce-evaluations when checkpoints or processed data are missing.
---

# Onboard the FORGE Repo (fresh machine)

Ordered, idempotent. Each step skips itself if already done. The source of truth for paths is
`release/manifest.yaml`.

## 1. Preflight
- `source .venv/bin/activate` (create env first if missing: `uv sync`).
- `hf auth whoami` — confirm logged in (token needs read access; the public repos need none).
- Check free disk: eval data + weights need ~120 GB. Warn and stop if insufficient.

## 2. Download released weights
`hf download Liornis/forge-fog --repo-type model --local-dir release/forge-fog`
That is the whole step: files land at their `manifest.yaml > *.local` paths and are read from
there. The release is weights-only `.safetensors`; each model is rebuilt from the `experiment`
config the manifest names for it (`utils/released_weights.py`), with the DeFOG fold from its
`splits` entry.

> **Encoders are only needed for training.** Each released classification file already contains
> its encoder, so evaluation needs no `encoders/*.safetensors` and nothing has to be copied
> into `checkpoints/`. Download the encoders when you intend to train new heads.

## 3. Download evaluation datasets
Download only the eval splits (pretraining's `kaggle_unlabeled`, 64 GB, is OPT-IN).
Use a **recursive** glob (`/**`): `kaggle_labeled/*` silently skips the nested
`defog/tdcsfog/notype` session data and you end up unable to build the Kaggle zarr.
`hf download Liornis/fog-dataset --repo-type dataset --include "kaggle_labeled/**" "fogathome/**" "fogathome_dailyliving/**" --local-dir ~/Datasets/fog-dataset`
Ensure `data/raw` points there: `ln -sfn ~/Datasets/fog-dataset data/raw`.

## 4. Build evaluation zarrs
The eval reads physical zarrs from `scripts/eval/eval_comprehensive.py` (`ZARR` map) and applies
the `validity == 1.0` filter in-code (no view files needed). For the MC headline build the three
eval zarrs directly — **skip** `generate_medcontext_zarrs.sh` step 1 (the 64 GB `kaggle_unlabeled`
pretraining build, which `set -euo pipefail` would otherwise abort on):
```
uv run python -m data.process paths=kaggle_defog_medcontext process=kaggle_medcontext
uv run python -m data.process paths=fogathome_medcontext process=kaggle_medcontext
uv run python -m data.process paths=fogathome_dailyliving_medcontext process=kaggle_medcontext
```
(For SC/LC, use the `*_shortcontext` / `*_longcontext` paths configs analogously.) Each takes
a few minutes; GPU not required.

## 5. Smoke test (acceptance)
Run one context end-to-end and confirm the headline reproduces:
`source .venv/bin/activate && python scripts/eval/eval_comprehensive.py --datasets fogathome --contexts mc --models probe --output logs/smoke_eval.csv --cache-dir logs/comprehensive_eval_cache`
Then check the MC-probe `seg` AUC row: it should be ≈0.908, i.e. within ~0.03 of the
manuscript's 0.887 in `manifest.yaml > results` (the manifest reports the nine-head ensemble
on annotator-verified frames; this smoke test runs seed-42 three-fold over the full window
grid, hence the small offset). If it lands there, onboarding is verified end-to-end. Full reproduction → use the `reproduce-evaluations` skill.

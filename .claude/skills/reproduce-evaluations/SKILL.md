---
name: reproduce-evaluations
description: Use when asked to reproduce or verify any FORGE paper result (FogAtHome clinical ICC, AP/AUC, Kaggle in-distribution, daily-living head-to-head). Resolves which checkpoint, dataset, config, and command produce each reported number.
---

# Reproduce FORGE Evaluations

The single source of truth is `release/manifest.yaml` (the 30 released checkpoints + the
headline `results:` table). Read it first; it tells you every available model and the
command + expected value for each reported number.

## How models map to checkpoints
- `manifest.yaml > classification[]` lists each downstream model: `{context, phase, fold,
  local, hf_path, config}`. The headline model is **mc / probe** (`headline: true`), reported
  as the **3-fold ensemble** (average the fold probabilities by frame index).
- `manifest.yaml > encoders` lists the three FORGE backbones (frozen during `probe`).
- The authoritative checkpoint pattern in code is `scripts/eval/eval_comprehensive.py` (`CKPT`
  dict); the manifest mirrors its entries.

## Threshold protocol (read before any clinical metric)
Clinical ICC uses **`defog_val_pr11`** — the PR-(1,1) operating point chosen on held-out DeFOG
validation and applied unchanged to FogAtHome (no test peeking). Only **MC** is
calibration-robust across domains; SC/supervised collapse under validation tuning and are NOT
reported as clinical results.

## Commands
- **Comprehensive AP/AUC/NormAP** (all datasets × contexts × models):
  `source .venv/bin/activate && python scripts/eval/eval_comprehensive.py --datasets fogathome dailyliving kaggle --contexts lc mc sc --models probe finetune supervised --output logs/comprehensive_eval.csv --cache-dir logs/comprehensive_eval_cache`
- **Clinical ICC table** (both threshold protocols — reads the fogathome+kaggle parquets the command above caches):
  `source .venv/bin/activate && python scripts/eval/compute_icc_thresholds.py --cache-dir logs/comprehensive_eval_cache --out logs/RESULTS_icc.md`
- **Daily-living head-to-head vs Kaggle winners:** `scripts/eval/kaggle_full_filters.py`.
  **Author-only.** This and 10 sibling supplementary scripts read prediction vectors from the
  gitignored `research/paper_final/data/` tree, so they cannot run in a clone. They exit with an
  `[author-only]` message naming what is missing; set `FORGE_PRIVATE_DATA` if you have the tree.
  None of the headline numbers depend on them.

### GPU / environment notes
- **Small GPU?** Set `EVAL_BATCH_SCALE=0.25` (and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) to shrink inference batches — does NOT change results, only memory/speed. The headline numbers were re-verified bit-identically at scale 0.25 on a ~6 GB free GPU (2026-06-22).
- **Gait-filtered ICC dependency:** the Salomon-matched walking-only ICC needs a FogAtHome `labels.csv` with an `Activity` column, passed via `--gait-labels`. That file is NOT in the release; the main %TF ICC does not need it. **Careful:** that walking-only lens yields ≈0.899, the same digits as the manuscript's participant-level ICC in the manifest. The agreement is coincidental — different estimands (see `compute_icc_thresholds.py:44`). Do not report one as confirming the other.

## Expected numbers

`manifest.yaml > results` holds the **manuscript's** released-detector values — the nine-head
ensemble (3 folds x 3 seeds) scored on annotator-verified frames. The commands above run a
**seed-42 three-fold ensemble over the full window grid**, so they land near those values, not
on them. `reproduce.py` checks the gap in a ballpark (AP/AUC ±0.03; ICC within the manuscript CI).

| Check | Manuscript (manifest) | This pipeline emits |
|---|---|---|
| FogAtHome frame AUROC | 0.887 [0.830, 0.922] | ≈0.908 (`seg` row, full grid) |
| FogAtHome frame AP | 0.804 [0.573, 0.902] | ≈0.81 (`seg` row) |
| FogAtHome ICC(%TF) | 0.899 [0.700, 0.970] | ≈0.909 (default {1,3,4} lens; the walking-only lens also gives ≈0.899 — coincidence, not a match) |
| DeFOG window AP (in-dist) | 0.730 | 0.730 (exact) |

Also produced by the ICC table, not checked automatically: #FOG ICC 0.717, Duration ICC 0.959.
Anything off by more than ~0.03 means wrong checkpoints or wrong data — not a basis difference.

- **Reproduction status (2026-06-22):** FogAtHome + Kaggle AP/AUC and the full clinical-ICC table reproduce *bit-identically* from a fresh recompute of the parquets. DailyLiving uses the identical mechanism.
- Outputs land in `logs/comprehensive_eval.csv` and `logs/RESULTS_icc.md`.

If checkpoints or zarr views are missing, run the `onboard-repo` skill first.

# Provenance of this snapshot

## What this repository is

A **public reproducibility snapshot** of the FORGE code, prepared for the submitted
manuscript. The authors developed the code in a separate private research repository.
This repository was created from the final audited state of that code as a **fresh Git
history with a single initial commit**.

- The private research-development history (intermediate experiments, commit history,
  drafts) is **not** included and cannot be recovered from this repository.
- This snapshot is **not** a Git-level record of when each experiment ran. The authors
  keep that record privately.
- Snapshot prepared: 2026-09-13.

## Relationship to the manuscript's experiments

- The released weights (`Liornis/forge-fog`, revision
  `fbdf75f6fda83937d8a0a079043631d43b1f8f47`) hold the same trained tensors that were
  first published on 2026-06-21; only the file format changed (see "Release format"
  below). No weights were retrained or reselected for this snapshot.
- **Split definitions:** the released heads were trained on the DeFOG participant-level
  3-fold CV over 57 participants, one split per context, committed as
  `configs/data/splits/kaggle_labeled/kfold_defog_fogcount_valid_{lc,mc,sc}{0,1,2}.yaml`.
  `release/manifest.yaml` names the split file belonging to each released file.
- **Model and training code:** since the released models were trained, the code for
  model architecture, losses, training pipelines and preprocessing has not changed in
  ways that alter computation. The code was changed in these ways:
  - *Data selection:* training and evaluation used to read the DeFOG "valid" subset
    through intermediate view files, which were not versioned. They now read the
    physical zarr store with the equivalent filter `metadata_query: "validity == 1.0"`
    (`configs/data/paths/kaggle_defog_valid_*.yaml`, `data/dataset/base.py`).
  - *Patient normalisation:* an unused `stats_path` argument was removed from
    `configs/model/preprocessors/hierarchical_norm_{pure,fan}.yaml`. The preprocessor
    never read it. Patient statistics were, and still are, computed from the training
    fold and injected at train start (`pipeline/base.py`).
  - *Evaluation:* an inference batch-size override (`EVAL_BATCH_SCALE`) and
    Windows-safe zarr writing were added. Model-selection keys in
    `scripts/eval/eval_comprehensive.py` were renamed; checkpoint paths are unchanged.
  - *Tooling:* reproduction tooling (`reproduce.*`, CI, `release/`) and further
    SSL-ablation configs were added, and unused scripts were removed.
- **Results:** the headline evaluation results were re-verified from the released
  weights and data with the scripts in this repository (see `REPRODUCE.md` and the
  `verification` field in `release/manifest.yaml`). Retraining from scratch with this
  snapshot was not re-run end to end for the release.
- **Release format:** the published weights were converted from the authors' Lightning
  checkpoints to weights-only `.safetensors` files. The conversion was verified for all
  30 artifacts: every released tensor is bit-identical to the tensor in the source
  checkpoint, and each of the 27 classification models, rebuilt from this repository's
  experiment config and the released weights, produces bit-identical outputs to the same
  model rebuilt from the source checkpoint's own stored configuration. The only tensors
  not carried over are the RevIN session-normaliser buffers, which hold the statistics of
  the last batch seen and are recomputed on every forward pass.

## Changes made only in this public copy

These edits were applied to the exported snapshot. None of them changes a computation.

1. **Personal filesystem paths removed.** Hard-coded absolute paths under the author's
   home directory were replaced:
   - raw-dataset locations in 11 scripts under `scripts/eval/`, `scripts/analysis/` and
     `scripts/data/` now resolve from `$FORGE_DATASETS_ROOT` (default `~/Datasets`);
   - `cd <absolute repo path>` in 16 runners under `scripts/shell/` is now
     `cd "$(dirname "$0")/../.."`;
   - the SBATCH template in `utils/downstream_task_callback.py` now uses the working
     directory.
2. **Files not included:** `notebooks/onboarding.ipynb`, an outdated tutorial that
   referenced files and configs that no longer exist, and two `*.yaml.unused` configs.
3. **Documentation:** new `README.md` and this file; `REPRODUCE.md`, `scripts/README.md`
   and `configs/README.md` were updated to current paths and the new repository URL; the
   model-card generator (`scripts/release/hf_cards.py`) now points to this repository.
4. **`.gitignore`** replaced with a release-specific version.
5. **Weights-only loading.** `utils/released_weights.py` was added, and
   `scripts/eval/eval_comprehensive.py`, `reproduce.py` and the release tooling now
   rebuild each released model from its `experiment` config instead of from a
   configuration pickled inside a checkpoint. Locally trained Lightning checkpoints are
   still accepted, which is what the SSL-ablation arms use.

Kept as provenance, but runnable only by the authors:

- `scripts/release/` contains the tooling that built the Hugging Face release (manifest
  builder, checkpoint slimming, card generators, uploaders). The uploaders do nothing
  unless given `--execute` and need write access to the authors' Hugging Face account.
  They contain no credentials.
- `scripts/analysis/normalized_ap_comparison.py` and `report_biomarkers.py` query the
  authors' Weights & Biases workspace by entity and run ID. That workspace is not
  public.

## Not included, by design

- Git history of the private research repository.
- Raw or processed participant data (distributed separately; see README, "Data
  availability").
- Model checkpoints (distributed on Hugging Face).
- Author-private derived prediction vectors used by eleven supplementary scripts. Those
  scripts exit with an `[author-only]` message.
- Experiment logs, W&B run data, local caches, drafts and internal notes.

## Security and privacy review

Before the initial commit, the snapshot was scanned for credentials (API keys, tokens,
W&B / Hugging Face keys, passwords, private keys, `.env` files), personal filesystem
paths, embedded notebook outputs, and participant data. The scan found no credentials,
notebook outputs or participant-level data. Split files contain only de-identified
participant codes from the source datasets.

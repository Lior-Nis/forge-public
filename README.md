# FORGE

Code for the manuscript **"Self-supervised learning improves cross-cohort freezing-of-gait
detection from a single lower-back accelerometer"** (submitted to *npj Digital Medicine*,
2026).

FORGE (FOG Representation via Generative Encoding) detects freezing of gait (FOG) in
Parkinson's disease from one lower-back tri-axial accelerometer. A spectral-patch
Vision-Transformer encoder is pretrained by masked autoencoding (MAE) on unlabeled at-home
recordings. A lightweight BiGRU head is then trained on the labeled DeFOG cohort, with the
encoder frozen ("probe"), fine-tuned, or trained from scratch ("supervised"). The encoder
comes in three temporal contexts: SC (2 s), MC (5 s) and LC (10 s). The manuscript's
headline detector is the **MC frozen-probe ensemble**, evaluated with no target-cohort
training on external cohorts at a fixed operating point chosen on DeFOG validation data.

> **About this repository.** This is a clean reproducibility snapshot of the code
> behind the submitted manuscript. It was exported from the authors' private research
> repository and starts from a single commit; **it does not contain the private
> development history**. See [`PROVENANCE.md`](PROVENANCE.md) for what the snapshot
> contains, how it was prepared, and what it can and cannot reproduce.

---

## Installation

Requires [uv](https://docs.astral.sh/uv/). The project pins Python `>=3.11,<3.12`, and
`uv` provisions that interpreter automatically.

```bash
git clone https://github.com/Lior-Nis/forge-public.git
cd forge-public
uv sync --frozen          # installs the exact versions recorded in uv.lock
```

Verified environment for this snapshot: **Python 3.11.14**, uv 0.9.14, macOS (arm64).
CI (`.github/workflows/ci.yml`) installs from the same lockfile on Ubuntu, macOS and
Windows.

| Package | Version | Package | Version |
|---|---|---|---|
| torch | 2.9.1 | pytorch-lightning | 2.6.0 |
| numpy | 1.26.4 | hydra-core / omegaconf | 1.3.2 / 2.3.0 |
| scikit-learn | 1.6.1 | pydantic | 2.12.5 |
| scipy | 1.15.3 | zarr | 3.1.5 |
| pandas | 2.3.1 | pingouin (ICC) | 0.5.5 |
| huggingface-hub | 1.3.1 | | |

The full resolved environment is in `uv.lock`. On Windows, `pyproject.toml` pulls the
CUDA 12.6 PyTorch wheels. On Linux and macOS it uses the default PyPI wheels.

---

## Released model weights

Weights are on Hugging Face: **[`Liornis/forge-fog`](https://huggingface.co/Liornis/forge-fog)** (public, MIT).

- Format: **weights only** (`.safetensors`). Each file holds the trained tensors and the
  normalisation buffers derived from its training fold, plus small string metadata (name,
  context, phase, fold, seed, and the `experiment` config that rebuilds the model).
  No training configuration, optimiser state or local file path is included, and loading
  executes no pickled code.
- Contents: 3 MAE encoders (`encoders/{lc,mc,sc}.safetensors`) plus 27 classification heads
  (`classification/{lc,mc,sc}_{probe,finetune,supervised}_fold{0,1,2}.safetensors`), all
  seed 42.
- Revision published with this snapshot:
  **`fbdf75f6fda83937d8a0a079043631d43b1f8f47`**. Pin it with `--revision` for an exact
  match; `checksums.json` in the same repo lists each file's SHA-256.
- The manuscript's headline detector is a nine-head ensemble: 3 participant folds × 3
  seeds over one shared frozen MC encoder. This release ships the **seed-42** heads.
  Encoder + `mc_probe_fold{0,1,2}` rebuild the seed-42 three-fold ensemble that the
  scripts here evaluate. It lands within about 0.02 of the nine-head manuscript values.

Download them (`reproduce.py` does this for you):

```bash
uv run hf download Liornis/forge-fog --repo-type model \
  --revision fbdf75f6fda83937d8a0a079043631d43b1f8f47 --local-dir release/forge-fog
```

Each classification file already contains its encoder, so `encoders/*.safetensors` are
needed only to train new heads. To load one, rebuild the model from the `experiment`
config that `release/manifest.yaml` records for it:

```python
from utils.released_weights import load_released_model

model, config = load_released_model(
    "release/forge-fog/classification/mc_probe_fold0.safetensors",
    experiment="classification/spectral_patch_mae_mc_valid_defog_soft",
    overrides=["data/splits=kaggle_labeled/kfold_defog_fogcount_valid_mc0"],
)
```

## Data availability

Processed datasets (standardised session CSVs for DeFOG/Kaggle-labeled, FogAtHome, and
FogAtHome daily living) are hosted on Hugging Face as
**[`Liornis/fog-dataset`](https://huggingface.co/datasets/Liornis/fog-dataset)**.

> **Access is gated.** The dataset repository is visible to everyone, and downloads
> require an access request that the authors review. Sign in to Hugging Face, open the
> dataset page, and accept the terms; once access is granted, `hf auth login` on the
> machine that runs the reproduction is enough. Until access is granted, the
> data-dependent steps (`reproduce.sh`/`reproduce.bat`, the zarr builds and all
> evaluations) fail with an authentication error. The code, configs, split definitions
> and released weights are public and need no request.
>
> The recordings come from third-party clinical studies and remain subject to the
> data-use terms of each source study. They are **not** covered by this repository's MIT
> license.

The raw public source for the in-distribution cohorts is the Kaggle *Parkinson's Freezing
of Gait Prediction* competition data (tDCS-FOG, DeFOG, unlabeled daily-living). The
tDCS-FOG and Stanford external results in the manuscript are **not** recomputable from
this repository (see below).

---

## Reproducing the manuscript's evaluation

One-command driver (downloads weights + data, builds zarrs, runs the evaluation, and
checks the numbers against manuscript tolerances):

```bash
./reproduce.sh            # headline MC results
./reproduce.sh --full     # full LC/MC/SC × probe/finetune/supervised table
```

Windows: double-click `reproduce.bat`. Step-by-step instructions and expected output are
in [`REPRODUCE.md`](REPRODUCE.md).

Canonical evaluation entry points (see [`scripts/eval/README.md`](scripts/eval/README.md)):

| Script | Produces |
|---|---|
| `scripts/eval/eval_comprehensive.py` | Frame- and window-level AUROC / AP / NormAP for every dataset × context × model. Its `CKPT`/`ZARR` maps are authoritative. |
| `scripts/eval/compute_icc_thresholds.py` | Clinical agreement: ICC(%TF), #FOG, duration, under both threshold protocols. |
| `scripts/eval/eval_dailyliving_walkstand.py`, `eval_fogathome_walkstand.py` | Activity-conditioned (walking + standing) evaluation. These need the per-frame `Activity` annotation, which is read from `$FORGE_DATASETS_ROOT` (default `~/Datasets`) and is **not** in the HF release. |
| `scripts/eval/bootstrap_significance.py` | Paired bootstrap significance tests. |

**The manuscript reports two model sets; don't read them as one.**

| Model set | What it is | Where |
|---|---|---|
| `mc_probe_ensemble` | The released detector: nine BiGRU heads (3 folds × 3 seeds) over one shared frozen 5 s encoder, applied to every external cohort at the unchanged DeFOG operating point of 0.35. | Table 3 |
| `mc_matched_arms` | The controlled comparison behind the abstract's headline effect: two arms differing only in encoder initialisation under matched downstream training (self-supervised, i.e. joint encoder–classifier optimisation, vs supervised from scratch). | Table 2 |

On FogAtHome-provoking the released detector reaches AUROC 0.887, while the matched
self-supervised arm reaches 0.861 against 0.752 for supervised from scratch. Those are
different model sets, not competing estimates of the same quantity. Every entry in
`release/manifest.yaml` names its `model` set and the manuscript `table` it comes from.

[`release/manifest.yaml`](release/manifest.yaml) maps every reported number to its
weights, config and command. Each entry also carries a `verification` field:

| `verification` | Meaning | Results |
|---|---|---|
| `one_click` | Recomputed by `reproduce.sh` | FogAtHome AUROC / AP / ICC(%TF); DeFOG window AP |
| `scripted` | Recomputable with the listed extra commands | FogAtHome daily-living AUROC / AP / ICC(%TF) |
| `recorded` | Reported value only; cohort not in the public release | tDCS-FOG and Stanford results |

Expected agreement: the scripts run the seed-42 three-fold ensemble over the full window
grid. They match the manuscript values to about 0.01–0.02, not exactly (details in
`REPRODUCE.md`).

**Supplementary scripts that cannot run from a clone.** Eleven supplementary analyses
read derived prediction vectors from an author-private directory that is not included:
`scripts/eval/kaggle_*.py`, `eval_dl_comparison_table.py`, `build_kaggle_idmap.py`,
`scripts/analysis/fig_kaggle_filter_robustness.py` and `fig_ssl_collapse.py`. They
include the daily-living comparison against the Kaggle competition winners. They exit
with an `[author-only]` message. No `one_click` or `scripted` result depends on them.
Several figure scripts under `scripts/analysis/` likewise write to or read from that
private directory.

---

## Training

Configuration is Hydra-based (`configs/`) with Pydantic validation. Configurations used in
the manuscript:

| Stage | Command / config |
|---|---|
| MAE pretraining, MC (5 s) | `uv run python scripts/train/pretrain_mae.py experiment=pretraining/spectral_patch_mae_medcontext_daily` |
| MAE pretraining, SC (2 s) / LC (10 s) | `experiment=pretraining/spectral_patch_mae_shortcontext_daily` / `experiment=pretraining/spectral_patch_mae_daily` |
| Probe / fine-tune heads | `experiment=classification/spectral_patch_mae_{lc,mc,sc}_valid_defog_soft` |
| Supervised-from-scratch heads | `experiment=classification/supervised_{lc,mc,sc}_fogr025_defog` |
| Probe / fine-tune runners | `scripts/shell/compare_context_lengths_fogr.sh` (per-context, per-fold CLI overrides) |
| Supervised-from-scratch runner | `scripts/shell/run_fogratio_supervised.sh` |
| SSL-objective ablation | `scripts/train/pretrain_{simclr,jepa}.py`, `configs/experiment/pretraining/spectral_patch_*`, `scripts/shell/run_ssl_ablation_all128.sh` |
| Label-efficiency curves | `scripts/shell/run_label_efficiency*.sh`, `scripts/eval/eval_label_efficiency*.py` |

Pretraining uses the Kaggle unlabeled daily-living recordings (about 64 GB). The
dataset-processing entry point is `uv run python -m data.process paths=<paths> process=<process>`
(examples in the `onboard-repo` skill). Training logs to Weights & Biases by default. Set
`WANDB_MODE=offline` or use `train/logger=no_logging` to run without an account.

### Cross-validation splits

| Split | Location |
|---|---|
| DeFOG 3-fold participant-level CV (57 participants; 19 train / 19 val / 19 test per fold) — used for the released heads, one split per context | `configs/data/splits/kaggle_labeled/kfold_defog_fogcount_valid_{lc,mc,sc}{0,1,2}.yaml` |
| FogAtHome leave-one-participant-out (12 folds) | `configs/data/splits/fogathome_lopo_fold{0..11}.yaml` |
| FogAtHome 3-fold fine-tuning curve | `configs/data/splits/fogathome_finetune_3fold{0,1,2}.yaml` |

Split files list de-identified participant codes from the source datasets, and
`release/manifest.yaml` names the split file belonging to each released head. Generators
for several of the split families are in `scripts/data/` (`create_fogcount_splits.py`,
`create_fogathome_lopo_splits.py`); the folds themselves are committed as the YAML files
above, which are the record of what ran.

---

## Statistical analysis

- **Agreement:** ICC computed with `pingouin` in `scripts/eval/compute_icc_thresholds.py`.
  The threshold protocol `defog_val_pr11` is a PR-(1,1) operating point chosen on
  held-out DeFOG validation and applied unchanged to external cohorts.
- **Discrimination:** AUROC, AP and prevalence-normalised AP in
  `scripts/eval/eval_comprehensive.py`.
- **Confidence intervals / significance:** participant-level bootstrap
  (`scripts/eval/bootstrap_significance.py`; manifest `ci:` fields record the reported
  intervals).

The provenance of each reported number is in `release/manifest.yaml`. How this snapshot
was prepared is in `PROVENANCE.md`.

## Repository layout

```
configs/    Hydra configs (experiments, model, training, data paths, CV splits)
data/       Data processing (raw → zarr), datasets, datamodule
model/      Backbones (spectral-patch ViT), heads, transforms, losses
pipeline/   PyTorch Lightning pipelines (classification, MAE, SimCLR, JEPA, …)
managers/   Logging, metrics, weight loading / freezing
scripts/    Entry points: train/, eval/, analysis/, data/, embed/, shell/, release/
release/    manifest.yaml (checkpoint + result provenance), release runbook
tests/      CPU test suite (uv run pytest tests/)
reproduce.* One-command evaluation reproduction
```

## Tests

```bash
uv run pytest tests/ -q
```

## Citation

If you use this code, please cite the manuscript (details will be updated on publication):

```bibtex
@article{nisimov2026forge,
  title   = {Self-supervised learning improves cross-cohort freezing-of-gait detection
             from a single lower-back accelerometer},
  author  = {Nisimov, Lior and Salomon, Amit and Gazit, Eran and Herman, Talia and
             Rokach, Lior and Hausdorff, Jeffrey M. and Shimoni, Nathaniel},
  journal = {Submitted to npj Digital Medicine},
  year    = {2026}
}
```

## License

MIT. See [`LICENSE`](LICENSE). The license covers this code and the released weights.
It does not cover the third-party datasets.

## Contact

Questions about the code: open a GitHub issue, or email lior3226@gmail.com.

Questions about the study, and data-access requests: the corresponding author named in the
manuscript. Dataset access is granted through the request form on the
[dataset page](https://huggingface.co/datasets/Liornis/fog-dataset).

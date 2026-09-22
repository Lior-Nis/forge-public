# FORGE — Self-Supervised FOG Detection

A PyTorch Lightning framework for detecting Freezing of Gait (FOG) episodes in Parkinson's disease patients using lower-back accelerometer data. Companion code for the FORGE paper.

**Released artifacts:** pretrained weights — [`Liornis/forge-fog`](https://huggingface.co/Liornis/forge-fog) · datasets — [`Liornis/fog-dataset`](https://huggingface.co/datasets/Liornis/fog-dataset). To reproduce the released detector from public assets, follow [`REPRODUCE.md`](REPRODUCE.md). Exact artifact provenance is recorded in `release/manifest.yaml`.

## Features

- **4 training pipelines**: supervised classification, MAE, SimCLR, and JEPA self-supervised pretraining
- **Signal transforms**: Wavelet, STFT, Mel spectrogram, GAF, raw, spectral-patch
- **Architectures**: FogFormer (ViT + ALiBi), ResNet, ConvNeXt, ensemble backbones
- **Config-driven**: Hydra for experiments, Pydantic for validation — no code changes needed for sweeps
- **Transfer learning**: flexible weight loading, backbone freezing, progressive unfreezing

## Installation

```bash
git clone https://github.com/Lior-Nis/forge.git
cd forge
uv sync --frozen   # or: pip install -e .
```

Requires Python 3.11 (the project pins `>=3.11,<3.12`); `uv` provisions it automatically. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

## Quick Start

```bash
# Supervised classification
uv run python scripts/train.py experiment=classification/baseline

# Override any parameter via CLI (preferred over creating new config files)
uv run python scripts/train.py experiment=classification/baseline \
  model/transform=wavelet \
  train.optimizer.lr=1e-4 \
  data.dataloader.batch_size=32

# Multi-run sweep
uv run python scripts/train.py -m experiment=classification/baseline \
  model/transform=raw,gaf,stft,wavelet

# MAE pretraining
uv run python scripts/train.py experiment=pretraining/spectral_patch_mae_daily

# JEPA pretraining
uv run python scripts/train.py experiment=pretraining/spectral_patch_jepa_daily

# Finetune from pretrained backbone
uv run python scripts/train.py experiment=classification/spectral_patch_supervised_defog \
  train.weights.load_from_registry="model:best" \
  train.weights.freeze_backbone=true
```

## Project Structure

```
forge/
├── configs/              # Hydra configs
│   ├── experiment/       # Complete experiment setups (start here)
│   │   ├── classification/
│   │   └── pretraining/
│   ├── model/            # Backbones, heads, transforms, preprocessors
│   ├── train/            # Optimizers, schedulers, losses, callbacks
│   └── data/             # Datasets, splits, dataloaders
│
├── src/                  # Importable Python packages
│   ├── pipeline/         # PyTorch Lightning pipelines
│   ├── model/            # Backbones, transforms, heads, and losses
│   ├── data/             # Data modules, datasets, and processing
│   ├── managers/         # Training logging, metrics, and weights
│   └── utils/            # Shared utilities
│
├── scripts/              # Entry points and focused utilities
│   ├── train.py          # Unified supervised and SSL training entrypoint
│   ├── eval/             # Evaluation and clinical metrics
│   ├── analysis/         # plot_*.py, analyze_*.py, aggregate_*.py
│   ├── data/             # generate_splits.py, standardize_datasets.py, …
│   ├── embed/            # extract_embeddings.py, train_embedding_probe.py
│   └── release/          # Artifact export and publication utilities
│
├── data/                 # Local downloaded/processed data (gitignored)
├── release/              # Public artifact manifest and release tooling
└── tests/                # Test suite
```

## Configuration

The framework uses [Hydra](https://hydra.cc/) for configuration with Pydantic validation.

**Key principle: use CLI overrides instead of creating new config files** for parameter variations.

```bash
# ✅ Correct — override parameters at the CLI
uv run python scripts/train.py experiment=classification/baseline \
  model/transform=gaf train.optimizer.lr=1e-3

# ❌ Wrong — don't create configs/experiment/classification/baseline_lr_1e3.yaml
```

### Configuration groups

| Group | Examples |
|-------|---------|
| `experiment` | `baseline`, `best`, `quick_win`, `spectral_patch_supervised_defog` |
| `model/backbone` | `fogformer`, `vit`, `resnet18`, `convnext_tiny`, `ensemble` |
| `model/transform` | `wavelet`, `stft`, `mel`, `gaf`, `raw`, `patch` |
| `train/loss` | `cross_entropy`, `focal_loss`, `mae`, `simclr`, `jepa`, `lejepa` |
| `train/optimizer` | `adam`, `adamw`, `adamw_differential_lr` |

## Transfer Learning

```bash
# Linear probe (frozen backbone)
uv run python scripts/train.py experiment=classification/baseline \
  train.weights.load_from_registry="model:best" \
  train.weights.freeze_backbone=true \
  train.optimizer.lr=1e-3

# Full finetuning
uv run python scripts/train.py experiment=classification/baseline \
  train.weights.load_from_registry="model:best" \
  train.weights.freeze_backbone=false \
  train.optimizer.lr=1e-5

# Progressive unfreezing
uv run python scripts/train.py experiment=classification/baseline \
  train.weights.load_from_registry="model:best" \
  train.weights.freeze_backbone=true \
  train.weights.unfreeze_schedule.enabled=true \
  train.weights.unfreeze_schedule.unfreeze_at_epoch=15
```

## Development

```bash
# Run tests (cross-platform; no Makefile required)
uv sync --frozen --extra dev
uv run pytest tests/

# Lint and format (install dev extras first: uv sync --frozen --extra dev)
uv run ruff check .
uv run ruff format .
```

Interactive analysis tools are optional: install them with
`uv sync --frozen --extra analysis`.

Dataset provenance and licensing notes are in [`DATASETS.md`](DATASETS.md).

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

For questions or issues, please open a GitHub issue or email lior3226@gmail.com.

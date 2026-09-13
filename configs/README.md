# Configuration Directory

## Overview

Hydra configuration files for the FoG detection framework with Pydantic validation.

**Paper configurations**: see the "Configurations used in the manuscript" section of the top-level README.
**Validation**: Pydantic models in `data/config.py`, `data/dataset/config.py`, `data/datamodule/config.py`

---

## Directory Structure

```
configs/
├── config.yaml                # Root config with global defaults
├── data/                      # Data configurations
│   ├── dataloader/            # DataLoader settings (batch size, workers, sampling)
│   ├── dataset/               # Dataset types (binary, multiclass, MAE, SimCLR)
│   ├── hierarchical_norm/     # Normalization strategies
│   ├── process/               # Processing pipelines
│   ├── splits/                # Train/val/test splits
│   └── paths/                 # Data paths
├── experiment/                # Complete experiment configs
│   ├── classification/        # Supervised / probe / fine-tune experiments
│   ├── downstream/            # Downstream task experiments
│   └── pretraining/           # Self-supervised pretraining
├── model/                     # Model component configs
│   ├── backbone/              # Architectures (FogFormer, ViT, ResNet, ConvNeXt)
│   ├── head/                  # Task-specific heads
│   ├── preprocessors/         # Preprocessing pipelines
│   ├── transform/             # Signal transforms (wavelet, STFT, GAF, etc.)
│   └── *_augmentor/           # Data augmentation
└── train/                     # Training configurations
    ├── callback/              # PyTorch Lightning callbacks
    ├── logger/                # Logging configurations
    ├── loss/                  # Loss functions
    ├── optimizer/             # 3 optimizers
    ├── registry/              # Model registry configs
    ├── scheduler/             # Learning rate schedulers
    ├── task_manager/          # Task-specific managers
    └── trainer/               # PyTorch Lightning Trainer settings
```

---

## Usage Patterns

### Running Experiments

Use experiment configs as entry points:

```bash
# Baseline experiment
python scripts/train/train_classification.py experiment=classification/baseline

# Best known configuration
python scripts/train/train_classification.py experiment=classification/best

# Multiclass classification
python scripts/train/train_classification.py experiment=classification/multiclass_base
```

### CLI Overrides (Preferred for Parameter Sweeps)

Instead of creating new config files, use CLI overrides:

```bash
# Override transform
python scripts/train/train_classification.py experiment=classification/baseline model/transform=gaf

# Override optimizer params
python scripts/train/train_classification.py experiment=classification/baseline \
  train.optimizer.lr=1e-3 \
  train.optimizer.weight_decay=0.0

# Multiple overrides
python scripts/train/train_classification.py experiment=classification/baseline \
  model/transform=stft \
  train/loss=focal_binary_bce \
  train.optimizer.lr=5e-4
```

### Hydra Multirun for Parameter Sweeps

Run multiple experiments with different parameters:

```bash
# Transform sweep
python scripts/train/train_classification.py -m experiment=classification/baseline \
  model/transform=raw,gaf,stft,wavelet,melspec,hilbert

# Optimizer sweep
python scripts/train/train_classification.py -m experiment=classification/baseline \
  train.optimizer.lr=1e-4,5e-4,1e-3 \
  train.optimizer.weight_decay=0.0,0.01

# Grid search
python scripts/train/train_classification.py -m experiment=classification/baseline \
  model/transform=wavelet,gaf \
  train.optimizer.lr=1e-4,1e-3
```

### Viewing Resolved Config

Check what your final config looks like:

```bash
python scripts/train/train_classification.py experiment=classification/baseline --cfg job
```

---

## Configuration Guidelines

### When to Create a New Config File

Create a new config file when:
- Representing a **semantically different** component (e.g., new backbone architecture)
- Creating a **fundamentally different** experimental setup
- Defining a **new task type** (classification vs pretraining)

### When to Use CLI Overrides

Use CLI overrides for:
- **Parameter tuning** (learning rate, batch size, dropout)
- **Component swapping** (different transforms, optimizers, losses)
- **Quick experiments** and ablations
- **Hyperparameter sweeps**

### Example: Transform Comparison

❌ **Don't do this** (creates 6 config files):
```
experiment/classification/
  - binary_fog_transform_raw.yaml
  - binary_fog_transform_gaf.yaml
  - binary_fog_transform_stft.yaml
  # ... etc
```

✅ **Do this instead** (1 config file + CLI):
```bash
python scripts/train/train_classification.py -m experiment=classification/baseline \
  model/transform=raw,gaf,stft,wavelet,melspec,hilbert
```

---

## Config Composition

Experiment configs use Hydra's defaults mechanism:

```yaml
# experiment/classification/baseline.yaml
defaults:
  - /model/preprocessors@model.preprocessors: minimal
  - /model/transform@model.transform: wavelet
  - /model/backbone@model.backbone: fogformer
  - /model/head@model.head: gru_sequence_binary
  - /train/optimizer@train.optimizer: adamw
  - /train/loss@train.loss: simple_ce_masked
  - /data: kaggle_labeled
  - _self_

global:
  experiment_name: baseline
  seed: 42

train:
  trainer:
    max_epochs: 30
```

---

## Validation

Configuration validation is handled by Pydantic models (`pipeline/config.py`, `model/config.py`, `data/config.py`) — type checking, constraints, and field validation.

---

## Troubleshooting

### Config Not Found

```bash
Error: Could not find 'foo' in config search path
```

Check if the config was removed during cleanup. Use CLI override instead:
```bash
python scripts/train/train_classification.py experiment=classification/baseline model/transform=foo
```

### Missing Key Error

```bash
Error: Missing key 'train.optimizer.lr'
```

Ensure the experiment config includes the optimizer:
```yaml
defaults:
  - /train/optimizer@train.optimizer: adamw
```

### Validation Error

```bash
ValidationError: weight_decay must be non-negative
```

Fix the invalid value in your config or CLI override. Pydantic validation catches errors early.

---

## References

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

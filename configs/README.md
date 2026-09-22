# Configuration

FORGE uses Hydra composition with Pydantic validation. Start from an experiment
config, then use command-line overrides for folds, seeds, hyperparameters, and
component swaps.

```bash
# Inspect a resolved experiment without training
uv run python scripts/train.py experiment=classification/best_fixed --cfg job

# Train one run
uv run python scripts/train.py experiment=classification/best_fixed \
  global.seed=42 train.optimizer.lr=1e-4

# Run a Hydra sweep instead of adding wrapper scripts or near-duplicate YAMLs
uv run python scripts/train.py -m experiment=classification/best_fixed \
  global.seed=42,43,44 train.optimizer.lr=1e-4,5e-4
```

## Groups

| Group | Responsibility |
|---|---|
| `experiment/` | Complete, meaningful experiment recipes. |
| `model/` | Backbones, transforms, heads, preprocessing, and augmentation. |
| `train/` | Pipeline target, loss, optimizer, scheduler, callbacks, and trainer. |
| `data/` | Dataset adapter, processing recipe, paths, and immutable splits. |

The selected `train` config declares `pipeline_target`; `scripts/train.py`
resolves that class and provides the common training lifecycle. This separates
the data task type from the pipeline implementation—for example, TFC uses the
SimCLR data interface while selecting its own pipeline class.

## When to add a config

Add a config for a new architecture, dataset adapter, task, or fundamentally
different experiment. Use overrides for parameter changes such as learning
rate, batch size, dropout, seed, and fold.

Keep split membership files explicit. They are provenance artifacts, not
parameters to regenerate at runtime. If two experiments use identical split
membership, both should reference the same split config.

Configuration validation is defined in `pipeline/config.py`, `model/config.py`,
and the `data/` config models. The test suite composes and validates every
experiment config.

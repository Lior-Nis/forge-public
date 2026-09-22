# Command-line entrypoints

The supported command surface is intentionally small. Experiment behavior is
selected with Hydra configs and overrides rather than additional wrapper
scripts.

## Train

`train.py` launches every supervised and self-supervised pipeline. The chosen
experiment config supplies the concrete pipeline class.

```bash
# Classification
uv run python scripts/train.py experiment=classification/baseline

# MAE, SimCLR/VICReg, or JEPA pretraining
uv run python scripts/train.py experiment=pretraining/spectral_patch_mae_daily
uv run python scripts/train.py experiment=pretraining/spectral_patch_simclr_daily
uv run python scripts/train.py experiment=pretraining/spectral_patch_jepa_daily

# Sweeps use Hydra directly
uv run python scripts/train.py -m experiment=classification/baseline \
  global.seed=42,43,44 train.optimizer.lr=1e-4,5e-4
```

Use `ckpt_path=/path/to/checkpoint.ckpt` to resume full training state or
`weights_path=/path/to/checkpoint.ckpt` to initialize model weights with fresh
optimizer and scheduler state.

## Prepare data

The data processor is already a module entrypoint:

```bash
python -m data.process paths=fogathome_medcontext process=kaggle_medcontext
```

Dataset conversion and release-maintenance utilities live in `data/` and
`release/`, respectively.

## Evaluate and reproduce

- `../reproduce.py` is the public released-detector reproduction command.
- `eval/eval_comprehensive.py` is the general evaluation engine.
- `eval/eval_external_cohorts.py` computes the released external-cohort table.
- Other files under `eval/` and `analysis/` implement named analyses rather
  than acting as alternate top-level launchers.

One-off shell loops and machine-monitoring scripts are intentionally omitted.
Use Hydra multiruns for parameter sweeps; Git history retains the old wrappers.

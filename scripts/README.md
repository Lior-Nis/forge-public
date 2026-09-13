# Scripts

Entry points, grouped by stage. Every command runs from the repository root inside the
uv environment (`uv run python …` or `source .venv/bin/activate`).

| Directory | Contents |
|---|---|
| `train/` | `pretrain_mae.py` (FORGE masked-autoencoder pretraining), `pretrain_simclr.py`, `pretrain_jepa.py` (SSL-objective ablations), `train_classification.py` (probe / fine-tune / supervised heads), `test_classification.py`. |
| `data/` | Dataset standardisation, split generation (`create_fogcount_splits.py`, `create_fogathome_lopo_splits.py`, …), soft-label recomputation, context-length zarr views. |
| `eval/` | Evaluation and statistics. See [`eval/README.md`](eval/README.md) for canonical vs supplementary scripts. |
| `analysis/` | Figure generation and supplementary analyses. |
| `embed/` | Embedding extraction and embedding-probe training. |
| `shell/` | The batch runners used for the sweeps (label efficiency, SSL ablation, context comparison, …). They `cd` to the repository root and assume checkpoints under `checkpoints/`. |
| `release/` | Author-side tooling that built the Hugging Face release (manifest, checkpoint slimming, model/dataset cards, uploaders). |

Examples:

```bash
# FORGE MAE pretraining (medium context, 5 s)
uv run python scripts/train/pretrain_mae.py experiment=pretraining/spectral_patch_mae_medcontext_daily

# Frozen-encoder BiGRU probe on one DeFOG fold
uv run python scripts/train/train_classification.py experiment=classification/spectral_patch_mae_mc_valid_defog_soft

# Print the resolved Hydra config without running
uv run python scripts/train/train_classification.py experiment=classification/spectral_patch_mae_mc_valid_defog_soft --cfg job
```

`release/manifest.yaml` maps each reported number to its checkpoint, config and command.

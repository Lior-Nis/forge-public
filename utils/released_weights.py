"""Load a released FORGE model from weights-only files.

The public release ships plain tensors (`.safetensors`), not Lightning
checkpoints. Nothing about the training environment travels with the weights:
the model is rebuilt from this repository's own Hydra configs, and the tensors
are loaded into it.

    from utils.released_weights import load_released_model

    model, config = load_released_model(
        "release/forge-fog/classification/mc_probe_fold0.safetensors",
        experiment="classification/spectral_patch_mae_mc_valid_defog_soft",
        overrides=["data/splits=kaggle_labeled/kfold_defog_fogcount_valid_mc0"],
    )

`release/manifest.yaml` records the `experiment` and `splits` that belong to
each released file, so callers normally read them from there rather than
hard-coding them.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import torch

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(REPO_ROOT / "configs")


def runtime_buffer_keys(state_dict: dict) -> list[str]:
    """Keys holding per-batch runtime state rather than trained parameters.

    The RevIN session normaliser stores the statistics of the last batch it saw
    in `mean` / `stdev` buffers whose first dimension is the batch size. They are
    recomputed on every forward pass, so they are not part of the released
    weights; keeping them would bake a batch size into the file.

    Identified structurally: only RevIN registers a `stdev` buffer (the patient
    normaliser uses `std`), so a `<prefix>.stdev` key marks `<prefix>.mean` and
    `<prefix>.stdev` as runtime state.
    """
    keys = []
    for key in state_dict:
        if key.endswith(".stdev"):
            prefix = key[: -len(".stdev")]
            keys.append(key)
            if f"{prefix}.mean" in state_dict:
                keys.append(f"{prefix}.mean")
    return sorted(keys)


def strip_runtime_buffers(state_dict: dict) -> dict:
    """Return the released tensor set: trained weights and data-derived buffers."""
    drop = set(runtime_buffer_keys(state_dict))
    return {k: v for k, v in state_dict.items() if k not in drop}


def compose_config(experiment: str, overrides: Optional[Iterable[str]] = None):
    """Compose a Hydra config for an experiment, as the training entry point does."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from utils.paths import normalize_data_paths

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={experiment}", *(list(overrides) if overrides else [])],
        )
    normalize_data_paths(cfg.data.paths)
    return cfg


def build_model(experiment: str, overrides: Optional[Iterable[str]] = None):
    """Build an untrained pipeline from the repo config, ready for released weights.

    Weight-loading directives are cleared first: `load_from` would make the
    pipeline pull an encoder checkpoint off local disk at build time, which the
    released weights make unnecessary (they already contain the encoder).
    """
    from utils.config_loaders import load_config

    cfg = compose_config(experiment, overrides)
    config = load_config(cfg)
    # The config models are frozen, so rebuild rather than assign.
    weights = config.train.weights.model_copy(
        update={"load_from": None, "load_from_registry": None}
    )
    config = config.model_copy(
        update={"train": config.train.model_copy(update={"weights": weights})}
    )

    if config.train.pipeline_type == "segmentation":
        from pipeline.segmentation import SegmentationPipeline as Pipeline
    else:
        from pipeline.classification import ClassificationPipeline as Pipeline
    return Pipeline(config), config


def load_released_model(
    weights_path: str | Path,
    experiment: str,
    overrides: Optional[Iterable[str]] = None,
    device: str = "cpu",
):
    """Rebuild a released model and load its weights. Returns `(model, config)`.

    Raises if the file does not fit the config: any missing key other than a
    runtime buffer, or any unexpected key, means the weights and the config
    describe different models.
    """
    from safetensors.torch import load_file

    model, config = build_model(experiment, overrides)
    state_dict = load_file(str(weights_path))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    runtime = set(runtime_buffer_keys(model.state_dict()))
    unexplained = [k for k in missing if k not in runtime]
    if unexplained or unexpected:
        raise RuntimeError(
            f"{Path(weights_path).name} does not match experiment {experiment!r}: "
            f"missing={unexplained[:8]} unexpected={list(unexpected)[:8]}"
        )
    if missing:
        logger.debug("runtime buffers left at init values: %s", sorted(missing))

    model.eval()
    return model.to(device), config


def save_released_weights(state_dict: dict, path: str | Path, metadata: dict) -> dict:
    """Write a weights-only file. Returns the tensor set that was written."""
    from safetensors.torch import save_file

    released = strip_runtime_buffers(state_dict)
    contiguous = {k: v.contiguous().cpu() for k, v in released.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(contiguous, str(path), metadata={k: str(v) for k, v in metadata.items()})
    return contiguous


def load_encoder_state_dict(path: str | Path) -> dict:
    """Read an encoder's tensors from either release format.

    `.safetensors` is the released format; `.ckpt` is a locally trained Lightning
    checkpoint. Training runs point `train.weights.load_from` at one of these.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("state_dict", ckpt)

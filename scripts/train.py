"""Train any FORGE pipeline through one Hydra entrypoint.

The selected ``train`` config owns both the data task type and the concrete
pipeline class. This keeps experiment commands stable as new SSL methods are
added and avoids one near-identical launcher per method.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from data.datamodule.datamodule import FOGDataModule
from utils.callback_utils import instantiate_callbacks
from utils.config_loaders import load_config
from utils.debug_utils import get_wandb_tags_with_debug
from utils.paths import normalize_data_paths

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


def resolve_pipeline(target: str):
    """Resolve the configured pipeline class at the training seam."""
    pipeline_class = hydra.utils.get_class(target)
    if not issubclass(pipeline_class, pl.LightningModule):
        raise TypeError(f"train.pipeline_target must be a LightningModule, got {target}")
    return pipeline_class


def load_initial_weights(model: pl.LightningModule, weights_path: str | None) -> None:
    """Load model weights without restoring optimizer or scheduler state."""
    if weights_path is None:
        return
    checkpoint = torch.load(Path(weights_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded initial weights from %s (%d missing, %d unexpected)",
        weights_path,
        len(missing),
        len(unexpected),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(hydra_config: DictConfig) -> None:
    pl.seed_everything(hydra_config["global"]["seed"], workers=True)
    normalize_data_paths(hydra_config.data.paths)
    config = load_config(hydra_config)

    callbacks = instantiate_callbacks(hydra_config.train.callbacks)
    task_type = config.train.pipeline_type
    tags = get_wandb_tags_with_debug(
        [task_type, "training"] + list(hydra_config.train.logger.get("tags", []))
    )
    training_logger = hydra.utils.instantiate(hydra_config.train.logger, tags=tags)
    if training_logger is not None:
        resolved_config = OmegaConf.to_container(hydra_config, resolve=True)
        training_logger.experiment.config.update(resolved_config)

    trainer = hydra.utils.instantiate(
        hydra_config.train.trainer,
        callbacks=callbacks,
        logger=training_logger,
        deterministic=True,
    )
    data_module = FOGDataModule(data_cfg=config.data, task_type=task_type)
    pipeline_class = resolve_pipeline(config.train.pipeline_target)
    model = pipeline_class(config)
    load_initial_weights(model, hydra_config.get("weights_path"))

    trainer.fit(
        model,
        data_module,
        ckpt_path=hydra_config.get("ckpt_path"),
        weights_only=False,
    )

    if task_type in {"classification", "segmentation"}:
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        trainer.test(model, data_module)

    if training_logger is not None:
        training_logger.experiment.finish()


if __name__ == "__main__":
    main()

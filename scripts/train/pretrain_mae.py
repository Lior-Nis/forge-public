import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from data.datamodule.datamodule import FOGDataModule
from pipeline.mae import MAEPipeline
from utils.callback_utils import instantiate_callbacks
from utils.debug_utils import get_wandb_tags_with_debug
from utils.config_loaders import load_config
from utils.paths import normalize_data_paths

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(hydra_config: DictConfig) -> None:
    pl.seed_everything(hydra_config['global']['seed'], workers=True)
    normalize_data_paths(hydra_config.data.paths)
    config = load_config(hydra_config)
    logger.info("Configuration validation passed")

    callbacks = instantiate_callbacks(hydra_config.train.callbacks)

    config_tags = hydra_config.train.logger.get("tags", []) if hydra_config.train.logger else []
    script_tags = ["mae", "pretraining", "self-supervised"]
    tags = get_wandb_tags_with_debug(script_tags + config_tags)
    resolved_config = OmegaConf.to_container(hydra_config, resolve=True)
    training_logger = hydra.utils.instantiate(hydra_config.train.logger, tags=tags)
    if training_logger is not None:
        training_logger.experiment.config.update(resolved_config)

    ckpt_path = hydra_config.get("ckpt_path", None)
    weights_path = hydra_config.get("weights_path", None)
    trainer = hydra.utils.instantiate(
        hydra_config.train.trainer,
        callbacks=callbacks,
        logger=training_logger,
        deterministic=True,
    )

    data_module = FOGDataModule(data_cfg=config.data, task_type=config.train.pipeline_type)
    model = MAEPipeline(config)

    if weights_path is not None:
        # Load model weights only — fresh optimizer and scheduler (clean LR schedule).
        # Use this instead of ckpt_path when the checkpoint's scheduler state is stale.
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(f"Loaded weights from {weights_path}: {len(checkpoint['state_dict']) - len(missing)} keys loaded, {len(missing)} missing, {len(unexpected)} unexpected")

    trainer.fit(model, data_module, ckpt_path=ckpt_path, weights_only=False)

    if training_logger is not None:
        training_logger.experiment.finish()


if __name__ == "__main__":
    main()

"""Run classification inference on a dataset using a saved checkpoint.

Loads a trained model checkpoint and runs test evaluation on a specified dataset.
Normalization stats from the training fold are preserved in the checkpoint.

Usage:
    # Test on fogstar dataset
    python scripts/test_classification.py \
        experiment=classification/best_fixed \
        data/paths=fogstar_longcontext \
        data/splits=fogstar \
        ckpt_path=checkpoints/classification/apricot-wind-234/last.ckpt

    # Test on fogathome dataset
    python scripts/test_classification.py \
        experiment=classification/best_fixed \
        data/paths=fogathome_longcontext \
        data/splits=fogathome \
        ckpt_path=checkpoints/classification/apricot-wind-234/last.ckpt
"""

import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from data.datamodule.datamodule import FOGDataModule
from pipeline.classification import ClassificationPipeline
from utils.callback_utils import instantiate_callbacks
from utils.debug_utils import get_wandb_tags_with_debug
from utils.config_loaders import load_config
from utils.paths import normalize_data_paths

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(hydra_config: DictConfig) -> None:
    ckpt_path = hydra_config.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("ckpt_path is required. Pass it as: ckpt_path=/path/to/checkpoint.ckpt")

    pl.seed_everything(hydra_config["global"]["seed"], workers=True)
    normalize_data_paths(hydra_config.data.paths)
    config = load_config(hydra_config)
    logger.info("Configuration validation passed")

    # Build model from config, then load state dict from checkpoint
    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = ClassificationPipeline(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Filter out RevIN normalizer buffers — they have batch-dependent shapes
    # and are recomputed per-batch at inference time
    state_dict = checkpoint["state_dict"]
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k not in model_state or v.shape == model_state[k].shape
    }
    skipped = set(state_dict.keys()) - set(filtered.keys())
    if skipped:
        logger.info(f"Skipped {len(skipped)} keys with shape mismatch (dynamic buffers): {skipped}")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
    logger.info("Checkpoint loaded successfully")

    # Setup data module (test stage only)
    data_module = FOGDataModule(data_cfg=config.data, task_type="classification")
    data_module.setup(stage="test")
    logger.info(f"Test dataset: {len(data_module.test_dataset)} samples")

    # Setup trainer (no training, just test)
    callbacks = instantiate_callbacks(hydra_config.train.callbacks)

    config_tags = hydra_config.train.logger.get("tags", [])
    script_tags = ["classification", "inference", "external-eval"]
    tags = get_wandb_tags_with_debug(script_tags + config_tags)
    resolved_config = OmegaConf.to_container(hydra_config, resolve=True)
    training_logger = hydra.utils.instantiate(hydra_config.train.logger, tags=tags)
    if training_logger is not None:
        training_logger.experiment.config.update(resolved_config)

    trainer = hydra.utils.instantiate(
        hydra_config.train.trainer,
        callbacks=callbacks,
        logger=training_logger,
        deterministic=True,
    )

    trainer.test(model, datamodule=data_module)

    if training_logger is not None:
        training_logger.experiment.finish()


if __name__ == "__main__":
    main()

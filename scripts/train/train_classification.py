import gc
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
    pl.seed_everything(hydra_config['global']['seed'], workers=True)
    normalize_data_paths(hydra_config.data.paths)
    config = load_config(hydra_config)
    logger.info("Configuration validation passed")


    callbacks = instantiate_callbacks(hydra_config.train.callbacks)

    config_tags = hydra_config.train.logger.get("tags", [])
    script_tags = ["classification", "supervised"]
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

    data_module = FOGDataModule(data_cfg=config.data, task_type=config.train.pipeline_type)
    if config.train.adversarial is not None:
        from pipeline.adversarial_classification import AdversarialClassificationPipeline
        model = AdversarialClassificationPipeline(config)
    elif config.train.pipeline_type == "segmentation":
        from pipeline.segmentation import SegmentationPipeline
        model = SegmentationPipeline(config)
    else:
        model = ClassificationPipeline(config)
    trainer.fit(model, data_module)

    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    trainer.test(model, data_module)

    if training_logger is not None:
        training_logger.experiment.finish()


if __name__ == "__main__":
    main()

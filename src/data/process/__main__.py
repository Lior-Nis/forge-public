"""Dataset processing CLI entry point."""

import logging

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs/data", config_name="process_pipeline")
def main(cfg: DictConfig) -> None:
    """Hydra-based CLI interface for data processing."""
    input_dir = cfg.paths.input_dir

    logger.info("Processing data from: %s", input_dir)
    logger.info("Block length: %s", cfg.process.lengths.block_len)
    logger.info("Stride length: %s", cfg.process.lengths.stride_len)

    processor = hydra.utils.instantiate(cfg.process.processor, cfg)
    output_path = processor.process()
    logger.info("Post-creation validation complete for %s", output_path)


if __name__ == "__main__":
    main()

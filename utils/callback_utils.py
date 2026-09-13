"""
Callback utility functions for training scripts.

Provides a unified way to instantiate callbacks from Hydra config.
"""

import logging
from typing import List

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def instantiate_callbacks(callbacks_config: DictConfig) -> List[pl.Callback]:
    """
    Instantiate callbacks from a config dict.

    Args:
        callbacks_config: Dict of callback_name -> callback config with _target_

    Returns:
        List of instantiated callbacks
    """
    if callbacks_config is None:
        return []

    callbacks = []
    for name, cfg in callbacks_config.items():
        if cfg is None:
            continue
        try:
            callbacks.append(hydra.utils.instantiate(cfg))
            logger.debug(f"Instantiated callback: {name}")
        except Exception as e:
            logger.warning(f"Failed to instantiate callback '{name}': {e}")

    return callbacks

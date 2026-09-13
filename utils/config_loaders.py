"""Utilities for loading Hydra configs into Pydantic models."""
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError
from typing import Type, TypeVar, Any, Dict, Optional

from pipeline.config import Config, TrainConfig
from model.config import ModelConfig
from data.datamodule.config import DataConfig

T = TypeVar('T', bound=BaseModel)


def load_pydantic_config(hydra_config: DictConfig, pydantic_class: Type[T]) -> T:
    """
    Convert Hydra DictConfig to Pydantic model.

    Args:
        hydra_config: Hydra configuration object (OmegaConf DictConfig)
        pydantic_class: Pydantic model class to instantiate

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If the config doesn't match the Pydantic schema
    """
    config_dict = OmegaConf.to_container(hydra_config, resolve=True)

    try:
        return pydantic_class(**config_dict)
    except ValidationError as e:
        print(f"Error validating {pydantic_class.__name__}:")
        print(f"Config dict: {config_dict}")
        raise e


def load_config(hydra_config: DictConfig) -> Config:
    """
    Load unified configuration from Hydra config.

    Args:
        hydra_config: Root Hydra config containing model, train, and data sections

    Returns:
        Validated Config instance with all sub-configs
    """
    model_cfg = load_pydantic_config(hydra_config.model, ModelConfig)
    train_cfg = load_pydantic_config(hydra_config.train, TrainConfig)
    data_cfg = load_pydantic_config(hydra_config.data, DataConfig)

    return Config(model=model_cfg, train=train_cfg, data=data_cfg)


def ensure_dict(config: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Convert Pydantic model to dict if needed, otherwise return as-is."""
    return config.model_dump() if hasattr(config, 'model_dump') else config

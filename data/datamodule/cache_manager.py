"""Cache key generation for dataset caching."""

import hashlib
import json
import logging
import os
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def get_cache_key(data_cfg, cache_type: str) -> str:
    """
    Generate cache key for dataset configuration.

    Cache keys uniquely identify dataset configurations and invalidate when:
    - Data file content changes (via size validation)
    - Processing parameters change (block_len, stride_len, etc.)
    - Split configuration changes
    - Metadata files change

    Args:
        data_cfg: Data configuration (DictConfig or Pydantic DataConfig)
        cache_type: Type of cache (e.g., 'balanced_probabilistic')

    Returns:
        Cache key string (underscore-separated hash components)
    """
    from utils.paths import build_processed_dataset_path

    components = []

    data_path = build_processed_dataset_path(processed_dir=data_cfg.paths.processed_dir,
                                             block_len=data_cfg.process.lengths.block_len,
                                             stride_len=data_cfg.process.lengths.stride_len)
    components.append(data_path)

    if os.path.exists(data_path):
        components.append(f"size:{os.path.getsize(data_path)}")

    components.append(f"cache_type:{cache_type}")
    components.append(f"classification_strategy:{data_cfg.dataset.classification_strategy}")
    components.append(f"block_len:{data_cfg.process.lengths.block_len}")
    components.append(f"stride_len:{data_cfg.process.lengths.stride_len}")
    components.append(f"axis:{str(data_cfg.dataset.axis)}")

    # Handle both DictConfig and Pydantic models
    if isinstance(data_cfg.splits, DictConfig):
        splits_dict = OmegaConf.to_container(data_cfg.splits)
    else:
        splits_dict = data_cfg.splits.model_dump()
    components.append(f"splits: {json.dumps(splits_dict, sort_keys=True, separators=(',', ':'))}")

    hashes = [hashlib.sha256(comp.encode()).hexdigest()[:8] for comp in components]
    hash_of_hashes = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:8]
    return hash_of_hashes

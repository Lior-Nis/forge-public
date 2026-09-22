"""
FoG MAE Dataset - For Masked Autoencoder pretraining.

Handles:
- Signal loading for reconstruction tasks
- Mask creation (can be overridden by TaskManager)
- Clean MAE-specific data loading
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

from data.config import PathsConfig
from data.dataset.config import DatasetConfig
from .base import BaseFOGDataset
from .schema import DatasetSample

logger = logging.getLogger(__name__)


class FOGMAEDataset(BaseFOGDataset):
    """
    FoG dataset for MAE (Masked Autoencoder) pretraining.
    
    Returns:
        (signal, mask, metadata)
    """
    
    def __init__(
        self,
        paths_cfg: PathsConfig,
        dataset_cfg: DatasetConfig,
        stage: str,
        patient_ids: Optional[List[str]] = None
    ) -> None:
        """
        Initialize MAE dataset.

        Args:
            paths_cfg: Path configuration (shared across layers)
            dataset_cfg: Dataset-specific configuration
            stage: Dataset stage (train/val/test)
            patient_ids: Patient IDs for this stage (e.g., ['bf608b', 'bae0ce'])

        Note: Masking ratio is configured in model config and applied in MAE pipeline forward pass.
        """
        super().__init__(paths_cfg, dataset_cfg, stage, patient_ids)
        logger.info(f"Initialized MAE dataset for stage={stage}")
    
    def __getitem__(self, idx: int) -> DatasetSample:
        """
        Get MAE sample.

        Args:
            idx: Dataset index

        Returns:
            DatasetSample with signal and metadata (no labels for MAE)
        """
        # Get raw data from base class
        x, labels, patch_labels, valid_mask, metadata = self._get_raw_data(idx)

        # Return raw signal - masking will be done after spectral transform in pipeline
        # No labels needed for MAE reconstruction task
        return DatasetSample(
            signal=x,
            metadata=metadata
        )
    

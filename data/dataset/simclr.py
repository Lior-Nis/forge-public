"""
FoG SimCLR Dataset - For contrastive learning pretraining.

Returns raw signals without augmentation. Augmentation is applied
in the model pipeline on GPU for better performance.
"""

import logging
from typing import Dict, List, Optional

from data.config import PathsConfig
from data.dataset.config import DatasetConfig
from .base import BaseFOGDataset
from .schema import DatasetSample

logger = logging.getLogger(__name__)


class FOGSimCLRDataset(BaseFOGDataset):
    """
    FoG dataset for SimCLR contrastive learning.

    Returns raw signal without augmentation. The model pipeline creates
    two augmented views on GPU for efficient contrastive learning.
    """

    def __init__(
        self,
        paths_cfg: PathsConfig,
        dataset_cfg: DatasetConfig,
        stage: str,
        patient_ids: Optional[List[str]] = None
    ) -> None:
        """
        Initialize SimCLR dataset.

        Args:
            paths_cfg: Path configuration (shared across layers)
            dataset_cfg: Dataset-specific configuration
            stage: Dataset stage (train/val/test)
            patient_ids: Patient IDs for this stage (e.g., ['bf608b', 'bae0ce'])
        """
        super().__init__(paths_cfg, dataset_cfg, stage, patient_ids)
        logger.info(f"Initialized SimCLR dataset for {stage} stage")

    def __getitem__(self, idx: int) -> DatasetSample:
        """
        Get SimCLR sample.

        Args:
            idx: Dataset index

        Returns:
            DatasetSample with raw signal (no augmentation)
        """
        # Get raw data from base class
        x, labels, patch_labels, valid_mask, metadata = self._get_raw_data(idx)

        # Return raw signal - augmentation happens in model pipeline
        return DatasetSample(
            signal=x,
            metadata=metadata
        )
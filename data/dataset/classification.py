"""
FoG Classification Dataset - For supervised classification tasks.

Handles:
- Clean classification-specific data loading
- Consistent return type for classification
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from data.config import PathsConfig
from data.dataset.config import DatasetConfig
from .base import BaseFOGDataset
from .schema import DatasetSample

logger = logging.getLogger(__name__)


class FOGClassificationDataset(BaseFOGDataset):
    """
    FoG dataset for classification tasks.

    Returns:
        Dictionary with keys:
            - 'x': Signal tensor (shape: [channels, seq_len])
            - 'y': Sequence labels tensor
            - 'patch_y': Patch-level label tensor
            - 'session_id': Session identifier string
            - 'valid_mask': Boolean mask tensor for valid timesteps
            - 'metadata': Additional metadata dictionary
    """

    def __init__(
        self,
        paths_cfg: PathsConfig,
        dataset_cfg: DatasetConfig,
        stage: str,
        patient_ids: Optional[List[str]] = None
    ) -> None:
        """
        Initialize classification dataset.

        Args:
            paths_cfg: Path configuration (shared across layers)
            dataset_cfg: Dataset-specific configuration
            stage: Dataset stage (train/val/test)
            patient_ids: Patient IDs for this stage (e.g., ['bf608b', 'bae0ce'])
        """
        super().__init__(paths_cfg, dataset_cfg, stage, patient_ids)

    def __getitem__(self, idx: int) -> DatasetSample:
        """
        Get classification sample.

        Args:
            idx: Dataset index

        Returns:
            DatasetSample with signal, labels, patch_label, valid_mask, and metadata.

        Raises:
            ValueError: If labels are missing (classification requires labeled data)
        """
        # Get raw data from base class
        x, labels, patch_labels, valid_mask, metadata = self._get_raw_data(idx)

        # Check if we have labels (required for classification)
        if labels is None or patch_labels is None:
            raise ValueError(
                f"FOGClassificationDataset requires labeled data, but labels are missing "
                f"for session {metadata['session_id']}. Use FOGMAEDataset for unlabeled data."
            )

        # Convert labels to tensors
        y = torch.from_numpy(labels).long()
        # patch_labels may be float32 fog_ratio or legacy int — keep as float for comparison
        patch_y = torch.tensor(patch_labels, dtype=torch.float32)
        valid_mask_tensor = torch.from_numpy(valid_mask).bool()

        if self.dataset_cfg.classification_strategy == "binary_any_fog":
            y = (y > 0).long()
            patch_y = (patch_y > 0).long()  # works for both float fog_ratio and legacy int
        elif self.dataset_cfg.classification_strategy == "true_any_fog_from_labels":
            # Compute patch label from raw per-timestep labels: positive if ANY timestep is FOG.
            # Ignores stored patch_labels (which are dominant-class-based), reads y directly.
            y = (y > 0).long()
            patch_y = (y > 0).any().long()
        elif self.dataset_cfg.classification_strategy == "fog_ratio":
            # Soft label: fraction of frames that are FOG.
            # patch_labels is stored as float32 fog_ratio in zarr (after recompute script).
            # Fall back to on-the-fly computation for legacy zarrs that lack it.
            y = (y > 0).long()
            if patch_y.item() > 1.0:
                # Legacy multiclass value — zarr not yet updated; compute on-the-fly
                patch_y = (y > 0).float().mean()
            # else: patch_y is already fog_ratio (0.0–1.0) from updated zarr
        elif self.dataset_cfg.classification_strategy == "fog_ratio_025":
            # Binary label: positive if ≥25% of frames are FOG.
            # More principled than any_fog: consistent across context lengths,
            # robust to single-frame annotation noise at patch boundaries.
            y = (y > 0).long()
            if patch_y.item() > 1.0:
                patch_y = (y > 0).float().mean()
            patch_y = (patch_y >= 0.25).long()

        return DatasetSample(
            signal=x,
            metadata=metadata,
            labels=y,
            patch_label=patch_y,
            valid_mask=valid_mask_tensor
        )

    def get_sampling_metadata(self) -> "pd.DataFrame":
        """
        Return metadata for balanced sampling with labels in correct form.

        For binary classification, binarizes the class_label column.
        For multiclass classification, returns labels as-is.

        Returns:
            DataFrame with sampling metadata (class_label reflects task type)
        """
        import pandas as pd

        # Get base metadata
        metadata = super().get_sampling_metadata()

        if self.dataset_cfg.classification_strategy in ("binary_any_fog", "true_any_fog_from_labels", "fog_ratio_025"):
            metadata = metadata.copy()
            metadata['class_label'] = (metadata['class_label'] > 0).astype(np.int64)

        return metadata
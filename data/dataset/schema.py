"""Standardized data structures for FOG datasets."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch


@dataclass
class DatasetSample:
    """
    Standardized dataset sample across all tasks.

    This ensures all dataset types (Classification, MAE, SimCLR) return
    a consistent structure, enabling polymorphic batch processing.

    All datasets return this type from __getitem__, with task-specific
    fields set as needed.
    """
    # Core data (always present)
    signal: torch.Tensor  # Shape: (channels, seq_len), e.g., (3, 200)
    metadata: Dict[str, Any]  # Session info, patient_id, protocol, indices

    # Optional task-specific fields
    labels: Optional[torch.Tensor] = None  # Sequence labels for classification
    patch_label: Optional[torch.Tensor] = None  # Patch-level label
    valid_mask: Optional[torch.Tensor] = None  # Valid timesteps mask
    signal_augmented: Optional[torch.Tensor] = None  # Second view for SimCLR

    # Utility properties
    @property
    def is_labeled(self) -> bool:
        """Whether this sample has labels."""
        return self.labels is not None

    @property
    def is_contrastive_pair(self) -> bool:
        """Whether this is a contrastive learning pair (SimCLR)."""
        return self.signal_augmented is not None

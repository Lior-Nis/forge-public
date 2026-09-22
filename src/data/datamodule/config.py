"""DataModule-specific configuration."""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any

from data.config import PathsConfig
from data.dataset.config import DatasetConfig
from data.process.schemas import ProcessingConfig


class DataLoaderConfig(BaseModel):
    """DataLoader configuration."""

    batch_size: int = Field(gt=0, default=1024)
    num_workers: int = Field(ge=0, default=3)
    drop_last: bool = True
    prefetch_factor: int = Field(ge=1, default=2)
    pin_memory: bool = True
    persistent_workers: bool = True
    cache_data_dir: str

    # DataLoader behavior
    shuffle: bool = Field(default=True, description="Shuffle data (used when not using custom sampler)")

    # Sampling configuration
    balanced_sampling: bool = False  # Opt-in for advanced hierarchical sampling
    sampler_type: str = Field(default="balanced")
    balance_weights: Optional[Dict[str, bool]] = Field(
        default_factory=lambda: {
            'balance_protocols': True,
            'balance_patients': True
        }
    )

    @field_validator('sampler_type')
    @classmethod
    def validate_sampler_type(cls, v: str) -> str:
        """Validate sampler type is one of the supported types."""
        valid_types = {'balanced', 'quota'}
        if v not in valid_types:
            raise ValueError(
                f"sampler_type must be one of {valid_types}, got '{v}'"
            )
        return v

    model_config = {"frozen": True, "extra": "forbid"}


class SplitsConfig(BaseModel):
    """Train/val/test patient splits."""

    train: List[str] = Field(default_factory=list)
    val: List[str] = Field(default_factory=list)
    test: List[str] = Field(default_factory=list)

    @field_validator('train', 'val', 'test')
    @classmethod
    def validate_no_duplicates(cls, v: List[str]) -> List[str]:
        """Validate that split contains no duplicate patient IDs."""
        if len(v) != len(set(v)):
            raise ValueError("Split contains duplicate patient IDs")
        return v

    model_config = {"frozen": True, "extra": "forbid"}


class HierarchicalNormConfig(BaseModel):
    """Hierarchical normalization configuration."""

    enabled: bool = Field(default=True)
    protocol_harmonization: bool = Field(default=True)
    patient_normalization: bool = Field(default=True)
    session_normalization: bool = Field(default=True)

    model_config = {"frozen": True, "extra": "forbid"}


class PseudoLabelConfig(BaseModel):
    """Configuration for pseudo-label semi-supervised training."""

    enabled: bool = Field(default=False, description="Enable pseudo-label training")
    file_path: Optional[str] = Field(default=None, description="Path to pseudo-label .pt file")
    confidence_threshold: float = Field(
        default=0.9, ge=0.5, le=1.0,
        description="Minimum confidence to include a pseudo-labeled sample"
    )
    max_samples: Optional[int] = Field(
        default=None, ge=1,
        description="Optional cap on number of pseudo-labeled samples"
    )
    unlabeled_paths: Optional[str] = Field(
        default=None,
        description="Hydra paths config name for unlabeled data (e.g. kaggle_daily). "
        "If None, reuses the main data paths."
    )

    model_config = {"frozen": True, "extra": "forbid"}


class DataConfig(BaseModel):
    """Unified data configuration matching Hydra's 'data' section."""

    dataloader: DataLoaderConfig
    dataset: DatasetConfig
    splits: SplitsConfig
    paths: PathsConfig
    process: ProcessingConfig
    hierarchical_norm: HierarchicalNormConfig = Field(default_factory=HierarchicalNormConfig)
    pseudo_labels: PseudoLabelConfig = Field(default_factory=PseudoLabelConfig)

    model_config = {"frozen": True, "extra": "forbid"}

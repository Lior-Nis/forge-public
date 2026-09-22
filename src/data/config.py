"""Shared configuration classes across data pipeline."""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
import os


class PathsConfig(BaseModel):
    """Shared paths configuration across all layers."""

    # Input paths
    input_dir: str = Field(description="Directory containing raw data")

    # Output paths
    processed_dir: str = Field(description="Directory for processed datasets")

    # Explicit dataset path (relative to processed_dir)
    dataset_path: str = Field(
        description="Path to dataset relative to processed_dir. "
        "Can be physical (.zarr) or view (.yaml in views/ subdir). "
        "Examples: 'len200_stride100_kaggle.zarr', 'views/valid100.yaml'"
    )

    # Optional cached paths
    cache_dir: Optional[str] = None
    cache_root: Optional[str] = None

    # Optional pandas-query filter on patch metadata (e.g. "validity == 1.0").
    # Lets a "valid" path config apply a filter directly on a physical .zarr,
    # replacing the legacy gitignored views/defog_valid_*.yaml files.
    metadata_query: Optional[str] = None

    @field_validator('processed_dir', 'input_dir')
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate that path strings are not empty."""
        if not v or not v.strip():
            raise ValueError("Path string cannot be empty")
        return v.strip()

    @field_validator('dataset_path')
    @classmethod
    def validate_dataset_path(cls, v: str) -> str:
        """Validate dataset path format."""
        if not v or not v.strip():
            raise ValueError("dataset_path cannot be empty")

        v = v.strip()

        # Must be either .zarr or .yaml in views/
        if not (v.endswith('.zarr') or (v.startswith('views/') and v.endswith('.yaml'))):
            raise ValueError(
                f"dataset_path must be either a .zarr dataset or a view "
                f"(.yaml in views/ subdirectory). Got: '{v}'. "
                f"Examples: 'len200_stride100_kaggle.zarr' or 'views/valid100.yaml'"
            )
        return v

    @property
    def processed_dataset_path(self) -> str:
        """Full absolute path to the dataset (physical or view)."""
        return os.path.join(self.processed_dir, self.dataset_path)

    model_config = {
        "frozen": True,  # Immutable
        "extra": "allow"  # Allow extra fields (for config helper fields like data_root, raw_root, metadata paths, etc.)
    }

"""Virtual dataset views for zero-copy dataset subsetting.

This module provides functionality to create filtered "views" of existing Zarr
datasets without duplicating data. Views are defined via lightweight YAML configs
and are transparent to downstream code (Dataset classes, DataModule).

Key concepts:
- Physical Dataset: Original .zarr file containing all data
- View: Lightweight .yaml config defining filters on a physical dataset
- ResolvedDataset: Unified abstraction for both physical datasets and views

Example usage:
    # Create a view via CLI
    python -m data.views create \\
        --source data/processed/len200_stride100_kaggle.zarr \\
        --name defog_only \\
        --output data/processed/views/defog_only.yaml \\
        --filter protocol=defog

    # Use view in config (transparent to downstream code)
    paths:
      processed_dataset_path: data/processed/views/defog_only.yaml

See README.md for detailed documentation.
"""

# Import public API
from data.views.resolver import resolve_dataset_path, ResolvedDataset
from data.views.filters import apply_filters, summarize_filtered_metadata
from data.views.schemas import ViewConfig, ViewMetadata, SourceConfig, FilterConfig
from data.views.validator import validate_view_config

__all__ = [
    # Core functionality
    'resolve_dataset_path',
    'ResolvedDataset',
    # Filter logic
    'apply_filters',
    'summarize_filtered_metadata',
    # Validation
    'validate_view_config',
    # Schemas
    'ViewConfig',
    'ViewMetadata',
    'SourceConfig',
    'FilterConfig',
]

__version__ = '1.0.0'

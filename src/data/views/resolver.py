"""Dataset path resolution for transparent view support.

This module provides the core abstraction that makes views transparent to
downstream code. Both physical datasets and views resolve to a unified
ResolvedDataset interface.
"""
import os
import logging
import yaml
from dataclasses import dataclass
from typing import Optional, Callable
import pandas as pd

from utils.zarr_helpers import read_zarr_metadata
from data.views.filters import apply_filters
from data.views.validator import validate_view_config

logger = logging.getLogger(__name__)


@dataclass
class ResolvedDataset:
    """
    Unified representation of physical dataset or view.

    This abstraction makes views transparent to BaseFOGDataset and all
    downstream code. Both physical datasets and views provide:
    - Path to physical Zarr store
    - Filtered metadata DataFrame
    - Index translation function

    Attributes:
        zarr_path: Path to physical Zarr store (even for views)
        metadata_df: Filtered metadata DataFrame
        index_translator: Function mapping dataset_idx -> global_idx in Zarr
        is_view: Whether this is a view (True) or physical dataset (False)
        source_path: Path to view config file (None for physical datasets)
        view_name: Name of view (None for physical datasets)
    """
    zarr_path: str
    metadata_df: pd.DataFrame
    index_translator: Callable[[int], int]
    is_view: bool
    source_path: Optional[str] = None
    view_name: Optional[str] = None


def resolve_dataset_path(path: str) -> ResolvedDataset:
    """
    Resolve a path to either a physical dataset or view.

    This is the main entry point for dataset resolution. It automatically
    detects whether the path points to a physical .zarr dataset or a
    view .yaml config and returns a unified ResolvedDataset interface.

    Args:
        path: Path to .zarr directory or .yaml view config

    Returns:
        ResolvedDataset with unified interface

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If path format is invalid, view source not found,
                   or view produces empty result

    Example:
        # Physical dataset
        resolved = resolve_dataset_path("data/processed/kaggle.zarr")

        # View
        resolved = resolve_dataset_path("data/processed/views/defog_only.yaml")

        # Both provide same interface
        metadata = resolved.metadata_df
        zarr_path = resolved.zarr_path
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset path not found: {path}. "
            "Please ensure the dataset or view config exists."
        )

    # Detection logic based on file extension
    if path.endswith('.yaml') or path.endswith('.yml'):
        logger.debug(f"Detected view config: {path}")
        return _resolve_view(path)
    elif path.endswith('.zarr') or os.path.isdir(path):
        logger.debug(f"Detected physical dataset: {path}")
        return _resolve_physical(path)
    else:
        raise ValueError(
            f"Cannot resolve path: {path}. "
            "Path must be a .zarr dataset or .yaml view config."
        )


def _resolve_physical(zarr_path: str) -> ResolvedDataset:
    """
    Load physical Zarr dataset as ResolvedDataset.

    Args:
        zarr_path: Path to physical Zarr dataset

    Returns:
        ResolvedDataset for physical dataset

    Raises:
        FileNotFoundError: If dataset doesn't exist
    """
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Physical dataset not found: {zarr_path}")

    # Load metadata
    logger.debug(f"Loading metadata from physical dataset: {zarr_path}")
    metadata_df = read_zarr_metadata(zarr_path, '/metadata')

    logger.info(
        f"Resolved physical dataset: {zarr_path} "
        f"({len(metadata_df)} patches)"
    )

    return ResolvedDataset(
        zarr_path=zarr_path,
        metadata_df=metadata_df,
        index_translator=lambda idx: metadata_df.index[idx],
        is_view=False,
        source_path=None,
        view_name=None
    )


def _resolve_view(view_config_path: str) -> ResolvedDataset:
    """
    Load view config and resolve to ResolvedDataset.

    Reads view YAML config, validates it, resolves source dataset path,
    loads source metadata, applies filters, and returns unified interface.

    Args:
        view_config_path: Path to view .yaml config file

    Returns:
        ResolvedDataset for view

    Raises:
        FileNotFoundError: If view config or source dataset doesn't exist
        ValueError: If view config invalid or produces empty result
    """
    if not os.path.exists(view_config_path):
        raise FileNotFoundError(f"View config not found: {view_config_path}")

    # Load view config
    logger.debug(f"Loading view config: {view_config_path}")
    with open(view_config_path, 'r') as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse view config YAML: {e}")

    # Validate config
    view_config = validate_view_config(config_dict, config_path=view_config_path)

    # Resolve source dataset path (handle relative paths)
    source_path = view_config.source.dataset_path

    if not os.path.isabs(source_path):
        # Resolve relative to view config directory
        config_dir = os.path.dirname(os.path.abspath(view_config_path))
        source_path = os.path.abspath(os.path.join(config_dir, source_path))
        logger.debug(f"Resolved relative source path: {source_path}")

    # Validate source is physical dataset
    if not source_path.endswith('.zarr'):
        # Could be directory without .zarr extension - check for .zgroup
        zgroup_path = os.path.join(source_path, '.zgroup')
        if not os.path.exists(zgroup_path):
            raise ValueError(
                f"View source must be a Zarr dataset. Got: {source_path}. "
                f"Expected .zarr extension or directory with .zgroup file."
            )

    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"View '{view_config.view_metadata.name}' references missing source dataset: {source_path}. "
            f"View config: {view_config_path}. "
            "Please ensure the source dataset exists."
        )

    # Load source metadata
    logger.debug(f"Loading metadata from source dataset: {source_path}")
    full_metadata_df = read_zarr_metadata(source_path, '/metadata')
    logger.debug(f"Source dataset has {len(full_metadata_df)} patches")

    # Apply filters
    logger.debug(f"Applying filters for view '{view_config.view_metadata.name}'")
    filters_dict = view_config.filters.model_dump()
    filtered_metadata_df = apply_filters(full_metadata_df, filters_dict)

    # Validate non-empty result
    if len(filtered_metadata_df) == 0:
        raise ValueError(
            f"View '{view_config.view_metadata.name}' produced empty result after filtering. "
            f"Source dataset has {len(full_metadata_df)} patches, but filters matched none. "
            f"Check filter criteria in {view_config_path}"
        )

    logger.info(
        f"Resolved view '{view_config.view_metadata.name}': "
        f"{len(filtered_metadata_df)}/{len(full_metadata_df)} patches "
        f"({100.0 * len(filtered_metadata_df) / len(full_metadata_df):.1f}% of source) "
        f"from {source_path}"
    )

    return ResolvedDataset(
        zarr_path=source_path,  # Points to physical dataset
        metadata_df=filtered_metadata_df,  # Filtered subset
        index_translator=lambda idx: filtered_metadata_df.index[idx],
        is_view=True,
        source_path=view_config_path,
        view_name=view_config.view_metadata.name
    )

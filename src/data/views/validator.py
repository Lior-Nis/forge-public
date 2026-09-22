"""View configuration validation utilities."""
import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from data.views.schemas import ViewConfig

logger = logging.getLogger(__name__)


def validate_view_config(config: Dict[str, Any], config_path: Optional[str] = None) -> ViewConfig:
    """
    Validate view configuration against schema and business rules.

    Args:
        config: View configuration dictionary (from YAML)
        config_path: Optional path to config file (for resolving relative paths)

    Returns:
        Validated ViewConfig object

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If source dataset doesn't exist
    """
    # Validate schema using Pydantic
    try:
        view_config = ViewConfig(**config)
    except Exception as e:
        raise ValueError(f"View config validation failed: {e}")

    # Resolve source dataset path
    source_path = view_config.source.dataset_path

    # If path is relative and config_path provided, resolve relative to config
    if not os.path.isabs(source_path) and config_path is not None:
        config_dir = os.path.dirname(os.path.abspath(config_path))
        source_path = os.path.abspath(os.path.join(config_dir, source_path))

    # Check source dataset exists
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"View source dataset not found: {source_path}. "
            f"View: '{view_config.view_metadata.name}'. "
            "Ensure the source dataset exists before creating the view."
        )

    # Check source is a Zarr dataset (directory or .zarr)
    if not source_path.endswith('.zarr'):
        # Could be directory without .zarr extension - check for .zgroup
        zgroup_path = os.path.join(source_path, '.zgroup')
        if not os.path.exists(zgroup_path):
            raise ValueError(
                f"Source path '{source_path}' does not appear to be a Zarr dataset. "
                "Expected .zarr extension or directory with .zgroup file."
            )

    # Validate no nested views (already checked in schema, but double-check)
    if source_path.endswith('.yaml') or source_path.endswith('.yml'):
        raise ValueError(
            f"Views cannot reference other views. "
            f"Source must be a physical Zarr dataset, got: {source_path}"
        )

    logger.debug(f"View config validation passed: {view_config.view_metadata.name}")

    return view_config


def validate_filters_against_metadata(filters: Dict[str, Any], metadata_columns: List[str]) -> None:
    """
    Validate that filter fields reference existing metadata columns.

    Args:
        filters: Filter configuration
        metadata_columns: Available metadata column names

    Raises:
        ValueError: If filter references non-existent column
    """
    # Map filter fields to required metadata columns
    filter_column_map = {
        'protocol': 'protocol',
        'class_label': 'class_label',
        'pure_patches_only': 'purity',
        'patient_ids': 'patient_id',
    }

    for filter_field, filter_value in filters.items():
        if filter_value is None:
            continue  # Skip null filters

        if filter_field == 'custom_query':
            continue  # Can't pre-validate custom queries

        # Check required column exists
        required_column = filter_column_map.get(filter_field)
        if required_column and required_column not in metadata_columns:
            raise ValueError(
                f"Filter '{filter_field}' requires metadata column '{required_column}', "
                f"but it's not present. Available columns: {metadata_columns}"
            )

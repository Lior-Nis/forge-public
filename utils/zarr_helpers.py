"""Zarr helper utilities for reading datasets."""

import logging
import os
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import zarr

logger = logging.getLogger(__name__)


def read_zarr_metadata(zarr_path: str, group_path: str = '/metadata') -> pd.DataFrame:
    """
    Read metadata DataFrame from Zarr store.

    Replacement for pd.read_hdf(path, '/metadata/patches').
    """
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    try:
        root = zarr.open(zarr_path, mode='r')
    except Exception as e:
        raise ValueError(f"Failed to open Zarr store at {zarr_path}: {e}")

    group_path = group_path.strip('/')
    if group_path not in root:
        raise KeyError(f"Metadata group '{group_path}' not found in Zarr store")

    metadata_group = root[group_path]

    if 'columns' not in metadata_group.attrs:
        raise ValueError("Metadata group missing 'columns' attribute")

    columns = metadata_group.attrs['columns']

    # Decode integer-encoded string columns (written by append_metadata_batch)
    encodings = metadata_group.attrs.get("encodings", {})
    reverse_encodings = {
        col: {int(v): k for k, v in enc.items()}
        for col, enc in encodings.items()
    }

    # Reconstruct DataFrame from column arrays
    data = {}
    for col in columns:
        if col not in metadata_group:
            raise KeyError(f"Column '{col}' not found in metadata group")
        values = metadata_group[col][:]
        if col in reverse_encodings:
            dec = reverse_encodings[col]
            values = np.array([dec[int(v)] for v in values])
        data[col] = values

    # Handle index if present
    if 'global_idx' in metadata_group:
        index = metadata_group['global_idx'][:]
        df = pd.DataFrame(data, index=index)
        df.index.name = 'global_idx'
    else:
        df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("category")

    logger.debug(f"Loaded metadata: {len(df)} records, {len(df.columns)} columns")
    return df


def open_zarr_store(zarr_path: str, mode: str = 'r') -> zarr.Group:
    """Open Zarr store with error handling."""
    if mode == 'r' and not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    try:
        return zarr.open(zarr_path, mode=mode)
    except Exception as e:
        raise ValueError(f"Failed to open Zarr store at {zarr_path}: {e}")


def read_zarr_stats(zarr_path: str, level: str, entity_id: str) -> Dict[str, Any]:
    """Read statistics for a specific session, patient, or protocol."""
    valid_levels = ['sessions', 'patients', 'protocols']
    if level not in valid_levels:
        raise ValueError(f"Invalid level '{level}'. Must be one of {valid_levels}")

    root = open_zarr_store(zarr_path, mode='r')

    if 'stats' not in root:
        raise KeyError(f"Stats group not found in Zarr store")

    stats_group = root['stats']
    if level not in stats_group:
        raise KeyError(f"Stats level '{level}' not found")

    level_group = stats_group[level]
    if str(entity_id) not in level_group:
        raise KeyError(f"Entity '{entity_id}' not found in {level} stats")

    entity_group = level_group[str(entity_id)]

    # Load arrays and attributes
    stats = {
        'mean': entity_group['mean'][:],
        'std': entity_group['std'][:],
        'median': entity_group['median'][:],
        'mad': entity_group['mad'][:]
    }
    stats.update(entity_group.attrs.asdict())

    return stats


def detect_dataset_format(path: str) -> str:
    """Detect if dataset is Zarr or HDF5 format. Returns 'zarr', 'hdf5', or 'unknown'."""
    if not os.path.exists(path):
        return 'unknown'

    # Check for Zarr directory store
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, '.zgroup')) or \
           os.path.exists(os.path.join(path, '.zarray')):
            return 'zarr'

    # Check for HDF5 file
    if os.path.isfile(path) and (path.endswith('.h5') or path.endswith('.hdf5')):
        return 'hdf5'

    return 'unknown'


def validate_zarr_structure(zarr_path: str, is_unlabeled: bool = False) -> None:
    """Validate that Zarr store has the required structure."""
    root = open_zarr_store(zarr_path, mode='r')

    # Check required arrays
    if 'accs' not in root:
        raise KeyError(f"Missing required array 'accs'")

    if 'metadata' not in root:
        raise KeyError(f"Missing required group 'metadata'")

    # Verify shapes are consistent
    n_patches = root['accs'].shape[0]
    metadata_df = read_zarr_metadata(zarr_path, '/metadata')

    if len(metadata_df) != n_patches:
        raise ValueError(
            f"Metadata count mismatch: {len(metadata_df)} records "
            f"vs {n_patches} patches in 'accs' array"
        )

    # Check label arrays for labeled datasets
    if not is_unlabeled:
        required_label_arrays = ['labels', 'patch_labels', 'valid_masks']
        for arr_name in required_label_arrays:
            if arr_name not in root:
                logger.warning(f"Array '{arr_name}' not found - treating as unlabeled dataset")

    if 'stats' not in root:
        logger.warning("No stats group found - normalization may not work")

    logger.debug(f"Zarr structure validated: {n_patches} patches")

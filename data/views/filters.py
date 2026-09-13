"""Filter application logic for view subsetting."""
import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


def apply_filters(metadata_df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply filter criteria to metadata DataFrame.

    Filters are applied sequentially with AND logic (all filters must pass).
    None/null filter values are skipped.

    Args:
        metadata_df: Full metadata DataFrame from physical dataset
        filters: Filter dictionary from view config

    Returns:
        Filtered metadata DataFrame (copy)

    Raises:
        ValueError: If filter references non-existent column or custom query fails
    """
    result = metadata_df.copy()
    original_count = len(result)

    # Protocol filter
    if filters.get('protocol') is not None:
        protocols = filters['protocol']
        # Handle single value or list
        if not isinstance(protocols, list):
            protocols = [protocols]

        if 'protocol' not in result.columns:
            raise ValueError(
                f"Filter 'protocol' requires column 'protocol', "
                f"but it's not in metadata. Available: {list(result.columns)}"
            )

        result = result[result['protocol'].isin(protocols)]
        logger.debug(f"Protocol filter: {len(result)}/{original_count} patches (protocols={protocols})")

    # Class label filter
    if filters.get('class_label') is not None:
        labels = filters['class_label']
        # Handle single value or list
        if not isinstance(labels, list):
            labels = [labels]

        if 'class_label' not in result.columns:
            raise ValueError(
                f"Filter 'class_label' requires column 'class_label', "
                f"but it's not in metadata. Available: {list(result.columns)}"
            )

        result = result[result['class_label'].isin(labels)]
        logger.debug(f"Class label filter: {len(result)}/{original_count} patches (labels={labels})")

    # Purity filter (pure patches only)
    if filters.get('pure_patches_only'):
        if 'purity' not in result.columns:
            raise ValueError(
                f"Filter 'pure_patches_only' requires column 'purity', "
                f"but it's not in metadata. Available: {list(result.columns)}"
            )

        result = result[result['purity'] == 1.0]
        logger.debug(f"Purity filter: {len(result)}/{original_count} patches (pure only)")

    # Patient ID filter
    if filters.get('patient_ids') is not None:
        patient_ids = filters['patient_ids']

        if 'patient_id' not in result.columns:
            raise ValueError(
                f"Filter 'patient_ids' requires column 'patient_id', "
                f"but it's not in metadata. Available: {list(result.columns)}"
            )

        result = result[result['patient_id'].isin(patient_ids)]
        logger.debug(f"Patient ID filter: {len(result)}/{original_count} patches ({len(patient_ids)} patients)")

    # Custom pandas query (advanced)
    if filters.get('custom_query'):
        query_str = filters['custom_query']
        try:
            result = result.query(query_str)
            logger.debug(f"Custom query filter: {len(result)}/{original_count} patches (query='{query_str}')")
        except Exception as e:
            raise ValueError(
                f"Custom query failed: '{query_str}'. "
                f"Error: {e}. "
                f"Available columns: {list(result.columns)}"
            )

    logger.info(
        f"Applied filters: {len(result)}/{original_count} patches "
        f"({100.0 * len(result) / original_count if original_count > 0 else 0:.1f}% retained)"
    )

    return result


def summarize_filtered_metadata(metadata_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary statistics for filtered metadata.

    Used to populate 'computed' section of view config.

    Args:
        metadata_df: Filtered metadata DataFrame

    Returns:
        Dictionary with computed statistics
    """
    stats = {
        'total_patches': len(metadata_df),
    }

    # Patient count
    if 'patient_id' in metadata_df.columns:
        stats['patient_count'] = metadata_df['patient_id'].nunique()

    # Session count
    if 'session_id' in metadata_df.columns:
        stats['session_count'] = metadata_df['session_id'].nunique()

    # Class distribution
    if 'class_label' in metadata_df.columns:
        class_dist = metadata_df['class_label'].value_counts().to_dict()
        # Convert keys to int for JSON serialization
        stats['class_distribution'] = {int(k): int(v) for k, v in class_dist.items()}

    # Protocol distribution
    if 'protocol' in metadata_df.columns:
        protocol_dist = metadata_df['protocol'].value_counts().to_dict()
        stats['protocol_distribution'] = {str(k): int(v) for k, v in protocol_dist.items()}

    # Purity statistics (if available)
    if 'purity' in metadata_df.columns:
        stats['purity_stats'] = {
            'mean': float(metadata_df['purity'].mean()),
            'pure_count': int((metadata_df['purity'] == 1.0).sum()),
        }

    return stats

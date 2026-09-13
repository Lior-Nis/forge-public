"""
Generate patient-level stratified splits from Zarr datasets.

This script supports:
- Random splits (no stratification)
- Stratified splits by metadata column (protocol, session count, etc.)
- K-fold cross-validation with stratification
- Flexible filtering (exclude patients, include specific protocols)

Usage Examples:

1. Random split:
    python scripts/generate_splits.py \\
        --zarr-path data/processed/len200_stride100_kaggle.zarr \\
        --output configs/data/splits/my_dataset/random.yaml \\
        --ratios 0.7 0.15 0.15 \\
        --seed 42

2. Protocol-stratified split:
    python scripts/generate_splits.py \\
        --zarr-path data/processed/len200_stride100_kaggle.zarr \\
        --output configs/data/splits/my_dataset/protocol_stratified.yaml \\
        --stratify-by protocol \\
        --ratios 0.7 0.15 0.15 \\
        --seed 42

3. K-fold cross-validation:
    python scripts/generate_splits.py \\
        --zarr-path data/processed/len200_stride100_kaggle.zarr \\
        --output configs/data/splits/my_dataset/fold \\
        --mode kfold \\
        --n-folds 3 \\
        --stratify-by protocol \\
        --seed 42
"""

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    StratifiedKFold,
    KFold,
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.zarr_helpers import read_zarr_metadata
from data.datamodule.config import SplitsConfig
from pydantic import ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ========== Exceptions ==========

class SplitGenerationError(Exception):
    """Base exception for split generation."""
    pass


class InsufficientPatientsError(SplitGenerationError):
    """Not enough patients for requested split."""
    pass


class InvalidStratificationError(SplitGenerationError):
    """Stratification column invalid or missing."""
    pass


# ========== Patient Metadata Extraction ==========

def load_patient_metadata(zarr_path: str) -> pd.DataFrame:
    """
    Aggregate patch-level metadata to patient level.

    Returns DataFrame with:
    - patient_id (index)
    - protocol (most common protocol for this patient)
    - n_sessions (unique sessions)
    - n_patches (total patches)
    - fog_ratio (% patches with FOG, if labeled)
    - mean_purity (average purity, if labeled)
    - protocols_list (all protocols this patient participated in)

    Args:
        zarr_path: Path to Zarr dataset

    Returns:
        Patient-level metadata DataFrame
    """
    logger.info(f"Loading metadata from {zarr_path}")

    try:
        patch_df = read_zarr_metadata(zarr_path, '/metadata')
    except Exception as e:
        raise SplitGenerationError(f"Failed to load metadata: {e}")

    logger.info(f"Loaded {len(patch_df)} patches")

    # Check for required columns
    required_cols = ['patient_id', 'session_id', 'protocol']
    missing_cols = [col for col in required_cols if col not in patch_df.columns]
    if missing_cols:
        raise SplitGenerationError(f"Missing required columns: {missing_cols}")

    # Check for optional label columns
    has_labels = 'class_label' in patch_df.columns
    has_purity = 'purity' in patch_df.columns

    if not has_labels:
        logger.info("Dataset appears to be unlabeled (no class_label column)")

    # Aggregate to patient level
    grouped = patch_df.groupby('patient_id')

    patient_df = pd.DataFrame({
        'n_sessions': grouped['session_id'].nunique(),
        'n_patches': grouped.size(),
        'protocol': grouped['protocol'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]),
    })

    if has_labels:
        patient_df['fog_ratio'] = grouped['class_label'].agg(lambda x: (x > 0).mean())

    if has_purity:
        patient_df['mean_purity'] = grouped['purity'].mean()

    # Add protocols_list (all protocols this patient participated in)
    protocols_list = patch_df.groupby('patient_id')['protocol'].apply(
        lambda x: sorted(x.unique())
    )
    patient_df['protocols_list'] = protocols_list

    logger.info(f"Aggregated to {len(patient_df)} patients")
    logger.info(f"Columns: {list(patient_df.columns)}")

    return patient_df


# ========== Preprocessing for Stratification ==========

def preprocess_metadata(patient_df: pd.DataFrame, stratify_by: str) -> pd.DataFrame:
    """
    Prepare stratification column, handling special cases.

    Special preprocessing:
    - 'session_bin': Bin n_sessions into low/medium/high quantiles
    - 'has_fog': Create binary column from fog_ratio (threshold=0.1)
    - 'fog_protocol': Composite of protocol + fog_ratio bin (none/low/medium/high)
    - Others: Use column directly

    Args:
        patient_df: Patient metadata DataFrame
        stratify_by: Column to stratify by

    Returns:
        DataFrame with preprocessed stratification column
    """
    patient_df = patient_df.copy()

    if stratify_by == 'session_bin':
        logger.info("Binning session counts into quantiles")
        try:
            patient_df['session_bin'] = pd.qcut(
                patient_df['n_sessions'],
                q=3,
                labels=['low', 'medium', 'high'],
                duplicates='drop'
            )
        except ValueError as e:
            logger.warning(f"Failed to create 3 quantile bins: {e}. Trying 2 bins.")
            try:
                patient_df['session_bin'] = pd.qcut(
                    patient_df['n_sessions'],
                    q=2,
                    labels=['low', 'high'],
                    duplicates='drop'
                )
            except ValueError:
                raise InvalidStratificationError(
                    "Cannot create session bins - insufficient variation in session counts"
                )

        logger.info(f"Session bin distribution:\n{patient_df['session_bin'].value_counts()}")

    elif stratify_by == 'has_fog':
        if 'fog_ratio' not in patient_df.columns:
            raise InvalidStratificationError(
                "--stratify-by has_fog requires labeled dataset with fog_ratio column"
            )
        patient_df['has_fog'] = (patient_df['fog_ratio'] > 0.1).astype(str)
        logger.info(f"FOG presence distribution:\n{patient_df['has_fog'].value_counts()}")

    elif stratify_by == 'fog_protocol':
        if 'fog_ratio' not in patient_df.columns:
            raise InvalidStratificationError(
                "--stratify-by fog_protocol requires labeled dataset with fog_ratio column"
            )
        # Bin fog_ratio into categories
        fog_bins = pd.cut(
            patient_df['fog_ratio'],
            bins=[-0.001, 0.01, 0.1, 0.4, 1.0],
            labels=['none', 'low', 'medium', 'high']
        )
        # Composite: protocol + fog_bin
        patient_df['fog_protocol'] = patient_df['protocol'].astype(str) + '_' + fog_bins.astype(str)

        # Merge small groups (< 3 patients) into nearest bin to avoid stratification failure
        value_counts = patient_df['fog_protocol'].value_counts()
        small_groups = value_counts[value_counts < 3].index
        if len(small_groups) > 0:
            logger.info(f"Merging {len(small_groups)} small groups: {list(small_groups)}")
            # Fall back to protocol-only for patients in small composite groups
            for group in small_groups:
                mask = patient_df['fog_protocol'] == group
                protocol = group.split('_')[0]
                patient_df.loc[mask, 'fog_protocol'] = protocol + '_merged'

        logger.info(f"FOG-protocol composite distribution:\n{patient_df['fog_protocol'].value_counts()}")

    else:
        # Use column directly
        if stratify_by not in patient_df.columns:
            available = list(patient_df.columns)
            raise InvalidStratificationError(
                f"Column '{stratify_by}' not found in metadata. Available: {available}"
            )
        logger.info(f"Using column '{stratify_by}' for stratification")
        logger.info(f"Distribution:\n{patient_df[stratify_by].value_counts()}")

    return patient_df


# ========== Split Generation Strategies ==========

def generate_random_split(
    patient_ids: np.ndarray,
    ratios: List[float],
    seed: int
) -> Dict[str, List[str]]:
    """
    Generate random split without stratification.

    Args:
        patient_ids: Array of patient IDs
        ratios: [train_ratio, val_ratio, test_ratio]
        seed: Random seed

    Returns:
        Dict with 'train', 'val', 'test' patient lists
    """
    train_ratio, val_ratio, test_ratio = ratios

    # First split: train vs (val+test)
    train_ids, temp_ids = train_test_split(
        patient_ids,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True
    )

    # Second split: val vs test
    # Adjust val_ratio relative to temp size
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_ratio_adjusted,
        random_state=seed,
        shuffle=True
    )

    return {
        'train': sorted(train_ids.tolist()),
        'val': sorted(val_ids.tolist()),
        'test': sorted(test_ids.tolist())
    }


def generate_stratified_split(
    patient_df: pd.DataFrame,
    stratify_by: str,
    ratios: List[float],
    seed: int
) -> Dict[str, List[str]]:
    """
    Generate stratified split by metadata column.

    Args:
        patient_df: Patient metadata DataFrame
        stratify_by: Column to stratify by
        ratios: [train_ratio, val_ratio, test_ratio]
        seed: Random seed

    Returns:
        Dict with 'train', 'val', 'test' patient lists
    """
    train_ratio, val_ratio, test_ratio = ratios

    patient_ids = patient_df.index.values
    stratify_labels = patient_df[stratify_by].values

    # Check for single class (can't stratify)
    unique_labels = np.unique(stratify_labels)
    if len(unique_labels) == 1:
        logger.warning(
            f"Only one unique value in '{stratify_by}': {unique_labels[0]}. "
            "Falling back to random split."
        )
        return generate_random_split(patient_ids, ratios, seed)

    # Check for insufficient samples per class
    label_counts = Counter(stratify_labels)
    min_count = min(label_counts.values())
    if min_count < 2:
        logger.warning(
            f"Some classes have only {min_count} sample(s). "
            "Falling back to random split to avoid stratification failure."
        )
        return generate_random_split(patient_ids, ratios, seed)

    try:
        # First split: train vs (val+test)
        sss = StratifiedShuffleSplit(
            n_splits=1,
            train_size=train_ratio,
            random_state=seed
        )
        train_idx, temp_idx = next(sss.split(patient_ids, stratify_labels))
        train_ids = patient_ids[train_idx]
        temp_ids = patient_ids[temp_idx]
        temp_labels = stratify_labels[temp_idx]

        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        sss2 = StratifiedShuffleSplit(
            n_splits=1,
            train_size=val_ratio_adjusted,
            random_state=seed
        )
        val_idx, test_idx = next(sss2.split(temp_ids, temp_labels))
        val_ids = temp_ids[val_idx]
        test_ids = temp_ids[test_idx]

    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}. Falling back to random split.")
        return generate_random_split(patient_ids, ratios, seed)

    return {
        'train': sorted(train_ids.tolist()),
        'val': sorted(val_ids.tolist()),
        'test': sorted(test_ids.tolist())
    }


def generate_kfold_splits(
    patient_df: pd.DataFrame,
    stratify_by: Optional[str],
    n_folds: int,
    seed: int
) -> List[Dict[str, List[str]]]:
    """
    Generate K-fold cross-validation splits.

    For each fold i:
    - test = fold i patients
    - val = fold (i+1) % n_folds patients
    - train = remaining patients

    Args:
        patient_df: Patient metadata DataFrame
        stratify_by: Column to stratify by (None = random)
        n_folds: Number of folds
        seed: Random seed

    Returns:
        List of split dicts (one per fold)
    """
    patient_ids = patient_df.index.values

    if len(patient_ids) < n_folds:
        raise InsufficientPatientsError(
            f"Need at least {n_folds} patients for {n_folds}-fold CV, found {len(patient_ids)}"
        )

    # Create folder
    if stratify_by is None:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_indices = list(kf.split(patient_ids))
    else:
        stratify_labels = patient_df[stratify_by].values

        # Check if stratification is possible
        unique_labels = np.unique(stratify_labels)
        if len(unique_labels) == 1:
            logger.warning(
                f"Only one unique value in '{stratify_by}'. "
                "Using random K-fold instead of stratified."
            )
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            fold_indices = list(kf.split(patient_ids))
        else:
            label_counts = Counter(stratify_labels)
            min_count = min(label_counts.values())
            if min_count < n_folds:
                logger.warning(
                    f"Some classes have fewer samples ({min_count}) than folds ({n_folds}). "
                    "Using random K-fold instead of stratified."
                )
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                fold_indices = list(kf.split(patient_ids))
            else:
                try:
                    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                    fold_indices = list(skf.split(patient_ids, stratify_labels))
                except ValueError as e:
                    logger.warning(f"Stratified K-fold failed: {e}. Using random K-fold.")
                    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                    fold_indices = list(kf.split(patient_ids))

    # Create splits for each fold
    splits = []
    for i in range(n_folds):
        _, test_idx = fold_indices[i]
        _, val_idx = fold_indices[(i + 1) % n_folds]

        # Train = all patients not in test or val
        test_set = set(patient_ids[test_idx])
        val_set = set(patient_ids[val_idx])
        train_set = set(patient_ids) - test_set - val_set

        splits.append({
            'train': sorted(list(train_set)),
            'val': sorted(list(val_set)),
            'test': sorted(list(test_set))
        })

    return splits


# ========== Filtering ==========

def filter_patients(
    patient_df: pd.DataFrame,
    exclude_patients: List[str],
    include_protocols: Optional[List[str]]
) -> pd.DataFrame:
    """
    Filter patient DataFrame by exclusions and protocol inclusions.

    Args:
        patient_df: Patient metadata DataFrame
        exclude_patients: Patient IDs to exclude
        include_protocols: Only include these protocols (None = all)

    Returns:
        Filtered DataFrame
    """
    original_count = len(patient_df)

    # Apply protocol filter first
    if include_protocols is not None:
        logger.info(f"Filtering to protocols: {include_protocols}")
        patient_df = patient_df[patient_df['protocol'].isin(include_protocols)]
        logger.info(f"After protocol filter: {len(patient_df)} patients")

    # Apply patient exclusions
    if exclude_patients:
        logger.info(f"Excluding {len(exclude_patients)} patients")
        patient_df = patient_df[~patient_df.index.isin(exclude_patients)]
        logger.info(f"After exclusions: {len(patient_df)} patients")

    filtered_count = len(patient_df)
    if filtered_count == 0:
        raise InsufficientPatientsError("No patients remaining after filtering")

    if filtered_count < original_count:
        logger.info(f"Filtered: {original_count} → {filtered_count} patients")

    return patient_df


# ========== Validation ==========

def validate_split(
    split_dict: Dict[str, List[str]],
    patient_df: pd.DataFrame
) -> None:
    """
    Comprehensive validation (errors on critical issues, warnings on imbalances).

    ERRORS (raise ValueError):
    - Patient overlap between train/val/test
    - Patients in split not in dataset
    - Empty splits
    - Pydantic SplitsConfig validation failure

    WARNINGS (log only):
    - Protocol distribution imbalance (>20% difference)
    - Missing patients from dataset (not in any split)
    - Very small split sizes (<5 patients)

    Args:
        split_dict: Split dictionary
        patient_df: Patient metadata DataFrame
    """
    train_ids = set(split_dict['train'])
    val_ids = set(split_dict['val'])
    test_ids = set(split_dict['test'])
    all_patients = set(patient_df.index)

    # Check for empty splits
    if not train_ids:
        raise ValueError("Train split is empty")
    if not val_ids:
        raise ValueError("Val split is empty")
    if not test_ids:
        raise ValueError("Test split is empty")

    # Check for overlap
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    if train_val_overlap:
        raise ValueError(f"Train/val overlap: {sorted(train_val_overlap)}")
    if train_test_overlap:
        raise ValueError(f"Train/test overlap: {sorted(train_test_overlap)}")
    if val_test_overlap:
        raise ValueError(f"Val/test overlap: {sorted(val_test_overlap)}")

    # Check patients exist in dataset
    split_patients = train_ids | val_ids | test_ids
    unknown_patients = split_patients - all_patients
    if unknown_patients:
        raise ValueError(f"Patients in split not in dataset: {sorted(unknown_patients)}")

    # Check for missing patients (warning only)
    missing_patients = all_patients - split_patients
    if missing_patients:
        logger.warning(
            f"{len(missing_patients)} patients from dataset not in any split: "
            f"{sorted(list(missing_patients))[:10]}..."
        )

    # Check for very small splits (warning only)
    for split_name, split_ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        if len(split_ids) < 5:
            logger.warning(f"{split_name} split has only {len(split_ids)} patients")

    # Validate with Pydantic schema
    try:
        SplitsConfig(**split_dict)
    except ValidationError as e:
        raise ValueError(f"Split failed Pydantic validation: {e}")

    # Check protocol distribution balance (warning only)
    if 'protocol' in patient_df.columns:
        train_protocols = patient_df.loc[list(train_ids), 'protocol'].value_counts(normalize=True)
        val_protocols = patient_df.loc[list(val_ids), 'protocol'].value_counts(normalize=True)
        test_protocols = patient_df.loc[list(test_ids), 'protocol'].value_counts(normalize=True)

        all_protocols = set(train_protocols.index) | set(val_protocols.index) | set(test_protocols.index)
        max_diff = 0
        for protocol in all_protocols:
            train_pct = train_protocols.get(protocol, 0)
            val_pct = val_protocols.get(protocol, 0)
            test_pct = test_protocols.get(protocol, 0)

            max_pct = max(train_pct, val_pct, test_pct)
            min_pct = min(train_pct, val_pct, test_pct)
            diff = max_pct - min_pct
            max_diff = max(max_diff, diff)

        if max_diff > 0.2:
            logger.warning(
                f"Protocol distribution imbalance detected (max difference: {max_diff:.1%})"
            )
            logger.warning(f"Train:\n{train_protocols}")
            logger.warning(f"Val:\n{val_protocols}")
            logger.warning(f"Test:\n{test_protocols}")

    logger.info("✓ Split validation passed")


# ========== YAML Generation ==========

def compute_split_statistics(
    split_dict: Dict[str, List[str]],
    patient_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute statistics for YAML header.

    Args:
        split_dict: Split dictionary
        patient_df: Patient metadata DataFrame

    Returns:
        Dict of statistics
    """
    stats = {
        'total_patients': len(patient_df),
        'train_count': len(split_dict['train']),
        'val_count': len(split_dict['val']),
        'test_count': len(split_dict['test']),
    }

    # Protocol distribution
    if 'protocol' in patient_df.columns:
        for split_name in ['train', 'val', 'test']:
            split_ids = split_dict[split_name]
            protocols = patient_df.loc[split_ids, 'protocol'].value_counts()
            stats[f'{split_name}_protocols'] = protocols.to_dict()

    return stats


def format_protocol_distribution(protocol_dict: Dict[str, int]) -> str:
    """Format protocol distribution for YAML header."""
    items = [f"{protocol}: {count}" for protocol, count in sorted(protocol_dict.items())]
    return ", ".join(items)


def write_split_yaml(
    split_dict: Dict[str, List[str]],
    output_path: str,
    metadata: Dict[str, Any],
    patient_df: pd.DataFrame,
    force: bool = False
) -> None:
    """
    Write split to YAML with informative header.

    Args:
        split_dict: Split dictionary
        output_path: Output file path
        metadata: Metadata for header
        patient_df: Patient metadata DataFrame
        force: Overwrite without prompting
    """
    # Check if file exists
    if os.path.exists(output_path) and not force:
        response = input(f"File {output_path} already exists. Overwrite? [y/N] ")
        if response.lower() != 'y':
            logger.info("Aborted")
            return

    # Compute statistics
    stats = compute_split_statistics(split_dict, patient_df)

    # Build header
    header_lines = []
    header_lines.append(f"# {metadata.get('description', 'Patient-level split')}")
    header_lines.append(f"# Dataset: {metadata['zarr_path']}")
    header_lines.append(f"# Total patients: {stats['total_patients']}")
    header_lines.append(
        f"# Split: {stats['train_count']} train / "
        f"{stats['val_count']} val / "
        f"{stats['test_count']} test"
    )

    if metadata.get('stratify_by'):
        header_lines.append(f"# Stratification: {metadata['stratify_by']}")
    else:
        header_lines.append("# Stratification: None (random)")

    header_lines.append(f"# Random seed: {metadata['seed']}")

    # Protocol distribution
    if 'train_protocols' in stats:
        header_lines.append("# Protocol distribution:")
        for split_name in ['train', 'val', 'test']:
            dist = stats[f'{split_name}_protocols']
            header_lines.append(f"#   {split_name}: {format_protocol_distribution(dist)}")

    header = "\n".join(header_lines)

    # Build YAML content (sorted patient IDs with quotes)
    yaml_dict = {
        'train': sorted(split_dict['train']),
        'val': sorted(split_dict['val']),
        'test': sorted(split_dict['test'])
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write file with all strings quoted to avoid YAML parsing issues
    # (patient IDs like "203e85" look like scientific notation)
    with open(output_path, 'w') as f:
        f.write(header + "\n\n")
        yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False, default_style='"')

    logger.info(f"✓ Split written to {output_path}")


# ========== Main Generation Logic ==========

def generate_splits(args):
    """Main split generation logic."""

    # Load patient metadata
    try:
        patient_df = load_patient_metadata(args.zarr_path)
    except SplitGenerationError as e:
        logger.error(str(e))
        sys.exit(1)

    # Apply filters
    try:
        patient_df = filter_patients(
            patient_df,
            args.exclude_patients,
            args.include_protocols
        )
    except InsufficientPatientsError as e:
        logger.error(str(e))
        sys.exit(1)

    # Preprocess for stratification if needed
    if args.stratify_by is not None:
        try:
            patient_df = preprocess_metadata(patient_df, args.stratify_by)
        except InvalidStratificationError as e:
            logger.error(str(e))
            sys.exit(1)

    # Generate splits based on mode
    if args.mode == 'split':
        logger.info(f"Generating single split with ratios {args.ratios}")

        if args.stratify_by is None:
            split_dict = generate_random_split(
                patient_df.index.values,
                args.ratios,
                args.seed
            )
        else:
            split_dict = generate_stratified_split(
                patient_df,
                args.stratify_by,
                args.ratios,
                args.seed
            )

        # Validate
        try:
            validate_split(split_dict, patient_df)
        except ValueError as e:
            logger.error(f"Validation failed: {e}")
            sys.exit(1)

        # Display statistics
        logger.info(f"Train: {len(split_dict['train'])} patients")
        logger.info(f"Val: {len(split_dict['val'])} patients")
        logger.info(f"Test: {len(split_dict['test'])} patients")

        if args.verbose:
            stats = compute_split_statistics(split_dict, patient_df)
            if 'train_protocols' in stats:
                for split_name in ['train', 'val', 'test']:
                    logger.info(f"{split_name} protocols: {stats[f'{split_name}_protocols']}")

        # Write YAML
        if not args.dry_run:
            metadata = {
                'zarr_path': args.zarr_path,
                'stratify_by': args.stratify_by,
                'seed': args.seed,
                'description': args.description or (
                    f"{'Stratified' if args.stratify_by else 'Random'} split "
                    f"({args.ratios[0]:.0%}/{args.ratios[1]:.0%}/{args.ratios[2]:.0%})"
                )
            }
            write_split_yaml(split_dict, args.output, metadata, patient_df, args.force)
        else:
            logger.info("Dry run - no files written")

    elif args.mode == 'kfold':
        logger.info(f"Generating {args.n_folds}-fold cross-validation splits")

        try:
            splits = generate_kfold_splits(
                patient_df,
                args.stratify_by,
                args.n_folds,
                args.seed
            )
        except InsufficientPatientsError as e:
            logger.error(str(e))
            sys.exit(1)

        # Validate all splits
        for i, split_dict in enumerate(splits):
            try:
                validate_split(split_dict, patient_df)
            except ValueError as e:
                logger.error(f"Fold {i} validation failed: {e}")
                sys.exit(1)

        # Display statistics
        for i, split_dict in enumerate(splits):
            logger.info(
                f"Fold {i}: "
                f"{len(split_dict['train'])} train / "
                f"{len(split_dict['val'])} val / "
                f"{len(split_dict['test'])} test"
            )

        # Write YAMLs
        if not args.dry_run:
            # Determine output prefix (remove .yaml if present)
            output_prefix = args.output.replace('.yaml', '')

            for i, split_dict in enumerate(splits):
                output_path = f"{output_prefix}{i}.yaml"
                metadata = {
                    'zarr_path': args.zarr_path,
                    'stratify_by': args.stratify_by,
                    'seed': args.seed,
                    'description': args.description or (
                        f"{args.n_folds}-fold CV (fold {i})"
                        f"{' stratified by ' + args.stratify_by if args.stratify_by else ''}"
                    )
                }
                write_split_yaml(split_dict, output_path, metadata, patient_df, args.force)
        else:
            logger.info("Dry run - no files written")


# ========== CLI ==========

USAGE_EXAMPLES = """
Examples:

1. Random split:
    python scripts/generate_splits.py \\
        --zarr-path data/processed/len200_stride100_kaggle.zarr \\
        --output configs/data/splits/my_dataset/random.yaml \\
        --ratios 0.7 0.15 0.15 \\
        --seed 42

2. Protocol-stratified split:
    python scripts/generate_splits.py \\
        --zarr-path data/processed/len200_stride100_kaggle.zarr \\
        --output configs/data/splits/my_dataset/protocol_stratified.yaml \\
        --stratify-by protocol \\
        --ratios 0.7 0.15 0.15 \\
        --seed 42

3. K-fold cross-validation:
    python scripts/generate_splits.py \\
        --zarr-path data/processed/len200_stride100_kaggle.zarr \\
        --output configs/data/splits/my_dataset/fold \\
        --mode kfold \\
        --n-folds 3 \\
        --stratify-by protocol \\
        --seed 42

4. Session-count stratified:
    python scripts/generate_splits.py \\
        --zarr-path data/processed/len200_stride100_kaggle.zarr \\
        --output configs/data/splits/my_dataset/session_balanced.yaml \\
        --stratify-by session_bin \\
        --ratios 0.7 0.15 0.15 \\
        --seed 42
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate patient-level stratified splits from Zarr dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE_EXAMPLES
    )

    # === REQUIRED ===
    parser.add_argument(
        '--zarr-path',
        type=str,
        required=True,
        help='Path to Zarr dataset (e.g., data/processed/len200_stride100_kaggle.zarr)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output YAML path or prefix for k-fold (e.g., configs/data/splits/my_dataset/split.yaml)'
    )

    # === SPLIT MODE ===
    parser.add_argument(
        '--mode',
        type=str,
        choices=['split', 'kfold'],
        default='split',
        help='Generation mode: split (single) or kfold (multiple)'
    )

    # === SPLIT-SPECIFIC ===
    parser.add_argument(
        '--ratios',
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=('TRAIN', 'VAL', 'TEST'),
        help='Train/val/test ratios (must sum to 1.0, default: 0.7 0.15 0.15)'
    )

    # === KFOLD-SPECIFIC ===
    parser.add_argument(
        '--n-folds',
        type=int,
        default=3,
        help='Number of folds for k-fold CV (default: 3)'
    )

    # === STRATIFICATION ===
    parser.add_argument(
        '--stratify-by',
        type=str,
        default=None,
        help='Metadata column to stratify by (e.g., protocol, session_bin, has_fog). None = random.'
    )

    # === REPRODUCIBILITY ===
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    # === FILTERS ===
    parser.add_argument(
        '--exclude-patients',
        type=str,
        nargs='*',
        default=[],
        help='Patient IDs to exclude from splits'
    )
    parser.add_argument(
        '--include-protocols',
        type=str,
        nargs='*',
        default=None,
        help='Only include these protocols (e.g., defog tdcsfog)'
    )

    # === OUTPUT OPTIONS ===
    parser.add_argument(
        '--description',
        type=str,
        default=None,
        help='Description for YAML header (auto-generated if not provided)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate splits but do not write files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed statistics'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing files without prompting'
    )

    args = parser.parse_args()

    # Validate args
    if args.mode == 'split' and not np.isclose(sum(args.ratios), 1.0):
        parser.error(f"--ratios must sum to 1.0, got {sum(args.ratios)}")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Generate
    generate_splits(args)


if __name__ == "__main__":
    main()

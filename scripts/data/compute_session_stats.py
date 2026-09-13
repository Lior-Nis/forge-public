"""
Compute and save per-session normalization statistics (mean/std).

This script iterates through a Zarr dataset, calculates the per-channel mean and
standard deviation for the accelerometer data of each session, and saves these
statistics to a PyTorch file.

Usage:
    python scripts/compute_session_stats.py --input-file /path/to/dataset.zarr --output-file session_stats.pt
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import zarr
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.zarr_helpers import read_zarr_metadata


def compute_stats(input_file: str, output_file: str):
    """
    Computes per-session mean and std from a Zarr file and saves them.

    Args:
        input_file: Path to the input Zarr dataset
        output_file: Path to save the output .pt file
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    session_stats = {}

    # Open Zarr store
    root = zarr.open(input_file, mode='r')

    # Load patch metadata to get session_id for each patch
    try:
        metadata_df = read_zarr_metadata(input_file, '/metadata')
        print(f"Loaded metadata with {len(metadata_df)} patches")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # Get all accelerometer data
    all_accs = root['accs'][:]
    print(f"Loaded accelerometer data shape: {all_accs.shape}")

    # Group patches by session_id
    session_groups = metadata_df.groupby('session_id')
    print(f"Found {len(session_groups)} unique sessions. Computing statistics...")

    for session_id, group_df in tqdm(session_groups, desc="Processing sessions"):
        # Get indices for this session's patches
        patch_indices = group_df['global_idx'].values

        # Collect all data for this session
        session_data = []
        for idx in patch_indices:
            patch_data = all_accs[idx]  # shape: (time, channels)
            session_data.append(patch_data)

        # Concatenate all patches for this session
        session_data = np.concatenate(session_data, axis=0)  # (total_time, channels)

        # Calculate mean and std per channel
        mean = np.mean(session_data, axis=0)  # shape: (3,)
        std = np.std(session_data, axis=0)    # shape: (3,)

        # Store as torch tensors
        session_stats[session_id] = {
            'mean': torch.from_numpy(mean).float(),
            'std': torch.from_numpy(std).float()
        }

    # Save the computed statistics
    torch.save(session_stats, output_file)
    print(f"\nSuccessfully computed statistics for {len(session_stats)} sessions.")
    print(f"Saved statistics to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-session normalization statistics.")
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Path to the input Zarr dataset'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='session_stats.pt',
        help='Path to save the output statistics file'
    )

    args = parser.parse_args()
    compute_stats(args.input_file, args.output_file)

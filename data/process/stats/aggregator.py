"""Statistics aggregation for normalization across patient splits."""

import logging
from typing import Dict, List
import numpy as np
import torch
import zarr

logger = logging.getLogger(__name__)


class StatsAggregator:
    """
    Aggregate normalization statistics from Zarr storage.

    This class handles the computation of dataset-wide normalization statistics
    by combining per-patient statistics stored in Zarr format. It uses the
    combining formula for means and variances to aggregate across patients
    without loading all data into memory.
    """

    def __init__(self, zarr_path: str):
        """
        Initialize aggregator with path to processed Zarr dataset.

        Args:
            zarr_path: Path to Zarr dataset containing stats/patients group
        """
        self.zarr_path = zarr_path

    def compute_split_stats(self, patient_ids: List[str]) -> Dict[str, torch.Tensor]:
        """
        Aggregate normalization stats from specified patients.

        Uses the combining formula for means and variances:
        - combined_mean = Σ(mean_i * n_i) / Σ(n_i)
        - combined_var = Σ((var_i + mean_i²) * n_i) / Σ(n_i) - combined_mean²

        This prevents data leakage by only using statistics from the specified
        patient subset (e.g., training patients only).

        Args:
            patient_ids: List of patient IDs to include (e.g., ['bf608b', 'bae0ce'])

        Returns:
            Dictionary with 'mean' and 'std' tensors for normalization:
            {
                'mean': torch.Tensor of shape (3,),  # Per-channel means
                'std': torch.Tensor of shape (3,)    # Per-channel stds
            }

        Raises:
            FileNotFoundError: If zarr_path doesn't exist
            KeyError: If patient_id not found in stats
        """
        root = zarr.open(self.zarr_path, mode='r')
        patients_group = root['stats/patients']

        total_n = 0
        channel_sums = np.zeros(3, dtype=np.float64)
        channel_sq_sums = np.zeros(3, dtype=np.float64)

        for patient_id in patient_ids:
            if patient_id not in patients_group:
                logger.warning(f"Patient {patient_id} not found in stats, skipping")
                continue

            patient_group = patients_group[patient_id]
            patient_mean = patient_group['mean'][:]
            patient_std = patient_group['std'][:]
            patient_n = patient_group.attrs['n_samples']

            # Aggregate using combining formula
            channel_sums += patient_mean * patient_n
            channel_sq_sums += (patient_std**2 + patient_mean**2) * patient_n
            total_n += patient_n

        if total_n == 0:
            raise ValueError(f"No valid patients found in {patient_ids}")

        # Compute final aggregated stats
        mean = channel_sums / total_n
        variance = (channel_sq_sums / total_n) - (mean ** 2)
        std = np.sqrt(np.maximum(variance, 1e-8))  # Prevent negative variance

        logger.debug(f"Aggregated stats from {len(patient_ids)} patients ({total_n} total samples)")

        return {
            'mean': torch.from_numpy(mean).float(),
            'std': torch.from_numpy(std).float()
        }

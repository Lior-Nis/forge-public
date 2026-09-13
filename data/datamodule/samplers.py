"""
Probabilistic balanced sampler.

Replaces the complex hierarchical system with elegant probabilistic sampling:
- Every sample gets a weight based on class/protocol/patient frequency
- Sampling is just np.random.choice() with probability weights
- Cache is tiny (just the weight vector)
"""

import logging
import os
import pickle
import time
from typing import Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler

from data.dataset.protocols import SamplableDataset

logger = logging.getLogger(__name__)


class ProbabilisticBalancedSampler(Sampler):
    CACHE_VERSION = 4
    """
    Probabilistic sampler that balances across hierarchical dimensions.

    Much simpler than the previous hierarchical approach:
    1. Compute weight for each sample based on class/protocol/patient frequency
    2. Sample probabilistically using these weights
    3. Cache only the lightweight weight vector

    Args:
        dataset: Dataset implementing SamplableDataset protocol
        batch_size: Number of samples per batch
        weights_cache_path: Path to cache computed weights
        balance_config: Configuration for balancing importance
    """

    def __init__(
        self,
        dataset: SamplableDataset,
        batch_size: int,
        weights_cache_path: Optional[str] = None,
        balance_config: Optional[dict] = None
    ):
        # Validate dataset implements protocol
        if not isinstance(dataset, SamplableDataset):
            raise TypeError(
                f"Dataset must implement SamplableDataset protocol. "
                f"Got {type(dataset).__name__} which does not provide "
                f"get_sampling_metadata() method."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.weights_cache_path = weights_cache_path

        # Default balance configuration (hierarchical toggles)
        self.balance_config = self._canonicalize_balance_config(balance_config)

        # Load or compute weights
        self.weights = self._load_or_compute_weights()

        logger.info(f"ProbabilisticBalancedSampler initialized with {len(self.weights)} samples")
        logger.info(f"Weight stats: min={self.weights.min():.6f}, max={self.weights.max():.6f}, "
                   f"mean={self.weights.mean():.6f}")

    def _load_or_compute_weights(self) -> np.ndarray:
        """Load weights from cache or compute them."""
        # Try to load from cache
        if self.weights_cache_path and os.path.exists(self.weights_cache_path):
            try:
                weights = self._load_weights_cache()
                if weights is not None:
                    logger.info(f"Loaded weights from cache: {self.weights_cache_path}")
                    return weights
            except Exception as e:
                logger.warning(f"Failed to load weights cache: {e}")

        # Compute weights
        logger.info("Computing balanced weights for all samples...")
        weights = self._compute_balanced_weights()

        # Save to cache
        if self.weights_cache_path:
            try:
                self._save_weights_cache(weights)
                logger.info(f"Saved weights to cache: {self.weights_cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save weights cache: {e}")

        return weights

    def _compute_balanced_weights(self) -> np.ndarray:
        """Compute hierarchical balance weights in single pass."""
        # Extract metadata for all samples
        classes, protocols, patients = self._extract_sample_metadata()

        # Start with uniform weights
        weights = np.ones(len(self.dataset), dtype=np.float32)

        # Convert to int64 for indexing
        classes = classes.astype(np.int64, copy=False)

        # --- Level 1: Class balancing ---
        class_counts = np.bincount(classes).astype(np.float64)
        num_classes = max(class_counts.size, 1)
        total_samples = float(len(classes))

        # Ensure no zero division
        class_counts_safe = np.maximum(class_counts, 1.0)
        target_per_class = total_samples / num_classes
        class_factors = target_per_class / class_counts_safe
        weights *= class_factors[classes]

        # --- Level 2: Protocol balancing within each class ---
        balance_protocols = self.balance_config.get('balance_protocols', True)
        unique_protocols, protocol_indices = np.unique(protocols, return_inverse=True)
        num_protocols = max(unique_protocols.size, 1)
        combo_counts = None

        if balance_protocols and num_protocols > 1:
            class_protocol_indices = np.ravel_multi_index(
                (classes, protocol_indices),
                (num_classes, num_protocols)
            )
            combo_counts = np.bincount(
                class_protocol_indices,
                minlength=num_classes * num_protocols
            ).reshape(num_classes, num_protocols).astype(np.float64)

            protocols_per_class = np.maximum((combo_counts > 0).sum(axis=1), 1.0)
            cp_counts = combo_counts[classes, protocol_indices]
            target_per_cp = class_counts_safe[classes] / protocols_per_class[classes]
            cp_counts_safe = np.maximum(cp_counts, 1.0)
            weights *= target_per_cp / cp_counts_safe
        else:
            class_protocol_indices = None

        # --- Level 3: Patient balancing within each class-protocol pair ---
        balance_patients = self.balance_config.get('balance_patients', True)
        unique_patients, patient_indices = np.unique(patients, return_inverse=True)
        num_patients = max(unique_patients.size, 1)

        if balance_patients and num_patients > 1:
            if combo_counts is None:
                class_protocol_indices = np.ravel_multi_index(
                    (classes, protocol_indices),
                    (num_classes, num_protocols)
                )
                combo_counts = np.bincount(
                    class_protocol_indices,
                    minlength=num_classes * num_protocols
                ).reshape(num_classes, num_protocols).astype(np.float64)

            class_protocol_patient_indices = np.ravel_multi_index(
                (classes, protocol_indices, patient_indices),
                (num_classes, num_protocols, num_patients)
            )
            cpp_counts = np.bincount(
                class_protocol_patient_indices,
                minlength=num_classes * num_protocols * num_patients
            ).reshape(num_classes, num_protocols, num_patients).astype(np.float64)

            cp_counts = np.maximum(combo_counts, 1.0)
            patients_per_cp = np.maximum((cpp_counts > 0).sum(axis=2), 1.0)
            target_per_cpp = cp_counts / patients_per_cp
            cpp_counts_sample = cpp_counts[classes, protocol_indices, patient_indices]
            cpp_counts_safe = np.maximum(cpp_counts_sample, 1.0)
            weights *= target_per_cpp[classes, protocol_indices] / cpp_counts_safe

        # Normalize to probabilities
        weights = weights / weights.sum()

        # Log balance statistics
        self._log_balance_stats(classes, protocols, patients, weights)

        return weights

    def _extract_sample_metadata(self):
        """Extract class, protocol, patient for each sample using protocol method."""
        metadata_df = self.dataset.get_sampling_metadata()
        logger.info("Using dataset sampling metadata for initialization")
        return self._extract_from_metadata_df(metadata_df)

    def _extract_from_metadata_df(self, metadata_df):
        """Extract metadata from DataFrame (now passed as parameter, not accessed directly)."""
        # Validate required columns
        required_columns = {'protocol', 'patient_id'}
        missing_columns = required_columns - set(metadata_df.columns)
        if missing_columns:
            raise ValueError(
                f"Sampling metadata missing required columns: {missing_columns}. "
                f"Available columns: {list(metadata_df.columns)}"
            )

        # Extract protocols directly from DataFrame
        protocols = metadata_df['protocol'].values

        # Extract patient IDs directly from DataFrame
        patients = metadata_df['patient_id'].values

        # Extract class labels (required)
        if 'class_label' not in metadata_df.columns:
            raise ValueError(
                "Sampling metadata must contain 'class_label' column. "
                "Please regenerate your dataset with the updated processing pipeline."
            )
        classes = metadata_df['class_label'].values.astype(np.int64, copy=False)

        logger.info(f"Extracted metadata for {len(metadata_df)} samples")
        logger.info(f"Protocols: {np.unique(protocols, return_counts=True)}")
        logger.info(f"Patients: {len(np.unique(patients))} unique patients")
        logger.info(f"Classes: {np.unique(classes, return_counts=True)}")

        return classes, protocols, patients


    def _log_balance_stats(self, classes, protocols, patients, weights):
        """Log balancing statistics."""
        logger.info("Balance statistics:")

        # Class distribution
        unique_classes, class_counts = np.unique(classes, return_counts=True)
        logger.info(f"Classes: {dict(zip(unique_classes, class_counts))}")

        # Protocol distribution
        unique_protocols, protocol_counts = np.unique(protocols, return_counts=True)
        logger.info(f"Protocols: {dict(zip(unique_protocols, protocol_counts))}")

        # Patient distribution
        unique_patients, patient_counts = np.unique(patients, return_counts=True)
        logger.info(f"Patients: {len(unique_patients)} total")

        # Weight distribution by class
        for class_id in unique_classes:
            class_mask = classes == class_id
            class_weight_mean = weights[class_mask].mean()
            logger.info(f"Class {class_id} average weight: {class_weight_mean:.6f}")

    def _load_weights_cache(self) -> Optional[np.ndarray]:
        """Load weights from cache with validation."""
        with open(self.weights_cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        # Validate cache
        if cache_data.get('dataset_size') != len(self.dataset):
            logger.info("Cache invalid: dataset size mismatch")
            return None

        if cache_data.get('balance_config') != self.balance_config:
            logger.info("Cache invalid: balance config mismatch")
            return None

        if cache_data.get('sampler_version', 1) != self.CACHE_VERSION:
            logger.info("Cache invalid: sampler version mismatch")
            return None


        return cache_data['weights']

    def _save_weights_cache(self, weights: np.ndarray):
        """Save weights to cache."""
        cache_data = {
            'weights': weights,
            'dataset_size': len(self.dataset),
            'balance_config': self.balance_config,
            'sampler_version': self.CACHE_VERSION,
            'timestamp': time.time()
        }

        os.makedirs(os.path.dirname(self.weights_cache_path), exist_ok=True)
        with open(self.weights_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches using probabilistic sampling."""
        while True:  # Infinite generator - training loop controls stopping
            indices = np.random.choice(
                len(self.weights),
                size=self.batch_size,
                p=self.weights,
                replace=True  # Bootstrap sampling
            )
            yield indices.tolist()

    def __len__(self) -> int:
        """Return large number for training loop control."""
        return len(self.dataset) // self.batch_size
    
    @staticmethod
    def _canonicalize_balance_config(balance_config: Optional[dict]) -> dict:
        """Normalize balance configuration to hierarchical toggle format."""
        if balance_config is None:
            return {
                'balance_protocols': True,
                'balance_patients': True,
            }

        if any(
            key in balance_config
            for key in ('balance_protocols', 'balance_patients')
        ):
            return {
                'balance_protocols': balance_config.get('balance_protocols', True),
                'balance_patients': balance_config.get('balance_patients', True),
            }

        # Backward compatibility for legacy importance-based configs
        return {
            'balance_protocols': balance_config.get('protocol_importance', 0.0) > 0,
            'balance_patients': balance_config.get('patient_importance', 0.0) > 0,
        }

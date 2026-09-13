"""
Hierarchical Statistics Computer for Normalization.

This module computes normalization statistics at multiple granularities:
- Session-level: Per-session mean/std/median/MAD
- Patient-level: Aggregated across all sessions for a patient
- Protocol-level: Aggregated across all patients in a protocol

Statistics are computed using online algorithms for memory efficiency
and stored in Zarr dataset for automatic loading during training.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd

from data.process.schemas import SessionStats

logger = logging.getLogger(__name__)


class OnlineStatsComputer:
    """
    Compute statistics online using Welford's algorithm.

    This allows computing mean and variance in a single pass without
    storing all data in memory. Particularly useful for large datasets.

    Algorithm:
        Welford's online algorithm for numerically stable variance computation.

    Attributes:
        num_channels: Number of channels (e.g., 3 for AccV, AccML, AccAP)
        n: Number of samples processed
        mean: Running mean per channel
        M2: Sum of squared differences from mean (for variance)
    """

    def __init__(self, num_channels: int = 3) -> None:
        self.num_channels: int = num_channels
        self.n: int = 0
        self.mean: npt.NDArray[np.float64] = np.zeros(num_channels, dtype=np.float64)
        self.M2: npt.NDArray[np.float64] = np.zeros(num_channels, dtype=np.float64)

        # For median/MAD, use numpy array for reservoir sampling (10x faster)
        self.max_samples_for_median: int = 10000
        self.samples: npt.NDArray[np.float32] = np.zeros((self.max_samples_for_median, num_channels), dtype=np.float32)
        self.samples_count: int = 0  # Track how many samples stored

    def update(self, data: npt.NDArray[np.floating[Any]]) -> None:
        """
        Update statistics with new data using vectorized Welford algorithm.

        Args:
            data: Array of shape (seq_len, num_channels) or (num_channels,)
        """
        # Ensure 2D array
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Vectorized Welford algorithm for numerical stability
        n_new = len(data)
        n_old = self.n
        n_total = n_old + n_new

        if n_old == 0:
            # First batch
            self.mean = np.mean(data, axis=0)
            self.M2 = np.sum((data - self.mean) ** 2, axis=0)
            self.n = n_new
        else:
            # Combine old and new statistics
            mean_new = np.mean(data, axis=0)
            delta = mean_new - self.mean

            # Update mean
            self.mean = self.mean + delta * n_new / n_total

            # Update M2 (sum of squared differences)
            M2_new = np.sum((data - mean_new) ** 2, axis=0)
            self.M2 = self.M2 + M2_new + delta ** 2 * n_old * n_new / n_total
            self.n = n_total

        # Vectorized reservoir sampling (10x faster than loop with .tolist())
        for sample in data:
            if self.samples_count < self.max_samples_for_median:
                # Fill phase: add samples until reservoir is full
                self.samples[self.samples_count] = sample
                self.samples_count += 1
            else:
                # Reservoir phase: randomly replace with correct probability
                # Use Algorithm R: replace with probability k/n
                j = np.random.randint(0, self.n)
                if j < self.max_samples_for_median:
                    self.samples[j] = sample

    def get_mean(self) -> npt.NDArray[np.float32]:
        """Get current mean."""
        return self.mean.astype(np.float32)

    def get_std(self, ddof: int = 0) -> npt.NDArray[np.float32]:
        """Get current standard deviation."""
        if self.n < 2:
            return np.zeros(self.num_channels, dtype=np.float32)
        variance = self.M2 / (self.n - ddof)
        return np.sqrt(variance).astype(np.float32)

    def get_median(self) -> npt.NDArray[np.float32]:
        """Get median from stored samples."""
        if self.samples_count == 0:
            return np.zeros(self.num_channels, dtype=np.float32)
        # Only use filled portion of reservoir
        valid_samples = self.samples[:self.samples_count]
        return np.median(valid_samples, axis=0).astype(np.float32)

    def get_mad(self) -> npt.NDArray[np.float32]:
        """Get Median Absolute Deviation from stored samples."""
        if self.samples_count == 0:
            return np.zeros(self.num_channels, dtype=np.float32)
        # Only use filled portion of reservoir
        valid_samples = self.samples[:self.samples_count]
        median = np.median(valid_samples, axis=0)
        mad = np.median(np.abs(valid_samples - median), axis=0)
        return mad.astype(np.float32)

    def get_stats(self) -> Dict[str, npt.NDArray[np.float32] | int | List[List[float]]]:
        """Get all statistics including reservoir samples for pooling."""
        # Convert samples to list for JSON serialization and compatibility
        valid_samples = self.samples[:self.samples_count]
        return {
            'mean': self.get_mean(),
            'std': self.get_std(),
            'median': self.get_median(),
            'mad': self.get_mad(),
            'n_samples': self.n,
            'samples': valid_samples.tolist()  # Convert to list for storage
        }


class HierarchicalStatsComputer:
    """
    Compute and aggregate statistics at session, patient, and protocol levels.

    This class manages the hierarchical computation of normalization statistics:
    1. Session-level: Raw statistics from each session
    2. Patient-level: Aggregated from all sessions of a patient
    3. Protocol-level: Aggregated from all patients in a protocol

    Session-to-patient and session-to-protocol mappings are resolved lazily
    from the MetadataProvider (loaded from the standardized metadata CSVs).

    Usage:
        >>> from data.processors.metadata_provider import MetadataProvider
        >>> provider = MetadataProvider("data/raw/kaggle_labeled")
        >>> computer = HierarchicalStatsComputer(metadata_provider=provider, num_channels=3)
        >>> computer.add_session('session_01', acc_data)
        >>> computer.add_session('session_02', acc_data)
        >>> stats = computer.aggregate()
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        num_channels: int = 3,
        max_samples_per_session: int = 100_000,
    ) -> None:
        """
        Initialize HierarchicalStatsComputer.

        Args:
            metadata_provider: Provider that resolves session_id -> patient_id/protocol
                               from the standardized metadata CSVs.
            num_channels: Number of channels (e.g., 3 for AccV, AccML, AccAP)
            max_samples_per_session: Maximum samples to use per session for stats computation.
                                     If None, uses all samples. Recommended: 100000 for large datasets.
        """
        self.metadata_df = metadata_df
        self.num_channels: int = num_channels
        self.max_samples_per_session: int = max_samples_per_session

        # Session-level storage: session_id -> stats_dict
        self.session_stats: Dict[str, Dict[str, Any]] = {}

        if max_samples_per_session:
            logger.info(f"Initialized HierarchicalStatsComputer with {num_channels} channels "
                       f"(subsampling to {max_samples_per_session:,} samples per session)")
        else:
            logger.info(f"Initialized HierarchicalStatsComputer with {num_channels} channels")

    def add_session(self, session_id: str, data: npt.NDArray[np.floating[Any]]) -> None:
        """
        Add a session's data and compute its statistics.

        Patient and protocol mappings are resolved lazily from the MetadataProvider
        during aggregation, using the standardized metadata CSVs.

        Args:
            session_id: Unique session identifier
            data: Accelerometer data, shape (n_patches, seq_len, num_channels)
                  or (seq_len, num_channels)
        """
        # Flatten data to (n_samples, num_channels)
        if data.ndim == 3:
            # (n_patches, seq_len, num_channels) -> (n_patches * seq_len, num_channels)
            data_flat = data.reshape(-1, self.num_channels)
        elif data.ndim == 2:
            data_flat = data
        else:
            raise ValueError(f"Data must be 2D or 3D, got shape {data.shape}")

        # Apply stratified subsampling if configured
        if self.max_samples_per_session and len(data_flat) > self.max_samples_per_session:
            # Stratified sampling: sample uniformly across the session
            n_samples = len(data_flat)
            indices = np.linspace(0, n_samples - 1, self.max_samples_per_session, dtype=int)
            data_flat = data_flat[indices]
            logger.debug(f"Subsampled session {session_id} from {n_samples:,} to "
                        f"{self.max_samples_per_session:,} samples")

        # Compute session stats
        computer = OnlineStatsComputer(num_channels=self.num_channels)
        computer.update(data_flat)
        stats = computer.get_stats()

        # Store session stats (mappings resolved lazily in aggregate())
        self.session_stats[session_id] = stats

        logger.debug(f"Added session {session_id} ({len(data_flat)} samples)")


    def _aggregate_sessions(self, session_ids: List[str]) -> Dict[str, npt.NDArray[np.float32] | int]:
        """
        Aggregate statistics from multiple sessions.

        Uses combining formula for means and variances:
        mean_combined = Σ(mean_i * n_i) / Σ(n_i)
        var_combined = Σ(var_i * n_i + mean_i²  * n_i) / Σ(n_i) - mean_combined²

        For median/MAD: pools reservoir samples from all sessions and computes
        median/MAD on the pooled distribution. This is statistically correct,
        unlike taking median of medians which biases toward sessions with fewer samples.

        Args:
            session_ids: List of session IDs to aggregate

        Returns:
            Dictionary with aggregated stats
        """
        if not session_ids:
            return {
                'mean': np.zeros(self.num_channels, dtype=np.float32),
                'std': np.zeros(self.num_channels, dtype=np.float32),
                'median': np.zeros(self.num_channels, dtype=np.float32),
                'mad': np.zeros(self.num_channels, dtype=np.float32),
                'n_sessions': 0,
                'n_samples': 0
            }

        # Collect session stats
        total_n = 0
        sum_mean_weighted = np.zeros(self.num_channels, dtype=np.float64)
        sum_sq_weighted = np.zeros(self.num_channels, dtype=np.float64)

        # Incremental reservoir sampling for median/MAD (memory-efficient)
        # Pre-allocate fixed-size reservoir to avoid unbounded growth
        max_pooled = 1_000_000
        pooled_samples = np.zeros((max_pooled, self.num_channels), dtype=np.float32)
        n_pooled = 0  # Track total samples processed

        for sess_id in session_ids:
            stats = self.session_stats[sess_id]
            n = stats['n_samples']
            mean = stats['mean']
            std = stats['std']

            # Aggregate mean and variance
            sum_mean_weighted += mean * n
            sum_sq_weighted += (std ** 2 + mean ** 2) * n
            total_n += n

            # Incremental reservoir sampling from session samples
            if 'samples' in stats and stats['samples']:
                sess_samples = np.array(stats['samples'], dtype=np.float32)

                for sample in sess_samples:
                    if n_pooled < max_pooled:
                        # Fill phase: add to reservoir
                        pooled_samples[n_pooled] = sample
                        n_pooled += 1
                    else:
                        # Reservoir phase: random replacement with probability max_pooled/(n_pooled+1)
                        j = np.random.randint(0, n_pooled + 1)
                        if j < max_pooled:
                            pooled_samples[j] = sample
                        n_pooled += 1

        # Compute aggregated stats
        if total_n > 0:
            agg_mean = sum_mean_weighted / total_n
            agg_variance = (sum_sq_weighted / total_n) - (agg_mean ** 2)
            agg_std = np.sqrt(np.maximum(agg_variance, 0))  # Prevent negative due to numerical errors
        else:
            agg_mean = np.zeros(self.num_channels, dtype=np.float64)
            agg_std = np.zeros(self.num_channels, dtype=np.float64)

        # Compute median/MAD from reservoir (only filled portion)
        if n_pooled > 0:
            actual_pooled = min(n_pooled, max_pooled)
            valid_samples = pooled_samples[:actual_pooled]
            agg_median = np.median(valid_samples, axis=0)
            agg_mad = np.median(np.abs(valid_samples - agg_median), axis=0)
        else:
            agg_median = np.zeros(self.num_channels)
            agg_mad = np.zeros(self.num_channels)

        return {
            'mean': agg_mean.astype(np.float32),
            'std': agg_std.astype(np.float32),
            'median': agg_median.astype(np.float32),
            'mad': agg_mad.astype(np.float32),
            'n_sessions': len(session_ids),
            'n_samples': int(total_n)
        }

    def aggregate_from_temp_storage(
        self,
        temp_stats_writer,
        chunk_size: int = 100
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate statistics from temporary storage in streaming fashion.

        Memory-efficient alternative to aggregate_from_list() that streams
        stats from disk in chunks, enabling unlimited dataset sizes.

        Args:
            temp_stats_writer: TemporaryStatsWriter instance with session stats
            chunk_size: Number of sessions to load in memory at once (default: 100)

        Returns:
            Hierarchical stats dict with sessions/patients/protocols levels
        """
        logger.info(f"Streaming stats aggregation with chunk_size={chunk_size}")

        # Build session stats dict by streaming chunks
        self.session_stats = {}
        total_loaded = 0

        for chunk in temp_stats_writer.iter_session_stats(chunk_size=chunk_size):
            for stats in chunk:
                self.session_stats[stats.session_id] = {
                    'mean': stats.mean,
                    'std': stats.std,
                    'median': stats.median,
                    'mad': stats.mad,
                    'n_samples': stats.n_samples,
                    'samples': stats.samples,
                }
                total_loaded += 1

            # Log progress every 10 chunks
            if total_loaded % (chunk_size * 10) == 0:
                logger.debug(f"Loaded {total_loaded} session stats from temporary storage")

        logger.info(f"Loaded {total_loaded} session stats, starting hierarchical aggregation")

        # Use existing aggregate() method for hierarchical aggregation
        return self.aggregate()

    def aggregate(self) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate all statistics into hierarchical structure.

        Resolves session-to-patient and session-to-protocol mappings lazily
        from the MetadataProvider (loaded from standardized metadata CSVs).

        Returns:
            Dictionary with keys:
                - 'sessions': session_id -> stats dict
                - 'patients': patient_id -> stats dict
                - 'protocols': protocol -> stats dict
        """
        logger.info("Aggregating hierarchical statistics...")

        # Resolve mappings from MetadataProvider (loaded from metadata CSVs)

        # Build reverse mappings for aggregation
        patient_sessions: Dict[str, List[str]] = defaultdict(list)
        protocol_patients: Dict[str, set] = defaultdict(set)

        for session_id in self.session_stats:
            session_row = self.metadata_df.loc[session_id]
            patient_id = session_row["patient_id"]
            protocol = session_row["protocol"]
            patient_sessions[patient_id].append(session_id)
            protocol_patients[protocol].add(patient_id)

        # Patient-level aggregation
        patient_stats = {}
        for patient_id, session_ids in patient_sessions.items():
            agg_stats = self._aggregate_sessions(session_ids)
            # Add protocols for this patient
            protocols = self.metadata_df.loc[session_ids]["protocol"].unique().tolist()
            agg_stats['protocols'] = protocols
            patient_stats[patient_id] = agg_stats

        # Protocol-level aggregation
        protocol_stats = {}
        for protocol, patient_ids in protocol_patients.items():
            # Collect all sessions for this protocol
            protocol_session_ids = [
                sid for sid, prot in self.metadata_df["protocol"].to_dict().items()
                if prot == protocol
            ]
            agg_stats = self._aggregate_sessions(protocol_session_ids)
            agg_stats['n_patients'] = len(patient_ids)
            protocol_stats[protocol] = agg_stats

        logger.info(f"Aggregated stats: {len(self.session_stats)} sessions, "
                   f"{len(patient_stats)} patients, {len(protocol_stats)} protocols")

        return {
            'sessions': self.session_stats,
            'patients': patient_stats,
            'protocols': protocol_stats
        }

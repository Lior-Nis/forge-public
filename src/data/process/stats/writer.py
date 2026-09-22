"""Temporary stats storage for streaming aggregation.

This module provides disk-based temporary storage for session statistics,
enabling constant memory usage regardless of dataset size.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, List

import numpy as np
import zarr

from data.process.schemas import SessionStats

logger = logging.getLogger(__name__)


class TemporaryStatsWriter:
    """Write session stats to temporary storage for streaming aggregation.

    This enables processing unlimited sessions with O(1) memory usage by:
    1. Writing stats to disk immediately after computation
    2. Streaming stats back in chunks during aggregation
    3. Cleaning up temporary storage automatically
    """

    def __init__(self, temp_path: str):
        """Initialize temporary stats writer.

        Args:
            temp_path: Path for temporary Zarr store
        """
        self.temp_path = temp_path
        self.store = zarr.DirectoryStore(temp_path)
        self.root = zarr.group(store=self.store, overwrite=True)
        self.session_count = 0
        logger.debug(f"Created temporary stats storage at {temp_path}")

    def write_session_stats(self, stats: SessionStats) -> None:
        """Write a single session's stats to disk.

        Args:
            stats: SessionStats object to persist
        """
        # Create group for this session
        grp = self.root.create_group(f"session_{self.session_count}")

        # Store metadata as attributes
        grp.attrs['session_id'] = stats.session_id
        grp.attrs['n_samples'] = int(stats.n_samples)

        # Store arrays as datasets (more efficient than attrs for large arrays)
        grp.create_dataset('mean', data=stats.mean, dtype=np.float32)
        grp.create_dataset('std', data=stats.std, dtype=np.float32)
        grp.create_dataset('median', data=stats.median, dtype=np.float32)
        grp.create_dataset('mad', data=stats.mad, dtype=np.float32)

        # Store reservoir samples (convert list to numpy array for efficiency)
        if stats.samples:
            samples_array = np.array(stats.samples, dtype=np.float32)
            grp.create_dataset('samples', data=samples_array, dtype=np.float32)
        else:
            # Empty placeholder
            grp.create_dataset('samples', shape=(0, 3), dtype=np.float32)

        self.session_count += 1

        if self.session_count % 100 == 0:
            logger.debug(f"Written {self.session_count} session stats to temporary storage")

    def iter_session_stats(self, chunk_size: int = 100) -> Iterator[List[SessionStats]]:
        """Yield session stats in chunks for memory-efficient aggregation.

        Args:
            chunk_size: Number of sessions to load in memory at once (default: 100)

        Yields:
            Lists of SessionStats objects, each list containing up to chunk_size items
        """
        for i in range(0, self.session_count, chunk_size):
            chunk_stats = []

            for j in range(i, min(i + chunk_size, self.session_count)):
                grp = self.root[f"session_{j}"]

                # Reconstruct SessionStats from stored data
                stats = SessionStats(
                    session_id=grp.attrs['session_id'],
                    mean=grp['mean'][:],
                    std=grp['std'][:],
                    median=grp['median'][:],
                    mad=grp['mad'][:],
                    n_samples=int(grp.attrs['n_samples']),
                    samples=grp['samples'][:].tolist(),  # Convert back to list
                )
                chunk_stats.append(stats)

            yield chunk_stats

        logger.debug(f"Streamed {self.session_count} session stats in chunks of {chunk_size}")

    def close(self) -> None:
        """Close and cleanup temporary storage."""
        if Path(self.temp_path).exists():
            shutil.rmtree(self.temp_path)
            logger.debug(f"Cleaned up temporary stats storage at {self.temp_path}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temporary files."""
        self.close()


def create_temp_stats_writer() -> TemporaryStatsWriter:
    """Factory function to create a temporary stats writer with auto-generated path.

    Returns:
        TemporaryStatsWriter instance with temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix="acc_stats_")
    return TemporaryStatsWriter(temp_dir)

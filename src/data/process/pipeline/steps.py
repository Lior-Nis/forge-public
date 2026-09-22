"""Pipeline step definitions for data processing."""

import logging
from typing import Callable, Optional, Protocol, Tuple
import numpy as np
import pandas as pd

from data.process.schemas import (
    SessionInfo,
    ProcessedBatch,
    PatchMetadata,
    SessionStats,
)

logger = logging.getLogger(__name__)


class DataLoaderStep(Protocol):
    """Step that loads data from session info."""

    def process(self, session_info: SessionInfo) -> dict:
        """Load data from session and return blocks dict."""
        ...


class TransformationStep(Protocol):
    """Step that transforms data."""

    def process(self, data: dict) -> ProcessedBatch:
        """Transform blocks dict into ProcessedBatch."""
        ...


class FilterStep(Protocol):
    """Step that filters batches."""

    def process(self, batch: ProcessedBatch) -> ProcessedBatch:
        """Filter batch and return filtered ProcessedBatch."""
        ...


class EnrichmentStep(Protocol):
    """Step that enriches batches with metadata."""

    def process(
        self, session_info: SessionInfo, batch: ProcessedBatch
    ) -> ProcessedBatch:
        """Enrich batch with metadata and return enriched ProcessedBatch."""
        ...


class SessionFileLoader:
    """Step 1: Load raw session data and extract blocks."""

    def __init__(
        self,
        file_reader: Callable,
        block_len: int,
        stride_len: int,
        fog_stride_len: Optional[int] = None,
    ):
        """Initialize session file loader.

        Args:
            file_reader: Function to read session files (pd.read_csv or pd.read_parquet)
            block_len: Block length in samples
            stride_len: Stride length in samples
            fog_stride_len: Denser stride for FOG-containing patches (labeled data only)
        """
        self.file_reader = file_reader
        self.block_len = block_len
        self.stride_len = stride_len
        self.fog_stride_len = fog_stride_len

    def process(self, session_info: SessionInfo) -> dict:
        """Load session file and extract blocks.

        Args:
            session_info: Session information with file path

        Returns:
            Dictionary with keys "accs", "labels", "valid" containing block lists
        """
        from utils.data import get_blocks

        sess_df = self.file_reader(session_info.path)
        return get_blocks(sess_df, self.block_len, self.stride_len, self.fog_stride_len)


class StatsCollector:
    """Step 3: Collect statistics for normalization from ProcessedBatch."""

    def __init__(self, num_channels: int = 3):
        """Initialize stats collector.

        Args:
            num_channels: Number of channels (default: 3 for accelerometer)
        """
        self.num_channels = num_channels

    def process(
        self, session_info: SessionInfo, batch: ProcessedBatch
    ) -> ProcessedBatch:
        """Compute stats from batch and attach to batch.stats.

        Args:
            session_info: Session information
            batch: ProcessedBatch with acc_blocks

        Returns:
            ProcessedBatch with stats attached
        """
        from data.process.stats.computer import OnlineStatsComputer

        # Compute stats using OnlineStatsComputer
        computer = OnlineStatsComputer(num_channels=self.num_channels)

        # Flatten and optionally subsample
        data_flat = batch.acc_blocks.reshape(-1, self.num_channels)
        max_samples = 100_000
        if len(data_flat) > max_samples:
            indices = np.linspace(0, len(data_flat) - 1, max_samples, dtype=int)
            data_flat = data_flat[indices]

        computer.update(data_flat)
        stats_dict = computer.get_stats()

        # Build SessionStats
        session_stats = SessionStats(
            session_id=session_info.id,
            mean=stats_dict["mean"],
            std=stats_dict["std"],
            median=stats_dict["median"],
            mad=stats_dict["mad"],
            n_samples=stats_dict["n_samples"],
            samples=stats_dict["samples"],
        )

        # Attach stats to batch
        batch.stats = session_stats
        return batch


class BlockArrayBuilder:
    """Step 2: Convert block lists to numpy arrays and compute patch labels."""

    def __init__(
        self,
        require_labels: bool = True,
        any_fog_labeling: bool = False,
        fog_ratio_labeling: bool = False,
    ):
        """Initialize block array builder.

        Args:
            require_labels: Whether labels are required (True for labeled, False for unlabeled)
            any_fog_labeling: If True, label patches as positive when ANY frame is FOG
                              (instead of majority-vote multiclass)
            fog_ratio_labeling: If True, store continuous fog_ratio (float32) as patch_labels
        """
        self.require_labels = require_labels
        self.any_fog_labeling = any_fog_labeling
        self.fog_ratio_labeling = fog_ratio_labeling

    def process(self, blocks: dict) -> ProcessedBatch:
        """Stack blocks into ProcessedBatch with arrays.

        Args:
            blocks: Dictionary with "accs", "labels", "valid" lists

        Returns:
            ProcessedBatch with stacked numpy arrays
        """
        acc_blocks, label_blocks, valid_blocks = self._stack_blocks(
            blocks,
            require_labels=self.require_labels,
            require_valid=self.require_labels,
        )

        # Compute patch labels if we have label blocks
        patch_labels = None
        if label_blocks is not None:
            if self.fog_ratio_labeling:
                # Store continuous fog_ratio: fraction of frames with label > 0
                patch_labels = np.array(
                    [(label_block > 0).mean() for label_block in label_blocks],
                    dtype=np.float32,
                )
            elif self.any_fog_labeling:
                from utils.data import get_patch_label_any_fog
                patch_labels = np.array(
                    [get_patch_label_any_fog(label_block) for label_block in label_blocks],
                    dtype=np.int32,
                )
            else:
                from utils.data import get_patch_label
                patch_labels = np.array(
                    [get_patch_label(label_block) for label_block in label_blocks],
                    dtype=np.int32,
                )

        start_frames = blocks.get("start_frames")
        start_frames_arr = np.array(start_frames, dtype=np.int64) if start_frames else None

        return ProcessedBatch(
            acc_blocks=acc_blocks,
            label_blocks=label_blocks,
            patch_labels=patch_labels,
            valid_blocks=valid_blocks,
            start_frames=start_frames_arr,
        )

    @staticmethod
    def _stack_blocks(
        blocks_dict: dict, require_labels: bool = True, require_valid: bool = True
    ):
        """Stack block lists into numpy arrays.

        Args:
            blocks_dict: Dictionary with "accs", "labels", "valid" lists
            require_labels: Whether to require label blocks
            require_valid: Whether to require valid masks

        Returns:
            Tuple of (acc_blocks, label_blocks, valid_blocks)

        Raises:
            ValueError: If required blocks are missing
        """
        if not blocks_dict["accs"]:
            raise ValueError("No blocks to process")

        labels = blocks_dict.get("labels", [])
        valids = blocks_dict.get("valid", [])

        if require_labels and not labels:
            raise ValueError("Label blocks required but none were provided")
        if require_valid and not valids:
            raise ValueError("Valid masks required but none were provided")

        acc_blocks = np.stack(blocks_dict["accs"], axis=0, dtype=np.float32)
        label_blocks = np.stack(labels, axis=0, dtype=np.int32) if labels else None
        valid_blocks = np.stack(valids, axis=0, dtype=bool) if valids else None

        return acc_blocks, label_blocks, valid_blocks

# TODO: remove it and use it in the views
class PatchPurityFilter:
    """Step 4: Filter patches by label purity (labeled data only)."""

    def __init__(self, pure_patches_only: bool):
        """Initialize purity filter.

        Args:
            pure_patches_only: Whether to filter out impure patches
        """
        self.pure_patches_only = pure_patches_only

    def process(self, batch: ProcessedBatch) -> ProcessedBatch:
        """Filter batch by purity, keeping only pure patches.

        Caches purity values for all patches BEFORE filtering so PatchMetadataBuilder
        can reuse them without recomputation.

        Args:
            batch: ProcessedBatch to filter

        Returns:
            Filtered ProcessedBatch with purity_cache populated
        """
        if not self.pure_patches_only or batch.label_blocks is None:
            return batch  # Pass through

        from utils.data import compute_patch_metadata

        keep_indices = []
        purity_values = {}  # Cache: idx -> purity

        # Compute purity for all patches (before filtering)
        for idx, label_block in enumerate(batch.label_blocks):
            metadata = compute_patch_metadata(label_block)
            purity = metadata["purity"]
            purity_values[idx] = purity

            if purity == 1.0:
                keep_indices.append(idx)

        if not keep_indices:
            # Return empty batch (preserve stats from original batch)
            logger.debug("All patches filtered out due to purity requirement")
            return ProcessedBatch(
                acc_blocks=np.array([]).reshape(0, *batch.acc_blocks.shape[1:]),
                label_blocks=None,
                patch_labels=None,
                valid_blocks=None,
                metadata=None,
                stats=batch.stats,  # Preserve stats for aggregation
                purity_cache={},  # Empty cache
            )

        # Build purity cache for surviving patches (reindex to new positions)
        filtered_purity_cache = {
            new_idx: purity_values[old_idx]
            for new_idx, old_idx in enumerate(keep_indices)
        }

        # Filter arrays and attach purity cache
        return ProcessedBatch(
            acc_blocks=batch.acc_blocks[keep_indices],
            label_blocks=batch.label_blocks[keep_indices]
            if batch.label_blocks is not None
            else None,
            patch_labels=batch.patch_labels[keep_indices]
            if batch.patch_labels is not None
            else None,
            valid_blocks=batch.valid_blocks[keep_indices]
            if batch.valid_blocks is not None
            else None,
            metadata=None,  # Will be built by PatchMetadataBuilder
            stats=batch.stats,  # Preserve stats for aggregation
            purity_cache=filtered_purity_cache,  # Cache purity values for metadata builder
        )


class PatchMetadataBuilder:
    """Step 5: Build complete metadata for each patch."""

    def __init__(self, metadata_df: pd.DataFrame, any_fog_labeling: bool = False):
        """Initialize metadata builder.

        Args:
            metadata_df: DataFrame indexed by session_id with patient_id and protocol columns
            any_fog_labeling: If True, class_label in metadata uses any-fog logic (consistent
                              with BlockArrayBuilder when any_fog_labeling=True)
        """
        self.metadata_df = metadata_df
        self.any_fog_labeling = any_fog_labeling
        self.current_global_idx = 0  # Track global patch index across all sessions

    def process(
        self, session_info: SessionInfo, batch: ProcessedBatch
    ) -> ProcessedBatch:
        """Build complete metadata for each patch with correct global indices.

        Args:
            session_info: Session information
            batch: ProcessedBatch to enrich

        Returns:
            ProcessedBatch with complete metadata
        """
        session_row = self.metadata_df.loc[session_info.id]
        patient_id = session_row["patient_id"]
        protocol = session_info.protocol

        metadata_list = []
        for session_idx in range(len(batch.acc_blocks)):
            # Compute purity and class_label if label blocks exist
            class_label = None
            purity = None
            validity = None

            if batch.label_blocks is not None:
                # Check if purity is already cached (from PatchPurityFilter)
                if batch.purity_cache is not None and session_idx in batch.purity_cache:
                    # Use cached purity value - no recomputation needed!
                    purity = batch.purity_cache[session_idx]

                    # Still need class_label and validity, compute them
                    from utils.data import compute_patch_metadata

                    patch_meta = compute_patch_metadata(
                        batch.label_blocks[session_idx],
                        batch.valid_blocks[session_idx]
                        if batch.valid_blocks is not None
                        else None,
                    )
                    class_label = (
                        int(patch_meta["fog_percentage"] > 0)
                        if self.any_fog_labeling
                        else patch_meta.get("class_label")
                    )
                    validity = patch_meta.get("validity", 1.0)
                else:
                    # No cache available, compute purity, class_label, and validity
                    from utils.data import compute_patch_metadata

                    patch_meta = compute_patch_metadata(
                        batch.label_blocks[session_idx],
                        batch.valid_blocks[session_idx]
                        if batch.valid_blocks is not None
                        else None,
                    )
                    class_label = (
                        int(patch_meta["fog_percentage"] > 0)
                        if self.any_fog_labeling
                        else patch_meta.get("class_label")
                    )
                    purity = patch_meta.get("purity")
                    validity = patch_meta.get("validity", 1.0)

            # Create complete metadata with proper global indices
            start_frame = (
                int(batch.start_frames[session_idx])
                if batch.start_frames is not None
                else session_idx  # fallback: wrong for fog-biased but safe for fixed-stride
            )
            metadata = PatchMetadata(
                # patch_idx=self.current_global_idx,
                global_idx=self.current_global_idx,
                session_id=session_info.id,
                patient_id=patient_id,
                protocol=protocol,
                session_idx=session_idx,
                start_frame=start_frame,
                class_label=class_label,
                purity=purity,
                validity=validity,
            )
            metadata_list.append(metadata)
            self.current_global_idx += 1  # Increment for next patch

        batch.metadata = metadata_list
        return batch

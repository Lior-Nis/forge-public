"""Pipeline execution orchestrator."""

import logging
from typing import Optional

from data.process.schemas import SessionInfo, ProcessedBatch
from data.process.pipeline.steps import (
    SessionFileLoader,
    StatsCollector,
    BlockArrayBuilder,
    PatchPurityFilter,
    PatchMetadataBuilder,
)

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Composable pipeline executor for data processing.

    This class orchestrates the execution of multiple processing steps
    across all sessions, handling data flow, error recovery, and memory management.

    Pure transformation pipeline - no I/O dependencies.
    """

    def __init__(
        self,
        reader: SessionFileLoader,
        array_builder: BlockArrayBuilder,
        stats_collector: StatsCollector,
        metadata_builder: PatchMetadataBuilder,
        purity_filter: Optional[PatchPurityFilter] = None,
    ):
        """Initialize processing pipeline with named stages.

        Pipeline order:
        1. SessionFileLoader - Load data and extract blocks (returns dict)
        2. BlockArrayBuilder - Convert to ProcessedBatch (returns ProcessedBatch)
        3. StatsCollector - Compute statistics (ProcessedBatch -> ProcessedBatch with stats)
        4. PatchPurityFilter - Filter by purity (optional, ProcessedBatch -> ProcessedBatch)
        5. PatchMetadataBuilder - Build metadata (ProcessedBatch -> ProcessedBatch with metadata)

        Args:
            reader: SessionFileLoader for loading session data
            array_builder: BlockArrayBuilder for stacking blocks into arrays
            stats_collector: StatsCollector for computing statistics
            metadata_builder: PatchMetadataBuilder for building metadata
            purity_filter: Optional PatchPurityFilter for filtering pure patches (labeled only)
        """
        self.reader = reader
        self.array_builder = array_builder
        self.stats_collector = stats_collector
        self.metadata_builder = metadata_builder
        self.purity_filter = purity_filter

    def process_session(
        self,
        session_info: SessionInfo
    ) -> Optional[ProcessedBatch]:
        """Execute pipeline for a single session.

        Args:
            session_info: Session information

        Returns:
            ProcessedBatch with complete metadata and stats, or None if session was filtered out

        Raises:
            ValueError: If no blocks to process
        """
        # Stage 1: Load session data (dict)
        blocks = self.reader.process(session_info)

        # Stage 2: Build arrays from blocks (dict -> ProcessedBatch)
        batch = self.array_builder.process(blocks)

        # Stage 3: Collect stats (ProcessedBatch -> ProcessedBatch with stats)
        batch = self.stats_collector.process(session_info, batch)

        # Stage 4: Filter by purity (optional, labeled only)
        if self.purity_filter is not None:
            batch = self.purity_filter.process(batch)

        # Check if batch is empty after filtering
        if len(batch.acc_blocks) == 0:
            logger.debug(f"Session {session_info.id} filtered out completely")
            return None

        # Stage 5: Build complete metadata
        batch = self.metadata_builder.process(session_info, batch)

        return batch

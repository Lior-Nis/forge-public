"""Processor for unlabeled dataset creation."""

import logging
from typing import Callable

import pandas as pd
from omegaconf import DictConfig

from data.process.processor.base import BaseDataProcessor
from data.process.pipeline import (
    SessionFileLoader,
    StatsCollector,
    BlockArrayBuilder,
    PatchMetadataBuilder,
    ProcessingPipeline,
)

logger = logging.getLogger(__name__)


class UnlabeledDataProcessor(BaseDataProcessor):
    """Processor for unlabeled data.

    Unlabeled data includes:
    - Accelerometer data only
    - No labels, patch labels, or valid masks
    - No purity filtering (not applicable)
    """

    def __init__(self, cfg: DictConfig):
        """Initialize unlabeled data processor.

        Args:
            cfg: Hydra configuration
        """
        super().__init__(cfg)

    @property
    def file_reader(self) -> Callable:
        """Return Parquet reader for unlabeled data."""
        return pd.read_parquet

    def build_pipeline(self) -> ProcessingPipeline:
        """Build processing pipeline for unlabeled data.

        Pipeline stages:
        1. SessionFileLoader - Load Parquet and extract blocks
        2. StatsCollector - Compute normalization statistics
        3. BlockArrayBuilder - Convert to numpy arrays (no labels required)
        4. PatchMetadataBuilder - Build metadata (no purity computation)

        Returns:
            ProcessingPipeline configured for unlabeled data
        """
        return ProcessingPipeline(
            reader=SessionFileLoader(
                file_reader=self.file_reader,
                block_len=self.processing_config.lengths.block_len,
                stride_len=self.processing_config.lengths.stride_len,
            ),
            array_builder=BlockArrayBuilder(require_labels=False),
            stats_collector=StatsCollector(num_channels=3),
            metadata_builder=PatchMetadataBuilder(metadata_df=self.metadata_df),
            purity_filter=None,  # Not applicable for unlabeled data
        )

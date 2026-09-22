"""Processor for labeled dataset creation."""

import logging
from typing import Callable, List, Tuple

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from data.process.processor.base import BaseDataProcessor
from data.process.schemas import SessionInfo
from data.process.pipeline import (
    SessionFileLoader,
    StatsCollector,
    BlockArrayBuilder,
    PatchPurityFilter,
    PatchMetadataBuilder,
    ProcessingPipeline,
)

logger = logging.getLogger(__name__)


class LabeledDataProcessor(BaseDataProcessor):
    """Processor for labeled data.

    Labeled data includes:
    - Accelerometer data
    - Label blocks (activity class per timestep)
    - Patch labels (majority class per patch)
    - Valid masks

    Note: Creates comprehensive datasets with ALL patches.
    Use views for filtering (purity, validity, protocol, etc.)
    """

    def __init__(self, cfg: DictConfig):
        """Initialize labeled data processor.

        Args:
            cfg: Hydra configuration
        """
        super().__init__(cfg)

    @property
    def file_reader(self) -> Callable:
        """Return CSV reader for labeled data."""
        return pd.read_csv

    def build_pipeline(self) -> ProcessingPipeline:
        """Build processing pipeline for labeled data.

        Pipeline stages:
        1. SessionFileLoader - Load CSV and extract blocks
        2. StatsCollector - Compute normalization statistics
        3. BlockArrayBuilder - Convert to numpy arrays (requires labels)
        4. PatchMetadataBuilder - Build complete metadata

        Note: Purity filtering removed - all patches included.
        Use views for filtering after dataset creation.

        Returns:
            ProcessingPipeline configured for labeled data
        """
        fog_stride_len = self.processing_config.lengths.fog_stride_len
        any_fog_labeling = self.processing_config.any_fog_labeling
        fog_ratio_labeling = self.processing_config.fog_ratio_labeling
        return ProcessingPipeline(
            reader=SessionFileLoader(
                file_reader=self.file_reader,
                block_len=self.processing_config.lengths.block_len,
                stride_len=self.processing_config.lengths.stride_len,
                fog_stride_len=fog_stride_len,
            ),
            array_builder=BlockArrayBuilder(
                require_labels=True,
                any_fog_labeling=any_fog_labeling,
                fog_ratio_labeling=fog_ratio_labeling,
            ),
            stats_collector=StatsCollector(num_channels=3),
            metadata_builder=PatchMetadataBuilder(
                metadata_df=self.metadata_df,
                any_fog_labeling=any_fog_labeling,
            ),
            purity_filter=None,  # Removed - use views for filtering
        )

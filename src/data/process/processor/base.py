"""Base data processor with shared logic for labeled and unlabeled datasets."""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data.config import PathsConfig
from data.process.schemas import SessionInfo, ProcessingConfig
from data.process.pipeline import ProcessingPipeline
from data.process.stats.computer import HierarchicalStatsComputer
from data.process.stats.writer import create_temp_stats_writer
from data.process.writer.zarr import ZarrWriter
from utils.paths import build_processed_dataset_path

logger = logging.getLogger(__name__)


class BaseDataProcessor(ABC):
    """Base class for dataset processors.

    Provides shared functionality for both labeled and unlabeled data processing.
    Subclasses must implement file_reader and build_pipeline.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize base data processor.

        Args:
            cfg: Hydra configuration (must contain data_type field)
        """
        # Validate paths config - only extract fields needed by PathsConfig
        paths_dict = OmegaConf.to_container(cfg.paths, resolve=True)
        self.paths_cfg = PathsConfig(
            input_dir=paths_dict["input_dir"],
            processed_dir=paths_dict["processed_dir"],
            dataset_path=paths_dict["dataset_path"],
        )

        # Extract processing config (always under cfg.process in Hydra composition)
        process_cfg = cfg.process

        # Convert OmegaConf to dict and validate with Pydantic
        process_dict = OmegaConf.to_container(process_cfg, resolve=True)

        self.processing_config = ProcessingConfig(**process_dict)

        # Keep raw config for compatibility
        self.cfg = cfg
        self.data_type = process_cfg.data_type

        # Use validated configs
        self.input_dir = Path(self.paths_cfg.input_dir)

        # Initialize processing components
        self.metadata_df = self._load_metadata()
        self.stats_computer = HierarchicalStatsComputer(self.metadata_df)
        self.writer_cls = ZarrWriter

        self.block_len = self.processing_config.lengths.block_len
        self.stride_len = self.processing_config.lengths.stride_len
        self.fog_stride_len = self.processing_config.lengths.fog_stride_len

        # Use explicit output filepath from config
        self.output_filepath = self.paths_cfg.processed_dataset_path
        os.makedirs(os.path.dirname(self.output_filepath), exist_ok=True)

    def process(self) -> str:
        """Run the processing pipeline."""
        session_infos: List[SessionInfo] = self._load_session_infos()
        output_path, written_patches = self._process_impl(session_infos=session_infos)
        return output_path

    @property
    @abstractmethod
    def file_reader(self) -> Callable:
        """Return file reader function (pd.read_csv or pd.read_parquet)."""
        raise NotImplementedError

    @abstractmethod
    def build_pipeline(self) -> ProcessingPipeline:
        """Build processing pipeline - must be implemented by subclass.

        Returns:
            ProcessingPipeline configured for this processor type
        """
        raise NotImplementedError

    def _process_impl(self, session_infos: List[SessionInfo]) -> Tuple[str, int]:
        """Execute processing pipeline.

        Args:
            session_infos: List of session information objects

        Returns:
            Tuple of (output_filepath, total_patches)
        """
        is_unlabeled = (self.data_type == "unlabeled")
        logger.info(f"Processing {self.data_type} sessions with pipeline...")

        zarr_cfg = self.processing_config.zarr_config
        with self._create_writer(
            output_filepath=self.output_filepath,
            block_len=self.block_len,
            stride_len=self.stride_len,
            fog_stride_len=self.fog_stride_len,
            is_unlabeled=is_unlabeled,
            fog_ratio_labeling=self.processing_config.fog_ratio_labeling,
            chunk_size=zarr_cfg.chunk_size,
            compressor=zarr_cfg.compressor,
            compression_level=zarr_cfg.compression_level,
        ) as writer:
            # Store session info upfront
            writer.store_session_info([s.to_dict() for s in session_infos])

            # Build pipeline (pure transformation, no I/O dependencies)
            pipeline = self.build_pipeline()

            # Session iteration happens here
            total_patches = 0
            all_stats = []

            for session_info in tqdm(session_infos, desc="Processing sessions"):
                try:
                    batch = pipeline.process_session(session_info)

                    if batch is None:
                        continue  # Empty after filtering

                    # Global indices already set correctly by PatchMetadataBuilder
                    # No mutation needed - PatchMetadata is now immutable (frozen)

                    # Update metadata_df with correct protocol from actual file location
                    # This ensures stats aggregation uses the correct protocol
                    if session_info.id in self.metadata_df.index:
                        self.metadata_df.loc[session_info.id, 'protocol'] = session_info.protocol

                    # Write batch to storage
                    writer.append_batch(
                        acc_blocks=batch.acc_blocks,
                        label_blocks=batch.label_blocks,
                        patch_labels=batch.patch_labels,
                        valid_blocks=batch.valid_blocks,
                    )

                    # Write metadata incrementally per session to avoid OOM on large
                    # datasets (e.g. 69M patches would require ~34 GB if accumulated)
                    records = [m.to_dict() for m in batch.metadata]
                    writer.append_metadata_batch(records)
                    all_stats.append(batch.stats)
                    total_patches += len(batch.acc_blocks)

                    # No explicit memory cleanup needed - Python GC handles it naturally
                    # Previous gc.collect() was a symptom of memory leaks, now fixed

                except ValueError as e:
                    if "No blocks to process" in str(e):
                        logger.warning(f"Session {session_info.id} had no blocks, skipping")
                        continue
                    raise

            # Validate that we generated some patches
            if total_patches == 0:
                raise ValueError(
                    f"No patches generated. All {len(session_infos)} sessions "
                    f"were either empty or filtered out."
                )

            # Aggregate stats from collected list
            # Convert list to dict format and aggregate
            for stats in all_stats:
                self.stats_computer.session_stats[stats.session_id] = {
                    'mean': stats.mean,
                    'std': stats.std,
                    'median': stats.median,
                    'mad': stats.mad,
                    'n_samples': stats.n_samples,
                    'samples': stats.samples,
                }
            aggregated_stats = self.stats_computer.aggregate()
            writer.store_hierarchical_stats(aggregated_stats)

            # Finalize dataset (metadata already written incrementally per session)
            self._finalize_dataset(
                writer=writer,
                session_infos=session_infos,
            )

        logger.info(f"Successfully created {self.data_type} dataset: %s", self.output_filepath)
        return self.output_filepath, total_patches

    def _load_session_infos(self) -> List[SessionInfo]:
        """Load session file paths from {protocol}/sessions/ structure with type safety."""
        session_infos = []

        for protocol_dir in sorted(self.input_dir.iterdir()):
            if not protocol_dir.is_dir():
                continue

            sessions_dir = protocol_dir / "sessions"
            if not sessions_dir.exists():
                logger.warning(f"No sessions directory in {protocol_dir}")
                continue

            for file_path in sorted(sessions_dir.iterdir()):
                if not file_path.is_file():
                    logger.warning(f"Note that you have a non-file item at {file_path}. Please remove it.")
                    continue

                # Skip hidden files (like .DS_Store) and non-CSV files
                if file_path.name.startswith('.'):
                    continue
                if not (file_path.suffix == '.csv' or file_path.suffix == '.parquet'):
                    logger.warning(f"Skipping non-CSV/parquet file: {file_path}")
                    continue

                session_info = SessionInfo(
                    path=str(file_path),
                    filename=file_path.name,
                    protocol=protocol_dir.name,
                    id=file_path.stem
                )
                session_infos.append(session_info)

        if not session_infos:
            raise ValueError(f"No session files found in {self.input_dir}")

        logger.info(f"Loaded {len(session_infos)} session files")
        return session_infos

    def _finalize_dataset(
        self,
        writer: ZarrWriter,
        session_infos: List[SessionInfo],
    ) -> None:
        """Finalize dataset creation: store aggregated stats and config.
        Metadata is written incrementally per session via append_metadata_batch.
        """
        # Metadata was written incrementally — just log the final count
        if "metadata" in writer.root:
            n = writer.root["metadata"].attrs.get("num_records", "?")
            n_patients = len(set(writer.root["metadata"]["patient_id"][:]))
            logger.info(
                f"Patch metadata: {n} patches across "
                f"{len(session_infos)} sessions, {n_patients} patients"
            )

        # Aggregate and store stats after all sessions processed
        hierarchical_stats = self.stats_computer.aggregate()
        writer.store_hierarchical_stats(hierarchical_stats)
        logger.info(
            "Stored normalization stats: "
            f"{len(hierarchical_stats['sessions'])} sessions, "
            f"{len(hierarchical_stats['patients'])} patients, "
            f"{len(hierarchical_stats['protocols'])} protocols"
        )

        # Store full configuration for reproducibility
        writer.store_config(self.cfg)

    def _create_writer(self, **kwargs):
        """Create writer instance."""
        return self.writer_cls(**kwargs)

    def _load_metadata(self) -> pd.DataFrame:
        """Load and combine metadata from all protocols."""
        protocls = os.listdir(self.input_dir)

        metadata_dfs = []
        for protocol in protocls:
            protocol_path = os.path.join(self.input_dir, protocol)
            # Skip non-directory items (like .DS_Store)
            if not os.path.isdir(protocol_path):
                continue
            metadata_path = os.path.join(protocol_path, 'metadata.csv')
            df = pd.read_csv(metadata_path)
            df['protocol'] = protocol
            metadata_dfs.append(df)

        metadata_df = pd.concat(metadata_dfs, ignore_index=True)
        columns_map = {"Id": "session_id", "Subject": "patient_id"}
        metadata_df = metadata_df.rename(columns=columns_map)
        metadata_df = metadata_df.drop_duplicates(subset="session_id")
        metadata_df = metadata_df.set_index("session_id")
        return metadata_df

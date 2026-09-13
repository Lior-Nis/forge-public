import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset
from tqdm import tqdm

from data.config import PathsConfig
from data.dataset.config import DatasetConfig
from data.dataset.protocols import SamplableDataset
from utils.zarr_helpers import read_zarr_metadata

logger = logging.getLogger(__name__)

# Enable Blosc multi-threading for faster decompression (4 threads per chunk)
try:
    import numcodecs
    numcodecs.blosc.set_nthreads(4)
except (ImportError, AttributeError):
    logger.warning("Could not set Blosc threads (numcodecs not available or incompatible version)")


class BaseFOGDataset(Dataset, ABC):
    """
    Base dataset class for Freezing of Gait (FoG) detection.

    This abstract base class provides common functionality for all FoG datasets,
    including:
    - Loading accelerometer data and labels from Zarr storage
    - Per-session normalization (optional)
    - Metadata management with patient/protocol/session information
    - Efficient O(1) patch-level data access

    Task-specific datasets (Classification, MAE, SimCLR) inherit from this class
    and implement their own __getitem__ methods with appropriate return types.

    All datasets return DatasetSample instances for consistent polymorphic processing.

    Implements SamplableDataset protocol to support balanced sampling.
    """
    
    def __init__(
        self,
        paths_cfg: PathsConfig,
        dataset_cfg: DatasetConfig,
        stage: Literal["train", "val", "test"],
        patient_ids: Optional[List[str]] = None
    ) -> None:
        """
        Initialize base dataset functionality with simplified patch-level metadata access.

        Args:
            paths_cfg: Path configuration (shared across layers)
            dataset_cfg: Dataset-specific configuration
            stage: Dataset stage determining which cross-validation split to use.
                Must be one of: "train", "val", "test"
            patient_ids: Patient IDs from cross-validation fold assignment.
                These are patient identifiers (e.g., ['bf608b', 'bae0ce'])
                used to filter patches by patient_id to maintain patient-level splits
                and prevent data leakage.

        Raises:
            FileNotFoundError: If Zarr data store doesn't exist at configured path
            ValueError: If patient_ids is None/empty or no matching patches found
            RuntimeError: If Zarr store is corrupted or has invalid structure
        """
        self.stage = stage
        self.patient_ids = patient_ids or []

        # Store validated configs
        self.paths_cfg = paths_cfg
        self.dataset_cfg = dataset_cfg

        # Extract commonly used config values for easy access
        self.processed_dir = paths_cfg.processed_dir
        self.axis = dataset_cfg.axis

        # Setup normalization strategy
        self._setup_normalization()

        # Initialize data source and metadata
        self._init_data_source()
        self._init_patch_metadata()

        # Free resolved metadata (only needed during init, not after filtering)
        del self._resolved_metadata_df
        self._resolved_metadata_df = None

        # Validate dataset has data
        self._validate_dataset()

    def _setup_normalization(self) -> None:
        """
        Setup per-session normalization if enabled.

        Loads session-level statistics from disk if available.
        Falls back to global normalization if stats not found.
        """
        self.normalize_per_session = self.dataset_cfg.normalize_per_session
        self.session_stats_path = self.dataset_cfg.session_stats_path
        self.session_stats = None

        if self.normalize_per_session:
            if self.session_stats_path and os.path.exists(self.session_stats_path):
                self.session_stats = torch.load(self.session_stats_path)

                # Pre-reshape stats for broadcasting to avoid repeated reshaping per sample
                for session_id in self.session_stats:
                    stats = self.session_stats[session_id]
                    stats['mean'] = stats['mean'].reshape(1, -1)
                    stats['std'] = stats['std'].reshape(1, -1)

                logger.debug(
                    f"Loaded per-session stats for {len(self.session_stats)} sessions"
                )
            else:
                logger.warning(
                    f"Per-session normalization enabled, but stats file not found: "
                    f"{self.session_stats_path}"
                )

    def _validate_dataset(self) -> None:
        """
        Validate dataset contains patches.

        Raises:
            ValueError: If no patches found for this stage
        """
        self.length = len(self.metadata_df)

        if self.length == 0:
            logger.error(
                f"No patches found for stage '{self.stage}' "
                f"with patient_ids: {self.patient_ids[:5]}..."
            )
            raise ValueError(f"No valid patches found for stage {self.stage}")

        logger.info(f"{self.stage}: {self.length} patches")
        
    def _init_data_source(self) -> None:
        """Initialize Zarr data store path with view support."""
        from data.views.resolver import resolve_dataset_path

        # Get raw path from config (could be .zarr or .yaml)
        raw_path = self.paths_cfg.processed_dataset_path

        # Resolve to physical dataset or view
        try:
            resolved = resolve_dataset_path(raw_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to resolve dataset path: {raw_path}. Error: {e}"
            )

        # Store resolved components
        self.data_file_path = resolved.zarr_path  # Points to physical Zarr
        # Store only picklable components from resolved dataset (avoid lambda)
        self._resolved_metadata_df = resolved.metadata_df
        self._resolved_is_view = resolved.is_view
        self._resolved_view_name = resolved.view_name

        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(
                f"Physical dataset not found: {self.data_file_path}. "
                f"Please ensure your data has been processed and saved in Zarr format."
            )

        # Log view information
        if resolved.is_view:
            logger.info(
                f"Using view '{resolved.view_name}' -> {resolved.zarr_path} "
                f"({len(resolved.metadata_df)} patches in view)"
            )

        # Zarr store handle will be created lazily in _get_raw_data() per worker
        self._zarr_store = None
        
    def _init_patch_metadata(self) -> None:
        """Load patch-level metadata - now using resolved (pre-filtered) metadata."""
        # Use pre-filtered metadata from resolved dataset
        # For physical datasets: this is all metadata
        # For views: this is already filtered by view criteria
        full_metadata_df = self._resolved_metadata_df

        # Apply patient_id filter for cross-validation splits
        mask = full_metadata_df['patient_id'].isin(self.patient_ids)
        self.metadata_df = full_metadata_df[mask].copy()

        # Optional metadata filter (replaces the legacy view-file mechanism). Applies
        # the same pandas-query semantics a view's custom_query used, so pointing at a
        # physical .zarr + metadata_query="validity == 1.0" is identical to the old
        # defog_valid_* views. Settable on the dataset config (eval) or the paths
        # config (training "valid" path configs). Applies to every stage.
        metadata_query = (
            getattr(self.dataset_cfg, "metadata_query", None)
            or getattr(self.paths_cfg, "metadata_query", None)
        )
        if metadata_query:
            from data.views.filters import apply_filters
            before = len(self.metadata_df)
            self.metadata_df = apply_filters(self.metadata_df, {"custom_query": metadata_query})
            logger.info(
                f"metadata_query='{metadata_query}': kept {len(self.metadata_df)}/{before} patches"
            )

        # Optional train-time data budget: keep only the first N minutes of each
        # patient's recording (fine-tuning-data-size curve, Fig 12). Train-only.
        max_minutes = getattr(self.dataset_cfg, "max_train_minutes", None)
        if self.stage == "train" and max_minutes is not None:
            before = len(self.metadata_df)
            self.metadata_df = self._subset_first_minutes(self.metadata_df, float(max_minutes))
            logger.info(
                f"max_train_minutes={max_minutes}: kept {len(self.metadata_df)}/{before} "
                f"train patches (first {max_minutes} min/patient)"
            )

        # Optional train-time PATIENT budget: keep only K patients (label-efficiency
        # curve, Fig S4). FOG-stratified + seeded so the same K patients are chosen
        # regardless of arm (probe/random/scratch) → fair comparison. Train-only.
        max_patients = getattr(self.dataset_cfg, "max_train_patients", None)
        if self.stage == "train" and max_patients is not None:
            seed = int(getattr(self.dataset_cfg, "train_subset_seed", 0))
            before = len(self.metadata_df)
            kept = self._subset_n_patients(self.metadata_df, int(max_patients), seed)
            self.metadata_df = kept
            logger.info(
                f"max_train_patients={max_patients} (seed={seed}): kept "
                f"{self.metadata_df['patient_id'].nunique()} patients / "
                f"{len(self.metadata_df)}/{before} train patches"
            )

        # Consolidated logging
        if len(self.metadata_df) > 0:
            protocols = self.metadata_df['protocol'].value_counts().to_dict()
            n_sessions = self.metadata_df['session_id'].nunique()
            view_info = f" (view: {self._resolved_view_name})" if self._resolved_is_view else ""
            logger.debug(
                f"{self.stage}{view_info}: {len(self.metadata_df)}/{len(full_metadata_df)} patches, "
                f"{n_sessions} sessions, protocols={protocols}"
            )
        else:
            logger.warning(f"No patches found for {self.stage} split with patient_ids: {self.patient_ids[:5]}...")

    def _subset_first_minutes(self, df: pd.DataFrame, minutes: float) -> pd.DataFrame:
        """Keep only patches within the first `minutes` of each patient's recording.

        Time is accumulated per patient across sessions (start_frame resets per
        session), giving a stratified per-patient budget rather than a global
        first-K slice. Requires 'patient_id', 'session_id', 'start_frame' columns.
        """
        if "start_frame" not in df.columns:
            logger.warning("max_train_minutes set but no 'start_frame' metadata; skipping subset")
            return df
        sr = getattr(self.dataset_cfg, "sampling_rate_hz", 100)
        cutoff_frames = minutes * 60.0 * sr
        keep_idx = []
        for _pid, pgrp in df.groupby("patient_id", sort=False):
            elapsed = 0  # frames consumed for this patient across prior sessions
            for _sid, sgrp in pgrp.sort_values(["session_id", "start_frame"]).groupby(
                "session_id", sort=False
            ):
                sf = sgrp["start_frame"].to_numpy()
                if sf.size == 0:
                    continue
                within = (elapsed + sf) < cutoff_frames
                keep_idx.extend(sgrp.index[within].tolist())
                elapsed += int(sf.max()) + 1  # advance by this session's span
                if elapsed >= cutoff_frames:
                    break
        return df.loc[keep_idx].copy()

    def _subset_n_patients(self, df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
        """Keep all patches of K patients, selected FOG-stratified + seeded.

        For a label-efficiency curve the budget is the NUMBER of labeled patients.
        A naive random draw is dangerous here: ~16% of DeFOG patients have ~0% FOG,
        so a small random subset can be positive-starved. We therefore rank patients
        by their per-patient FOG rate (mean class_label) and pick K patients spread
        across that ranking (systematic stratified sampling), preserving the
        cohort's FOG-rate distribution. Deterministic in `seed`, so the SAME K
        patients are selected for every arm at a given (budget, seed) → fair
        probe/random/scratch comparison.
        """
        pats = df["patient_id"].unique()
        n = len(pats)
        if k >= n:
            return df
        # Per-patient FOG rate (fall back to 0 if class_label absent).
        if "class_label" in df.columns:
            rate = df.groupby("patient_id")["class_label"].mean()
        else:
            logger.warning("max_train_patients: no 'class_label'; selecting patients unstratified")
            rate = pd.Series(0.0, index=pats)
        order = rate.reindex(pats).sort_values(kind="stable").index.to_numpy()  # low→high FOG
        rng = np.random.default_rng(seed)
        # Split the FOG-ordered patients into K contiguous strata; draw one per
        # stratum so the kept set spans the full FOG-rate range.
        bins = np.array_split(np.arange(n), k)
        chosen = [order[b[rng.integers(len(b))]] for b in bins]
        kept_rates = rate.reindex(chosen)
        logger.info(
            f"  selected {k} patients, FOG-rate span "
            f"[{kept_rates.min():.3f}, {kept_rates.max():.3f}] mean {kept_rates.mean():.3f} "
            f"(cohort mean {rate.mean():.3f})"
        )
        return df[df["patient_id"].isin(chosen)].copy()

    def _get_raw_data(self, idx: int) -> Tuple[torch.Tensor, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Dict]:
        """
        Get raw accelerometer data and labels for a given dataset index.

        Args:
            idx: Dataset index in range [0, len(dataset))

        Returns:
            Tuple containing:
                - signal: Accelerometer tensor (channels, seq_len)
                - labels: Timestep-level labels (seq_len,) or None
                - patch_labels: Patch-level labels (seq_len,) or None
                - valid_mask: Boolean mask (seq_len,)
                - metadata: Patch and session information

        Raises:
            IndexError: If idx is out of bounds or Zarr access fails
        """
        # Validate index
        if idx < 0 or idx >= self.length:
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {self.length}"
            )

        # Get patch metadata
        patch_row = self.metadata_df.iloc[idx]

        # Load raw data from Zarr
        signal, labels, patch_labels, valid_mask = self._load_from_zarr(
            patch_row.name , patch_row['session_id'], idx
        )   # global_idx

        # Preprocess signal (convert to tensor, normalize, transpose)
        signal = self._preprocess_signal(signal, patch_row)

        # Build metadata dictionary
        metadata = {
            "session_id": patch_row['session_id'],
            "patient_id": patch_row['patient_id'],
            "protocol": patch_row['protocol'],
            "session_idx": patch_row['session_idx'],
            "global_idx": patch_row.name,  # global_idx is the DataFrame index
            "dataset_idx": idx,
        }

        return signal, labels, patch_labels, valid_mask, metadata

    def _load_from_zarr(
        self, global_idx: int, session_id: str, dataset_idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Load data arrays from Zarr store.

        Args:
            global_idx: Global patch index in Zarr arrays
            session_id: Session ID for error reporting
            dataset_idx: Dataset index for error reporting

        Returns:
            Tuple of (signal, labels, patch_labels, valid_mask)
        """
        # Get Zarr store handle (cached per worker)
        if not hasattr(self, '_zarr_store') or self._zarr_store is None:
            self._zarr_store = zarr.open(self.data_file_path, mode='r')

        try:
            # Load accelerometer data (always required)
            signal = self._zarr_store['accs'][global_idx]

            # Try to load labels (may not exist for unlabeled datasets)
            if 'labels' in self._zarr_store:
                labels = self._zarr_store['labels'][global_idx]
                patch_labels = self._zarr_store['patch_labels'][global_idx]
                valid_mask = self._zarr_store['valid_masks'][global_idx]
            else:
                # Unlabeled dataset - no labels available
                labels = None
                patch_labels = None
                valid_mask = np.ones(len(signal), dtype=bool)

        except (KeyError, IndexError) as e:
            raise IndexError(
                f"Failed to load patch at global_idx={global_idx} (dataset_idx={dataset_idx}). "
                f"Session: {session_id}, Error: {e}"
            )

        return signal, labels, patch_labels, valid_mask

    def _preprocess_signal(self, signal: np.ndarray, patch_row) -> torch.Tensor:
        """
        Convert signal to tensor, apply normalization, and transpose.

        Args:
            signal: Raw signal array (time, channels)
            patch_row: Patch metadata row from DataFrame

        Returns:
            Preprocessed signal tensor (channels, time)
        """
        # Convert to tensor
        x = torch.from_numpy(signal).float()

        # Apply per-session normalization if enabled
        if self.normalize_per_session and self.session_stats:
            session_id = patch_row['session_id']
            if session_id in self.session_stats:
                stats = self.session_stats[session_id]
                # x is (time, channels) - stats already reshaped to (1, channels) in _setup_normalization
                x = (x - stats['mean']) / (stats['std'] + 1e-8)

        # Transpose to (channels, time)
        return x.T

    def __len__(self) -> int:
        """Return dataset length."""
        return self.length

    def get_sampling_metadata(self) -> pd.DataFrame:
        """
        Return metadata for balanced sampling.

        Implements SamplableDataset protocol. Returns a view of the metadata
        DataFrame with columns required for hierarchical sampling:
        - class_label: Class labels for class balancing
        - protocol: Protocol identifiers for protocol balancing
        - patient_id: Patient identifiers for patient balancing

        Returns:
            DataFrame with sampling metadata (length matches dataset)

        Raises:
            RuntimeError: If called before dataset initialization

        Note:
            Returns a reference to internal DataFrame for performance.
            Callers should not modify the returned DataFrame.
        """
        if self.metadata_df is None:
            raise RuntimeError(
                "Dataset metadata not initialized. "
                "This should not happen if dataset was properly constructed."
            )

        # Return the full metadata DataFrame
        # Sampler will extract the columns it needs
        return self.metadata_df

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Task-specific data loading - implemented by subclasses."""
        pass

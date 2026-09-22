"""
Zarr writer module for efficient streaming data storage.

Provides dynamic append capabilities without requiring upfront size allocation,
enabling true single-pass streaming processing.
"""

import functools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import zarr
from omegaconf import DictConfig, OmegaConf
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.core.dtype.npy.string import VariableLengthUTF8

logger = logging.getLogger(__name__)

# Constants
NUM_ACC_CHANNELS = 3


# ---------------------------------------------------------------------------
# Windows atomic-write resilience
# ---------------------------------------------------------------------------
# zarr v3's LocalStore writes each metadata/chunk file by creating a temporary
# "*.partial" file and then os.replace()-ing it onto the final name. On Windows
# this atomic rename fails with WinError 5 (ERROR_ACCESS_DENIED) or WinError 32
# (ERROR_SHARING_VIOLATION) whenever another process holds a transient handle on
# the freshly written file -- OneDrive (this repo can live under OneDrive\Desktop),
# Windows Defender real-time scanning, and the Search Indexer all do this. Because
# the metadata arrays are resized once per session, the tiny zarr.json files get
# rewritten hundreds of times in quick succession, making a collision almost
# certain. The locks clear within ~1s, so we retry the whole write op with
# exponential backoff. This is a no-op on POSIX, where these winerrors never occur.
_WINDOWS_LOCK_WINERRORS = frozenset({5, 32})


def _retry_on_windows_lock(fn):
    """Retry a ZarrWriter write method when a transient Windows file lock makes
    zarr's atomic os.replace fail. No behavioural change on POSIX.

    Safe because every wrapped method is idempotent on retry: it overwrites the
    same array slices and only advances its length/attrs counters at the very end.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        delay = 0.25
        max_attempts = 8
        for attempt in range(1, max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except OSError as exc:  # PermissionError is an OSError subclass
                winerror = getattr(exc, "winerror", None)
                if (
                    os.name != "nt"
                    or winerror not in _WINDOWS_LOCK_WINERRORS
                    or attempt == max_attempts
                ):
                    raise
                logger.warning(
                    "%s: transient Windows file lock (WinError %s); retry %d/%d in %.2fs",
                    fn.__name__, winerror, attempt, max_attempts - 1, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 5.0)

    return wrapper


class ZarrWriter:
    """
    Zarr-based writer with dynamic append support for streaming data processing.

    Provides methods for:
    - Creating Zarr stores with optimal chunking and compression
    - Incremental data appending for memory efficiency
    - Metadata and session information storage
    - Support for both labeled and unlabeled datasets
    """

    def __init__(
        self,
        output_filepath: str,
        block_len: int,
        stride_len: int,
        fog_stride_len: Optional[int] = None,
        is_unlabeled: bool = False,
        fog_ratio_labeling: bool = False,
        chunk_size: int = 1000,
        compressor = "zstd",
        compression_level: int = 3,
    ):
        """
        Initialize Zarr writer.

        Args:
            output_filepath: Path for output Zarr store (directory or .zarr)
            block_len: Length of each block
            stride_len: Stride length used for background patches
            fog_stride_len: Denser stride used for FOG patches (None = fixed stride)
            is_unlabeled: Whether this is an unlabeled dataset
            chunk_size: Number of patches per chunk (default: 1000)
            compressor: Compression algorithm ('zstd', 'lz4', 'blosc', None)
            compression_level: Compression level (1-9, default: 3)

        Note: Creates comprehensive datasets with ALL patches.
        Use views for filtering (purity, validity, protocol, etc.)
        """
        self.output_filepath = output_filepath
        self.block_len = block_len
        self.stride_len = stride_len
        self.fog_stride_len = fog_stride_len
        self.is_unlabeled = is_unlabeled
        self.fog_ratio_labeling = fog_ratio_labeling
        self.chunk_size = chunk_size
        self.compressors = BloscCodec(cname=compressor,
                                      clevel=compression_level,
                                      shuffle=BloscShuffle.shuffle)

        self.store = None
        self.root = None
        self.arrays = {}
        self.current_patch_count = 0
        # Authoritative length of the metadata arrays. Kept as a plain instance
        # counter (like current_patch_count) instead of reading the zarr
        # `num_records` attr, because the attr is updated in memory BEFORE its
        # durable write -- so on a Windows lock retry it could double-count. See
        # append_metadata_batch.
        self._metadata_len = 0

        logger.info(
            f"ZarrWriter initialized: {Path(output_filepath).name}, "
            f"{'unlabeled' if is_unlabeled else 'labeled'}, "
            f"compressor={compressor}, level={compression_level}"
        )

    def __enter__(self):
        """Context manager entry."""
        self._create_store()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @_retry_on_windows_lock
    def _create_store(self) -> None:
        """Create Zarr store and initialize arrays."""
        # Create Zarr group (zarr v3 API)
        self.root = zarr.open_group(self.output_filepath, mode='w')
        self.store = self.root.store

        # Create accelerometer array (always present)
        # Shape starts at (0, block_len, channels) and grows dynamically
        self.arrays["accs"] = self.root.create_array(
            "accs",
            shape=(0, self.block_len, NUM_ACC_CHANNELS),
            chunks=(self.chunk_size, self.block_len, NUM_ACC_CHANNELS),
            dtype=np.float32,
            compressors=self.compressors,
        )

        # Create label arrays only for labeled data
        if not self.is_unlabeled:
            self.arrays["labels"] = self.root.create_array(
                "labels",
                shape=(0, self.block_len),
                chunks=(self.chunk_size, self.block_len),
                dtype=np.int32,
                compressors=self.compressors,
            )

            patch_label_dtype = np.float32 if self.fog_ratio_labeling else np.int32
            self.arrays["patch_labels"] = self.root.create_array(
                "patch_labels",
                shape=(0,),
                chunks=(self.chunk_size,),
                dtype=patch_label_dtype,
                compressors=self.compressors,
            )

            self.arrays["valid_masks"] = self.root.create_array(
                "valid_masks",
                shape=(0, self.block_len),
                chunks=(self.chunk_size, self.block_len),
                dtype=bool,
                compressors=self.compressors,
            )

        # Store metadata as attributes
        self.root.attrs["block_len"] = self.block_len
        self.root.attrs["stride_len"] = self.stride_len
        self.root.attrs["fog_stride_len"] = self.fog_stride_len
        self.root.attrs["is_unlabeled"] = self.is_unlabeled
        self.root.attrs["fog_ratio_labeling"] = self.fog_ratio_labeling
        # Note: Dataset name extracted from filepath, purity filtering removed (use views)

        logger.debug(f"Created Zarr store at {self.output_filepath}")

    @_retry_on_windows_lock
    def append_batch(
        self,
        acc_blocks: np.ndarray,
        label_blocks: Optional[np.ndarray] = None,
        patch_labels: Optional[np.ndarray] = None,
        valid_blocks: Optional[np.ndarray] = None,
    ) -> int:
        """
        Append a batch of data to the Zarr arrays.

        This is the key method that enables streaming: no need to know
        total size upfront, just append as data becomes available.

        Args:
            acc_blocks: Accelerometer data (n_patches, block_len, 3)
            label_blocks: Label data (n_patches, block_len) - labeled only
            patch_labels: Patch-level labels (n_patches,) - labeled only
            valid_blocks: Valid masks (n_patches, block_len) - labeled only

        Returns:
            New current patch count after append
        """
        n_patches = len(acc_blocks)
        if n_patches == 0:
            return self.current_patch_count

        # Calculate new size
        new_size = self.current_patch_count + n_patches

        # Resize arrays and append data
        self.arrays["accs"].resize((new_size, self.block_len, NUM_ACC_CHANNELS))
        self.arrays["accs"][self.current_patch_count:new_size] = acc_blocks

        if not self.is_unlabeled:
            if label_blocks is None or patch_labels is None or valid_blocks is None:
                raise ValueError("Labeled dataset requires label_blocks, patch_labels, and valid_blocks")

            self.arrays["labels"].resize((new_size, self.block_len))
            self.arrays["labels"][self.current_patch_count:new_size] = label_blocks

            self.arrays["patch_labels"].resize((new_size,))
            self.arrays["patch_labels"][self.current_patch_count:new_size] = patch_labels

            self.arrays["valid_masks"].resize((new_size, self.block_len))
            self.arrays["valid_masks"][self.current_patch_count:new_size] = valid_blocks

        self.current_patch_count = new_size
        logger.debug(f"Appended {n_patches} patches, total: {self.current_patch_count}")

        return self.current_patch_count

    @_retry_on_windows_lock
    def store_session_info(self, session_info_list: List[Dict[str, Any]]) -> None:
        """
        Store session information as JSON in root attributes.

        Args:
            session_info_list: List of session info dictionaries
        """
        self.root.attrs["session_infos"] = json.dumps(session_info_list)
        logger.debug(f"Stored session info for {len(session_info_list)} sessions")

    @_retry_on_windows_lock
    def append_metadata_batch(self, records: list) -> None:
        """
        Incrementally append a batch of metadata records to the zarr store.

        Call this after processing each session to avoid accumulating all records
        in memory. Safe to call multiple times; creates the group on first call
        and appends on subsequent calls.

        String columns are stored as fixed-length unicode ('U32') which supports
        zarr v3 resize; VariableLengthUTF8 does not support resize.

        Args:
            records: list of dicts from PatchMetadata.to_dict()
        """
        if not records:
            return

        df = pd.DataFrame(records).set_index("global_idx")

        # Encode string columns as int16 to avoid zarr v3 string dtype bugs
        # (VariableLengthUTF8 doesn't support resize; FixedLengthUTF32 has .partial cleanup errors)
        # Encoding dicts are stored in attrs for decoding/debugging.
        string_cols = [c for c in df.columns if df[c].dtype == object]

        if self._metadata_len == 0:
            # First batch: create the metadata group + arrays. overwrite=True makes a
            # retried partial-create idempotent (a transient Windows lock mid-create
            # would otherwise leave a half-written group that the retry must replace).
            metadata_group = self.root.create_group("metadata", overwrite=True)
            encodings = {}
            for col in df.columns:
                data = df[col].values
                if data.dtype == object:
                    enc = {v: i for i, v in enumerate(sorted(set(data.tolist())))}
                    encodings[col] = enc
                    data = np.array([enc[v] for v in data], dtype=np.int16)
                metadata_group.create_array(
                    col, data=data,
                    chunks=(self.chunk_size,),
                    compressors=self.compressors,
                )
            metadata_group.create_array(
                "global_idx", data=df.index.values,
                chunks=(self.chunk_size,),
                compressors=self.compressors,
            )
            metadata_group.attrs["num_records"] = len(df)
            metadata_group.attrs["columns"] = list(df.columns)
            metadata_group.attrs["encodings"] = encodings
            # Advance the counter LAST (pure Python, cannot raise): if any zarr write
            # above failed and the @_retry_on_windows_lock decorator re-runs this
            # method, _metadata_len is still 0 so we recreate cleanly — never double.
            self._metadata_len = len(df)
        else:
            metadata_group = self.root["metadata"]
            # Authoritative length is the instance counter, NOT
            # metadata_group.attrs["num_records"]: zarr updates that attr in memory
            # BEFORE its durable atomic write, so if that write hits a transient
            # Windows lock and this method is retried, reading the attr back would
            # return the already-incremented value and append the batch twice
            # (observed: a daily-living build grew 548196 -> 550638 with duplicates).
            current_len = self._metadata_len
            n = len(df)
            encodings = dict(metadata_group.attrs.get("encodings", {}))
            for col in df.columns:
                arr = metadata_group[col]
                data = df[col].values
                if data.dtype == object:
                    enc = encodings.get(col, {})
                    # Register any new values
                    for v in data:
                        if v not in enc:
                            enc[v] = len(enc)
                    encodings[col] = enc
                    data = np.array([enc[v] for v in data], dtype=np.int16)
                arr.resize(current_len + n)
                arr[current_len:current_len + n] = data
            idx_arr = metadata_group["global_idx"]
            idx_arr.resize(current_len + n)
            idx_arr[current_len:current_len + n] = df.index.values
            metadata_group.attrs["num_records"] = current_len + n
            metadata_group.attrs["encodings"] = encodings
            # Advance the counter LAST — see note above; keeps retries idempotent.
            self._metadata_len = current_len + n

    @_retry_on_windows_lock
    def store_patch_metadata(self, metadata_df: pd.DataFrame) -> None:
        """
        Store patch-level metadata as a Zarr group with separate arrays per column.

        For large datasets prefer append_metadata_batch() called per session to
        avoid loading all records into memory at once.

        Args:
            metadata_df: DataFrame with patch-level metadata
        """
        if metadata_df.empty:
            logger.warning("Empty metadata DataFrame, skipping storage")
            return

        if "metadata" in self.root:
            # Already written incrementally — nothing to do
            logger.info(f"Metadata already written incrementally ({len(metadata_df)} records)")
            return

        # Create metadata group
        metadata_group = self.root.create_group("metadata", overwrite=True)

        # Store each column as a separate array for efficient access
        for col in metadata_df.columns:
            data = metadata_df[col].values
            dtype = data.dtype

            # Special handling for object/string columns
            if dtype == object:
                # Use variable-length UTF-8 strings (Zarr V3 compatible)
                string_data = [str(x) for x in data]
                arr = metadata_group.create_array(
                    col,
                    shape=(len(string_data),),
                    dtype=VariableLengthUTF8(),
                    chunks=(self.chunk_size,),
                    compressors=self.compressors,
                )
                arr[:] = string_data
            else:
                # For numeric types, use the data parameter directly
                metadata_group.create_array(
                    col,
                    data=data,
                    chunks=(self.chunk_size,),
                    compressors=self.compressors,
                )

        # Store index separately
        metadata_group.create_array(
            "global_idx",
            data=metadata_df.index.values,
            chunks=(self.chunk_size,),
            compressors=self.compressors,
        )

        metadata_group.attrs["num_records"] = len(metadata_df)
        metadata_group.attrs["columns"] = list(metadata_df.columns)

        logger.info(f"Stored patch metadata: {len(metadata_df)} records")

    @_retry_on_windows_lock
    def store_hierarchical_stats(self, hierarchical_stats: Dict[str, Dict[str, Any]]) -> None:
        """
        Store hierarchical normalization statistics in Zarr groups.

        Creates /stats/ group with session, patient, and protocol level statistics.
        Each level stores mean, std, median, and MAD for normalization.

        Args:
            hierarchical_stats: Dictionary with keys 'sessions', 'patients', 'protocols'

        Structure:
            /stats/sessions/{session_id}/{mean, std, median, mad, n_samples}
            /stats/patients/{patient_id}/{mean, std, median, mad, n_sessions, n_samples, protocols}
            /stats/protocols/{protocol}/{mean, std, median, mad, n_patients, n_sessions, n_samples}
        """
        # Create stats root group
        stats_group = self.root.create_group('stats', overwrite=True)

        # Store session-level stats
        sessions_group = stats_group.create_group('sessions')
        for session_id, session_stats in hierarchical_stats['sessions'].items():
            sess_grp = sessions_group.create_group(str(session_id))
            sess_grp.create_array('mean', data=session_stats['mean'])
            sess_grp.create_array('std', data=session_stats['std'])
            sess_grp.create_array('median', data=session_stats['median'])
            sess_grp.create_array('mad', data=session_stats['mad'])
            sess_grp.attrs['n_samples'] = int(session_stats['n_samples'])

        # Store patient-level stats
        patients_group = stats_group.create_group('patients')
        for patient_id, patient_stats in hierarchical_stats['patients'].items():
            pat_grp = patients_group.create_group(str(patient_id))
            pat_grp.create_array('mean', data=patient_stats['mean'])
            pat_grp.create_array('std', data=patient_stats['std'])
            pat_grp.create_array('median', data=patient_stats['median'])
            pat_grp.create_array('mad', data=patient_stats['mad'])
            pat_grp.attrs['n_sessions'] = int(patient_stats['n_sessions'])
            pat_grp.attrs['n_samples'] = int(patient_stats['n_samples'])
            pat_grp.attrs['protocols'] = json.dumps(patient_stats['protocols'])

        # Store protocol-level stats
        protocols_group = stats_group.create_group('protocols')
        for protocol, protocol_stats in hierarchical_stats['protocols'].items():
            prot_grp = protocols_group.create_group(str(protocol))
            prot_grp.create_array('mean', data=protocol_stats['mean'])
            prot_grp.create_array('std', data=protocol_stats['std'])
            prot_grp.create_array('median', data=protocol_stats['median'])
            prot_grp.create_array('mad', data=protocol_stats['mad'])
            prot_grp.attrs['n_patients'] = int(protocol_stats['n_patients'])
            prot_grp.attrs['n_sessions'] = int(protocol_stats['n_sessions'])
            prot_grp.attrs['n_samples'] = int(protocol_stats['n_samples'])

        logger.info(
            f"Stored hierarchical statistics: {len(hierarchical_stats['sessions'])} sessions, "
            f"{len(hierarchical_stats['patients'])} patients, {len(hierarchical_stats['protocols'])} protocols"
        )

    def store_config(self, cfg: DictConfig) -> None:
        """
        Store the full processing configuration as a YAML file inside the Zarr directory.

        This provides complete observability of how the dataset was created.

        Args:
            cfg: Full Hydra configuration used to create the dataset
        """
        try:
            # Write config to YAML file inside the Zarr directory
            config_path = Path(self.output_filepath) / 'dataset_config.yaml'
            with open(config_path, 'w') as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

            logger.info(f"Stored configuration to {config_path}")

        except Exception as e:
            logger.warning(f"Failed to store configuration: {e}")

    @_retry_on_windows_lock
    def close(self) -> None:
        """Close the Zarr store and finalize writes."""
        if self.store is not None:
            # Update final metadata
            self.root.attrs["total_patches"] = self.current_patch_count

            # Zarr handles finalization automatically, just log
            logger.info(
                f"Closed Zarr store: {self.current_patch_count} patches written to {self.output_filepath}"
            )

            # Clear references
            self.arrays = {}
            self.root = None
            self.store = None

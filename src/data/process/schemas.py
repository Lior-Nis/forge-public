"""Type-safe schemas for data processing pipeline."""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class ZarrConfig(BaseModel):
    """Configuration for Zarr compression and chunking."""

    chunk_size: int = Field(default=1000, gt=0, description="Number of samples per chunk")
    compressor: str = Field(default="zstd", description="Compression codec (zstd, lz4, blosc)")
    compression_level: int = Field(default=3, ge=0, le=9, description="Compression level (0-9)")


class LengthsConfig(BaseModel):
    """Configuration for data windowing lengths."""

    block_len: int = Field(gt=0, description="Block length in samples")
    stride_len: int = Field(gt=0, description="Stride length in samples")
    fog_stride_len: Optional[int] = Field(
        default=None, gt=0,
        description="Denser stride for FOG-containing patches (labeled data only)"
    )

    @field_validator("block_len")
    @classmethod
    def block_len_reasonable(cls, v):
        """Validate block length is reasonable for accelerometer data."""
        if v < 10:
            raise ValueError(f"block_len ({v}) too small, minimum 10 samples")
        if v > 100000:
            raise ValueError(f"block_len ({v}) too large, maximum 100000 samples")
        return v

    @field_validator("stride_len")
    @classmethod
    def stride_not_greater_than_block(cls, v, info):
        """Validate stride doesn't exceed block length."""
        if "block_len" in info.data and v > info.data["block_len"]:
            raise ValueError(
                f"stride_len ({v}) cannot exceed block_len ({info.data['block_len']})"
            )
        return v

    @model_validator(mode="after")
    def fog_stride_not_greater_than_stride(self):
        """Validate fog_stride_len is strictly less than stride_len (must be denser)."""
        if self.fog_stride_len is not None and self.fog_stride_len >= self.stride_len:
            raise ValueError(
                f"fog_stride_len ({self.fog_stride_len}) must be less than "
                f"stride_len ({self.stride_len}) to produce denser FOG sampling"
            )
        return self


class ProcessingConfig(BaseModel):
    """Validate processing configuration with business rules."""

    sampling_rate: int = Field(default=100, gt=0)
    lengths: LengthsConfig
    data_type: str = Field(pattern="^(labeled|unlabeled)$")
    any_fog_labeling: bool = Field(
        default=False,
        description="Label patches as positive if ANY timestep is FOG (vs majority vote)"
    )
    fog_ratio_labeling: bool = Field(
        default=False,
        description="Store continuous fog_ratio (fraction of FOG frames) as float32 patch_labels"
    )
    zarr_config: ZarrConfig = Field(default_factory=ZarrConfig)

    processor: Optional[dict] = Field(
        default=None, description="Hydra processor configuration"
    )


@dataclass(frozen=True)  # Immutable
class SessionInfo:
    """Type-safe session information."""

    path: str
    filename: str
    protocol: str
    id: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "filename": self.filename,
            "protocol": self.protocol,
            "id": self.id,
        }


@dataclass
class SessionStats:
    """Stats computed for a single session."""

    session_id: str
    mean: np.ndarray  # shape: (3,)
    std: np.ndarray
    median: np.ndarray
    mad: np.ndarray
    n_samples: int
    samples: List[List[float]]  # Reservoir samples for aggregation


@dataclass(frozen=True)
class PatchMetadata:
    """Type-safe patch metadata (immutable)."""

    # patch_idx: int
    global_idx: int
    session_id: str
    patient_id: str
    protocol: str
    session_idx: int
    start_frame: int = 0
    class_label: Optional[int] = None
    purity: Optional[float] = None
    validity: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ProcessedBatch:
    """Container for processed data batch."""

    acc_blocks: np.ndarray
    label_blocks: Optional[np.ndarray] = None
    patch_labels: Optional[np.ndarray] = None
    valid_blocks: Optional[np.ndarray] = None
    start_frames: Optional[np.ndarray] = None  # Actual frame start per patch
    metadata: Optional[list["PatchMetadata"]] = None  # Type-safe metadata list
    stats: Optional[SessionStats] = None  # Explicit stats from stats collection
    purity_cache: Optional[dict] = (
        None  # Cache of patch purity values {session_idx: purity}
    )

    def __post_init__(self):
        # Validate shapes
        n_patches = len(self.acc_blocks)
        if self.label_blocks is not None:
            assert len(self.label_blocks) == n_patches, "label_blocks shape mismatch"
        if self.patch_labels is not None:
            assert len(self.patch_labels) == n_patches, "patch_labels shape mismatch"
        if self.valid_blocks is not None:
            assert len(self.valid_blocks) == n_patches, "valid_blocks shape mismatch"

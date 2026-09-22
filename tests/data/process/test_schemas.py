"""Tests for data processing schemas and Pydantic validation."""

# Direct import to avoid circular dependency
# Import the module file directly, not through package __init__.py
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Load schemas module directly
schemas_path = project_root / "src" / "data" / "process" / "schemas.py"
spec = importlib.util.spec_from_file_location("schemas_module", schemas_path)
schemas_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schemas_module)

ProcessingConfig = schemas_module.ProcessingConfig
SessionInfo = schemas_module.SessionInfo
SessionStats = schemas_module.SessionStats
PatchMetadata = schemas_module.PatchMetadata
ProcessedBatch = schemas_module.ProcessedBatch


class TestProcessingConfig:
    """Test ProcessingConfig validation."""

    def test_block_len_too_small(self):
        """Test validation fails for block_len < 10."""
        with pytest.raises(ValidationError, match="block_len"):
            ProcessingConfig(
                block_len=5,
                stride_len=2,
                data_type="labeled"
            )

    def test_block_len_too_large(self):
        """Test validation fails for block_len > 100000."""
        with pytest.raises(ValidationError, match="block_len"):
            ProcessingConfig(
                block_len=200000,
                stride_len=1000,
                data_type="labeled"
            )

    def test_invalid_data_type(self):
        """Test invalid data_type fails pattern validation."""
        with pytest.raises(ValidationError):
            ProcessingConfig(
                block_len=100,
                stride_len=50,
                data_type="invalid_type"
            )


class TestSessionInfo:
    """Test SessionInfo dataclass immutability."""

    def test_session_info_creation(self):
        """Test SessionInfo creation."""
        session = SessionInfo(
            path="/data/protocol_a/sessions",
            filename="session_001.csv",
            protocol="protocol_a",
            id="session_001"
        )
        assert session.id == "session_001"
        assert session.protocol == "protocol_a"

    def test_session_info_frozen(self):
        """Test SessionInfo is immutable (frozen=True)."""
        session = SessionInfo(
            path="/data/protocol_a/sessions",
            filename="session_001.csv",
            protocol="protocol_a",
            id="session_001"
        )
        with pytest.raises(AttributeError):
            session.id = "new_id"

    def test_to_dict(self):
        """Test SessionInfo serialization."""
        session = SessionInfo(
            path="/data/protocol_a/sessions",
            filename="session_001.csv",
            protocol="protocol_a",
            id="session_001"
        )
        result = session.to_dict()
        assert isinstance(result, dict)
        assert result['id'] == "session_001"
        assert result['protocol'] == "protocol_a"


class TestSessionStats:
    """Test SessionStats dataclass."""

    def test_session_stats_creation(self):
        """Test SessionStats creation."""
        stats = SessionStats(
            session_id="session_001",
            mean=np.array([0.1, 0.2, 0.3]),
            std=np.array([0.5, 0.6, 0.7]),
            median=np.array([0.0, 0.1, 0.2]),
            mad=np.array([0.4, 0.5, 0.6]),
            n_samples=1000,
            samples=np.random.randn(100, 3)
        )
        assert stats.session_id == "session_001"
        assert stats.n_samples == 1000
        assert len(stats.mean) == 3

    def test_session_stats_without_samples(self):
        """Test SessionStats without reservoir samples."""
        stats = SessionStats(
            session_id="session_001",
            mean=np.array([0.1, 0.2, 0.3]),
            std=np.array([0.5, 0.6, 0.7]),
            median=np.array([0.0, 0.1, 0.2]),
            mad=np.array([0.4, 0.5, 0.6]),
            n_samples=1000,
            samples=None
        )
        assert stats.samples is None


class TestPatchMetadata:
    """Test PatchMetadata dataclass."""

    def test_patch_metadata_labeled(self):
        """Test PatchMetadata for labeled data."""
        metadata = PatchMetadata(
            global_idx=0,
            session_id="session_001",
            patient_id="patient_a",
            protocol="protocol_a",
            session_idx=0,
            class_label=1,
            purity=1.0
        )
        assert metadata.global_idx == 0
        assert metadata.class_label == 1
        assert metadata.purity == 1.0

    def test_patch_metadata_unlabeled(self):
        """Test PatchMetadata for unlabeled data."""
        metadata = PatchMetadata(
            global_idx=0,
            session_id="session_001",
            patient_id="patient_a",
            protocol="protocol_a",
            session_idx=0,
            class_label=None,
            purity=None
        )
        assert metadata.class_label is None
        assert metadata.purity is None

    def test_patch_metadata_frozen(self):
        """Test PatchMetadata is immutable."""
        metadata = PatchMetadata(
            global_idx=0,
            session_id="session_001",
            patient_id="patient_a",
            protocol="protocol_a",
            session_idx=0
        )
        with pytest.raises(AttributeError):
            metadata.global_idx = 10

    def test_to_dict_excludes_none(self):
        """Test None values excluded from dict."""
        metadata = PatchMetadata(
            global_idx=0,
            session_id="session_001",
            patient_id="patient_a",
            protocol="protocol_a",
            session_idx=0,
            class_label=None,
            purity=None
        )
        result = metadata.to_dict()
        assert 'class_label' not in result
        assert 'purity' not in result
        assert result['global_idx'] == 0


class TestProcessedBatch:
    """Test ProcessedBatch validation."""

    def test_valid_batch_labeled(self):
        """Test valid labeled batch passes validation."""
        batch = ProcessedBatch(
            acc_blocks=np.random.randn(10, 100, 3),
            label_blocks=np.random.randint(0, 2, (10, 100)),
            patch_labels=np.random.randint(0, 2, 10),
            valid_blocks=np.ones((10, 100), dtype=bool)
        )
        assert batch.acc_blocks.shape[0] == 10
        assert batch.label_blocks.shape[0] == 10

    def test_valid_batch_unlabeled(self):
        """Test valid unlabeled batch passes validation."""
        batch = ProcessedBatch(
            acc_blocks=np.random.randn(10, 100, 3),
            label_blocks=None,
            patch_labels=None,
            valid_blocks=None
        )
        assert batch.acc_blocks.shape[0] == 10
        assert batch.label_blocks is None

    def test_shape_mismatch_labels(self):
        """Test __post_init__ catches shape mismatches."""
        with pytest.raises(AssertionError, match="label_blocks"):
            ProcessedBatch(
                acc_blocks=np.random.randn(10, 100, 3),
                label_blocks=np.random.randint(0, 2, (5, 100)),  # Wrong first dim
                patch_labels=np.random.randint(0, 2, 10),
                valid_blocks=np.ones((10, 100), dtype=bool)
            )

    def test_shape_mismatch_patch_labels(self):
        """Test shape mismatch for patch_labels."""
        with pytest.raises(AssertionError, match="patch_labels"):
            ProcessedBatch(
                acc_blocks=np.random.randn(10, 100, 3),
                label_blocks=np.random.randint(0, 2, (10, 100)),
                patch_labels=np.random.randint(0, 2, 5),  # Wrong length
                valid_blocks=np.ones((10, 100), dtype=bool)
            )

    def test_valid_blocks_first_dim(self):
        """Test valid_blocks first dimension validated."""
        with pytest.raises(AssertionError, match="valid_blocks"):
            ProcessedBatch(
                acc_blocks=np.random.randn(10, 100, 3),
                label_blocks=np.random.randint(0, 2, (10, 100)),
                patch_labels=np.random.randint(0, 2, 10),
                valid_blocks=np.ones((5, 100), dtype=bool)  # Wrong first dim
            )

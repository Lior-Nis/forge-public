"""Shared pytest fixtures for data pipeline tests."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from typing import Tuple


# ============================================================================
# Helper Functions
# ============================================================================

def generate_accelerometer_data(
    n_samples: int = 200,
    n_channels: int = 3,
    with_labels: bool = True,
    class_pattern: str = "alternating"
) -> pd.DataFrame:
    """Generate synthetic accelerometer data with realistic patterns.

    Args:
        n_samples: Number of timesteps to generate
        n_channels: Number of accelerometer channels (default 3: V, ML, AP)
        with_labels: Whether to include Label and Valid columns
        class_pattern: How to generate labels ("alternating", "blocks", "single")

    Returns:
        DataFrame with columns [AccV, AccML, AccAP] and optionally [Label, Valid]
    """
    # Generate sinusoidal patterns with noise for realistic accelerometer data
    t = np.linspace(0, 10, n_samples)

    # Create different frequency patterns for each channel
    data = {
        'AccV': 0.5 * np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(n_samples),
        'AccML': 0.3 * np.sin(2 * np.pi * 1.5 * t) + 0.1 * np.random.randn(n_samples),
        'AccAP': 0.4 * np.sin(2 * np.pi * 0.8 * t) + 0.1 * np.random.randn(n_samples),
    }

    if with_labels:
        # Generate labels based on pattern
        if class_pattern == "alternating":
            # Alternate between 0 and 1 every 50 samples
            labels = np.array([i // 50 % 2 for i in range(n_samples)])
        elif class_pattern == "blocks":
            # First half class 0, second half class 1
            labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
        elif class_pattern == "single":
            # All same class
            labels = np.zeros(n_samples, dtype=int)
        else:
            labels = np.zeros(n_samples, dtype=int)

        data['Label'] = labels
        # Valid mask (all valid for synthetic data)
        data['Valid'] = np.ones(n_samples, dtype=int)

    return pd.DataFrame(data)


def generate_session_stats(session_id: str, data: np.ndarray):
    """Generate SessionStats from accelerometer data.

    Args:
        session_id: Session identifier
        data: Numpy array of shape (n_samples, n_channels)

    Returns:
        SessionStats with computed statistics
    """
    from data.process.stats.computer import OnlineStatsComputer
    from data.process.schemas import SessionStats

    computer = OnlineStatsComputer()
    computer.update(data)
    stats_dict = computer.get_stats()

    return SessionStats(
        session_id=session_id,
        mean=stats_dict['mean'],
        std=stats_dict['std'],
        median=stats_dict['median'],
        mad=stats_dict['mad'],
        n_samples=len(data),
        samples=stats_dict.get('samples', None)
    )


# ============================================================================
# Session-Scoped Fixtures (Expensive, Reused Across Tests)
# ============================================================================

@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    """Create temporary directory for all test fixtures (session-scoped)."""
    return tmp_path_factory.mktemp("fixtures")


@pytest.fixture(scope="session")
def raw_csv_data() -> pd.DataFrame:
    """Generate synthetic accelerometer CSV data for labeled datasets.

    Returns 200 timesteps, 3 channels, with labels and valid masks.
    """
    return generate_accelerometer_data(n_samples=200, with_labels=True)


@pytest.fixture(scope="session")
def raw_parquet_data() -> pd.DataFrame:
    """Generate synthetic accelerometer Parquet data for unlabeled datasets.

    Returns 200 timesteps, 3 channels, no labels.
    """
    return generate_accelerometer_data(n_samples=200, with_labels=False)


@pytest.fixture(scope="session")
def metadata_df() -> pd.DataFrame:
    """Generate metadata DataFrame for test sessions.

    Returns DataFrame mapping session_id → patient_id and protocol.
    """
    return pd.DataFrame({
        'Id': ['session_001', 'session_002', 'session_003'],
        'Subject': ['patient_a', 'patient_a', 'patient_b'],
        'protocol': ['protocol_a', 'protocol_a', 'protocol_b']
    })


@pytest.fixture(scope="session")
def raw_data_dir(
    fixture_dir: Path,
    raw_csv_data: pd.DataFrame,
    raw_parquet_data: pd.DataFrame,
    metadata_df: pd.DataFrame
) -> Path:
    """Create complete raw data directory structure on disk.

    Creates:
    - protocol_a/metadata.csv
    - protocol_a/sessions/session_001.csv
    - protocol_a/sessions/session_002.csv
    - protocol_b/metadata.csv
    - protocol_b/sessions/session_003.parquet

    Returns:
        Path to raw data directory
    """
    raw_dir = fixture_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    # Create protocol_a directory
    protocol_a_dir = raw_dir / "protocol_a"
    protocol_a_dir.mkdir(exist_ok=True)

    # Write protocol_a metadata
    protocol_a_metadata = metadata_df[metadata_df['protocol'] == 'protocol_a']
    protocol_a_metadata.to_csv(protocol_a_dir / "metadata.csv", index=False)

    # Create protocol_a sessions directory
    sessions_a_dir = protocol_a_dir / "sessions"
    sessions_a_dir.mkdir(exist_ok=True)

    # Write session CSV files
    raw_csv_data.to_csv(sessions_a_dir / "session_001.csv", index=False)
    raw_csv_data.to_csv(sessions_a_dir / "session_002.csv", index=False)

    # Create protocol_b directory
    protocol_b_dir = raw_dir / "protocol_b"
    protocol_b_dir.mkdir(exist_ok=True)

    # Write protocol_b metadata
    protocol_b_metadata = metadata_df[metadata_df['protocol'] == 'protocol_b']
    protocol_b_metadata.to_csv(protocol_b_dir / "metadata.csv", index=False)

    # Create protocol_b sessions directory
    sessions_b_dir = protocol_b_dir / "sessions"
    sessions_b_dir.mkdir(exist_ok=True)

    # Write session Parquet file
    raw_parquet_data.to_parquet(sessions_b_dir / "session_003.parquet", index=False)

    return raw_dir


# ============================================================================
# Function-Scoped Fixtures (Fast, Isolated)
# ============================================================================

@pytest.fixture(scope="function")
def processing_config():
    """Valid ProcessingConfig for testing."""
    from data.process.schemas import ProcessingConfig

    return ProcessingConfig(
        block_len=100,
        stride_len=50,
        data_type="labeled",
        pure_patches_only=False
    )


@pytest.fixture(scope="function")
def zarr_store_path(tmp_path: Path) -> Path:
    """Temporary path for Zarr store."""
    return tmp_path / "test.zarr"


@pytest.fixture(scope="function")
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for processed data."""
    output_dir = tmp_path / "processed"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (integration tests)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests"
    )

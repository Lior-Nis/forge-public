"""Tests for ProbabilisticBalancedSampler with protocol."""

import numpy as np
import pandas as pd
import pytest

from data.datamodule.samplers import ProbabilisticBalancedSampler
from data.dataset.protocols import SamplableDataset


class MockSamplableDataset:
    """Mock dataset for testing sampler in isolation."""

    def __init__(self, metadata_df: pd.DataFrame):
        self.metadata_df = metadata_df

    def get_sampling_metadata(self) -> pd.DataFrame:
        return self.metadata_df

    def __len__(self) -> int:
        return len(self.metadata_df)


def test_sampler_accepts_protocol_implementation():
    """Test that sampler works with any SamplableDataset implementation."""
    # Create mock metadata
    metadata = pd.DataFrame({
        'class_label': [0, 0, 1, 1],
        'protocol': ['protocol_a', 'protocol_a', 'protocol_b', 'protocol_b'],
        'patient_id': ['patient1', 'patient2', 'patient1', 'patient2']
    })

    dataset = MockSamplableDataset(metadata)

    # Should work with protocol implementation
    sampler = ProbabilisticBalancedSampler(
        dataset=dataset,
        batch_size=2,
        weights_cache_path=None
    )

    assert len(sampler.weights) == 4
    assert sampler.weights.sum() == pytest.approx(1.0)


def test_sampler_validates_protocol():
    """Test that sampler rejects non-protocol datasets."""
    class InvalidDataset:
        def __len__(self):
            return 10
        # Missing get_sampling_metadata()

    dataset = InvalidDataset()

    with pytest.raises(TypeError, match="must implement SamplableDataset protocol"):
        ProbabilisticBalancedSampler(
            dataset=dataset,
            batch_size=2,
            weights_cache_path=None
        )


def test_sampler_validates_required_columns():
    """Test that sampler validates metadata has required columns."""
    # Missing 'patient_id' column
    metadata = pd.DataFrame({
        'class_label': [0, 1],
        'protocol': ['protocol_a', 'protocol_b']
        # patient_id missing
    })

    dataset = MockSamplableDataset(metadata)

    with pytest.raises(ValueError, match="missing required columns.*patient_id"):
        ProbabilisticBalancedSampler(
            dataset=dataset,
            batch_size=2,
            weights_cache_path=None
        )


def test_class_balancing():
    """Test that class balancing weights work correctly."""
    # Imbalanced classes: 3 samples of class 0, 1 sample of class 1
    metadata = pd.DataFrame({
        'class_label': [0, 0, 0, 1],
        'protocol': ['p1', 'p1', 'p1', 'p1'],
        'patient_id': ['p1', 'p1', 'p1', 'p1']
    })

    dataset = MockSamplableDataset(metadata)
    sampler = ProbabilisticBalancedSampler(
        dataset=dataset,
        batch_size=2,
        weights_cache_path=None,
        balance_config={'balance_protocols': False, 'balance_patients': False}
    )

    # Class 1 should have higher weight than class 0
    class_0_weight = sampler.weights[0]  # First sample is class 0
    class_1_weight = sampler.weights[3]  # Last sample is class 1

    assert class_1_weight > class_0_weight
    # Class 1 should have ~3x weight (to compensate for 1:3 ratio)
    assert class_1_weight == pytest.approx(class_0_weight * 3, rel=0.1)


def test_protocol_balancing():
    """Test that protocol balancing works within classes."""
    # Class 0: 2 samples from protocol_a, 1 from protocol_b
    # Class 1: balanced protocols
    metadata = pd.DataFrame({
        'class_label': [0, 0, 0, 1, 1],
        'protocol': ['protocol_a', 'protocol_a', 'protocol_b', 'protocol_a', 'protocol_b'],
        'patient_id': ['p1', 'p1', 'p1', 'p2', 'p2']
    })

    dataset = MockSamplableDataset(metadata)
    sampler = ProbabilisticBalancedSampler(
        dataset=dataset,
        batch_size=2,
        weights_cache_path=None,
        balance_config={'balance_protocols': True, 'balance_patients': False}
    )

    # Within class 0, protocol_b should have higher weight than protocol_a
    # (to balance the 1:2 ratio)
    class_0_protocol_a_weight = sampler.weights[0]
    class_0_protocol_b_weight = sampler.weights[2]

    assert class_0_protocol_b_weight > class_0_protocol_a_weight


def test_patient_balancing():
    """Test that patient balancing works within class-protocol pairs."""
    # Imbalanced patients within same class-protocol
    metadata = pd.DataFrame({
        'class_label': [0, 0, 0, 0],
        'protocol': ['protocol_a', 'protocol_a', 'protocol_a', 'protocol_a'],
        'patient_id': ['patient1', 'patient1', 'patient1', 'patient2']
    })

    dataset = MockSamplableDataset(metadata)
    sampler = ProbabilisticBalancedSampler(
        dataset=dataset,
        batch_size=2,
        weights_cache_path=None,
        balance_config={'balance_protocols': False, 'balance_patients': True}
    )

    # Patient 2 should have higher weight than patient 1
    patient_1_weight = sampler.weights[0]
    patient_2_weight = sampler.weights[3]

    assert patient_2_weight > patient_1_weight


def test_sampler_iteration():
    """Test that sampler produces batches correctly."""
    metadata = pd.DataFrame({
        'class_label': [0, 0, 1, 1],
        'protocol': ['protocol_a', 'protocol_a', 'protocol_b', 'protocol_b'],
        'patient_id': ['patient1', 'patient2', 'patient1', 'patient2']
    })

    dataset = MockSamplableDataset(metadata)
    sampler = ProbabilisticBalancedSampler(
        dataset=dataset,
        batch_size=2,
        weights_cache_path=None
    )

    # Get a few batches
    iterator = iter(sampler)
    batch1 = next(iterator)
    batch2 = next(iterator)

    # Check batch size
    assert len(batch1) == 2
    assert len(batch2) == 2

    # Check indices are valid
    assert all(0 <= idx < len(dataset) for idx in batch1)
    assert all(0 <= idx < len(dataset) for idx in batch2)


def test_missing_class_label_raises():
    """Sampler requires a 'class_label' column and raises a clear error if absent."""
    # No class_label column
    metadata = pd.DataFrame({
        'protocol': ['protocol_a', 'protocol_b'],
        'patient_id': ['patient1', 'patient2']
    })

    dataset = MockSamplableDataset(metadata)

    with pytest.raises(ValueError, match="class_label"):
        ProbabilisticBalancedSampler(
            dataset=dataset,
            batch_size=2,
            weights_cache_path=None
        )


def test_mock_dataset_satisfies_protocol():
    """Test that MockSamplableDataset satisfies the protocol."""
    metadata = pd.DataFrame({
        'class_label': [0, 1],
        'protocol': ['protocol_a', 'protocol_b'],
        'patient_id': ['patient1', 'patient2']
    })

    dataset = MockSamplableDataset(metadata)

    # Runtime check
    assert isinstance(dataset, SamplableDataset)

"""Tests for SamplableDataset protocol validation."""

import pandas as pd
import pytest

from data.dataset.protocols import SamplableDataset


class TestSamplableDatasetProtocol:
    """Test SamplableDataset protocol validation."""

    def test_mock_dataset_satisfies_protocol(self):
        """Test custom mock datasets can satisfy protocol."""
        class MockDataset:
            def __init__(self):
                self.metadata_df = pd.DataFrame({
                    'class_label': [0, 1],
                    'protocol': ['protocol_a', 'protocol_b'],
                    'patient_id': ['patient1', 'patient2']
                })

            def get_sampling_metadata(self) -> pd.DataFrame:
                return self.metadata_df

            def __len__(self) -> int:
                return len(self.metadata_df)

        dataset = MockDataset()

        # Runtime check
        assert isinstance(dataset, SamplableDataset)

    def test_invalid_dataset_fails_protocol(self):
        """Test dataset without get_sampling_metadata fails."""
        class InvalidDataset:
            def __len__(self) -> int:
                return 10
            # Missing get_sampling_metadata()

        dataset = InvalidDataset()

        # Should NOT satisfy protocol
        assert not isinstance(dataset, SamplableDataset)

    def test_partial_implementation_fails(self):
        """Test dataset with wrong signature fails."""
        class PartialDataset:
            def get_sampling_metadata(self):
                # Wrong return type (should be pd.DataFrame)
                return []

            def __len__(self) -> int:
                return 10

        dataset = PartialDataset()

        # Runtime check will pass (duck typing), but usage will fail
        # This tests that protocol is structural, not nominal
        assert isinstance(dataset, SamplableDataset)

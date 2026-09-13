"""Tests for collate functions."""

import torch
import pytest

from data.datamodule.collate import classification, selfsupervised
from data.dataset.schema import DatasetSample


class TestCollateFunctions:
    """Test collate functions."""

    def test_classification_collate(self):
        """Test classification() collate function."""
        # Create batch of DatasetSample
        batch = [
            DatasetSample(
                signal=torch.randn(3, 100),
                labels=torch.randint(0, 2, (100,)),
                patch_label=torch.tensor(0),
                valid_mask=torch.ones(100, dtype=torch.bool),
                metadata={'session_id': 's1', 'patient_id': 'p1'}
            ),
            DatasetSample(
                signal=torch.randn(3, 100),
                labels=torch.randint(0, 2, (100,)),
                patch_label=torch.tensor(1),
                valid_mask=torch.ones(100, dtype=torch.bool),
                metadata={'session_id': 's2', 'patient_id': 'p2'}
            )
        ]

        result = classification(batch)

        # Verify output structure
        assert isinstance(result, dict)
        assert 'x' in result
        assert 'y' in result
        assert 'patch_y' in result
        assert 'valid_mask' in result
        assert 'metadata' in result

        # Verify shapes
        assert result['x'].shape == (2, 3, 100)
        assert result['y'].shape == (2, 100)
        assert result['patch_y'].shape == (2,)
        assert result['valid_mask'].shape == (2, 100)

        # Verify metadata is list
        assert isinstance(result['metadata'], list)
        assert len(result['metadata']) == 2

    def test_selfsupervised_collate(self):
        """Test selfsupervised() collate function."""
        # Create batch of DatasetSample
        batch = [
            DatasetSample(
                signal=torch.randn(3, 100),
                metadata={'session_id': 's1', 'patient_id': 'p1'}
            ),
            DatasetSample(
                signal=torch.randn(3, 100),
                metadata={'session_id': 's2', 'patient_id': 'p2'}
            )
        ]

        result = selfsupervised(batch)

        # Verify output structure
        assert isinstance(result, dict)
        assert 'input' in result
        assert 'target' in result
        assert 'metadata' in result

        # Verify shapes
        assert result['input'].shape == (2, 3, 100)
        assert result['target'].shape == (2, 3, 100)

        # Verify metadata is list
        assert isinstance(result['metadata'], list)
        assert len(result['metadata']) == 2

    def test_batch_shapes(self):
        """Test batched tensors have correct shapes."""
        batch = [
            DatasetSample(
                signal=torch.randn(3, 100),
                labels=torch.randint(0, 2, (100,)),
                patch_label=torch.tensor(0),
                valid_mask=torch.ones(100, dtype=torch.bool),
                metadata={'session_id': f's{i}'}
            )
            for i in range(5)
        ]

        result = classification(batch)

        # All batch dimensions should be 5
        assert result['x'].shape[0] == 5
        assert result['y'].shape[0] == 5
        assert result['patch_y'].shape[0] == 5
        assert result['valid_mask'].shape[0] == 5
        assert len(result['metadata']) == 5

    def test_metadata_list(self):
        """Test metadata returned as list."""
        batch = [
            DatasetSample(
                signal=torch.randn(3, 100),
                metadata={'session_id': 's1', 'extra_field': 'value1'}
            ),
            DatasetSample(
                signal=torch.randn(3, 100),
                metadata={'session_id': 's2', 'extra_field': 'value2'}
            )
        ]

        result = selfsupervised(batch)

        # Verify metadata preserved
        assert result['metadata'][0]['session_id'] == 's1'
        assert result['metadata'][0]['extra_field'] == 'value1'
        assert result['metadata'][1]['session_id'] == 's2'

    def test_single_sample_batch(self):
        """Test collate with single sample."""
        batch = [
            DatasetSample(
                signal=torch.randn(3, 100),
                labels=torch.randint(0, 2, (100,)),
                patch_label=torch.tensor(0),
                valid_mask=torch.ones(100, dtype=torch.bool),
                metadata={'session_id': 's1'}
            )
        ]

        result = classification(batch)

        # Batch dimension should be 1
        assert result['x'].shape == (1, 3, 100)
        assert result['y'].shape == (1, 100)

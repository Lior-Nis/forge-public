"""Tests for SimCLR logging manager."""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from managers.logging.simclr import SimCLRLoggingManager


class TestSimCLRLoggingManager:
    """Test suite for SimCLR logging manager."""

    def test_initialization(self, mock_simclr_config, mock_trainer):
        """Test SimCLR logging manager can be initialized without errors."""
        manager = SimCLRLoggingManager(
            config=mock_simclr_config,
            device="cpu",
            trainer=mock_trainer
        )

        assert manager.representation_interval == 1
        assert manager.max_samples_per_epoch == 5
        assert isinstance(manager.validation_embeddings, list)
        assert isinstance(manager.test_embeddings, list)
        assert isinstance(manager.contrastive_statistics, dict)

    def test_embedding_visualization_uses_z1_z2(self, mock_simclr_config, mock_trainer):
        """Test that embedding visualization uses z1/z2 keys correctly (Bug 1 fix)."""
        manager = SimCLRLoggingManager(
            config=mock_simclr_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Simulate data accumulation with z1/z2 structure
        sample_data = {
            "z1": torch.randn(1, 128),
            "z2": torch.randn(1, 128),
            "metadata": [{"patient_id": "P001"}]
        }
        manager.validation_embeddings.append(sample_data)
        manager.validation_embeddings.append(sample_data)  # Need at least 2 for t-SNE

        # Mock matplotlib and sklearn
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('sklearn.manifold.TSNE') as mock_tsne, \
             patch('wandb.Image') as mock_wandb_image:

            # Setup mocks
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            mock_reducer = Mock()
            mock_reducer.fit_transform = Mock(return_value=np.random.randn(2, 2))
            mock_tsne.return_value = mock_reducer

            mock_wandb_image.return_value = "mock_image"

            # Should not raise KeyError
            fig = manager._create_embedding_visualization(stage="val", epoch=0)

            # Verify it used z1 data
            assert fig is not None or len(manager.validation_embeddings) < 2

    def test_embedding_visualization_min_samples(self, mock_simclr_config, mock_trainer):
        """Test that visualization handles <2 samples gracefully (Bug 3 fix)."""
        manager = SimCLRLoggingManager(
            config=mock_simclr_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Single sample - should return None with warning
        single_sample = {
            "z1": torch.randn(1, 128),
            "z2": torch.randn(1, 128),
            "metadata": [{}]
        }
        manager.validation_embeddings.append(single_sample)

        # Should return None for <2 samples
        fig = manager._create_embedding_visualization(stage="val", epoch=0)
        assert fig is None

        # Add second sample - should work now
        manager.validation_embeddings.append(single_sample)

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('sklearn.manifold.TSNE') as mock_tsne, \
             patch('wandb.Image'):

            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            mock_reducer = Mock()
            mock_reducer.fit_transform = Mock(return_value=np.random.randn(2, 2))
            mock_tsne.return_value = mock_reducer

            fig = manager._create_embedding_visualization(stage="val", epoch=0)
            assert fig is not None

    def test_perplexity_calculation_safe(self, mock_simclr_config, mock_trainer):
        """Test that perplexity uses max(1, ...) for very small datasets."""
        manager = SimCLRLoggingManager(
            config=mock_simclr_config,
            device="cpu",
            trainer=mock_trainer
        )

        # 2 samples - perplexity should be max(1, min(30, 1)) = 1
        for _ in range(2):
            sample = {
                "z1": torch.randn(1, 128),
                "z2": torch.randn(1, 128),
                "metadata": [{}]
            }
            manager.validation_embeddings.append(sample)

        with patch('sklearn.manifold.TSNE') as mock_tsne, \
             patch('matplotlib.pyplot.subplots'), \
             patch('wandb.Image'):

            mock_reducer = Mock()
            mock_reducer.fit_transform = Mock(return_value=np.random.randn(2, 2))
            mock_tsne.return_value = mock_reducer

            manager._create_embedding_visualization(stage="val", epoch=0)

            # Verify TSNE was called with perplexity=1 (not 0 or negative)
            call_kwargs = mock_tsne.call_args[1]
            assert call_kwargs['perplexity'] >= 1

    def test_patch_table_has_sequential_index(self, mock_simclr_config, mock_trainer):
        """Test that patch tables use reset_index for WandB compatibility (Bug 5 fix)."""
        manager = SimCLRLoggingManager(
            config=mock_simclr_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Create patch data
        patch_data = [
            {"patient_id": "P001", "loss": 0.5},
            {"patient_id": "P002", "loss": 0.3},
        ]
        manager.validation_patch_data = patch_data

        # Mock WandB - patch where it's imported (locally in the method)
        with patch('wandb.Table') as mock_wandb_table, \
             patch('pandas.DataFrame') as mock_df_class:

            mock_df = Mock()
            mock_df.reset_index = Mock(return_value=mock_df)
            mock_df_class.return_value = mock_df

            mock_wandb_table.return_value = "mock_table"

            manager.log_patch_tables(stage="val")

            # Verify reset_index was called with drop=True
            mock_df.reset_index.assert_called_once_with(drop=True)

    def test_contrastive_analysis_checks_negative_similarities(self, mock_simclr_config, mock_trainer):
        """Test that contrastive analysis checks for empty negative_similarities (Bug 7 fix)."""
        manager = SimCLRLoggingManager(
            config=mock_simclr_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Add only positive similarities
        manager.contrastive_statistics["positive_similarities"] = [0.8, 0.9, 0.85]
        manager.contrastive_statistics["negative_similarities"] = []  # Empty!

        # Should return None with warning, not crash
        fig = manager._create_contrastive_analysis_plots(epoch=0)
        assert fig is None

    def test_temperature_analysis_filters_empty_data(self, mock_simclr_config, mock_trainer):
        """Test that temperature analysis filters out empty data (Bug 8 fix)."""
        manager = SimCLRLoggingManager(
            config=mock_simclr_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Add temperature data with some empty entries
        manager.contrastive_statistics["temperature_effects"] = {
            0.1: [0.5, 0.6, 0.55],  # Valid
            0.5: [],                 # Empty - should be filtered
            1.0: [0.3, 0.35]         # Valid
        }

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('wandb.Image'):

            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            fig = manager._create_temperature_analysis_plots(epoch=0)

            # Should create plot with only valid temperatures
            assert fig is not None

            # Verify errorbar was called with only valid temps (0.1 and 1.0)
            call_args = mock_ax.errorbar.call_args
            temps = call_args[0][0]
            assert len(temps) == 2  # Only 2 valid temperatures
            assert 0.5 not in temps  # Empty one filtered out

    def test_temperature_analysis_returns_none_if_all_empty(self, mock_simclr_config, mock_trainer):
        """Test that temperature analysis returns None if all data is empty."""
        manager = SimCLRLoggingManager(
            config=mock_simclr_config,
            device="cpu",
            trainer=mock_trainer
        )

        # All empty
        manager.contrastive_statistics["temperature_effects"] = {
            0.1: [],
            0.5: [],
            1.0: []
        }

        fig = manager._create_temperature_analysis_plots(epoch=0)
        assert fig is None

    def test_loss_extraction_fails_with_bad_shape(self, mock_simclr_config, mock_trainer):
        """Test that loss extraction raises ValueError for unexpected shapes (Bug 11 fix)."""
        manager = SimCLRLoggingManager(
            config=mock_simclr_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Mock data with bad loss shape
        losses = torch.randn(2, 3)  # Wrong shape!
        metadata = [{"patient_id": "P001"}, {"patient_id": "P002"}]

        mock_data = Mock()
        mock_data.losses = losses
        mock_data.metadata = metadata

        # Should raise ValueError, not silently fallback
        with pytest.raises(ValueError, match="Unexpected loss shape"):
            manager.accumulate_patches(mock_data, stage="val")

    def test_representation_samples_storage(self, mock_simclr_config, mock_trainer):
        """Test that representation samples are stored with correct structure."""
        manager = SimCLRLoggingManager(
            config=mock_simclr_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Create mock batch data
        mock_data = Mock()
        mock_data.z1 = torch.randn(4, 128)
        mock_data.z2 = torch.randn(4, 128)
        mock_data.metadata = [{"patient_id": f"P{i:03d}"} for i in range(4)]

        # Log samples
        manager.log_representation_samples(mock_data, stage="val")

        assert len(manager.validation_embeddings) == 1
        sample = manager.validation_embeddings[0]
        assert "z1" in sample
        assert "z2" in sample
        assert "metadata" in sample
        assert sample["z1"].shape[0] == 1  # First sample only

    def test_clear_validation_data(self, mock_simclr_config, mock_trainer):
        """Test that validation data is cleared correctly."""
        manager = SimCLRLoggingManager(
            config=mock_simclr_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Add some data
        manager.validation_embeddings.append({"z1": torch.randn(1, 128)})
        manager.validation_patch_data.append({"patch": "data"})

        # Clear
        manager.clear_validation_data()

        assert len(manager.validation_embeddings) == 0
        assert len(manager.validation_patch_data) == 0

    def test_contrastive_statistics_tracking(self, mock_simclr_config, mock_trainer):
        """Test that contrastive statistics are tracked correctly."""
        manager = SimCLRLoggingManager(
            config=mock_simclr_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Log statistics
        manager.log_contrastive_statistics(pos_sim=0.85, neg_sim=0.15, separation=0.70)
        manager.log_temperature_effect(temperature=0.1, separation_score=0.75)

        assert len(manager.contrastive_statistics["positive_similarities"]) == 1
        assert len(manager.contrastive_statistics["negative_similarities"]) == 1
        assert len(manager.contrastive_statistics["separation_metrics"]) == 1
        assert 0.1 in manager.contrastive_statistics["temperature_effects"]
        assert manager.contrastive_statistics["temperature_effects"][0.1] == [0.75]

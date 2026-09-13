"""Tests for MAE logging manager."""
import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from managers.logging.mae import MAELoggingManager


class TestMAELoggingManager:
    """Test suite for MAE logging manager."""

    def test_initialization(self, mock_mae_config, mock_trainer):
        """Test MAE logging manager can be initialized without errors."""
        manager = MAELoggingManager(
            config=mock_mae_config,
            device="cpu",
            trainer=mock_trainer
        )

        assert manager.patch_size == 10
        assert manager.reconstruction_interval == 5
        assert manager.max_samples_per_epoch == 5
        assert isinstance(manager.validation_reconstructions, list)
        assert isinstance(manager.test_reconstructions, list)

    def test_reconstruction_sample_accumulation(self, mock_mae_config, mock_trainer):
        """Test that reconstruction samples are accumulated correctly."""
        manager = MAELoggingManager(
            config=mock_mae_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Create sample data
        original = torch.randn(2, 3, 1, 200)
        reconstructed = torch.randn(2, 3, 1, 200)
        mask = torch.randint(0, 2, (2, 20)).bool()

        # Log sample
        manager.log_reconstruction_sample(original, reconstructed, mask, stage="val")

        assert len(manager.validation_reconstructions) == 1
        sample = manager.validation_reconstructions[0]
        assert "original" in sample
        assert "reconstructed" in sample
        assert "mask" in sample
        assert sample["original"].shape[0] == 1  # First sample only

    @pytest.mark.parametrize("shape,expected_n_cols", [
        ((1, 3, 1, 200), 4),    # H=1 raw signal → C+1 = 4 columns
        ((1, 3, 200), 4),       # 2D time-series, normalized to H=1 → C+1 = 4 columns
        ((1, 3, 128, 200), 3),  # H>1 spectrogram → 3 columns
    ])
    def test_reconstruction_visualization_shapes(self, mock_mae_config, mock_trainer, shape, expected_n_cols):
        """Test that reconstruction visualization handles different shapes and column counts."""
        manager = MAELoggingManager(
            config=mock_mae_config,
            device="cpu",
            trainer=mock_trainer
        )

        sample = {
            "original": torch.randn(*shape),
            "reconstructed": torch.randn(*shape),
            "mask": None,
            "loss": 0.042,
        }
        manager.validation_reconstructions.append(sample)

        # Use side_effect to create properly-sized mock axes matching the subplot call
        def make_subplots(*args, **kwargs):
            nrows = args[0] if args else kwargs.get('nrows', 1)
            ncols = args[1] if len(args) > 1 else kwargs.get('ncols', 1)
            mock_axes = np.array([[Mock() for _ in range(ncols)] for _ in range(nrows)])
            return Mock(), mock_axes

        with patch('matplotlib.pyplot.subplots', side_effect=make_subplots), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.suptitle'), \
             patch('matplotlib.pyplot.colorbar'), \
             patch('wandb.Image', return_value="mock_image"):

            fig = manager._create_reconstruction_visualization(stage="val", epoch=0)
            # Method must succeed (not swallow an IndexError from wrong column count)
            assert fig is not None, (
                f"Visualization returned None for shape {shape} — likely an IndexError "
                f"from wrong column count (expected {expected_n_cols} cols)"
            )

    def test_patch_table_has_sequential_index(self, mock_mae_config, mock_trainer):
        """Test that patch tables use reset_index for WandB compatibility."""
        manager = MAELoggingManager(
            config=mock_mae_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Create patch data with potential gaps
        patch_data = [
            {"patient_id": "P001", "loss": 0.5},
            {"patient_id": "P002", "loss": 0.3},
        ]
        manager.validation_patch_data = patch_data

        # Mock WandB - patch where it's imported (locally in the method)
        with patch('wandb.Table') as mock_wandb_table, \
             patch('pandas.DataFrame') as mock_df_class:

            # Create mock DataFrame with non-sequential index
            mock_df = Mock()
            mock_df.reset_index = Mock(return_value=mock_df)
            mock_df_class.return_value = mock_df

            mock_wandb_table.return_value = "mock_table"

            # Call log_patch_tables
            manager.log_patch_tables(stage="val")

            # Verify reset_index was called with drop=True
            mock_df.reset_index.assert_called_once_with(drop=True)

    def test_loss_extraction_fails_with_bad_shape(self, mock_mae_config, mock_trainer):
        """Test that loss extraction raises ValueError for unexpected shapes."""
        manager = MAELoggingManager(
            config=mock_mae_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Create mock batch data with bad loss shape
        from pipeline.schemas import MAEBatchLogData

        # Mock data with 2D loss (unexpected)
        losses = torch.randn(2, 3)  # Wrong shape!
        metadata = [{"patient_id": "P001"}, {"patient_id": "P002"}]

        mock_data = Mock()
        mock_data.losses = losses
        mock_data.metadata = metadata

        # Should raise ValueError, not silently fallback
        with pytest.raises(ValueError, match="Unexpected loss shape"):
            manager.accumulate_patches(mock_data, stage="val")

    def test_clear_validation_data(self, mock_mae_config, mock_trainer):
        """Test that validation data is cleared correctly."""
        manager = MAELoggingManager(
            config=mock_mae_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Add some data
        manager.validation_reconstructions.append({"sample": "data"})
        manager.validation_patch_data.append({"patch": "data"})

        # Clear
        manager.clear_validation_data()

        assert len(manager.validation_reconstructions) == 0
        assert len(manager.validation_patch_data) == 0

    def test_mask_statistics_tracking(self, mock_mae_config, mock_trainer):
        """Test that mask statistics are tracked correctly."""
        manager = MAELoggingManager(
            config=mock_mae_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Create binary mask
        mask = torch.tensor([True, False, True, True, False])  # 60% masked
        reconstruction_quality = 0.85

        # Log statistics
        manager.log_mask_statistics(mask, reconstruction_quality)

        assert len(manager.mask_statistics["mask_ratios"]) == 1
        assert abs(manager.mask_statistics["mask_ratios"][0] - 0.6) < 0.01

        # Check reconstruction quality is binned
        assert 0.6 in manager.mask_statistics["reconstruction_quality_by_mask"]
        assert reconstruction_quality in manager.mask_statistics["reconstruction_quality_by_mask"][0.6]

    def test_spectral_reconstructions_removed(self, mock_mae_config, mock_trainer):
        """Test that dead log_spectral_reconstructions method was removed."""
        manager = MAELoggingManager(
            config=mock_mae_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Method should not exist
        assert not hasattr(manager, 'log_spectral_reconstructions')
        assert not hasattr(manager, '_create_mae_spectral_plot')
        assert not hasattr(manager, '_add_reconstruction_frames')
        assert not hasattr(manager, '_get_robust_colorbar_limits')

    def test_cleanup_matplotlib_uses_exception(self, mock_mae_config, mock_trainer):
        """Test that _cleanup_matplotlib_memory uses 'except Exception' not bare except."""
        manager = MAELoggingManager(
            config=mock_mae_config,
            device="cpu",
            trainer=mock_trainer
        )

        # Read the source to check exception type
        import inspect
        source = inspect.getsource(manager._cleanup_matplotlib_memory)

        # Should have 'except Exception:', not bare 'except:'
        assert 'except Exception:' in source
        assert 'except:\n' not in source  # No bare except

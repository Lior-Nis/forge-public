"""Pytest fixtures for manager tests."""
import pytest
from unittest.mock import Mock, MagicMock
from typing import Any, Dict


@pytest.fixture
def mock_wandb_logger():
    """Mock WandB logger for testing."""
    logger = Mock()
    logger.experiment = Mock()
    logger.experiment.log = Mock()
    logger.current_epoch = 0
    return logger


@pytest.fixture
def mock_trainer(mock_wandb_logger):
    """Mock PyTorch Lightning trainer."""
    trainer = Mock()
    trainer.logger = mock_wandb_logger
    trainer.global_step = 0
    trainer.current_epoch = 0
    return trainer


@pytest.fixture
def mock_mae_config():
    """Mock MAE configuration for testing."""
    config = Mock()

    # Model config
    config.model = Mock()
    config.model.backbone = Mock()
    config.model.backbone.patch_size = 10
    config.model.backbone._target_ = "model.backbones.FogFormer"

    # Data config
    config.data = Mock()
    config.data.process = Mock()
    config.data.process.lengths = Mock()
    config.data.process.lengths.block_len = 200
    config.data.dataset = Mock()
    config.data.dataset.seq_len = 200

    # Training config
    config.train = Mock()
    config.train.logging = Mock()
    config.train.logging.reconstruction_interval = 5
    config.train.logging.representation_interval = 1

    return config


@pytest.fixture
def mock_simclr_config():
    """Mock SimCLR configuration for testing."""
    config = Mock()

    # Model config
    config.model = Mock()
    config.model.backbone = Mock()
    config.model.backbone._target_ = "model.backbones.FogFormer"

    # Data config
    config.data = Mock()
    config.data.process = Mock()
    config.data.process.lengths = Mock()
    config.data.process.lengths.block_len = 200
    config.data.dataset = Mock()
    config.data.dataset.seq_len = 200

    # Training config
    config.train = Mock()
    config.train.logging = Mock()
    config.train.logging.representation_interval = 1

    return config


@pytest.fixture
def mock_datamodule():
    """Mock datamodule for testing."""
    datamodule = Mock()
    datamodule.train_dataset = Mock()
    datamodule.val_dataset = Mock()
    datamodule.test_dataset = Mock()
    return datamodule

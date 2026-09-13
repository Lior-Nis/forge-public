"""
Base logging management for all training paradigms.
Provides common functionality for device management, configuration, and basic I/O.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING, Literal

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot

if TYPE_CHECKING:
    from pipeline.config import Config

from utils.wandb_logging_keys import WandBKeys

logger = logging.getLogger(__name__)


class BaseLoggingManager(ABC):
    """Abstract base class for all logging managers."""

    def __init__(
        self,
        config: "Config",
        device: str,
        trainer: Any,
        datamodule: Optional[Any] = None,
    ):
        """
        Initialize base logging manager.

        Args:
            config: Unified pipeline configuration (Pydantic model)
            device: Device for tensor operations
            trainer: PyTorch Lightning trainer (provides logger and global_step)
            datamodule: Optional datamodule for accessing dataset metadata
        """
        self.config = config
        self.device = device
        self.trainer = trainer
        self.logger = trainer.logger  # Infer from trainer
        self.datamodule = datamodule
        self._data_initialized = False

        # Use attribute access for Pydantic Config
        self.seq_len = config.data.process.lengths.block_len
        self.stride_len = config.data.process.lengths.stride_len
        self._setup_logging_structures()

    def _ensure_data_initialized(self) -> bool:
        """
        DEPRECATED: Kept for backward compatibility during transition.
        Metadata is now loaded on-demand using inline caching.

        Returns:
            True (always, to maintain compatibility with existing checks)
        """
        return True

    def _populate_metadata(self, train_ds, val_ds, test_ds):
        """
        Populate metadata from datasets.

        Override in subclasses that need dataset metadata.
        """
        pass

    def _can_log_to_wandb(self) -> bool:
        """
        Check if wandb logging is available.

        Returns False if logger is None or doesn't have experiment attribute.
        This is a LEGITIMATE hasattr check - we're checking an external
        library interface (WandbLogger), not our own config.
        """
        return self.logger is not None and hasattr(self.logger, 'experiment')

    def log_calibration_plot(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        stage: Literal["val", "test"],
        n_bins: int = 10,
        class_name: Optional[str] = None
    ) -> None:
        """
        Create and log calibration plot (reliability diagram).

        Shows how well predicted probabilities match actual frequencies.
        Perfect calibration: predicted prob = observed frequency.

        Args:
            probabilities: Predicted probabilities for positive class [N]
            labels: Ground truth binary labels [N] (0 or 1)
            stage: Validation or test stage
            n_bins: Number of probability bins
            class_name: Optional class name for multiclass (e.g., "Class_2")
        """
        if not self._can_log_to_wandb():
            return

        from sklearn.calibration import calibration_curve
        import matplotlib.pyplot as plt
        import wandb

        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, probabilities, n_bins=n_bins, strategy='uniform'
        )

        # Compute Expected Calibration Error (ECE)
        bin_indices = np.digitize(probabilities, bins=np.linspace(0, 1, n_bins + 1))
        ece = 0.0
        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_acc = labels[mask].mean()
                bin_conf = probabilities[mask].mean()
                ece += mask.sum() / len(probabilities) * abs(bin_acc - bin_conf)

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)
        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            label=f"Model (ECE={ece:.3f})",
            linewidth=2,
            markersize=8
        )
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title(f"Calibration Plot - {stage.upper()}", fontsize=14)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])

        # Log to WandB (with optional class suffix for multiclass)
        plot_key = WandBKeys.calibration_plot(stage, class_name)
        ece_key = WandBKeys.calibration_ece(stage, class_name)

        self.logger.experiment.log({
            plot_key: wandb.Image(fig),
            ece_key: ece
        })

        plt.close(fig)
        log_suffix = f" ({class_name})" if class_name else ""
        logger.info(f"{stage.upper()} Calibration ECE{log_suffix}: {ece:.4f}")

    @abstractmethod
    def _setup_logging_structures(self):
        """Setup basic logging structures."""
        raise NotImplementedError

    @abstractmethod
    def clear_validation_data(self):
        """Clear validation data structures."""
        raise NotImplementedError

    @abstractmethod
    def clear_test_data(self):
        """Clear test data structures."""
        raise NotImplementedError

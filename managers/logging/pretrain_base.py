"""
Base pretraining logging management for self-supervised learning.
Provides common functionality for pretraining tasks (MAE, SimCLR, etc.).
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, Literal, TYPE_CHECKING
from collections import defaultdict

import torch
import numpy as np

if TYPE_CHECKING:
    from pipeline.config import Config

from .base import BaseLoggingManager

logger = logging.getLogger(__name__)


class PretrainingLoggingManager(BaseLoggingManager):
    """Abstract base class for pretraining logging managers."""

    def __init__(
        self,
        config: "Config",
        device: str,
        trainer: Any,
        datamodule: Optional[Any] = None,
    ):
        """
        Initialize pretraining logging manager.

        Args:
            config: Unified pipeline configuration (Pydantic model)
            device: Device for tensor operations
            trainer: PyTorch Lightning trainer (provides logger and global_step)
            datamodule: Optional datamodule for accessing dataset metadata
        """
        super().__init__(config, device, trainer, datamodule)
        self._setup_pretraining_logging()

    def _setup_logging_structures(self):
        """Satisfy BaseLoggingManager abstract method; actual setup runs in _setup_pretraining_logging."""
        pass

    def _setup_pretraining_logging(self):
        """Setup common pretraining logging structures."""
        # Training progress tracking
        self.epoch_reconstructions = {}  # Store sample reconstructions by epoch
        self.loss_components = defaultdict(list)  # Track different loss components
        self.learning_curves = defaultdict(list)  # Track various metrics over time
        
        # Model state tracking
        self.model_states = {}  # Store model checkpoints/states
        self.hyperparameter_logs = {}  # Track hyperparameter changes
        
        # Sample tracking for visualization
        self.validation_samples = []
        self.test_samples = []
        self.max_samples_per_epoch = 5  # Limit samples to prevent memory issues

    def clear_validation_data(self):
        """Clear validation data structures."""
        self.validation_samples.clear()
        if hasattr(self, 'validation_reconstructions'):
            self.validation_reconstructions.clear()

    def clear_test_data(self):
        """Clear test data structures.""" 
        self.test_samples.clear()
        if hasattr(self, 'test_reconstructions'):
            self.test_reconstructions.clear()

    def log_epoch_end(self, stage: Literal["val", "test"], epoch: int):
        """Log end-of-epoch visualizations and summaries for pretraining."""
        # Create learning curves
        self.create_learning_curves_plot()
        
        # Log task-specific visualizations
        self._log_task_specific_visualizations(stage, epoch)
        
        # Clear data to prevent memory accumulation
        if stage == "val":
            self.clear_validation_data()
        else:
            self.clear_test_data()

    @abstractmethod
    def _log_task_specific_visualizations(self, stage: str, epoch: int):
        """Log task-specific visualizations (MAE reconstructions, SimCLR embeddings, etc.)."""
        pass

    def log_loss_components(self, components: Dict[str, float], stage: str, step: int):
        """Log different components of the loss function."""
        for component_name, value in components.items():
            key = f"{stage}_{component_name}"
            self.loss_components[key].append((step, value))

    def log_hyperparameters(self, params: Dict[str, Any], epoch: int):
        """Log hyperparameter changes."""
        self.hyperparameter_logs[epoch] = params.copy()

    def log_model_state(self, model_state: Dict[str, Any], epoch: int):
        """Log model state information."""
        # Only store lightweight state info, not full model weights
        lightweight_state = {
            "epoch": epoch,
            "model_size": len(str(model_state)),  # Approximate size
            "keys": list(model_state.keys()) if isinstance(model_state, dict) else [],
        }
        self.model_states[epoch] = lightweight_state

    def accumulate_samples_for_visualization(self, 
                                           inputs: torch.Tensor,
                                           outputs: torch.Tensor, 
                                           stage: str,
                                           metadata: Optional[Dict] = None):
        """Accumulate samples for later visualization."""
        if stage == "val" and len(self.validation_samples) < self.max_samples_per_epoch:
            sample = {
                "inputs": inputs[:1].detach().cpu(),  # Take first sample only
                "outputs": outputs[:1].detach().cpu(),
                "metadata": metadata
            }
            self.validation_samples.append(sample)
        elif stage == "test" and len(self.test_samples) < self.max_samples_per_epoch:
            sample = {
                "inputs": inputs[:1].detach().cpu(),
                "outputs": outputs[:1].detach().cpu(),
                "metadata": metadata
            }
            self.test_samples.append(sample)

    def create_loss_components_plot(self, save_path: Optional[str] = None):
        """Create plot showing different loss components over time."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            # Group components by stage
            train_components = {}
            val_components = {}
            
            for key, values in self.loss_components.items():
                if key.startswith("train_"):
                    component_name = key[6:]  # Remove "train_" prefix
                    train_components[component_name] = values
                elif key.startswith("val_"):
                    component_name = key[4:]  # Remove "val_" prefix
                    val_components[component_name] = values
            
            # Plot training components
            ax = axes[0]
            for component_name, values in train_components.items():
                if values:
                    steps, losses = zip(*values)
                    ax.plot(steps, losses, label=component_name, alpha=0.7)
            ax.set_title('Training Loss Components')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot validation components
            ax = axes[1]
            for component_name, values in val_components.items():
                if values:
                    steps, losses = zip(*values)
                    ax.plot(steps, losses, label=component_name, alpha=0.7)
            ax.set_title('Validation Loss Components')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Combined view
            ax = axes[2]
            for component_name in set(train_components.keys()).union(set(val_components.keys())):
                if component_name in train_components and train_components[component_name]:
                    steps, losses = zip(*train_components[component_name])
                    ax.plot(steps, losses, label=f'Train {component_name}', alpha=0.7)
                if component_name in val_components and val_components[component_name]:
                    steps, losses = zip(*val_components[component_name])
                    ax.plot(steps, losses, label=f'Val {component_name}', alpha=0.7, linestyle='--')
            ax.set_title('Combined Loss Components')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Hide empty subplot
            axes[3].set_visible(False)
            
            plt.tight_layout()
            
            if save_path:
                self._ensure_directory(save_path)
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            self._cleanup_matplotlib_memory()
            return fig
            
        except ImportError:
            logger.warning("matplotlib not available for loss components plot")
            return None
        except Exception as e:
            logger.warning(f"Error creating loss components plot: {e}")
            return None

    def _cleanup_matplotlib_memory(self):
        """Clean up matplotlib memory to prevent memory leaks."""
        try:
            import matplotlib.pyplot as plt
            import gc
            plt.close('all')
            gc.collect()
        except Exception:
            pass

"""
Base pretraining metrics management for self-supervised learning.
Provides common functionality for pretraining tasks (MAE, SimCLR, etc.).
"""

import logging
from abc import abstractmethod
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from .base import BaseMetricsManager

logger = logging.getLogger(__name__)


class PretrainingMetricsManager(BaseMetricsManager):
    """Abstract base class for pretraining metrics managers."""

    def __init__(self, device: torch.device, track_gradients: bool = True):
        """
        Initialize pretraining metrics manager.

        Args:
            device: Device to place metrics on
            track_gradients: Whether to track gradient statistics
        """
        self.device = device
        self.track_gradients = track_gradients
        self.gradient_norms = []
        self.convergence_metrics = {
            "loss_smoothness": [],
            "loss_variance": [],
            "learning_progress": []
        }

        super().__init__()

    def _setup_metrics(self):
        """Setup common pretraining metrics."""
        # Common loss tracking - implemented in base class
        pass

    def _move_metrics_to_device(self):
        """Move metrics to device - most pretraining metrics are computed on demand."""
        pass

    def reset(self):
        """Reset pretraining metrics."""
        self.gradient_norms = []
        for key in self.convergence_metrics:
            self.convergence_metrics[key] = []

    @abstractmethod
    def compute_reconstruction_metrics(self, outputs: torch.Tensor, 
                                     targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute task-specific reconstruction/representation metrics.
        
        Args:
            outputs: Model outputs
            targets: Target data
            
        Returns:
            Dictionary of task-specific metrics
        """
        pass

    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, 
                       stage: str) -> Dict[str, float]:
        """
        Compute common pretraining metrics plus task-specific ones.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            stage: Training stage (train/val/test)
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Basic loss computation
        loss = F.mse_loss(outputs, targets)
        metrics["reconstruction_loss"] = loss.item()
        
        # Add task-specific reconstruction metrics
        task_metrics = self.compute_reconstruction_metrics(outputs, targets)
        metrics.update(task_metrics)
        
        # Training dynamics
        if stage == "train":
            self._update_convergence_metrics(loss.item())
            metrics.update(self._get_convergence_summary())
        
        return metrics

    def log_gradient_norms(self, model: torch.nn.Module):
        """Log gradient norms for convergence analysis."""
        if not self.track_gradients:
            return
        
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.gradient_norms.append(total_norm)

    def _update_convergence_metrics(self, loss: float):
        """Update convergence tracking metrics."""
        window_size = 10
        
        # Loss smoothness (variance in recent window)
        if len(self.loss_history["train"]) >= window_size:
            recent_losses = self.loss_history["train"][-window_size:]
            loss_variance = np.var(recent_losses)
            self.convergence_metrics["loss_variance"].append(loss_variance)
            
            # Loss smoothness (average change)
            loss_diffs = np.diff(recent_losses)
            smoothness = np.mean(np.abs(loss_diffs))
            self.convergence_metrics["loss_smoothness"].append(smoothness)
            
            # Learning progress (relative improvement)
            if len(recent_losses) > 1:
                progress = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
                self.convergence_metrics["learning_progress"].append(progress)

    def _get_convergence_summary(self) -> Dict[str, float]:
        """Get current convergence metrics summary."""
        summary = {}
        
        if self.gradient_norms:
            summary["gradient_norm"] = self.gradient_norms[-1]
            summary["avg_gradient_norm"] = np.mean(self.gradient_norms[-10:])
        
        for metric_name, values in self.convergence_metrics.items():
            if values:
                summary[f"convergence_{metric_name}"] = values[-1]
        
        return summary

    def compute_representation_quality(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """
        Compute representation quality metrics.
        
        Args:
            embeddings: Learned embeddings/representations
            
        Returns:
            Dictionary of representation quality metrics
        """
        metrics = {}
        
        with torch.no_grad():
            # Embedding statistics
            metrics["embedding_mean"] = embeddings.mean().item()
            metrics["embedding_std"] = embeddings.std().item()
            metrics["embedding_norm"] = torch.norm(embeddings, dim=-1).mean().item()
            
            # Representation collapse detection
            # Measure effective rank of embedding matrix
            if embeddings.dim() == 2:
                U, S, V = torch.svd(embeddings)
                # Effective rank using 95% of singular values
                cumsum_ratio = torch.cumsum(S, dim=0) / torch.sum(S)
                effective_rank = torch.sum(cumsum_ratio < 0.95).item() + 1
                metrics["effective_rank"] = effective_rank
                metrics["rank_ratio"] = effective_rank / min(embeddings.shape)
                
                # Condition number (measure of numerical stability)
                if S[-1] > 1e-10:  # Avoid division by zero
                    metrics["condition_number"] = (S[0] / S[-1]).item()
        
        return metrics

    def log_learning_curves(self, logger_fn=None, step: int = 0):
        """Log learning curves for visualization."""
        if logger_fn is None:
            return
        
        # Log loss history
        if self.loss_history["train"]:
            logger_fn("loss/train_curve", self.loss_history["train"], step)
        if self.loss_history["val"]:
            logger_fn("loss/val_curve", self.loss_history["val"], step)
        
        # Log gradient norms
        if self.gradient_norms:
            logger_fn("training/gradient_norms", self.gradient_norms, step)
        
        # Log convergence metrics
        for metric_name, values in self.convergence_metrics.items():
            if values:
                logger_fn(f"convergence/{metric_name}", values, step)

    def get_pretraining_summary(self) -> Dict[str, Any]:
        """Get comprehensive pretraining summary."""
        summary = super().get_training_summary()
        
        # Add pretraining-specific metrics
        if self.gradient_norms:
            summary["avg_gradient_norm"] = np.mean(self.gradient_norms)
            summary["final_gradient_norm"] = self.gradient_norms[-1]
        
        # Convergence metrics
        for metric_name, values in self.convergence_metrics.items():
            if values:
                summary[f"final_{metric_name}"] = values[-1]
                summary[f"avg_{metric_name}"] = np.mean(values[-10:])  # Last 10 values
        
        return summary
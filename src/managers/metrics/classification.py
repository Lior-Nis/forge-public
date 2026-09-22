"""
Classification metrics management for supervised FoG detection.
Handles metric computation, logging, and device management for classification tasks.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics
import torchmetrics.classification as tcls
import torch.nn.functional as F
import wandb

from .base import BaseMetricsManager
from utils.clinical_metrics import ClinicalMetricsManager

logger = logging.getLogger(__name__)


class ClassificationMetricsManager(BaseMetricsManager):
    """Manages all classification metrics computation and logging."""

    def __init__(self, num_classes: int = 2):
        """Initialize classification metrics manager.

        Args:
            num_classes: Number of classes for multiclass metrics
        """
        self.num_classes = num_classes
        super().__init__()

    def _build_metrics_dict(self) -> Dict[str, torchmetrics.Metric]:
        """Get dictionary of metrics for classification."""
        metrics = {
            "ap": tcls.BinaryAveragePrecision(),
            "f1": tcls.BinaryF1Score(),
            "accuracy": tcls.BinaryAccuracy(),
            "precision": tcls.BinaryPrecision(),
            "recall": tcls.BinaryRecall(),
            "specificity": tcls.BinarySpecificity(),
        }

        # Add per-class metrics using Multiclass metrics (works for binary too)
        metrics.update({
            "f1_per_class": tcls.MulticlassF1Score(num_classes=self.num_classes, average=None),
            "precision_per_class": tcls.MulticlassPrecision(num_classes=self.num_classes, average=None),
            "recall_per_class": tcls.MulticlassRecall(num_classes=self.num_classes, average=None),
        })

        return metrics

    def _setup_metrics(self):
        """Setup classification-specific metrics."""
        self.metrics = {
            "val": self._build_metrics_dict(),
            "test": self._build_metrics_dict()
        }

    def to(self, device: torch.device):
        """Move all metrics to the specified device."""
        for stage in self.metrics:
            for metric_name in self.metrics[stage]:
                self.metrics[stage][metric_name] = self.metrics[stage][metric_name].to(device)

    def reset_metrics(self, stage: Literal["val", "test"]):
        """Reset all metrics to their initial state."""
        for metric in self.metrics[stage].values():
            metric.reset()

    def compute_metrics(self, stage: Literal["val", "test"]) -> Dict[str, float]:
        """ Compute all classification metrics. """
        computed_metrics = {}
        for metric_name, metric in self.metrics[stage].items():
            value = metric.compute()

            # Handle per-class metrics (return as separate keys)
            if metric_name.endswith('_per_class'):
                base_name = metric_name.replace('_per_class', '')
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    for class_idx in range(len(value)):
                        key = f"{base_name}_class_{class_idx}"
                        computed_metrics[key] = value[class_idx].item()
                else:
                    computed_metrics[metric_name] = value.item()
            else:
                # Convert torch tensors to Python primitives
                computed_metrics[metric_name] = value.item() if isinstance(value, torch.Tensor) else value
        return computed_metrics
        

    def update_metrics(self, probabilities: torch.Tensor,
                       targets: torch.Tensor,
                       stage: Literal["val", "test"],
                       valid_masks: Optional[torch.Tensor] = None):
        """
        Update metrics with new batch of data.

        Args:
            probabilities: Model output probabilities [B, C] or [B, T, C]
            targets: Ground truth targets [B] or [B, T]
            stage: Current stage (val or test)
            valid_masks: Optional boolean mask [B] or [B, T]; invalid positions excluded
        """
        # Flatten sequence dimension before passing to torchmetrics
        if probabilities.dim() == 3:
            B, T, C = probabilities.shape
            probabilities = probabilities.reshape(B * T, C)
            targets = targets.reshape(B * T)
            if valid_masks is not None:
                valid_masks = valid_masks.reshape(B * T).bool()

        # Apply valid mask — drop padding / invalid timesteps
        if valid_masks is not None:
            probabilities = probabilities[valid_masks]
            targets = targets[valid_masks]

        if probabilities.shape[0] == 0:
            return

        # Ensure metrics are on the same device as input
        self.to(probabilities.device)

        # For binary metrics, extract positive class probability
        binary_probs = probabilities
        if probabilities.dim() == 2 and probabilities.shape[-1] == 2:
            binary_probs = probabilities[:, 1]

        for metric_name, metric in self.metrics[stage].items():
            # Per-class metrics need full probabilities (even for binary)
            if metric_name.endswith('_per_class'):
                metric.update(probabilities, targets)
            else:
                metric.update(binary_probs, targets)

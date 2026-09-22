"""
Base metrics management for all training paradigms.
Provides common functionality for device management, logging, and basic tracking.
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)

# TODO: make this utility functions instead?
class BaseMetricsManager(ABC):
    """Abstract base class for all metrics managers."""

    def __init__(self):
        """
        Initialize base metrics manager.

        Args:
            device: Device to place metrics on
        """
        self._setup_metrics()

    @abstractmethod
    def _setup_metrics(self):
        """Setup base metrics specific to each manager type."""
        raise NotImplementedError

    @abstractmethod
    def to(self, device: Optional[torch.device] = None):
        """Move all metrics to the specified device."""
        raise NotImplementedError

    @abstractmethod
    def reset_metrics(self):
        """Reset all metrics to their initial state."""
        raise NotImplementedError

    @abstractmethod
    def update_metrics(self, probas: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new batch predictions and targets."""
        raise NotImplementedError
        
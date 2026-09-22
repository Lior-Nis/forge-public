"""
Manager modules for the FoG detection framework.

This package contains all manager classes organized by functionality:
- base: Abstract base classes for logging and metrics
- classification: Supervised learning managers
- pretraining: Self-supervised learning managers (MAE, SimCLR)
- task_manager: Central training orchestration
"""

# Import commonly used managers for convenience
# from .task import TaskManager  # TODO: Fix - managers/task.py does not exist

# Import base classes
from .logging.base import BaseLoggingManager
from .metrics.base import BaseMetricsManager

# Import task-specific managers
from .logging.classification import ClassificationLoggingManager
from .metrics.classification import ClassificationMetricsManager

from .logging.mae import MAELoggingManager
from .metrics.mae import MAEMetricsManager
from .logging.simclr import SimCLRLoggingManager
from .metrics.simclr import SimCLRMetricsManager

__all__ = [
    # Core managers
    # 'TaskManager',  # TODO: Fix - managers/task.py does not exist

    # Base classes
    'BaseLoggingManager',
    'BaseMetricsManager',

    # Classification managers
    'ClassificationLoggingManager',
    'ClassificationMetricsManager',

    # Pretraining managers
    'MAELoggingManager',
    'MAEMetricsManager',
    'SimCLRLoggingManager',
    'SimCLRMetricsManager',
]
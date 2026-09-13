"""
Debug utilities for development and debugging workflows.
"""

import os
import logging
from typing import List

logger = logging.getLogger(__name__)


def get_wandb_tags_with_debug(base_tags: List[str]) -> List[str]:
    """
    Get W&B tags with debug tag added if running in debug mode.
    
    Args:
        base_tags: Base list of tags to include
        
    Returns:
        List of tags with 'debug' added if WANDB_DEBUG_MODE environment variable is set
    """
    tags = base_tags.copy()
    
    if os.getenv("WANDB_DEBUG_MODE") == "true":
        tags.append("debug")
        logger.info("Debug mode detected - adding 'debug' tag to W&B run")
    
    return tags


def is_debug_mode() -> bool:
    """
    Check if running in debug mode based on environment variables.
    
    Returns:
        True if running in debug mode, False otherwise
    """
    return os.getenv("WANDB_DEBUG_MODE") == "true"
"""
Loss Functions for FoG Detection

This package provides a comprehensive collection of loss functions organized by category:
- base: BaseLoss, utility functions
- classification: Focal losses, weighted cross-entropy
- reconstruction: MAE, SSIM losses
- contrastive: SimCLR contrastive learning
- segmentation: Segmentation loss, Soft Dice
- composite: JointLoss, MaskedLossWrapper

All losses are re-exported here for backward compatibility:
    from model.losses import FocalLoss, BCEFocalLoss, MAELoss, SimCLRLoss
"""

# Base classes and utilities
from model.losses.base import (
    BaseLoss,
    validate_tensor_inputs,
    compute_inverse_frequency_weights,
    compute_median_frequency_weights,
    apply_reduction,
)

# Classification losses
from model.losses.classification import (
    FocalLoss,
    BCEFocalLoss,
    CEFocalLoss,
    WeightedCrossEntropyLoss,
    WeightedBCELoss,
    SmoothAPLoss,
    SoftFBetaLoss,
)

# Reconstruction losses
from model.losses.reconstruction import (
    MAELoss,
    SpectralMAELoss,
    SSIMLoss,
)

# Contrastive losses
from model.losses.contrastive import (
    SimCLRLoss,
)
from model.losses.vicreg import (
    VICRegLoss,
)

# JEPA losses
from model.losses.jepa import (
    JEPALoss,
)
from model.losses.lejepa import (
    LeJEPALoss,
)

# iBOT loss
from model.losses.ibot import (
    IBOTLoss,
)

# Segmentation losses
from model.losses.segmentation import (
    SegmentationLoss,
    SoftDiceLoss,
)

# Composite losses
from model.losses.composite import (
    MaskedLossWrapper,
    JointLoss,
)

__all__ = [
    # Base
    "BaseLoss",
    "validate_tensor_inputs",
    "compute_inverse_frequency_weights",
    "compute_median_frequency_weights",
    "apply_reduction",
    # Classification
    "FocalLoss",
    "BCEFocalLoss",
    "CEFocalLoss",
    "WeightedCrossEntropyLoss",
    "WeightedBCELoss",
    "SmoothAPLoss",
    "SoftFBetaLoss",
    # Reconstruction
    "MAELoss",
    "SpectralMAELoss",
    "SSIMLoss",
    # Contrastive
    "SimCLRLoss",
    "VICRegLoss",
    # JEPA
    "JEPALoss",
    "LeJEPALoss",
    # iBOT
    "IBOTLoss",
    # Segmentation
    "SegmentationLoss",
    "SoftDiceLoss",
    # Composite
    "MaskedLossWrapper",
    "JointLoss",
]

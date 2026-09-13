"""
Pipeline data schemas for batch logging and data transfer.

This module contains Pydantic models for type-safe data transfer between
pipeline components (training steps, logging managers, metrics managers).

These schemas ensure:
- Type safety and validation
- Clear contracts between components
- Separation of concerns
- Easy serialization/deserialization
"""

from typing import Dict, List, Optional

import torch
from pydantic import BaseModel, ConfigDict, Field


class BatchLogData(BaseModel):
    """
    Base class for batch logging data.

    Provides common functionality for detaching and moving tensors to CPU.
    Subclasses should add task-specific fields.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def detach_cpu(self) -> 'BatchLogData':
        """
        Move all tensor data to CPU and detach from computation graph.

        This method should be overridden in subclasses to handle
        task-specific tensor fields.

        Returns:
            Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement detach_cpu")


class ClassificationBatchLogData(BatchLogData):
    """
    Context object for classification batch logging data.

    Contains all information needed for classification metrics,
    logging, and analysis after a training/validation/test step.
    """

    logits: torch.Tensor = Field(
        description="Model predictions [batch_size, num_classes] or [batch_size, seq_len, num_classes]"
    )
    labels: torch.Tensor = Field(
        description="Ground truth labels [batch_size] or [batch_size, seq_len]"
    )
    losses: torch.Tensor = Field(
        description="Per-sample or per-timestep losses"
    )
    valid_masks: torch.Tensor = Field(
        description="Boolean mask for valid timesteps/samples [batch_size] or [batch_size, seq_len]"
    )
    patches_metadata: List[Dict] = Field(
        description="Metadata for each patch/sample in the batch"
    )
    probabilities: Optional[torch.Tensor] = Field(
        default=None,
        description="Class probabilities from softmax [batch_size, num_classes] or [batch_size, seq_len, num_classes]"
    )
    x: Optional[torch.Tensor] = Field(
        default=None,
        description="Optional input tensor for spectral logging [batch_size, channels, height, width]"
    )

    def detach_cpu(self) -> 'ClassificationBatchLogData':
        """Move all tensor data to CPU and detach from graph."""
        self.logits = self.logits.detach().cpu()
        self.labels = self.labels.detach().cpu()
        self.losses = self.losses.detach().cpu()
        self.valid_masks = self.valid_masks.detach().cpu().to(torch.bool)
        if self.probabilities is not None:
            self.probabilities = self.probabilities.detach().cpu()
        if self.x is not None:
            self.x = self.x.detach().cpu()
        return self


class MAEBatchLogData(BatchLogData):
    """
    Context object for MAE (Masked Autoencoder) batch logging data.

    Contains all information needed for reconstruction metrics,
    logging, and analysis after a training/validation/test step.
    """

    reconstructed: torch.Tensor = Field(
        description="Reconstructed data [batch_size, channels, height, width]"
    )
    original: torch.Tensor = Field(
        description="Original unmasked data [batch_size, channels, height, width]"
    )
    spectral_mask: Optional[torch.Tensor] = Field(
        default=None,
        description="Boolean mask indicating which temporal patches were masked [batch_size, num_patches]"
    )
    losses: torch.Tensor = Field(
        description="Per-sample reconstruction losses [batch_size]"
    )
    metadata: List[Dict] = Field(
        description="Metadata for each sample in the batch"
    )

    def detach_cpu(self) -> 'MAEBatchLogData':
        """Move all tensor data to CPU and detach from graph."""
        self.reconstructed = self.reconstructed.detach().cpu()
        self.original = self.original.detach().cpu()
        if self.spectral_mask is not None:
            self.spectral_mask = self.spectral_mask.detach().cpu()
        self.losses = self.losses.detach().cpu()
        return self


class SimCLRBatchLogData(BatchLogData):
    """
    Context object for SimCLR contrastive learning batch logging data.

    Contains all information needed for contrastive learning metrics,
    logging, and analysis after a training/validation/test step.
    """

    z1: torch.Tensor = Field(
        description="Representations from first view [batch_size, feature_dim]"
    )
    z2: torch.Tensor = Field(
        description="Representations from second view [batch_size, feature_dim]"
    )
    losses: torch.Tensor = Field(
        description="Per-sample contrastive losses [batch_size]"
    )
    metadata: List[Dict] = Field(
        description="Metadata for each sample in the batch"
    )
    x_raw: Optional[torch.Tensor] = Field(
        default=None,
        description="Optional raw input signal for visualization [batch_size, channels, seq_len]"
    )
    z1_patches: Optional[torch.Tensor] = Field(
        default=None,
        description="Patch-level projections view 1 [batch_size, N, proj_dim]"
    )
    z2_patches: Optional[torch.Tensor] = Field(
        default=None,
        description="Patch-level projections view 2 [batch_size, N, proj_dim]"
    )
    per_position_losses: Optional[torch.Tensor] = Field(
        default=None,
        description="Mean NT-Xent loss per patch position [N]"
    )

    def detach_cpu(self) -> 'SimCLRBatchLogData':
        """Move all tensor data to CPU and detach from graph."""
        self.z1 = self.z1.detach().cpu()
        self.z2 = self.z2.detach().cpu()
        self.losses = self.losses.detach().cpu()
        if self.x_raw is not None:
            self.x_raw = self.x_raw.detach().cpu()
        if self.z1_patches is not None:
            self.z1_patches = self.z1_patches.detach().cpu()
            self.z2_patches = self.z2_patches.detach().cpu()
        if self.per_position_losses is not None:
            self.per_position_losses = self.per_position_losses.detach().cpu()
        return self


class JEPABatchLogData(BatchLogData):
    """
    Context object for JEPA (Joint Embedding Predictive Architecture) batch logging data.

    Contains all information needed for JEPA metrics, logging, and analysis.
    """

    predicted: torch.Tensor = Field(
        description="Predicted embeddings for masked positions [batch_size, num_masked, embed_dim]"
    )
    target: torch.Tensor = Field(
        description="Target embeddings for masked positions [batch_size, num_masked, embed_dim]"
    )
    losses: torch.Tensor = Field(
        description="Per-sample prediction losses [batch_size]"
    )
    metadata: List[Dict] = Field(
        description="Metadata for each sample in the batch"
    )
    patch_mask: Optional[torch.Tensor] = Field(
        default=None,
        description="Boolean mask indicating which patches were masked [batch_size, num_patches]"
    )

    def detach_cpu(self) -> 'JEPABatchLogData':
        """Move all tensor data to CPU and detach from graph."""
        self.predicted = self.predicted.detach().cpu()
        self.target = self.target.detach().cpu()
        self.losses = self.losses.detach().cpu()
        if self.patch_mask is not None:
            self.patch_mask = self.patch_mask.detach().cpu()
        return self


class IBOTBatchLogData(BatchLogData):
    """
    Context object for iBOT batch logging data.

    Contains all information needed for iBOT metrics, logging, and analysis.
    """

    student_tokens: torch.Tensor = Field(
        description="Student backbone output tokens [batch_size, nW, embed_dim]"
    )
    teacher_tokens: torch.Tensor = Field(
        description="EMA teacher backbone output tokens [batch_size, nW, embed_dim]"
    )
    patch_mask: torch.Tensor = Field(
        description="Boolean mask indicating masked temporal patches [batch_size, nW]"
    )
    losses: torch.Tensor = Field(
        description="Per-sample cosine losses [batch_size]"
    )
    metadata: List[Dict] = Field(
        description="Metadata for each sample in the batch"
    )

    def detach_cpu(self) -> 'IBOTBatchLogData':
        """Move all tensor data to CPU and detach from graph."""
        self.student_tokens = self.student_tokens.detach().cpu()
        self.teacher_tokens = self.teacher_tokens.detach().cpu()
        self.patch_mask = self.patch_mask.detach().cpu()
        self.losses = self.losses.detach().cpu()
        return self


# Export all schemas
__all__ = [
    'BatchLogData',
    'ClassificationBatchLogData',
    'MAEBatchLogData',
    'SimCLRBatchLogData',
    'JEPABatchLogData',
    'IBOTBatchLogData',
]

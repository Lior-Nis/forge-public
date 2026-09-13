"""
Classification loss functions for FoG detection.

This module provides loss functions optimized for classification tasks:
- FocalLoss: Handles class imbalance with focusing parameter
- BCEFocalLoss: Binary cross-entropy with focal weighting
- CEFocalLoss: Cross-entropy with focal weighting
- WeightedCrossEntropyLoss: Cross-entropy with class weighting
- WeightedBCELoss: Binary cross-entropy with positive class weighting
- SmoothAPLoss: Differentiable Average Precision surrogate (Brown et al. 2020)
- SoftFBetaLoss: Differentiable F-beta loss for tunable precision/recall tradeoff
"""

import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.losses.base import (
    BaseLoss,
    compute_inverse_frequency_weights,
    compute_median_frequency_weights,
)

logger = logging.getLogger(__name__)


class FocalLoss(BaseLoss):
    """
    Focal Loss for handling class imbalance in dense object detection.

    Automatically handles both multiclass and multilabel classification.
    Based on: https://arxiv.org/abs/1708.02002

    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Optional per-class weighting factor
        reduction: Reduction method ('mean', 'sum', 'none')
        weighted: Whether to use inverse frequency weights
        class_counts: Sample counts per class for weight computation
        background_weight_factor: Multiplier for background class weight (default: 0.2)
        num_event_classes: Number of event classes for multilabel detection (default: 3)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, List[float]]] = None,
        reduction: str = "mean",
        weighted: bool = False,
        class_counts: Optional[List[float]] = None,
        background_weight_factor: float = 0.2,
        num_event_classes: int = 3,
        **kwargs,
    ):
        super().__init__(reduction=reduction)
        self.gamma = gamma
        self.num_event_classes = num_event_classes

        # Setup alpha (per-class weights)
        self.alpha = None
        if alpha is not None:
            self.alpha = torch.tensor(alpha) if isinstance(alpha, list) else alpha

        # Setup class weights for weighted mode
        self.class_weights = None
        if weighted and class_counts is not None:
            self.class_weights = compute_inverse_frequency_weights(
                class_counts=class_counts,
                normalize=True,
                background_weight_factor=background_weight_factor,
                device=None  # Will be moved to correct device in forward
            )

    def forward(
        self,
        logits: torch.Tensor,
        target_ohe: torch.Tensor,
        reduction: Optional[str] = None
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model predictions
            target_ohe: One-hot encoded targets
            reduction: Override default reduction

        Returns:
            Focal loss value
        """
        # Validate inputs
        if not self._validate_inputs(logits, target_ohe):
            logger.warning("NaN or inf detected in inputs")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Determine if multilabel or multiclass based on target shape
        is_multilabel = target_ohe.shape[-1] == self.num_event_classes

        if is_multilabel:
            # Multilabel: BCE + focal weighting
            pos_weight = None
            if self.class_weights is not None:
                pos_weight = self.class_weights[1:].to(logits.device)

            bce_loss = F.binary_cross_entropy_with_logits(
                logits,
                target_ohe.float(),
                pos_weight=pos_weight,
                reduction="none"
            )

            pt = torch.sigmoid(logits)
            pt = torch.where(target_ohe == 1, pt, 1 - pt)
            focal_weight = (1 - pt).pow(self.gamma)
            weighted_loss = focal_weight * bce_loss
        else:
            # Multiclass: CE + focal weighting
            class_weights = None
            if self.class_weights is not None:
                class_weights = self.class_weights.to(logits.device)

            ce_loss = F.cross_entropy(
                logits,
                torch.argmax(target_ohe, dim=-1),
                weight=class_weights,
                reduction="none"
            )

            pt = F.softmax(logits, dim=-1)
            pt = pt.gather(dim=-1, index=torch.argmax(target_ohe, dim=-1).unsqueeze(-1)).squeeze(-1)
            focal_weight = (1 - pt).pow(self.gamma)
            weighted_loss = focal_weight * ce_loss

        # Apply alpha weighting if specified
        if self.alpha is not None and isinstance(self.alpha, torch.Tensor):
            alpha_weights = self.alpha.to(logits.device)
            if is_multilabel:
                alpha_weights = alpha_weights[1:]  # Skip background
            else:
                alpha_weights = alpha_weights[target_ohe.long()]
            weighted_loss = alpha_weights * weighted_loss

        # Validate output
        if not self._validate_inputs(weighted_loss):
            logger.warning("NaN or inf detected in focal loss output")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return self._apply_reduction(weighted_loss, reduction)


class WeightedCrossEntropyLoss(BaseLoss):
    """
    Cross-entropy loss with optional class weighting.

    Supports both static (pre-computed) and dynamic (batch-based) weight computation.

    Args:
        weighted: Whether to use class weights
        class_counts: Sample counts for static weight computation
        compute_class_weights: Use dynamic median-frequency weighting per batch
        reduction: Reduction method
    """

    def __init__(
        self,
        weighted: bool = False,
        class_counts: Optional[List[float]] = None,
        compute_class_weights: bool = False,
        reduction: str = "mean",
        **kwargs,
    ):
        super().__init__(reduction=reduction)
        self.compute_class_weights = compute_class_weights

        # Compute static weights if provided
        static_weights = None
        if weighted and class_counts is not None:
            static_weights = compute_inverse_frequency_weights(
                class_counts, normalize=True
            )

        self.static_weights = static_weights
        self.criterion = nn.CrossEntropyLoss(weight=static_weights, reduction=reduction)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy loss."""
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1).long()

        if self.compute_class_weights:
            # Dynamic weight computation per batch
            class_weights = compute_median_frequency_weights(labels_flat)
            class_weights = None if len(class_weights) == 1 else class_weights
            loss = F.cross_entropy(
                logits_flat, labels_flat,
                weight=class_weights,
                reduction=self.reduction
            )
        else:
            # Use static criterion
            loss = self.criterion(logits_flat, labels_flat)

        return loss


class WeightedBCELoss(BaseLoss):
    """
    Binary Cross-Entropy loss with optional positive class weighting.

    Designed for multilabel classification where background class is dropped.

    Args:
        weighted: Whether to use positive class weights
        pos_weights: Per-class positive weights
        reduction: Reduction method
    """

    def __init__(
        self,
        weighted: bool = False,
        pos_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(reduction='mean')

        pos_weight_tensor = None
        if weighted and pos_weights is not None:
            pos_weight_tensor = torch.tensor(pos_weights)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Note: Automatically drops background class from multilabel targets.
        """
        # Drop background class for multilabel
        y = y[..., 1:]
        return self.criterion(x, y.float())


class BCEFocalLoss(BaseLoss):
    """
    Binary Cross-Entropy Focal Loss variant.

    Combines BCE with focal weighting for multilabel classification.

    Args:
        gamma: Focusing parameter
        alpha: Per-class weighting
        num_classes: Number of classes (including background)
        reduction: Reduction method
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[List[float]] = None,
        num_classes: int = 4,
        reduction: str = "none",
        **kwargs,
    ):
        super().__init__(reduction=reduction)
        self.gamma = gamma
        self.num_classes = num_classes
        self._alpha_list = alpha
        self.alpha = None  # Lazy initialization on first forward
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute BCE focal loss with optional masking."""
        # Convert targets to one-hot (includes background class)
        targets_ohe = F.one_hot(targets, num_classes=self.num_classes)

        # Compute BCE loss
        bce_loss = self.bce_criterion(logits, targets_ohe.float())

        # Compute focal weighting
        p = torch.sigmoid(logits)
        pt = torch.where(targets_ohe == 1, p, 1 - p)

        # Initialize alpha on first forward
        if self.alpha is None:
            if self._alpha_list is None:
                self.alpha = torch.ones(self.num_classes, device=pt.device)
            else:
                self.alpha = torch.tensor(self._alpha_list, device=pt.device)

        # Apply focal weighting and sum across classes
        focal_loss = self.alpha.to(logits.device) * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss.sum(dim=-1)  # Per-sample loss

        # Apply masking if provided
        focal_loss = self._apply_mask(focal_loss, valid_mask)

        return focal_loss


class CEFocalLoss(BaseLoss):
    """
    Cross-Entropy Focal Loss variant.

    Combines cross-entropy with focal weighting for multiclass classification.

    Args:
        gamma: Focusing parameter
        alpha: Per-class weighting factors
        reduction: Reduction method
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: List[float] = [1.0, 1.0],
        reduction: str = "none",
        **kwargs,
    ):
        super().__init__(reduction=reduction)
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)
        self.ce_criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        targets = targets.long()

        # CrossEntropyLoss requires class dim at position 1: [B, C] or [B, C, T].
        # For sequence output [B, T, C], reshape to [B*T, C] and back.
        if logits.dim() == 3:
            B, T, C = logits.shape
            ce_loss = self.ce_criterion(logits.reshape(B * T, C), targets.reshape(B * T)).reshape(B, T)
        else:
            ce_loss = self.ce_criterion(logits, targets)

        # Compute focal weighting
        p = F.softmax(logits, dim=-1)
        pt = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Apply per-class alpha weighting
        alpha_weights = self.alpha.to(logits.device)[targets]
        focal_loss = alpha_weights * (1 - pt) ** self.gamma * ce_loss

        focal_loss = self._apply_mask(focal_loss, valid_mask)

        return focal_loss


class SmoothAPLoss(BaseLoss):
    """
    Differentiable surrogate for Average Precision.

    Uses sigmoid-smoothed pairwise ranking to approximate AP and optimize it
    directly. Maximizes the area under the Precision-Recall curve end-to-end.

    Reference: Brown et al., "Smooth-AP: Smoothing the Path Towards
    Large-Scale Image Retrieval", ECCV 2020.

    Args:
        temperature: Sigmoid temperature for soft ranking (lower = sharper,
            default 0.01). Larger values give smoother gradients early in
            training; smaller values approach true AP.
        num_classes: Number of output classes (default: 2)
        pos_class: Index of the positive class (default: 1)
        reduction: Not used (batch-level metric), kept for API compatibility
        use_masked_loss: Whether to apply valid_mask filtering
    """

    def __init__(
        self,
        temperature: float = 0.01,
        num_classes: int = 2,
        pos_class: int = 1,
        reduction: str = "mean",
        use_masked_loss: bool = True,
        **kwargs,
    ):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.num_classes = num_classes
        self.pos_class = pos_class
        self.use_masked_loss = use_masked_loss

    def _smooth_ap(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute Smooth-AP for a batch.

        For each positive example i, soft-precision at i is estimated as:
            pi_i = rank_among_positives_i / rank_among_all_i
        where ranks are computed via sigmoid-smoothed pairwise comparisons.
        AP = mean(pi_i) over all positives.

        Args:
            scores: [N] positive-class probability scores
            labels: [N] binary labels (1 = positive, 0 = negative)

        Returns:
            Scalar Smooth-AP in [0, 1]
        """
        n_pos = labels.sum()
        if n_pos == 0 or n_pos == labels.size(0):
            # Undefined AP — return 1.0 so loss = 0, no spurious gradient
            return scores.new_tensor(1.0)

        # diff[i, j] = score[i] - score[j]  →  [N, N]
        diff = scores.unsqueeze(1) - scores.unsqueeze(0)

        # xi[i, j] ≈ 1 if score[i] > score[j], 0.5 if equal, 0 if score[i] < score[j]
        xi = torch.sigmoid(diff / self.temperature)  # [N, N]

        # For each example j, how many examples i have score[i] > score[j]
        # (i.e., how many examples are ranked above j — determines j's rank from top)
        # sum over rows (dim=0) for each column j.
        rank_all = xi.sum(dim=0)  # [N]

        # For each example j, how many POSITIVES i are ranked above j
        # labels.unsqueeze(1) broadcasts as [N, 1] → masks rows by label
        rank_pos = (xi * labels.unsqueeze(1)).sum(dim=0)  # [N]

        # Precision at each position; only positives contribute to AP
        precision = rank_pos / rank_all.clamp(min=1e-6)  # [N]
        ap = (precision * labels).sum() / n_pos

        return ap

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:     [B, C] or [B, T, C]
            targets:    [B]    or [B, T]
            valid_mask: [B]    or [B, T]  optional validity mask

        Returns:
            [B] loss tensor — batch-level AP loss broadcast to per-sample shape,
            matching the contract of all other classification losses so that
            JointLoss and the logging manager work without modification.
        """
        B = logits.size(0)

        # Flatten sequence dimension
        if logits.dim() == 3:
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            mask_flat = valid_mask.reshape(-1) if valid_mask is not None else None
        else:
            logits_flat, targets_flat, mask_flat = logits, targets, valid_mask

        # Filter to valid samples for the AP computation
        if mask_flat is not None and self.use_masked_loss:
            keep = mask_flat.bool()
            logits_flat = logits_flat[keep]
            targets_flat = targets_flat[keep]

        if logits_flat.size(0) == 0:
            return logits.new_zeros(B)

        scores = F.softmax(logits_flat, dim=-1)[:, self.pos_class]
        labels = (targets_flat == self.pos_class).float()

        scalar_loss = 1.0 - self._smooth_ap(scores, labels)

        # Expand to [B]: all samples share the same batch-level loss.
        # Use clone() so autograd can track through the expansion.
        per_sample = scalar_loss.expand(B).clone()

        # Zero invalid positions — consistent with CEFocalLoss._apply_mask()
        if valid_mask is not None and self.use_masked_loss:
            sample_valid = valid_mask if valid_mask.dim() == 1 else valid_mask.all(dim=-1)
            per_sample = per_sample * sample_valid.float()

        return per_sample

    def compute_per_sample(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Delegates to forward() which already returns [B]."""
        return self.forward(logits, targets, valid_mask)


class FogRatioLoss(BaseLoss):
    """
    Soft-label BCE loss for patch-level FOG ratio targets.

    Treats the FOG ratio (fraction of frames that are FOG in the patch) as a
    continuous training target in [0, 1] instead of a hard binary label.
    Uses the positive-class logit via BCEWithLogitsLoss.

    Args:
        pos_weight: Optional scalar weight for positive class (handles imbalance)
        reduction: Reduction method
    """

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        reduction: str = "none",
        **kwargs,
    ):
        super().__init__(reduction=reduction)
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:  [B, C] — only the positive-class logit (index 1) is used
            targets: [B]    — float fog ratios in [0, 1]
            valid_mask: [B] optional

        Returns:
            [B] per-sample loss
        """
        pos_logits = logits[:, 1] if logits.dim() == 2 else logits
        loss = self.criterion(pos_logits, targets.float().to(logits.device))

        if valid_mask is not None:
            loss = loss * valid_mask.float().to(logits.device)

        return loss

    def compute_per_sample(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(logits, targets, valid_mask)


class SoftFBetaLoss(BaseLoss):
    """
    Differentiable F-beta loss using soft TP/FP/FN counts.

    Directly optimizes a smooth approximation of F_beta by replacing hard
    thresholding with predicted probabilities. Use beta to steer the
    precision/recall tradeoff:

        beta < 1  →  weights precision more  (e.g. beta=0.5 for high precision)
        beta = 1  →  F1 — balanced
        beta > 1  →  weights recall more     (e.g. beta=2 for high recall)

    To target the precision=85%, recall=90% operating point, start with
    beta ≈ 0.5–0.7 and monitor the val PR curve.

    Args:
        beta: F-beta parameter (default: 1.0)
        num_classes: Number of output classes (default: 2)
        pos_class: Index of the positive class (default: 1)
        smooth: Numerical stability epsilon (default: 1e-6)
        reduction: Not used (batch-level metric), kept for API compatibility
        use_masked_loss: Whether to apply valid_mask filtering
    """

    def __init__(
        self,
        beta: float = 1.0,
        num_classes: int = 2,
        pos_class: int = 1,
        smooth: float = 1e-6,
        reduction: str = "mean",
        use_masked_loss: bool = True,
        **kwargs,
    ):
        super().__init__(reduction=reduction)
        self.beta_sq = beta ** 2
        self.num_classes = num_classes
        self.pos_class = pos_class
        self.smooth = smooth
        self.use_masked_loss = use_masked_loss

    def _soft_fbeta(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute soft F-beta using differentiable TP/FP/FN.

        Args:
            probs:  [N] positive-class probabilities
            labels: [N] binary labels

        Returns:
            Scalar F-beta in [0, 1]
        """
        tp = (probs * labels).sum()
        fp = (probs * (1.0 - labels)).sum()
        fn = ((1.0 - probs) * labels).sum()

        precision = tp / (tp + fp + self.smooth)
        recall = tp / (tp + fn + self.smooth)

        f_beta = (
            (1.0 + self.beta_sq) * precision * recall
            / (self.beta_sq * precision + recall + self.smooth)
        )
        return f_beta

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:     [B, C] or [B, T, C]
            targets:    [B]    or [B, T]
            valid_mask: [B]    or [B, T]  optional validity mask

        Returns:
            [B] loss tensor — batch-level F-beta loss broadcast to per-sample shape,
            matching the contract of all other classification losses so that
            JointLoss and the logging manager work without modification.
        """
        B = logits.size(0)

        # Flatten sequence dimension
        if logits.dim() == 3:
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            mask_flat = valid_mask.reshape(-1) if valid_mask is not None else None
        else:
            logits_flat, targets_flat, mask_flat = logits, targets, valid_mask

        # Filter to valid samples for the F-beta computation
        if mask_flat is not None and self.use_masked_loss:
            keep = mask_flat.bool()
            logits_flat = logits_flat[keep]
            targets_flat = targets_flat[keep]

        if logits_flat.size(0) == 0:
            return logits.new_zeros(B)

        probs = F.softmax(logits_flat, dim=-1)[:, self.pos_class]
        labels = (targets_flat == self.pos_class).float()

        scalar_loss = 1.0 - self._soft_fbeta(probs, labels)

        # Expand to [B]: all samples share the same batch-level loss.
        # Use clone() so autograd can track through the expansion.
        per_sample = scalar_loss.expand(B).clone()

        # Zero invalid positions — consistent with CEFocalLoss._apply_mask()
        if valid_mask is not None and self.use_masked_loss:
            sample_valid = valid_mask if valid_mask.dim() == 1 else valid_mask.all(dim=-1)
            per_sample = per_sample * sample_valid.float()

        return per_sample

    def compute_per_sample(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Delegates to forward() which already returns [B]."""
        return self.forward(logits, targets, valid_mask)

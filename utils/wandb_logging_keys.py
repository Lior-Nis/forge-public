"""Centralized WandB logging key constants for consistent metric tracking.

This module provides a single source of truth for all WandB logging keys used
across classification, MAE, and SimCLR pipelines. Using these constants ensures:
- Consistent naming conventions across pipelines
- Type safety and autocomplete support
- Easy refactoring and discovery of all logging keys
- Prevention of typos in hardcoded strings

Usage:
    from utils.wandb_logging_keys import WandBKeys

    # Patch analysis table
    table_name = WandBKeys.patch_analysis_table("val")  # "patch_analysis/val_patch_details"

    # Summary metrics
    total_key = WandBKeys.patch_summary_total("train")  # "patch_summary/train_total_patches"
"""

from typing import Literal, Optional

Stage = Literal["train", "val", "test"]


class WandBKeys:
    """WandB logging key constants for all pipelines.

    Naming conventions:
    - patch_analysis/: Detailed patch-level tables
    - patch_summary/: Aggregated statistics
    - patch_metadata/: Dataset metadata (classification only)
    - spectrals/: Visualization plots
    - mae_*/simclr_*: Pipeline-specific visualizations
    - model_registry/: Model artifact registration
    """

    # -------------------------------------------------------------------------
    # Patch Analysis Tables
    # -------------------------------------------------------------------------

    @staticmethod
    def patch_analysis_table(stage: Stage) -> str:
        """Detailed patch-level performance table.

        Contains per-patch metrics like loss, predictions, patient IDs, etc.
        Used by all three pipelines (classification, MAE, SimCLR).

        Args:
            stage: Training stage ("train", "val", or "test")

        Returns:
            Key: "patch_analysis/{stage}_patch_details"
        """
        return f"patch_analysis/{stage}_patch_details"

    @staticmethod
    def patch_metadata_table(stage: Stage) -> str:
        """Patch metadata table (classification only).

        Contains patient/protocol/session distribution information.
        Only logged by classification pipeline.

        Args:
            stage: Training stage ("train", "val", or "test")

        Returns:
            Key: "patch_metadata/{stage}_metadata"
        """
        return f"patch_metadata/{stage}_metadata"

    # -------------------------------------------------------------------------
    # Patch Summary Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def patch_summary_total(stage: Stage) -> str:
        """Total number of patches in epoch.

        Args:
            stage: Training stage ("train", "val", or "test")

        Returns:
            Key: "patch_summary/{stage}_total_patches"
        """
        return f"patch_summary/{stage}_total_patches"

    @staticmethod
    def patch_summary_mean_loss(stage: Stage) -> str:
        """Mean loss across all patches.

        Args:
            stage: Training stage ("train", "val", or "test")

        Returns:
            Key: "patch_summary/{stage}_mean_loss"
        """
        return f"patch_summary/{stage}_mean_loss"

    @staticmethod
    def patch_summary_std_loss(stage: Stage) -> str:
        """Standard deviation of loss across patches.

        Args:
            stage: Training stage ("train", "val", or "test")

        Returns:
            Key: "patch_summary/{stage}_std_loss"
        """
        return f"patch_summary/{stage}_std_loss"

    # -------------------------------------------------------------------------
    # Classification-Specific Visualizations
    # -------------------------------------------------------------------------

    @staticmethod
    def spectral_plot(stage: Stage, status: str, ground_truth: int) -> str:
        """Spectral visualization plot for classification.

        Args:
            stage: Training stage ("train", "val", or "test")
            status: Prediction status ("correct", "incorrect", "false_positive", etc.)
            ground_truth: Ground truth label (0 or 1)

        Returns:
            Key: "spectrals/{stage}_{status}_gt={ground_truth}"
        """
        return f"spectrals/{stage}_{status}_gt={ground_truth}"

    @staticmethod
    def patient_histogram(stage: Stage) -> str:
        """Patient distribution histogram.

        Shows number of patches per patient for dataset balance analysis.

        Args:
            stage: Training stage ("train", "val", or "test")

        Returns:
            Key: "dataset_metadata/patches_per_patient_{stage}"
        """
        return f"dataset_metadata/patches_per_patient_{stage}"

    # -------------------------------------------------------------------------
    # Enhanced Classification Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def confusion_matrix(stage: Stage) -> str:
        """Confusion matrix visualization."""
        return f"classification_metrics/confusion_matrix_{stage}"

    @staticmethod
    def roc_curve(stage: Stage) -> str:
        """ROC curve visualization."""
        return f"classification_metrics/roc_curve_{stage}"

    @staticmethod
    def roc_auc(stage: Stage) -> str:
        """ROC AUC score."""
        return f"classification_metrics/roc_auc_{stage}"

    @staticmethod
    def pr_curve(stage: Stage) -> str:
        """Precision-Recall curve visualization."""
        return f"classification_metrics/pr_curve_{stage}"

    @staticmethod
    def average_precision(stage: Stage) -> str:
        """Average Precision score."""
        return f"classification_metrics/average_precision_{stage}"

    @staticmethod
    def patient_f1(stage: Stage) -> str:
        """Patient-level F1 score."""
        return f"patient_metrics/f1_{stage}"

    @staticmethod
    def patient_accuracy(stage: Stage) -> str:
        """Patient-level accuracy."""
        return f"patient_metrics/accuracy_{stage}"

    @staticmethod
    def patient_precision(stage: Stage) -> str:
        """Patient-level precision."""
        return f"patient_metrics/precision_{stage}"

    @staticmethod
    def patient_recall(stage: Stage) -> str:
        """Patient-level recall."""
        return f"patient_metrics/recall_{stage}"

    @staticmethod
    def patient_metrics_table(stage: Stage) -> str:
        """Patient-level metrics table."""
        return f"patient_metrics/per_patient_table_{stage}"

    @staticmethod
    def calibration_plot(stage: Stage, class_name: Optional[str] = None) -> str:
        """Calibration plot (reliability diagram).

        Args:
            stage: Training stage
            class_name: Optional class name for multiclass (e.g., "Class_2")

        Returns:
            "calibration/plot_{stage}" for binary
            "calibration/plot_{stage}_{class_name}" for multiclass
        """
        if class_name:
            return f"calibration/plot_{stage}_{class_name}"
        return f"calibration/plot_{stage}"

    @staticmethod
    def calibration_ece(stage: Stage, class_name: Optional[str] = None) -> str:
        """Expected Calibration Error (ECE).

        Args:
            stage: Training stage
            class_name: Optional class name for multiclass

        Returns:
            "calibration/ece_{stage}" for binary
            "calibration/ece_{stage}_{class_name}" for multiclass
        """
        if class_name:
            return f"calibration/ece_{stage}_{class_name}"
        return f"calibration/ece_{stage}"

    @staticmethod
    def protocol_f1(stage: Stage, protocol: str) -> str:
        """F1 score for specific protocol."""
        return f"protocol_metrics/{stage}/{protocol}/f1"

    @staticmethod
    def protocol_accuracy(stage: Stage, protocol: str) -> str:
        """Accuracy for specific protocol."""
        return f"protocol_metrics/{stage}/{protocol}/accuracy"

    @staticmethod
    def protocol_precision(stage: Stage, protocol: str) -> str:
        """Precision for specific protocol."""
        return f"protocol_metrics/{stage}/{protocol}/precision"

    @staticmethod
    def protocol_recall(stage: Stage, protocol: str) -> str:
        """Recall for specific protocol."""
        return f"protocol_metrics/{stage}/{protocol}/recall"

    @staticmethod
    def protocol_metrics_table(stage: Stage) -> str:
        """Protocol metrics summary table."""
        return f"protocol_metrics/summary_table_{stage}"

    @staticmethod
    def protocol_metrics_chart(stage: Stage) -> str:
        """Protocol metrics bar chart comparison."""
        return f"protocol_metrics/comparison_chart_{stage}"

    # -------------------------------------------------------------------------
    # MAE-Specific Visualizations
    # -------------------------------------------------------------------------

    @staticmethod
    def mae_reconstructions(stage: Stage) -> str:
        """MAE reconstruction comparison image grid.

        Shows original, masked, and reconstructed spectrograms side-by-side.

        Args:
            stage: Training stage ("train", "val", or "test")

        Returns:
            Key: "mae_reconstructions_{stage}"
        """
        return f"mae_reconstructions_{stage}"

    MAE_MASK_ANALYSIS = "mae_mask_analysis"
    """MAE mask statistics table (mask ratios, pattern analysis)."""

    MAE_SPECTRAL_FIDELITY = "mae_spectral_fidelity"
    """MAE spectral fidelity metrics table (frequency-domain quality)."""

    @staticmethod
    def mae_spectral_reconstruction(stage: Stage, sample_idx: int, batch_idx: int) -> str:
        """Individual MAE spectral reconstruction plot.

        Args:
            stage: Training stage ("train", "val", or "test")
            sample_idx: Sample index within batch
            batch_idx: Batch index

        Returns:
            Key: "mae_spectral_reconstructions/{stage}_sample_{sample_idx}_batch_{batch_idx}"
        """
        return f"mae_spectral_reconstructions/{stage}_sample_{sample_idx}_batch_{batch_idx}"

    # -------------------------------------------------------------------------
    # SimCLR-Specific Visualizations
    # -------------------------------------------------------------------------

    @staticmethod
    def simclr_embeddings(stage: Stage) -> str:
        """SimCLR embedding visualization (UMAP/t-SNE projection).

        Args:
            stage: Training stage ("train", "val", or "test")

        Returns:
            Key: "simclr_embeddings_{stage}"
        """
        return f"simclr_embeddings_{stage}"

    SIMCLR_CONTRASTIVE_ANALYSIS = "simclr_contrastive_analysis"
    """SimCLR contrastive analysis table (positive/negative pair statistics)."""

    SIMCLR_TEMPERATURE_ANALYSIS = "simclr_temperature_analysis"
    """SimCLR temperature scaling analysis table."""

    # -------------------------------------------------------------------------
    # Model Registry (from wandb_callbacks.py)
    # -------------------------------------------------------------------------

    MODEL_REGISTRY_REGISTERED = "model_registry/registered"
    """Boolean flag indicating successful model registration."""

    MODEL_REGISTRY_ARTIFACT_NAME = "model_registry/artifact_name"
    """Name of the registered WandB artifact."""

    MODEL_REGISTRY_REASON = "model_registry/reason"
    """Human-readable reason for registration."""

    MODEL_REGISTRY_FAILED = "model_registry/registration_failed"
    """Boolean flag indicating registration failure."""

    MODEL_REGISTRY_ERROR = "model_registry/error"
    """Error message if registration failed."""

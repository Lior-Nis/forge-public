"""
Classification logging management for supervised FoG detection.
Handles session/patient data accumulation, clinical evaluation, and spectral logging.
"""

from collections import defaultdict
import logging
import os
import gc
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
from matplotlib import pyplot as plt
import pandas as pd
import torch
import wandb
import numpy as np

if TYPE_CHECKING:
    from pipeline.config import Config

from pipeline.schemas import ClassificationBatchLogData
from .base import BaseLoggingManager
from utils.wandb_logging_keys import WandBKeys

logger = logging.getLogger(__name__)

# Backward compatibility alias
BatchLogData = ClassificationBatchLogData


class ClassificationLoggingManager(BaseLoggingManager):
    """Manages data logging, accumulation, and spectral visualization."""

    def __init__(
        self,
        config: "Config",
        device: str,
        num_classes: int,
        trainer: Any,
        datamodule: Optional[Any] = None,
    ):
        """
        Initialize classification logging manager.

        Args:
            config: Unified pipeline configuration (Pydantic model)
            device: Device for tensor operations
            num_classes: Total number of classes
            trainer: PyTorch Lightning trainer (provides logger and global_step)
            datamodule: Optional datamodule for accessing dataset metadata
        """
        self.num_classes = num_classes
        self.num_event_classes = num_classes - 1  # Event classes exclude background

        # Use attribute access for Pydantic Config
        paths_cfg = config.data.paths
        self.input_dir = paths_cfg.input_dir
        self.raw_root = paths_cfg.raw_root
        self.dataset_source = os.path.basename(os.path.normpath(self.input_dir))

        super().__init__(config, device, trainer, datamodule)

    def _setup_logging_structures(self):
        """Setup classification-specific logging data structures."""
        self.patch_logging_keys = ["global_idx", "labels", "logits", "losses", "valid_masks", "probabilities"]
        self.timestamps_logging_keys = ["labels", "logits", "valid_masks"]

        # Accumulation intervals from validated LoggingConfig
        logging_config = self.config.train.logging
        if logging_config:
            self.spectrals_interval = logging_config.accumulation_intervals.get("spectrals", 10)
            self.timestamps_interval = logging_config.accumulation_intervals.get("timestamps", 5)
        else:
            # Fallback defaults if logging config not provided
            self.spectrals_interval = 10
            self.timestamps_interval = 5

        self.logging_patch_storage = {
            split: {k: [] for k in self.patch_logging_keys}
                for split in ("train", "val", "test")
        }

        self.logging_timestamps_storage = {
            split: {k: defaultdict(list)
                    for k in self.timestamps_logging_keys}
                        for split in ("val", "test")
        }

        self.logging_spectral_storage = {
            split: {cw: {k: {} for k in range(self.num_classes)}
                    for cw in ("correct", "wrong")}
                        for split in ("val", "test")
        }

        # Initialize empty metadata structures (populated lazily)
        self.metadata_dataframes = {}
        self.session_metadata = pd.DataFrame()

        # Cache for patch performance DataFrames (reused by visualizations)
        self.patch_performance_dfs = {}

    def _populate_metadata(self, train_ds, val_ds, test_ds):
        """
        DEPRECATED: Kept for backward compatibility.
        Per-stage metadata is now loaded on-demand.
        This only builds session_metadata aggregation.
        """
        # Create unified session lookup for O(1) access (train+val only for clinical eval)
        dfs = [ds.metadata_df for ds in [train_ds, val_ds] if ds is not None]
        if dfs:
            full_df = pd.concat(dfs)
            self.session_metadata = full_df.drop_duplicates(
                subset=['session_id']
            ).set_index('session_id')
            logger.info("Built session metadata lookup from train+val datasets")
        else:
            self.session_metadata = pd.DataFrame()

    def _build_session_metadata(self):
        """Build session metadata lookup from train and val datasets."""
        if not self.session_metadata.empty:
            return  # Already built

        if self.datamodule is None:
            logger.warning("Datamodule not available for session metadata")
            return

        train_ds = getattr(self.datamodule, 'train_dataset', None)
        val_ds = getattr(self.datamodule, 'val_dataset', None)

        dfs = [ds.metadata_df for ds in [train_ds, val_ds] if ds is not None]
        if dfs:
            full_df = pd.concat(dfs)
            self.session_metadata = full_df.drop_duplicates(
                subset=['session_id']
            ).set_index('session_id')
            logger.info("Built session metadata lookup from train+val datasets")

    ### Accumulation & Aggregation Methods ###

    def accumulate_patches(
        self,
        data: ClassificationBatchLogData,
        stage: Literal["train", "val", "test"],
    ):
        """
        Simple classic validation on the patch level - accumulate logits, labels, and valid_masks.
        This is the closest measure to asses overfitting during training.

        All of the argument are already detached from computation graph and on CPU.
        Args:
            data: Batch data context
            stage: Validation stage ("val" or "test")
        """
        global_idxs = [md['global_idx'] for md in data.patches_metadata]
        self.logging_patch_storage[stage]["global_idx"].append(global_idxs)
        self.logging_patch_storage[stage]["logits"].append(data.logits)
        self.logging_patch_storage[stage]["labels"].append(data.labels)
        self.logging_patch_storage[stage]["losses"].append(data.losses)
        self.logging_patch_storage[stage]["valid_masks"].append(data.valid_masks)
        if data.probabilities is not None:
            self.logging_patch_storage[stage]["probabilities"].append(data.probabilities)

    def accumulate_spectrals(
        self,
        data: ClassificationBatchLogData,
        stage: Literal["val", "test"],
    ):
        if data.x is None:
            return

        predictions = data.logits.argmax(dim=-1)
        valid_indices = data.valid_masks.nonzero().squeeze(-1)

        for i in valid_indices:
            metadata = data.patches_metadata[i]
            gt = data.labels[i].item()
            prediction = predictions[i].item()
            status = "correct" if prediction == gt else "wrong"

            # Store only one sample per gt per status
            if not self.logging_spectral_storage[stage][status][gt]:
                spectral = {
                    'x': data.x[i],
                    'ground_truth': gt,
                    'prediction': prediction,
                    'status': status,
                    'protocol_type': metadata['protocol'],
                    'patient_id': metadata['patient_id'],
                    'session_id': metadata['session_id'],
                    'session_idx': metadata['session_idx'],
                }
                self.logging_spectral_storage[stage][status][gt] = spectral

    def accumulate_timestamps(
        self,
        data: ClassificationBatchLogData,
        stage: Literal["val", "test"],
    ):
        """Accumulate model outputs for later aggregation, preserving temporal structure."""
        logits = data.logits
        labels = data.labels
        valid_masks = data.valid_masks

        if logits.ndim == 2:  # if it is patch classification backbone
            logits = logits.unsqueeze(1).repeat(1, self.seq_len, 1)
            if valid_masks.ndim == 1:
                valid_masks = valid_masks.unsqueeze(1).repeat(1, self.seq_len)

        for i, metadata in enumerate(data.patches_metadata):  # iterate through the ordered batch
            sid = metadata['session_id']
            self.logging_timestamps_storage[stage]["labels"][sid].append(labels[i])
            self.logging_timestamps_storage[stage]["logits"][sid].append(logits[i])
            self.logging_timestamps_storage[stage]["valid_masks"][sid].append(valid_masks[i])

    def aggregate_timestamps(self, stage: Literal["val", "test"]):
        """Aggregate accumulated timestamps into session-level data."""
        self._build_session_metadata()  # Ensure session metadata is available

        data_dict = self.logging_timestamps_storage[stage]
        session_ids = list(data_dict["labels"].keys())

        for session_id in session_ids:
            self._aggregate_single_session(data_dict, session_id)

    def _aggregate_single_session(self, data_dict: Dict, session_id: str):
        """Helper to aggregate a single session's data using vectorized operations."""
        # Stack lists into tensors: (N_blocks, seq_len, C)
        session_logits = torch.stack(data_dict["logits"][session_id])
        session_labels = torch.stack(data_dict["labels"][session_id])
        session_masks = torch.stack(data_dict["valid_masks"][session_id])

        n_blocks, seq_len, n_classes = session_logits.shape

        # Calculate dimensions
        padded_session_length = (n_blocks - 1) * self.stride_len + self.seq_len
        session_length = self.get_session_length(session_id)

        if padded_session_length < session_length:
            raise ValueError(f"Padded session length {padded_session_length} < original {session_length} for {session_id}")

        # Create indices for scatter_add
        # shape: (N_blocks, seq_len) -> values are global indices
        block_starts = torch.arange(n_blocks) * self.stride_len
        indices = block_starts.unsqueeze(1) + torch.arange(seq_len)
        indices = indices.view(-1)  # Flatten to (N_blocks * seq_len)

        # Prepare aggregated tensors
        t = padded_session_length

        # 1. Logits aggregation (Sum then divide by counts)
        aggregated_logits = torch.zeros((t, n_classes), device=session_logits.device, dtype=session_logits.dtype)
        counts = torch.zeros((t,), device=session_logits.device, dtype=torch.float32)

        # Flatten source data for index_add
        flat_logits = session_logits.view(-1, n_classes)

        # Vectorized accumulation
        aggregated_logits.index_add_(0, indices, flat_logits)
        counts.index_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))

        # Normalize
        mask = counts > 0
        aggregated_logits[mask] /= counts[mask].unsqueeze(-1)

        # 2. Labels and Masks aggregation
        aggregated_labels = torch.full((t, session_labels.shape[-1]), fill_value=float('nan'))
        aggregated_valid_masks = torch.zeros((t, session_masks.shape[-1]), dtype=torch.bool)

        for i in range(n_blocks):
            start = i * self.stride_len
            end = start + self.seq_len
            aggregated_labels[start:end] = session_labels[i]
            aggregated_valid_masks[start:end] |= session_masks[i]

        # Trim to original length
        valid_mask = aggregated_valid_masks[:session_length]

        data_dict[session_id]["labels"] = aggregated_labels[:session_length][valid_mask]
        data_dict[session_id]["logits"] = aggregated_logits[:session_length][valid_mask]
        data_dict[session_id]["valid_masks"] = valid_mask

    ### Logging Methods ###

    def log_spectrals(self, stage: Literal["val", "test"]):
        """Log performance-based spectral visualizations to wandb."""
        for status, status_dict in self.logging_spectral_storage[stage].items():
            for class_idx, spectral_sample in status_dict.items():
                if not spectral_sample:
                    continue

                wandb_image = self.create_spectral_plot(spectral_sample)
                plot_tag = self.create_plot_tag(stage, spectral_sample)
                self.logger.experiment.log({plot_tag: wandb_image})

    def log_patch_performance_table(self, stage: Literal["train", "val", "test"]):
        """
        Create and log WandB table focused on patch performance tracking.

        :param stage: Stage of logging ("train", "val", or "test")
        """
        # Get metadata - cache it if first time accessing this stage
        metadata_df = self.metadata_dataframes.setdefault(
            stage,
            getattr(self.datamodule, f"{stage}_dataset").metadata_df
        )

        indices = np.concatenate(self.logging_patch_storage[stage]['global_idx'])
        losses = torch.cat(self.logging_patch_storage[stage]['losses']).numpy()

        performance_data = {'loss': losses}

        # Add ground truth labels (needed for confusion matrix, ROC, etc.)
        labels = torch.cat(self.logging_patch_storage[stage]['labels']).numpy()
        performance_data['label'] = labels

        # Add predicted label and predicted class probability (val/test only)
        if self.logging_patch_storage[stage]['probabilities']:
            probabilities = torch.cat(self.logging_patch_storage[stage]['probabilities'])
            predicted_labels = probabilities.argmax(dim=-1).numpy()
            predicted_class_probabilities = probabilities.max(dim=-1).values.numpy()
            performance_data['predicted_label'] = predicted_labels
            performance_data['predicted_class_probability'] = predicted_class_probabilities

            # Store ALL per-class probabilities (supports multiclass)
            # Creates columns: prob_class_0, prob_class_1, ..., prob_class_N
            for class_idx in range(self.num_classes):
                performance_data[f'prob_class_{class_idx}'] = probabilities[:, class_idx].numpy()

        performance_df = pd.DataFrame(performance_data, index=pd.Index(indices, name='global_idx'))
        merged_df = metadata_df.join(performance_df, how='inner')
        merged_df = merged_df.sort_values(by='loss', ascending=False)

        # Cache DataFrame for reuse by visualization methods
        self.patch_performance_dfs[stage] = merged_df

        df_for_table = merged_df.reset_index()
        if len(df_for_table) > 200000:
            df_for_table = df_for_table.head(200000)

        table_name = WandBKeys.patch_analysis_table(stage)
        table = wandb.Table(dataframe=df_for_table)
        self.logger.experiment.log({table_name: table})

        # Log summary statistics
        summary = {
            'total_patches': len(merged_df),
            'mean_loss': merged_df['loss'].mean(),
            'std_loss': merged_df['loss'].std(),
        }
        self.logger.experiment.log({
            WandBKeys.patch_summary_total(stage): summary['total_patches'],
            WandBKeys.patch_summary_mean_loss(stage): summary['mean_loss'],
            WandBKeys.patch_summary_std_loss(stage): summary['std_loss'],
        })

    def log_patch_metadata_table(self, stage: Literal["train", "val", "test"]):
        """Log patch metadata table to WandB."""
        metadata_df = self.metadata_dataframes.setdefault(
            stage,
            getattr(self.datamodule, f"{stage}_dataset").metadata_df
        )

        table_name = WandBKeys.patch_metadata_table(stage)
        table = wandb.Table(dataframe=metadata_df.reset_index())
        self.logger.experiment.log({table_name: table})

    def log_patient_histogram(self, stage: Literal["train", "val", "test"]):
        """Log patient distribution histogram to WandB."""
        # Get metadata - cache it if first time accessing this stage
        metadata_df = self.metadata_dataframes.setdefault(
            stage,
            getattr(self.datamodule, f"{stage}_dataset").metadata_df
        )

        patient_counts = metadata_df['patient_id'].value_counts().sort_index()
        self.logger.experiment.log({
            WandBKeys.patient_histogram(stage): wandb.plot.bar(
                wandb.Table(data=[[str(pid), count] for pid, count in patient_counts.items()],
                           columns=["patient_id", "patch_count"]),
                "patient_id", "patch_count",
                title=f"Patches per Patient - {stage.upper()}"
            )
        })

    def get_session_length(self, session_id: str) -> int:
        """Get the original length of a session from its raw file."""
        path = self.resolve_session_path(session_id)
        read_func = pd.read_csv if path.endswith('.csv') else pd.read_parquet
        df = read_func(path)
        return len(df)

    def resolve_session_path(self, session_id: str) -> str:
        """Resolve the file path for a given session ID."""
        self._build_session_metadata()  # Ensure session metadata is available

        if session_id not in self.session_metadata.index:
            raise KeyError(f"Session ID {session_id} not found in metadata lookup.")

        session_row = self.session_metadata.loc[session_id]
        path = os.path.join(self.raw_root,
                    self.dataset_source,
                    session_row['protocol'],
                    'sessions',
                    f"{session_id}.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Session file not found at resolved path: {path}")

        return path

    def clear_validation_data(self):
        """Clear validation data structures."""
        self.clear_logging_data("val")

    def clear_test_data(self):
        """Clear test data structures."""
        self.clear_logging_data("test")

    def clear_logging_data(self, stage: Literal["train", "val", "test"]):
        """Clear all logging data structures for a given stage."""
        self.logging_patch_storage[stage] = {k: [] for k in self.patch_logging_keys}
        self.logging_spectral_storage[stage] = {
            cw: {k: {} for k in range(self.num_classes)}
                for cw in ("correct", "wrong")
        }
        # Reset timestamps storage
        self.logging_timestamps_storage[stage] = {
            k: defaultdict(list) for k in self.timestamps_logging_keys
        }
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def create_plot_tag(self, stage: str, spectral_sample: Dict) -> str:
        gt = spectral_sample['ground_truth']
        status = spectral_sample['status']
        plot_name = WandBKeys.spectral_plot(stage, status, gt)
        return plot_name

    def create_spectral_plot(self, sample: Dict) -> wandb.Image:
        x = sample["x"].detach().cpu().numpy()  # (3, L)
        gt = sample["ground_truth"]
        pred = sample["prediction"]
        status = sample["status"]

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(x[0], label="X-axis")
        ax.plot(x[1], label="Y-axis")
        ax.plot(x[2], label="Z-axis")

        ax.set_title(f"Spectral Sample | GT={gt}, Pred={pred}, Status={status}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Acceleration")
        ax.legend()

        caption = (
            f"GT={gt}, Pred={pred}, Status={status}, "
            f"Protocol={sample['protocol_type']}, "
            f"Patient={sample['patient_id']}, "
            f"Session={sample['session_id']}, "
            f"Patch={sample['session_idx']}"
        )

        return wandb.Image(fig, caption=caption)

    ### Enhanced Classification Visualizations ###

    def log_confusion_matrix(self, stage: Literal["val", "test"]) -> None:
        """Create and log confusion matrix from cached DataFrame."""
        df = self.patch_performance_dfs.get(stage)
        if df is None or 'label' not in df.columns:
            return

        y_true = df['label'].values
        y_pred = df['predicted_label'].values

        class_names = [f"Class {i}" for i in range(self.num_classes)]
        wandb_cm = wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=class_names
        )

        self.logger.experiment.log({
            WandBKeys.confusion_matrix(stage): wandb_cm
        })
        logger.info(f"{stage.upper()} Confusion matrix logged")

    def log_roc_curve(self, stage: Literal["val", "test"]) -> None:
        """Create and log ROC curves (one-vs-rest for multiclass)."""
        df = self.patch_performance_dfs.get(stage)
        if df is None or 'label' not in df.columns:
            return

        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        y_true = df['label'].values

        # Binary classification
        if self.num_classes == 2:
            y_score = df['prob_class_1'].values
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            # Plot single ROC curve
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(fpr, tpr, label=f'ROC curve (AUC={roc_auc:.3f})', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Curve - {stage.upper()}', fontsize=14)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)

            self.logger.experiment.log({
                WandBKeys.roc_curve(stage): wandb.Image(fig),
                WandBKeys.roc_auc(stage): roc_auc
            })
            plt.close(fig)

        # Multiclass: one-vs-rest ROC curves
        else:
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            fig, ax = plt.subplots(figsize=(10, 8))

            for class_idx in range(self.num_classes):
                y_score = df[f'prob_class_{class_idx}'].values
                fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_score)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'Class {class_idx} (AUC={roc_auc:.3f})', linewidth=2)

            ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Curves (One-vs-Rest) - {stage.upper()}', fontsize=14)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)

            self.logger.experiment.log({
                WandBKeys.roc_curve(stage): wandb.Image(fig)
            })
            plt.close(fig)

        logger.info(f"{stage.upper()} ROC curve logged")

    def log_precision_recall_curve(self, stage: Literal["val", "test"]) -> None:
        """Create and log Precision-Recall curves (one-vs-rest for multiclass)."""
        df = self.patch_performance_dfs.get(stage)
        if df is None or 'label' not in df.columns:
            return

        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize

        y_true = df['label'].values

        # Binary classification
        if self.num_classes == 2:
            y_score = df['prob_class_1'].values
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap_score = average_precision_score(y_true, y_score)

            # Plot single PR curve
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(recall, precision, label=f'PR curve (AP={ap_score:.3f})', linewidth=2)
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title(f'Precision-Recall Curve - {stage.upper()}', fontsize=14)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

            self.logger.experiment.log({
                WandBKeys.pr_curve(stage): wandb.Image(fig),
                WandBKeys.average_precision(stage): ap_score
            })
            plt.close(fig)

        # Multiclass: one-vs-rest PR curves
        else:
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            fig, ax = plt.subplots(figsize=(10, 8))

            for class_idx in range(self.num_classes):
                y_score = df[f'prob_class_{class_idx}'].values
                precision, recall, _ = precision_recall_curve(y_true_bin[:, class_idx], y_score)
                ap_score = average_precision_score(y_true_bin[:, class_idx], y_score)
                ax.plot(recall, precision, label=f'Class {class_idx} (AP={ap_score:.3f})', linewidth=2)

            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title(f'Precision-Recall Curves (One-vs-Rest) - {stage.upper()}', fontsize=14)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

            self.logger.experiment.log({
                WandBKeys.pr_curve(stage): wandb.Image(fig)
            })
            plt.close(fig)

        logger.info(f"{stage.upper()} Precision-Recall curve logged")

    def log_per_patient_metrics(self, stage: Literal["val", "test"]) -> None:
        """
        Compute and log patient-level aggregated metrics.
        Aggregates patch-level predictions to patient level via majority voting.

        Args:
            stage: Validation or test stage
        """
        if not self._can_log_to_wandb():
            return

        df = self.patch_performance_dfs.get(stage)
        if df is None or 'label' not in df.columns or 'patient_id' not in df.columns:
            logger.warning(f"Cannot compute per-patient metrics for {stage}: missing required columns")
            return

        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

        # Aggregate by patient (majority voting)
        patient_metrics = []
        for patient_id, group in df.groupby('patient_id'):
            # Majority vote for patient-level label and prediction
            true_label = group['label'].mode()[0]  # Most common label
            pred_label = group['predicted_label'].mode()[0]  # Most common prediction
            mean_prob = group['predicted_class_probability'].mean()  # Average confidence

            patient_metrics.append({
                'patient_id': patient_id,
                'true_label': int(true_label),
                'predicted_label': int(pred_label),
                'mean_confidence': float(mean_prob),
                'num_patches': len(group)
            })

        patient_df = pd.DataFrame(patient_metrics)

        # Compute patient-level metrics
        y_true = patient_df['true_label'].values
        y_pred = patient_df['predicted_label'].values

        avg_type = 'binary' if self.num_classes == 2 else 'macro'
        patient_f1 = f1_score(y_true, y_pred, average=avg_type, zero_division=0)
        patient_accuracy = accuracy_score(y_true, y_pred)
        patient_precision = precision_score(y_true, y_pred, average=avg_type, zero_division=0)
        patient_recall = recall_score(y_true, y_pred, average=avg_type, zero_division=0)

        # Log scalar metrics
        self.logger.experiment.log({
            WandBKeys.patient_f1(stage): patient_f1,
            WandBKeys.patient_accuracy(stage): patient_accuracy,
            WandBKeys.patient_precision(stage): patient_precision,
            WandBKeys.patient_recall(stage): patient_recall,
        })

        table_name = WandBKeys.patient_metrics_table(stage)
        table = wandb.Table(dataframe=patient_df)
        self.logger.experiment.log({table_name: table})

        logger.info(
            f"{stage.upper()} Patient-Level Metrics: "
            f"F1={patient_f1:.4f}, Acc={patient_accuracy:.4f}, "
            f"n_patients={len(patient_df)}"
        )

    def log_calibration_plot(self, stage: Literal["val", "test"]) -> None:
        """
        Wrapper to call base class calibration plot with classification data.

        For binary: single calibration curve
        For multiclass: calibration per class (log multiple plots)

        Args:
            stage: Validation or test stage
        """
        if not self._can_log_to_wandb():
            return

        df = self.patch_performance_dfs.get(stage)
        if df is None or 'label' not in df.columns:
            logger.warning(f"Cannot compute calibration plot for {stage}: missing required columns")
            return

        # Binary: single plot
        if self.num_classes == 2:
            if 'prob_class_1' not in df.columns:
                logger.warning(f"Cannot compute calibration plot for {stage}: missing prob_class_1")
                return

            probabilities = df['prob_class_1'].values
            labels = df['label'].values
            super().log_calibration_plot(probabilities, labels, stage)

        # Multiclass: one calibration plot per class
        else:
            for class_idx in range(self.num_classes):
                prob_col = f'prob_class_{class_idx}'
                if prob_col not in df.columns:
                    logger.warning(f"Skipping calibration for class {class_idx}: missing {prob_col}")
                    continue

                # Get binary labels (class_idx vs rest)
                binary_labels = (df['label'].values == class_idx).astype(int)
                class_probs = df[prob_col].values

                # Log with class-specific key
                super().log_calibration_plot(
                    class_probs,
                    binary_labels,
                    stage,
                    class_name=f"Class_{class_idx}"
                )

        logger.info(f"{stage.upper()} Calibration plot(s) logged")

    def log_protocol_stratified_metrics(self, stage: Literal["val", "test"]) -> None:
        """
        Compute and log metrics stratified by protocol type.
        Shows performance breakdown across different data collection protocols.

        Args:
            stage: Validation or test stage
        """
        if not self._can_log_to_wandb():
            return

        df = self.patch_performance_dfs.get(stage)
        if df is None or 'label' not in df.columns or 'protocol' not in df.columns:
            logger.warning(f"Cannot compute protocol-stratified metrics for {stage}: missing required columns")
            return

        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

        # Compute per-protocol metrics
        protocol_metrics = []
        avg_type = 'binary' if self.num_classes == 2 else 'macro'

        for protocol, group in df.groupby('protocol'):
            y_true = group['label'].values
            y_pred = group['predicted_label'].values

            # Compute metrics
            f1 = f1_score(y_true, y_pred, average=avg_type, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average=avg_type, zero_division=0)
            rec = recall_score(y_true, y_pred, average=avg_type, zero_division=0)

            protocol_metrics.append({
                'protocol': protocol,
                'f1': float(f1),
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'num_samples': int(len(group))
            })

            # Log individual protocol metrics
            self.logger.experiment.log({
                WandBKeys.protocol_f1(stage, protocol): f1,
                WandBKeys.protocol_accuracy(stage, protocol): acc,
                WandBKeys.protocol_precision(stage, protocol): prec,
                WandBKeys.protocol_recall(stage, protocol): rec,
            })

        protocol_df = pd.DataFrame(protocol_metrics)
        table_name = WandBKeys.protocol_metrics_table(stage)
        table = wandb.Table(dataframe=protocol_df)
        self.logger.experiment.log({table_name: table})


        # Create bar chart comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics_to_plot = ['f1', 'accuracy', 'precision', 'recall']
        titles = ['F1 Score', 'Accuracy', 'Precision', 'Recall']

        for ax, metric, title in zip(axes.flatten(), metrics_to_plot, titles):
            protocol_df.plot(x='protocol', y=metric, kind='bar', ax=ax, legend=False, color='steelblue')
            ax.set_title(f'{title} by Protocol - {stage.upper()}', fontsize=12)
            ax.set_ylabel(title, fontsize=10)
            ax.set_xlabel('Protocol', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        self.logger.experiment.log({
            WandBKeys.protocol_metrics_chart(stage): wandb.Image(fig)
        })
        plt.close(fig)

        logger.info(
            f"{stage.upper()} Protocol-Stratified Metrics logged for "
            f"{len(protocol_df)} protocols"
        )

    def log_classification_visualizations(self, stage: Literal["val", "test"]) -> None:
        """
        Log all classification-specific visualizations and metrics.

        Orchestrates: confusion matrix, ROC, PR curves, per-patient metrics,
        calibration plot, protocol-stratified metrics.

        Args:
            stage: Validation or test stage
        """
        if not self._can_log_to_wandb():
            logger.debug("WandB logging not available, skipping classification visualizations")
            return

        # Check if we have cached DataFrame
        if stage not in self.patch_performance_dfs:
            logger.warning(f"No cached DataFrame for {stage}, skipping visualizations")
            return

        df = self.patch_performance_dfs[stage]
        if df.empty:
            logger.warning(f"Empty DataFrame for {stage}, skipping visualizations")
            return

        logger.info(f"Logging classification visualizations for {stage}")

        # Feature 1: Confusion Matrix
        try:
            self.log_confusion_matrix(stage)
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

        # Features 2-3: ROC and PR curves (all num_classes)
        try:
            self.log_roc_curve(stage)
        except Exception as e:
            logger.warning(f"Failed to log ROC curve: {e}")

        try:
            self.log_precision_recall_curve(stage)
        except Exception as e:
            logger.warning(f"Failed to log PR curve: {e}")

        # Feature 5: Per-patient metrics
        if 'patient_id' in df.columns:
            try:
                self.log_per_patient_metrics(stage)
            except Exception as e:
                logger.warning(f"Failed to log per-patient metrics: {e}")

        # Feature 6: Calibration plot
        try:
            self.log_calibration_plot(stage)
        except Exception as e:
            logger.warning(f"Failed to log calibration plot: {e}")

        # Feature 7: Protocol-stratified metrics
        if 'protocol' in df.columns:
            try:
                self.log_protocol_stratified_metrics(stage)
            except Exception as e:
                logger.warning(f"Failed to log protocol-stratified metrics: {e}")

        logger.info(f"Classification visualizations completed for {stage}")

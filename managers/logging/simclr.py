"""
SimCLR (Simple Framework for Contrastive Learning) logging management.
Handles embedding visualization, contrastive analysis, and clustering quality logging.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Literal, List

import torch
import torch.nn.functional as F
import numpy as np

from .pretrain_base import PretrainingLoggingManager
from utils.wandb_logging_keys import WandBKeys

if TYPE_CHECKING:
    from pipeline.config import Config
    from pipeline.schemas import SimCLRBatchLogData

logger = logging.getLogger(__name__)


class SimCLRLoggingManager(PretrainingLoggingManager):
    """Logging manager specialized for SimCLR contrastive pretraining tasks."""

    def __init__(
        self,
        config: "Config",
        device: str,
        trainer: Any,
        datamodule: Optional[Any] = None,
    ):
        """
        Initialize SimCLR logging manager.

        Args:
            config: Unified pipeline configuration (Pydantic model)
            device: Device for tensor operations
            trainer: PyTorch Lightning trainer (provides logger and global_step)
            datamodule: Optional datamodule for accessing dataset metadata
        """
        super().__init__(config, device, trainer, datamodule)
        # seq_len is inherited from BaseLoggingManager (config.data.process.lengths.block_len)
        self._setup_simclr_logging()

    def _setup_simclr_logging(self):
        """Setup SimCLR-specific logging structures."""
        # Embedding tracking
        self.validation_embeddings = []
        self.test_embeddings = []

        # Contrastive analysis
        self.contrastive_statistics = {
            "positive_similarities": [],
            "negative_similarities": [],
            "temperature_effects": {},
            "separation_metrics": []
        }

        # Clustering analysis
        self.clustering_metrics = []

        # Patch-level data storage for detailed analysis
        self.validation_patch_data = []
        self.test_patch_data = []

        # Representation interval for conditional accumulation (from config)
        logging_config = self.config.train.logging
        self.representation_interval = logging_config.representation_interval if logging_config else 1
        self.max_samples_per_epoch = 5  # Max representation samples to store

        # Patch-level metric accumulators (val/test only, cleared each epoch)
        self._val_pos_sims: list = []    # list of [N] tensors: per-position positive similarity
        self._val_neg_sims: list = []    # list of [N] tensors: per-position negative similarity
        self._val_pos_losses: list = []  # list of [N] tensors: per-position NT-Xent loss

    def log_embedding_sample(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None, stage: str = "val"):
        """Log embedding samples for visualization."""
        sample = {
            "embeddings": embeddings.detach().cpu(),
            "labels": labels.detach().cpu() if labels is not None else None,
        }
        
        if stage == "val" and len(self.validation_embeddings) < self.max_samples_per_epoch:
            self.validation_embeddings.append(sample)
        elif stage == "test" and len(self.test_embeddings) < self.max_samples_per_epoch:
            self.test_embeddings.append(sample)

    def log_contrastive_statistics(self, pos_sim: float, neg_sim: float, separation: float):
        """Log contrastive learning statistics."""
        self.contrastive_statistics["positive_similarities"].append(pos_sim)
        self.contrastive_statistics["negative_similarities"].append(neg_sim)
        self.contrastive_statistics["separation_metrics"].append(separation)

    def log_temperature_effect(self, temperature: float, separation_score: float):
        """Log temperature effect on contrastive learning."""
        if temperature not in self.contrastive_statistics["temperature_effects"]:
            self.contrastive_statistics["temperature_effects"][temperature] = []
        self.contrastive_statistics["temperature_effects"][temperature].append(separation_score)

    def log_clustering_metrics(self, metrics: Dict[str, float]):
        """Log clustering quality metrics."""
        self.clustering_metrics.append(metrics)

    def accumulate_patches(
        self,
        data: 'SimCLRBatchLogData',
        stage: Literal["train", "val", "test"]
    ):
        """
        Accumulate patch-level data for detailed analysis.

        Args:
            data: SimCLRBatchLogData containing batch information
            stage: Current stage ("train", "val", "test")
        """
        if stage == "train":
            return  # Don't accumulate patches during training

        patch_data_list = self.validation_patch_data if stage == "val" else self.test_patch_data

        # Get sequence length from config or default
        seq_len = self.seq_len

        # Extract data from batch context
        losses = data.losses
        metadata_list = data.metadata

        for i, metadata in enumerate(metadata_list):
            # Extract loss for this sample
            if losses.dim() == 0:  # Single scalar loss for entire batch
                patch_loss = losses.item()
            elif losses.dim() == 1 and i < len(losses):  # Per-sample losses
                patch_loss = losses[i].item()
            else:
                # Fail fast instead of silent fallback
                raise ValueError(
                    f"Unexpected loss shape: {losses.shape}. Expected scalar or [batch_size]. "
                    f"This indicates a bug in the loss computation."
                )

            # Extract session information from metadata
            session_id = metadata.get('session_id', 'unknown')
            patient_id = metadata.get('patient_id', 'unknown')
            global_idx = metadata.get('global_idx', -1)

            # Calculate frame range in session
            start_frame = global_idx * seq_len if global_idx >= 0 else -1
            end_frame = start_frame + seq_len if start_frame >= 0 else -1

            # Build patch entry
            patch_entry = {
                'patient_id': patient_id,
                'session_id': session_id,
                'full_session_id': f"{patient_id}_{session_id}",
                'session_idx': metadata.get('session_idx', -1),
                'global_idx': global_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'loss': patch_loss,
                'stage': stage,
                # SimCLR-specific metrics can be added here
                'contrastive_loss': patch_loss,
            }
            patch_data_list.append(patch_entry)

    def log_representation_samples(
        self,
        data: 'SimCLRBatchLogData',
        stage: Literal["val", "test"]
    ):
        """
        Log representation samples for visualization.

        Args:
            data: SimCLRBatchLogData containing representation information
            stage: Current stage ("val", "test")
        """
        sample_list = self.validation_embeddings if stage == "val" else self.test_embeddings

        # Only store a limited number of samples per epoch
        if len(sample_list) >= self.max_samples_per_epoch:
            return

        # Store first sample from batch for visualization
        sample = {
            "z1": data.z1[:1],  # Already on CPU from detach_cpu
            "z2": data.z2[:1],
            "metadata": data.metadata[:1]
        }
        sample_list.append(sample)

    def accumulate_patch_metrics(
        self,
        data: 'SimCLRBatchLogData',
        stage: Literal["val", "test"],
    ):
        """
        Accumulate per-position similarity and loss statistics for epoch-end plots.
        Only runs when z1_patches is populated (patch-level SimCLR mode).
        """
        if data.z1_patches is None:
            return

        z1 = data.z1_patches.float()   # [B, N, D] — already on CPU
        z2 = data.z2_patches.float()
        B, N, D = z1.shape

        z1_n = F.normalize(z1, dim=-1)
        z2_n = F.normalize(z2, dim=-1)

        # Per-position positive similarity: mean over samples
        pos_sims = (z1_n * z2_n).sum(dim=-1).mean(dim=0)  # [N]

        # Per-position negative similarity: mean of off-diagonal cross-similarities
        z1_t = z1_n.permute(1, 0, 2)   # [N, B, D]
        z2_t = z2_n.permute(1, 0, 2)
        cross = torch.bmm(z1_t, z2_t.transpose(1, 2))  # [N, B, B]
        eye = torch.eye(B, dtype=torch.bool)
        cross.masked_fill_(eye.unsqueeze(0), 0.0)
        neg_sims = cross.sum(dim=(1, 2)) / max(B * (B - 1), 1)  # [N]

        self._val_pos_sims.append(pos_sims)
        self._val_neg_sims.append(neg_sims)

        if data.per_position_losses is not None:
            self._val_pos_losses.append(data.per_position_losses.float())

    def _log_per_position_gap(self, stage: str, epoch: int):
        """
        Bar chart: per-position (positive − negative) similarity gap.
        Two panels: gap bars + pos/neg overlay.
        """
        if not self._val_pos_sims:
            return
        try:
            import matplotlib.pyplot as plt
            import wandb

            pos = torch.stack(self._val_pos_sims).mean(0).numpy()   # [N]
            neg = torch.stack(self._val_neg_sims).mean(0).numpy()   # [N]
            gap = pos - neg
            N = len(gap)

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            colors = ['green' if g > 0 else 'red' for g in gap]
            axes[0].bar(range(N), gap, color=colors, alpha=0.75)
            axes[0].axhline(0, color='black', lw=1, ls='--')
            axes[0].set_xlabel('Patch position')
            axes[0].set_ylabel('Positive − Negative similarity')
            axes[0].set_title(f'Per-position similarity gap  (epoch {epoch})')

            axes[1].plot(pos, label='positive', color='green', lw=1.5, alpha=0.8)
            axes[1].plot(neg, label='negative', color='red',   lw=1.5, alpha=0.8)
            axes[1].fill_between(range(N), neg, pos, alpha=0.15, color='blue')
            axes[1].set_xlabel('Patch position')
            axes[1].set_ylabel('Cosine similarity')
            axes[1].set_title('Positive vs Negative per position')
            axes[1].legend()

            plt.tight_layout()
            self.logger.experiment.log({f"patch/{stage}_position_gap": wandb.Image(fig)})
            self._cleanup_matplotlib_memory()

        except Exception as e:
            logger.warning(f"Failed to create per-position gap plot: {e}")

    def _log_per_position_loss_heatmap(self, stage: str, epoch: int):
        """
        5×20 heatmap of per-position losses mapped to spectrogram patch grid.
        Rows = frequency bands (low→high), columns = time windows (left→right).
        Hot colours = harder patches.
        """
        if not self._val_pos_losses:
            return
        try:
            import matplotlib.pyplot as plt
            import wandb

            losses = torch.stack(self._val_pos_losses).mean(0).numpy()  # [N]
            N = len(losses)

            # SpectralPatchEncoder default: freq_patch=20 → nH=5; time_patch=50 → nW=20
            nH, nW = 5, N // 5
            if N % 5 != 0:
                logger.warning(
                    f"N={N} patches not divisible by 5; heatmap will be 1×{N} instead of {nH}×{nW}"
                )
                nH, nW = 1, N

            heatmap = losses.reshape(nH, nW)

            fig, ax = plt.subplots(figsize=(10, 3))
            im = ax.imshow(heatmap, aspect='auto', cmap='hot_r', origin='lower')
            plt.colorbar(im, ax=ax, label='NT-Xent loss')
            ax.set_xlabel('Time window →')
            ax.set_ylabel('Freq band (low→high)')
            ax.set_title(f'Per-position loss heatmap  (epoch {epoch})  — hot = harder to align')
            plt.tight_layout()

            self.logger.experiment.log({f"patch/{stage}_position_loss_heatmap": wandb.Image(fig)})
            self._cleanup_matplotlib_memory()

        except Exception as e:
            logger.warning(f"Failed to create per-position loss heatmap: {e}")

    def log_representation_visualizations(self, stage: Literal["val", "test"]):
        """
        Log representation visualizations at epoch end.

        Args:
            stage: Current stage ("val", "test")
        """
        if not self._can_log_to_wandb():
            return

        epoch = getattr(self.logger, 'current_epoch', 0)

        # Create embedding visualization
        self._create_embedding_visualization(stage, epoch)

        # Create contrastive analysis plots if we have statistics
        if self.contrastive_statistics["positive_similarities"]:
            self._create_contrastive_analysis_plots(epoch)

    def log_patch_performance_table(self, stage: Literal["train", "val", "test"]):
        """
        Log patch-level performance table.

        Args:
            stage: Current stage ("train", "val", "test")
        """
        if stage == "train":
            return  # Don't log patch tables for training

        patch_data_list = self.validation_patch_data if stage == "val" else self.test_patch_data

        if not patch_data_list:
            logger.debug(f"No patch data to log for {stage} stage")
            return

        # Create DataFrame for easy analysis
        try:
            import pandas as pd

            df = pd.DataFrame(patch_data_list)

            # Compute summary statistics
            summary = {
                'total_patches': len(df),
                'mean_loss': df['loss'].mean(),
                'std_loss': df['loss'].std(),
                'min_loss': df['loss'].min(),
                'max_loss': df['loss'].max(),
            }

            # Log summary to logger
            if self._can_log_to_wandb():
                self.logger.experiment.log({
                    WandBKeys.patch_summary_total(stage): summary['total_patches'],
                    WandBKeys.patch_summary_mean_loss(stage): summary['mean_loss'],
                    WandBKeys.patch_summary_std_loss(stage): summary['std_loss'],
                })

            logger.info(f"{stage.upper()} Patch Summary: {summary}")

        except Exception as e:
            logger.warning(f"Failed to create patch performance table: {e}")

    def clear_validation_data(self):
        """Clear validation data structures."""
        super().clear_validation_data()
        self.validation_patch_data.clear()
        self.validation_embeddings.clear()
        self._val_pos_sims.clear()
        self._val_neg_sims.clear()
        self._val_pos_losses.clear()

    def clear_test_data(self):
        """Clear test data structures."""
        super().clear_test_data()
        self.test_patch_data.clear()
        self.test_embeddings.clear()

    def _log_task_specific_visualizations(self, stage: str, epoch: int):
        """Log SimCLR-specific visualizations."""
        self._create_embedding_visualization(stage, epoch)
        self._create_contrastive_analysis_plots(epoch)
        self._create_temperature_analysis_plots(epoch)
        self.log_patch_tables(stage)

        # Patch-level visualizations (only populated in patch-SimCLR mode)
        self._log_per_position_gap(stage, epoch)
        self._log_per_position_loss_heatmap(stage, epoch)

    def _create_embedding_visualization(self, stage: str, epoch: int):
        """Create t-SNE/PCA visualization of embedding space."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA

            embeddings_data = self.validation_embeddings if stage == "val" else self.test_embeddings

            if not embeddings_data:
                return None

            # Combine z1 embeddings from all samples
            # Note: z1 and z2 should be similar after training, so we use z1 for visualization
            all_z1 = []

            for sample in embeddings_data:
                all_z1.append(sample["z1"])

            if not all_z1:
                return None

            embeddings_combined = torch.cat(all_z1, dim=0).numpy()
            labels_combined = None  # Pretraining datasets don't have labels

            # Check minimum samples for dimensionality reduction
            if embeddings_combined.shape[0] < 2:
                logger.warning(
                    f"Need at least 2 samples for embedding visualization, got {embeddings_combined.shape[0]}. "
                    f"Skipping visualization for epoch {epoch}."
                )
                return None

            # Use PCA for large datasets, t-SNE for smaller ones
            if embeddings_combined.shape[0] > 1000:
                reducer = PCA(n_components=2)
                embeddings_2d = reducer.fit_transform(embeddings_combined)
                method = "PCA"
            else:
                # Ensure perplexity is at least 1 for small datasets
                perplexity = max(1, min(30, embeddings_combined.shape[0] - 1))
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                embeddings_2d = reducer.fit_transform(embeddings_combined)
                method = "t-SNE"
            
            fig, ax = plt.subplots(figsize=(10, 8))

            # No labels in pretraining datasets - simple scatter plot
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=20)
            
            ax.set_title(f'SimCLR Embedding Space ({method}) - Epoch {epoch} - {stage.upper()}')
            ax.set_xlabel(f'{method} Component 1')
            ax.set_ylabel(f'{method} Component 2')
            
            # Log to wandb via logger
            import wandb
            self.logger.experiment.log({WandBKeys.simclr_embeddings(stage): wandb.Image(fig)})

            self._cleanup_matplotlib_memory()
            return fig
            
        except ImportError:
            logger.warning("Required packages (matplotlib, sklearn) not available for embedding visualization")
            return None
        except Exception as e:
            logger.warning(f"Error creating embedding visualization: {e}")
            return None

    def _create_contrastive_analysis_plots(self, epoch: int):
        """Create plots analyzing contrastive learning effectiveness."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.contrastive_statistics["positive_similarities"]:
                return None

            # Check negative similarities too (needed for box plot)
            if not self.contrastive_statistics["negative_similarities"]:
                logger.warning("Missing negative similarity data, skipping contrastive analysis")
                return None

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Positive vs negative similarity distributions
            axes[0].hist(self.contrastive_statistics["positive_similarities"],
                        bins=20, alpha=0.7, label='Positive', edgecolor='black')
            axes[0].hist(self.contrastive_statistics["negative_similarities"],
                        bins=20, alpha=0.7, label='Negative', edgecolor='black')
            axes[0].set_title('Similarity Distributions')
            axes[0].set_xlabel('Cosine Similarity')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Separation metrics over time
            if self.contrastive_statistics["separation_metrics"]:
                axes[1].plot(self.contrastive_statistics["separation_metrics"], alpha=0.7)
                axes[1].set_title('Positive-Negative Separation')
                axes[1].set_xlabel('Training Step')
                axes[1].set_ylabel('Separation Score')
                axes[1].grid(True, alpha=0.3)
            
            # Box plot comparison
            pos_sims = self.contrastive_statistics["positive_similarities"]
            neg_sims = self.contrastive_statistics["negative_similarities"]
            axes[2].boxplot([pos_sims, neg_sims], labels=['Positive', 'Negative'])
            axes[2].set_title('Similarity Comparison')
            axes[2].set_ylabel('Cosine Similarity')
            axes[2].grid(True, alpha=0.3)
            
            plt.suptitle(f'SimCLR Contrastive Analysis - Epoch {epoch}')
            plt.tight_layout()
            
            # Save via wandb if available
            import wandb
            self.logger.experiment.log({WandBKeys.SIMCLR_CONTRASTIVE_ANALYSIS: wandb.Image(fig)})

            self._cleanup_matplotlib_memory()
            return fig
            
        except ImportError:
            logger.warning("matplotlib not available for contrastive analysis plots")
            return None
        except Exception as e:
            logger.warning(f"Error creating contrastive analysis plots: {e}")
            return None

    def _create_temperature_analysis_plots(self, epoch: int):
        """Create plots analyzing temperature effects."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.contrastive_statistics["temperature_effects"]:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))

            # Filter out temperatures with empty data to prevent NaN values
            temperatures = list(self.contrastive_statistics["temperature_effects"].keys())
            valid_temps = []
            avg_separations = []
            std_separations = []

            for temp in temperatures:
                scores = self.contrastive_statistics["temperature_effects"][temp]
                if len(scores) > 0:  # Only include if has data
                    valid_temps.append(temp)
                    avg_separations.append(np.mean(scores))
                    std_separations.append(np.std(scores))

            if not valid_temps:
                logger.warning("No valid temperature data for analysis")
                return None

            ax.errorbar(valid_temps, avg_separations, yerr=std_separations,
                       marker='o', capsize=5, capthick=2, linewidth=2)
            ax.set_title(f'Temperature Effect Analysis - Epoch {epoch}')
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Average Separation Score')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')

            # Highlight optimal temperature
            if valid_temps:
                optimal_idx = np.argmax(avg_separations)
                optimal_temp = valid_temps[optimal_idx]
                ax.axvline(x=optimal_temp, color='red', linestyle='--', alpha=0.7,
                          label=f'Optimal: {optimal_temp:.3f}')
                ax.legend()
            
            plt.tight_layout()
            
            # Log to wandb via logger
            import wandb
            self.logger.experiment.log({WandBKeys.SIMCLR_TEMPERATURE_ANALYSIS: wandb.Image(fig)})

            self._cleanup_matplotlib_memory()
            return fig
            
        except ImportError:
            logger.warning("matplotlib not available for temperature analysis plots")
            return None
        except Exception as e:
            logger.warning(f"Error creating temperature analysis plots: {e}")
            return None

    def log_patch_tables(self, stage: str):
        """Create WandB table from accumulated patch data."""
        patch_data = self.validation_patch_data if stage == "val" else self.test_patch_data
        
        if not patch_data:
            return
        
        try:
            # Convert to pandas DataFrame for WandB table
            import pandas as pd
            import wandb
            df = pd.DataFrame(patch_data)

            table_name = WandBKeys.patch_analysis_table(stage)
            # Reset index to ensure sequential 0, 1, 2, ... for WandB compatibility
            table = wandb.Table(dataframe=df.reset_index(drop=True))
            self.logger.experiment.log({table_name: table})

            logger.info(f"Logged {len(patch_data)} patch entries for {stage}")
            
        except Exception as e:
            logger.error(f"Error creating patch table for {stage}: {e}")

    def get_simclr_summary(self) -> Dict[str, Any]:
        """Get comprehensive SimCLR logging summary."""
        summary = super().get_training_summary()
        
        # Add SimCLR-specific summaries
        if self.contrastive_statistics["positive_similarities"]:
            summary["avg_positive_similarity"] = np.mean(self.contrastive_statistics["positive_similarities"])
            summary["avg_negative_similarity"] = np.mean(self.contrastive_statistics["negative_similarities"])
        
        if self.contrastive_statistics["separation_metrics"]:
            summary["final_separation"] = self.contrastive_statistics["separation_metrics"][-1]
            summary["avg_separation"] = np.mean(self.contrastive_statistics["separation_metrics"])
        
        if self.contrastive_statistics["temperature_effects"]:
            # Find optimal temperature
            temp_scores = {}
            for temp, scores in self.contrastive_statistics["temperature_effects"].items():
                temp_scores[temp] = np.mean(scores)
            
            if temp_scores:
                optimal_temp = max(temp_scores, key=temp_scores.get)
                summary["optimal_temperature"] = optimal_temp
                summary["max_separation_at_optimal"] = temp_scores[optimal_temp]
        
        if self.clustering_metrics:
            summary["total_clustering_evaluations"] = len(self.clustering_metrics)
        
        return summary
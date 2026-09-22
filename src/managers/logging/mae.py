"""
MAE (Masked Autoencoder) logging management for self-supervised pretraining.
Handles reconstruction visualizations, masking analysis, and spectral fidelity logging.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Literal, List

import torch
import numpy as np

from .pretrain_base import PretrainingLoggingManager
from utils.wandb_logging_keys import WandBKeys

if TYPE_CHECKING:
    from pipeline.config import Config
    from pipeline.schemas import MAEBatchLogData

logger = logging.getLogger(__name__)


class MAELoggingManager(PretrainingLoggingManager):
    """Logging manager specialized for MAE pretraining tasks."""

    def __init__(
        self,
        config: "Config",
        device: str,
        trainer: Any,
        datamodule: Optional[Any] = None,
    ):
        """
        Initialize MAE logging manager.

        Args:
            config: Unified pipeline configuration (Pydantic model)
            device: Device for tensor operations
            trainer: PyTorch Lightning trainer (provides logger and global_step)
            datamodule: Optional datamodule for accessing dataset metadata
        """
        super().__init__(config, device, trainer, datamodule)
        # seq_len is inherited from BaseLoggingManager (config.data.process.lengths.block_len)

        # Get patch_size from backbone config (Pydantic attribute access)
        self.patch_size = getattr(config.model.backbone, 'patch_size', 10)
        self.mask_mode = getattr(config.model, 'mask_mode', 'temporal')

        self._setup_mae_logging()

    def _setup_mae_logging(self):
        """Setup MAE-specific logging structures."""
        # Reconstruction tracking
        self.validation_reconstructions = []
        self.test_reconstructions = []

        # Masking analysis
        self.mask_statistics = {
            "mask_ratios": [],
            "reconstruction_quality_by_mask": {},
            "spectral_fidelity_scores": []
        }

        # Patch-level analysis
        self.patch_quality_metrics = []

        # Patch-level data storage for detailed analysis
        self.validation_patch_data = []
        self.test_patch_data = []

        # Reconstruction interval for conditional accumulation (from config)
        logging_config = self.config.train.logging
        self.reconstruction_interval = logging_config.reconstruction_interval if logging_config else 1
        self.max_samples_per_epoch = 5  # Max reconstruction samples to store

    def log_reconstruction_sample(self, 
                                original: torch.Tensor, 
                                reconstructed: torch.Tensor,
                                mask: Optional[torch.Tensor] = None,
                                stage: str = "val"):
        """Log a reconstruction sample for visualization."""
        sample = {
            "original": original[:1].detach().cpu(),  # Take first sample
            "reconstructed": reconstructed[:1].detach().cpu(),
            "mask": mask[:1].detach().cpu() if mask is not None else None,
        }
        
        if stage == "val" and len(self.validation_reconstructions) < self.max_samples_per_epoch:
            self.validation_reconstructions.append(sample)
        elif stage == "test" and len(self.test_reconstructions) < self.max_samples_per_epoch:
            self.test_reconstructions.append(sample)

    def log_mask_statistics(self, mask: torch.Tensor, reconstruction_quality: float):
        """Log statistics about masking effectiveness."""
        if mask is not None:
            with torch.no_grad():
                if mask.dtype == torch.bool:
                    mask_ratio = mask.float().mean().item()
                else:
                    mask_ratio = (mask > 0.5).float().mean().item()
                
                self.mask_statistics["mask_ratios"].append(mask_ratio)
                
                # Track reconstruction quality by mask ratio
                mask_bin = round(mask_ratio, 1)
                if mask_bin not in self.mask_statistics["reconstruction_quality_by_mask"]:
                    self.mask_statistics["reconstruction_quality_by_mask"][mask_bin] = []
                self.mask_statistics["reconstruction_quality_by_mask"][mask_bin].append(reconstruction_quality)

    def log_spectral_fidelity(self, fidelity_score: float):
        """Log spectral fidelity score."""
        self.mask_statistics["spectral_fidelity_scores"].append(fidelity_score)

    def log_patch_quality_metrics(self, metrics: Dict[str, float]):
        """Log patch-level quality metrics."""
        self.patch_quality_metrics.append(metrics)

    def accumulate_patches(
        self,
        data: 'MAEBatchLogData',
        stage: Literal["train", "val", "test"]
    ):
        """
        Accumulate patch-level data for detailed analysis.

        Args:
            data: MAEBatchLogData containing batch information
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
                # MAE-specific metrics can be added here
                'reconstruction_loss': patch_loss,
            }
            patch_data_list.append(patch_entry)

    def log_reconstruction_samples(
        self,
        data: 'MAEBatchLogData',
        stage: Literal["val", "test"]
    ):
        """
        Log reconstruction samples for visualization.

        Args:
            data: MAEBatchLogData containing reconstruction information
            stage: Current stage ("val", "test")
        """
        sample_list = self.validation_reconstructions if stage == "val" else self.test_reconstructions

        # Only store a limited number of samples per epoch
        if len(sample_list) >= self.max_samples_per_epoch:
            return

        # Pick a random sample from the batch for visual diversity across epochs
        batch_size = data.original.shape[0]
        idx = torch.randint(batch_size, (1,)).item()

        sample = {
            "original": data.original[idx:idx+1],
            "reconstructed": data.reconstructed[idx:idx+1],
            "mask": data.spectral_mask[idx:idx+1] if data.spectral_mask is not None else None,
            "loss": data.losses[idx].item() if data.losses is not None and len(data.losses) > idx else None,
        }
        sample_list.append(sample)

    def log_reconstruction_visualizations(self, stage: Literal["val", "test"]):
        """
        Log reconstruction visualizations at epoch end.

        Args:
            stage: Current stage ("val", "test")
        """
        if not self._can_log_to_wandb():
            return

        epoch = getattr(self.logger, 'current_epoch', 0)

        # Create reconstruction visualization
        self._create_reconstruction_visualization(stage, epoch)

        # Create mask analysis plots if we have mask statistics
        if self.mask_statistics["mask_ratios"]:
            self._create_mask_analysis_plots(epoch)

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
        self.validation_reconstructions.clear()

    def clear_test_data(self):
        """Clear test data structures."""
        super().clear_test_data()
        self.test_patch_data.clear()
        self.test_reconstructions.clear()

    def _log_task_specific_visualizations(self, stage: str, epoch: int):
        """Log MAE-specific visualizations."""
        # Create reconstruction comparisons
        self._create_reconstruction_visualization(stage, epoch)
        
        # Create masking analysis plots
        self._create_mask_analysis_plots(epoch)
        
        # Create spectral fidelity plots
        self._create_spectral_fidelity_plots(epoch)
        
        # Log patch-level data tables
        self.log_patch_tables(stage)

    def _create_reconstruction_visualization(self, stage: str, epoch: int):
        """Create visualization comparing original vs reconstructed signals.

        H==1 (raw signal): n_samples rows × (C+1) columns — one column per channel
            plus a mean-error column. Original (dashed) and reconstructed (solid)
            overlaid, masked patches shaded red, shared y-limits per channel,
            per-sample loss annotated on row ylabel.

        H>1 (spectrogram): n_samples rows × 3 columns — Original | Reconstructed |
            |Error|. Shared colorbar range for original/reconstructed, independent
            range for error. Masked patches indicated by white vertical lines.
        """
        try:
            import matplotlib.pyplot as plt
            import wandb

            samples = self.validation_reconstructions if stage == "val" else self.test_reconstructions

            if not samples:
                return None

            n_samples = min(3, len(samples))
            patch_size = self.patch_size

            # Determine layout from first sample shape
            first_orig = samples[0]["original"][0].numpy()
            if first_orig.ndim == 2:
                first_orig = first_orig[:, np.newaxis, :]  # treat [C, T] as [C, 1, T]
            C_0, H_0, _ = first_orig.shape

            if H_0 == 1:
                # Raw signal branch: one column per channel + mean-error column
                channel_labels = (
                    ["AccV", "AccML", "AccAP"] if C_0 == 3
                    else [f"Ch {i}" for i in range(C_0)]
                )
                n_cols = C_0 + 1

                fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 4 * n_samples))
                if n_samples == 1:
                    axes = axes.reshape(1, -1)

                for i, sample in enumerate(samples[:n_samples]):
                    original = sample["original"][0].numpy()
                    reconstructed = sample["reconstructed"][0].numpy()
                    mask = sample["mask"][0].numpy() if sample["mask"] is not None else None
                    loss = sample.get("loss")

                    # Normalize to [C, H, W]
                    if original.ndim == 2:
                        original = original[:, np.newaxis, :]
                        reconstructed = reconstructed[:, np.newaxis, :]

                    C = original.shape[0]
                    orig_2d = original[:, 0, :]       # [C, T]
                    recon_2d = reconstructed[:, 0, :]  # [C, T]
                    error_2d = np.abs(orig_2d - recon_2d)
                    mean_error = error_2d.mean(axis=0)  # [T]

                    for c in range(C):
                        ax = axes[i, c]

                        ymin = min(orig_2d[c].min(), recon_2d[c].min())
                        ymax = max(orig_2d[c].max(), recon_2d[c].max())
                        padding = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
                        ax.set_ylim(ymin - padding, ymax + padding)

                        ax.plot(orig_2d[c], '--', color='gray', alpha=0.5, label='original')
                        ax.plot(recon_2d[c], label='reconstructed')

                        if i == 0:
                            label = channel_labels[c] if c < len(channel_labels) else f"Ch {c}"
                            ax.set_title(label)

                        if c == 0:
                            ylabel = (
                                f"Sample {i+1}\nLoss={loss:.5f}" if loss is not None
                                else f"Sample {i+1}"
                            )
                            ax.set_ylabel(ylabel)

                        if mask is not None:
                            is_causal = self.mask_mode == "causal"
                            first_masked = next((p for p, m in enumerate(mask) if m), None)
                            for p, is_masked in enumerate(mask):
                                if is_masked:
                                    ax.axvspan(
                                        p * patch_size, (p + 1) * patch_size,
                                        alpha=0.15, color='red', lw=0
                                    )
                            if is_causal and first_masked is not None and c == 0:
                                ax.axvline(
                                    first_masked * patch_size,
                                    color='red', alpha=0.9, lw=2.0, ls='--', zorder=5,
                                )

                    # Error panel (last column)
                    ax_err = axes[i, C]
                    ax_err.plot(mean_error, color='black')
                    ax_err.grid(alpha=0.3)

                    if i == 0:
                        ax_err.set_title("|Error| (mean ch)")

                    if mask is not None:
                        is_causal = self.mask_mode == "causal"
                        first_masked = next((p for p, m in enumerate(mask) if m), None)
                        if is_causal and first_masked is not None:
                            # Causal: split line only — keep error magnitude readable
                            ax_err.axvline(
                                first_masked * patch_size,
                                color='red', alpha=0.9, lw=2.0, ls='--', zorder=5,
                            )
                            ax_err.text(
                                first_masked * patch_size - 1,
                                ax_err.get_ylim()[1] if ax_err.get_ylim()[1] != 1.0 else mean_error.max(),
                                'visible →',
                                ha='right', va='top', fontsize=6, color='red',
                            )
                            ax_err.text(
                                first_masked * patch_size + 1,
                                ax_err.get_ylim()[1] if ax_err.get_ylim()[1] != 1.0 else mean_error.max(),
                                '← predicted',
                                ha='left', va='top', fontsize=6, color='red',
                            )
                        else:
                            for p, is_masked in enumerate(mask):
                                if is_masked:
                                    ax_err.axvspan(
                                        p * patch_size, (p + 1) * patch_size,
                                        alpha=0.15, color='red', lw=0
                                    )

                    # Legend on first row, last column
                    if i == 0:
                        from matplotlib.patches import Patch
                        from matplotlib.lines import Line2D
                        legend_elements = [
                            Line2D([0], [0], linestyle='--', color='gray', alpha=0.5, label='original'),
                            Line2D([0], [0], color='C0', label='reconstructed'),
                            Patch(facecolor='red', alpha=0.15, label='masked region'),
                        ]
                        if self.mask_mode == "causal":
                            from matplotlib.lines import Line2D as _L2D
                            legend_elements.append(
                                _L2D([0], [0], color='red', lw=2, ls='--', label='causal split')
                            )
                        ax_err.legend(handles=legend_elements, loc='upper right', fontsize='small')

            else:
                # Spectrogram branch: one combined figure, n_samples*C_vis rows × 3 cols.
                # Consistent with the raw signal branch — single wandb.Image, one tile per epoch.
                _first = samples[0]["original"][0].numpy()  # [C, H, W]
                C_vis, _H, _W = _first.shape
                ch_names = ["X", "Y", "Z"][:C_vis] if C_vis <= 3 else [f"ch{c}" for c in range(C_vis)]
                col_width_in = 5.0
                row_height_in = max(1.5, col_width_in * _H / _W)
                n_rows = n_samples * C_vis

                fig, axes = plt.subplots(
                    n_rows, 3,
                    figsize=(3 * col_width_in, row_height_in * n_rows),
                    squeeze=False,
                )

                for i, sample in enumerate(samples[:n_samples]):
                    original = sample["original"][0].numpy()            # [C, H, W]
                    reconstructed = sample["reconstructed"][0].numpy()  # [C, H, W]
                    mask = sample["mask"][0].numpy() if sample["mask"] is not None else None
                    loss = sample.get("loss")

                    for c in range(C_vis):
                        row = i * C_vis + c
                        orig_ch = original[c]        # [H, W]
                        recon_ch = reconstructed[c]  # [H, W]
                        recon_ch = np.nan_to_num(recon_ch, nan=0.0, posinf=0.0, neginf=0.0)
                        error_ch = np.abs(orig_ch - recon_ch)
                        error_ch = np.nan_to_num(error_ch, nan=0.0, posinf=0.0, neginf=0.0)

                        orig_log = np.log1p(np.maximum(orig_ch, 0))
                        recon_log = np.log1p(np.maximum(recon_ch, 0))
                        p2, p98 = np.percentile(orig_log[orig_log > 0], [2, 98]) if (orig_log > 0).any() else (0, 1)
                        vmin, vmax = p2, max(p98, p2 + 1e-6)

                        ch_label = ch_names[c]
                        loss_str = f" — Loss={loss:.5f}" if (loss is not None and c == 0) else ""
                        axes[row, 0].set_title(f"S{i+1} Original — {ch_label}{loss_str}")
                        axes[row, 1].set_title(f"S{i+1} Reconstructed — {ch_label}")
                        axes[row, 2].set_title(f"S{i+1} |Error| — {ch_label}")

                        im0 = axes[row, 0].imshow(orig_log, aspect='auto', cmap='viridis',
                                                   vmin=vmin, vmax=vmax, origin='lower')
                        plt.colorbar(im0, ax=axes[row, 0])

                        im1 = axes[row, 1].imshow(recon_log, aspect='auto', cmap='viridis',
                                                   vmin=vmin, vmax=vmax, origin='lower')
                        plt.colorbar(im1, ax=axes[row, 1])

                        im2 = axes[row, 2].imshow(error_ch, aspect='auto', cmap='Reds', origin='lower')
                        plt.colorbar(im2, ax=axes[row, 2])

                        if mask is not None:
                            for col_idx, ax_col in enumerate([axes[row, 0], axes[row, 1], axes[row, 2]]):
                                if self.mask_mode == "frequency":
                                    for band_idx, is_masked in enumerate(mask):
                                        if is_masked:
                                            ax_col.axhspan(band_idx - 0.5, band_idx + 0.5, alpha=0.2, color='red', lw=0)
                                elif self.mask_mode == "2d_patch":
                                    H_img, W_img = orig_ch.shape
                                    nH_mask, nW_mask = mask.shape
                                    freq_patch_size = max(1, H_img // nH_mask)
                                    from matplotlib.patches import Rectangle
                                    for fh in range(nH_mask):
                                        for fw in range(nW_mask):
                                            if mask[fh, fw]:
                                                x = fw * patch_size
                                                y = fh * freq_patch_size
                                                rect = Rectangle(
                                                    (x - 0.5, y - 0.5),
                                                    patch_size, freq_patch_size,
                                                    linewidth=0, edgecolor='none',
                                                    facecolor='white', alpha=0.35,
                                                )
                                                ax_col.add_patch(rect)
                                else:
                                    is_causal = self.mask_mode == "causal"
                                    H_img = orig_ch.shape[0]
                                    first_masked = next(
                                        (p for p, m in enumerate(mask) if m), None
                                    )
                                    if col_idx == 2:
                                        if is_causal and first_masked is not None:
                                            ax_col.axvline(
                                                first_masked * patch_size,
                                                color='red', alpha=0.9, lw=2.0, ls='--', zorder=5,
                                            )
                                            split_x = first_masked * patch_size
                                            ax_col.text(
                                                split_x - 1, H_img * 0.97,
                                                'visible →', ha='right', va='top',
                                                fontsize=6, color='red', fontweight='bold',
                                            )
                                            ax_col.text(
                                                split_x + 1, H_img * 0.97,
                                                '← predicted', ha='left', va='top',
                                                fontsize=6, color='red', fontweight='bold',
                                            )
                                    else:
                                        for p, is_masked_patch in enumerate(mask):
                                            if is_masked_patch:
                                                ax_col.axvspan(
                                                    p * patch_size, (p + 1) * patch_size,
                                                    alpha=0.22, color='red', lw=0, zorder=3,
                                                )
                                        if is_causal and first_masked is not None:
                                            ax_col.axvline(
                                                first_masked * patch_size,
                                                color='red', alpha=0.9, lw=2.0, ls='--', zorder=5,
                                            )
                                        if is_causal and col_idx == 1 and first_masked is not None:
                                            split_x = first_masked * patch_size
                                            ax_col.text(
                                                split_x - 1, H_img * 0.97,
                                                'visible →', ha='right', va='top',
                                                fontsize=6, color='white', fontweight='bold',
                                            )
                                            ax_col.text(
                                                split_x + 1, H_img * 0.97,
                                                '← predicted', ha='left', va='top',
                                                fontsize=6, color='white', fontweight='bold',
                                            )

                plt.suptitle(f'MAE Reconstructions — Epoch {epoch} — {stage.upper()}')
                plt.tight_layout()
                self.logger.experiment.log({WandBKeys.mae_reconstructions(stage): wandb.Image(fig)})
                plt.close(fig)
                self._cleanup_matplotlib_memory()
                return fig

            plt.suptitle(f'MAE Reconstructions — Epoch {epoch} — {stage.upper()}')
            plt.tight_layout()

            self.logger.experiment.log({WandBKeys.mae_reconstructions(stage): wandb.Image(fig)})
            self._cleanup_matplotlib_memory()
            return fig

        except ImportError:
            logger.warning("matplotlib not available for reconstruction visualization")
            return None
        except Exception as e:
            logger.warning(f"Error creating reconstruction visualization: {e}")
            return None

    def _create_mask_analysis_plots(self, epoch: int):
        """Create plots analyzing mask effectiveness."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.mask_statistics["mask_ratios"]:
                return None
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Mask ratio distribution
            axes[0].hist(self.mask_statistics["mask_ratios"], bins=20, alpha=0.7, edgecolor='black')
            axes[0].set_title('Mask Ratio Distribution')
            axes[0].set_xlabel('Mask Ratio')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
            
            # Reconstruction quality by mask ratio
            if self.mask_statistics["reconstruction_quality_by_mask"]:
                mask_ratios = list(self.mask_statistics["reconstruction_quality_by_mask"].keys())
                avg_qualities = [np.mean(self.mask_statistics["reconstruction_quality_by_mask"][ratio]) 
                               for ratio in mask_ratios]
                std_qualities = [np.std(self.mask_statistics["reconstruction_quality_by_mask"][ratio]) 
                               for ratio in mask_ratios]
                
                axes[1].errorbar(mask_ratios, avg_qualities, yerr=std_qualities, 
                               marker='o', capsize=5, capthick=2)
                axes[1].set_title('Reconstruction Quality vs Mask Ratio')
                axes[1].set_xlabel('Mask Ratio')
                axes[1].set_ylabel('Reconstruction Quality')
                axes[1].grid(True, alpha=0.3)
            
            # Spectral fidelity over time
            if self.mask_statistics["spectral_fidelity_scores"]:
                axes[2].plot(self.mask_statistics["spectral_fidelity_scores"], alpha=0.7)
                axes[2].set_title('Spectral Fidelity Over Training')
                axes[2].set_xlabel('Step')
                axes[2].set_ylabel('Spectral Fidelity Score')
                axes[2].grid(True, alpha=0.3)
            
            plt.suptitle(f'MAE Mask Analysis - Epoch {epoch}')
            plt.tight_layout()
            
            # Save via wandb if available
            import wandb
            self.logger.experiment.log({WandBKeys.MAE_MASK_ANALYSIS: wandb.Image(fig)})

            self._cleanup_matplotlib_memory()
            return fig
            
        except ImportError:
            logger.warning("matplotlib not available for mask analysis plots")
            return None
        except Exception as e:
            logger.warning(f"Error creating mask analysis plots: {e}")
            return None

    def _create_spectral_fidelity_plots(self, epoch: int):
        """Create plots showing spectral reconstruction fidelity."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.mask_statistics["spectral_fidelity_scores"]:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scores = self.mask_statistics["spectral_fidelity_scores"]
            ax.plot(scores, alpha=0.7, linewidth=2)
            ax.set_title(f'Spectral Fidelity Progress - Epoch {epoch}')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Spectral Fidelity Score')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(scores) > 10:
                x = np.arange(len(scores))
                z = np.polyfit(x, scores, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "--", alpha=0.5, color='red', label='Trend')
                ax.legend()
            
            plt.tight_layout()
            
            # Log to wandb via logger
            import wandb
            self.logger.experiment.log({WandBKeys.MAE_SPECTRAL_FIDELITY: wandb.Image(fig)})

            self._cleanup_matplotlib_memory()
            return fig
            
        except ImportError:
            logger.warning("matplotlib not available for spectral fidelity plots")
            return None
        except Exception as e:
            logger.warning(f"Error creating spectral fidelity plots: {e}")
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

    def get_mae_summary(self) -> Dict[str, Any]:
        """Get comprehensive MAE logging summary."""
        summary = super().get_training_summary()
        
        # Add MAE-specific summaries
        if self.mask_statistics["mask_ratios"]:
            summary["avg_mask_ratio"] = np.mean(self.mask_statistics["mask_ratios"])
            summary["mask_ratio_std"] = np.std(self.mask_statistics["mask_ratios"])
        
        if self.mask_statistics["spectral_fidelity_scores"]:
            summary["final_spectral_fidelity"] = self.mask_statistics["spectral_fidelity_scores"][-1]
            summary["avg_spectral_fidelity"] = np.mean(self.mask_statistics["spectral_fidelity_scores"])
        
        if self.patch_quality_metrics:
            summary["total_patch_evaluations"] = len(self.patch_quality_metrics)
        
        return summary
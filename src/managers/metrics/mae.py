"""
MAE (Masked Autoencoder) metrics management for self-supervised pretraining.
Handles reconstruction quality, spectral fidelity, and masking effectiveness metrics.
"""

import logging
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr

from .pretrain_base import PretrainingMetricsManager

logger = logging.getLogger(__name__)


class MAEMetricsManager(PretrainingMetricsManager):
    """Metrics manager specialized for MAE pretraining tasks."""

    def __init__(self, device: torch.device, track_gradients: bool = True,
                 spectral_metrics: bool = True):
        """
        Initialize MAE metrics manager.

        Args:
            device: Device to place metrics on
            track_gradients: Whether to track gradient statistics
            spectral_metrics: Whether to compute spectral domain metrics
        """
        self.spectral_metrics = spectral_metrics
        self.mask_statistics = {
            "mask_ratios": [],
            "reconstruction_quality_by_mask": {},
            "masked_vs_unmasked_error": []
        }
        
        super().__init__(device, track_gradients)

    def to(self, device: Optional[torch.device] = None):
        """Move all metrics to the specified device."""
        if device is not None:
            self.device = device
        return self

    def reset_metrics(self):
        """Reset all metrics to their initial state."""
        self.mask_statistics = {
            "mask_ratios": [],
            "reconstruction_quality_by_mask": {},
            "masked_vs_unmasked_error": []
        }
        if hasattr(super(), 'reset'):
            super().reset()

    def update_metrics(self, probas: torch.Tensor, targets: torch.Tensor):
        """MAE does not use the standard probas/targets paradigm — no-op."""
        pass

    def compute_reconstruction_metrics(self, outputs: torch.Tensor,
                                     targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute MAE-specific reconstruction metrics.
        
        Args:
            outputs: Reconstructed data
            targets: Original data
            
        Returns:
            Dictionary of MAE reconstruction metrics
        """
        metrics = {}
        
        with torch.no_grad():
            # Basic reconstruction metrics
            metrics["mse_loss"] = F.mse_loss(outputs, targets).item()
            metrics["mae_loss"] = F.l1_loss(outputs, targets).item()
            
            # Peak Signal-to-Noise Ratio (PSNR)
            mse = F.mse_loss(outputs, targets)
            if mse > 0:
                max_val = torch.max(targets)
                psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
                metrics["psnr"] = psnr.item()
            
            # Structural similarity metrics
            metrics.update(self._compute_similarity_metrics(outputs, targets))
            
            # SSIM metrics for spectral data
            if outputs.dim() == 4 and targets.dim() == 4:  # Spectral data [B, C, H, W]
                ssim_metrics = self._compute_ssim_metrics(outputs, targets)
                metrics.update(ssim_metrics)
            
            # Spectral domain metrics if enabled
            if self.spectral_metrics:
                spectral_metrics = self._compute_spectral_metrics(outputs, targets)
                metrics.update(spectral_metrics)
        
        return metrics

    def _compute_similarity_metrics(self, outputs: torch.Tensor, 
                                  targets: torch.Tensor) -> Dict[str, float]:
        """Compute structural similarity metrics."""
        metrics = {}
        
        # Flatten for correlation computation
        outputs_flat = outputs.flatten()
        targets_flat = targets.flatten()
        
        # Pearson correlation
        if len(outputs_flat) > 1:
            outputs_np = outputs_flat.cpu().numpy()
            targets_np = targets_flat.cpu().numpy()
            
            if np.std(outputs_np) > 1e-10 and np.std(targets_np) > 1e-10:
                correlation, _ = pearsonr(outputs_np, targets_np)
                metrics["pearson_correlation"] = correlation
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(outputs_flat, targets_flat, dim=0)
        metrics["cosine_similarity"] = cos_sim.item()
        
        # Normalized cross-correlation
        outputs_norm = (outputs - outputs.mean()) / (outputs.std() + 1e-8)
        targets_norm = (targets - targets.mean()) / (targets.std() + 1e-8)
        ncc = F.cosine_similarity(outputs_norm.flatten(), targets_norm.flatten(), dim=0)
        metrics["normalized_cross_correlation"] = ncc.item()
        
        return metrics

    def _compute_ssim_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute SSIM metrics for spectral data evaluation.
        
        Args:
            outputs: Reconstructed spectral data [B, C, H, W]
            targets: Original spectral data [B, C, H, W]
            
        Returns:
            Dictionary with SSIM metrics
        """
        metrics = {}
        
        try:
            from model.losses import SSIMLoss
            
            # Create SSIM loss instance for metrics computation
            ssim_computer = SSIMLoss(window_size=11, data_range=1.0, channel_reduction='mean')
            ssim_computer = ssim_computer.to(outputs.device)
            
            # Compute overall SSIM (returns 1 - SSIM, so we need to invert)
            ssim_loss = ssim_computer(outputs, targets)
            ssim_value = 1.0 - ssim_loss.item()
            metrics["ssim"] = ssim_value
            
            # Compute per-channel SSIM for detailed analysis
            if outputs.shape[1] > 1:  # Multi-channel data
                channel_names = ['AccV', 'AccML', 'AccAP']  # Accelerometer channels
                for c in range(min(outputs.shape[1], len(channel_names))):
                    channel_outputs = outputs[:, c:c+1, :, :]
                    channel_targets = targets[:, c:c+1, :, :]
                    
                    try:
                        # Create a new SSIM computer for each channel to avoid state issues
                        channel_ssim_computer = SSIMLoss(window_size=11, data_range=1.0, channel_reduction='mean')
                        channel_ssim_computer = channel_ssim_computer.to(outputs.device)
                        channel_ssim_loss = channel_ssim_computer(channel_outputs, channel_targets)
                        channel_ssim_value = 1.0 - channel_ssim_loss.item()
                        metrics[f"ssim_{channel_names[c]}"] = channel_ssim_value
                    except Exception as e:
                        logger.warning(f"Failed to compute SSIM for channel {channel_names[c]}: {e}")
            
            # Compute multi-scale SSIM with different window sizes
            for window_size in [5, 7, 11]:
                try:
                    ms_ssim_computer = SSIMLoss(window_size=window_size, data_range=1.0, channel_reduction='mean')
                    ms_ssim_computer = ms_ssim_computer.to(outputs.device)
                    ms_ssim_loss = ms_ssim_computer(outputs, targets)
                    ms_ssim_value = 1.0 - ms_ssim_loss.item()
                    metrics[f"ssim_w{window_size}"] = ms_ssim_value
                except Exception as e:
                    logger.warning(f"Failed to compute SSIM with window size {window_size}: {e}")
                    
        except ImportError:
            logger.warning("SSIM loss not available for metrics computation")
        except Exception as e:
            logger.warning(f"Error computing SSIM metrics: {e}")
        
        return metrics

    def _compute_spectral_metrics(self, outputs: torch.Tensor, 
                                targets: torch.Tensor) -> Dict[str, float]:
        """Compute frequency domain reconstruction metrics."""
        metrics = {}
        
        try:
            # Handle different tensor shapes
            if outputs.dim() == 4:  # [B, C, H, W] - 2D spectrograms/wavelets
                # Average over batch and channels
                outputs_2d = outputs.mean(dim=(0, 1))  # [H, W]
                targets_2d = targets.mean(dim=(0, 1))
                
                # Compute 2D FFT
                outputs_fft = torch.fft.fft2(outputs_2d)
                targets_fft = torch.fft.fft2(targets_2d)
                
            elif outputs.dim() == 3:  # [B, C, T] - 1D signals
                # Average over batch and channels
                outputs_1d = outputs.mean(dim=(0, 1))  # [T]
                targets_1d = targets.mean(dim=(0, 1))
                
                # Compute 1D FFT
                outputs_fft = torch.fft.fft(outputs_1d)
                targets_fft = torch.fft.fft(targets_1d)
                
            else:
                return metrics  # Unsupported shape
            
            # Spectral magnitude comparison
            outputs_mag = torch.abs(outputs_fft)
            targets_mag = torch.abs(targets_fft)
            
            # Spectral MSE
            spectral_mse = F.mse_loss(outputs_mag, targets_mag)
            metrics["spectral_mse"] = spectral_mse.item()
            
            # Spectral correlation
            spectral_corr = F.cosine_similarity(outputs_mag.flatten(), 
                                              targets_mag.flatten(), dim=0)
            metrics["spectral_correlation"] = spectral_corr.item()
            
            # Phase coherence
            outputs_phase = torch.angle(outputs_fft)
            targets_phase = torch.angle(targets_fft)
            phase_diff = torch.abs(outputs_phase - targets_phase)
            # Wrap phase differences to [-π, π]
            phase_diff = torch.remainder(phase_diff + np.pi, 2*np.pi) - np.pi
            metrics["phase_coherence"] = (1 - torch.mean(torch.abs(phase_diff)) / np.pi).item()
            
            # Spectral energy preservation
            outputs_energy = torch.sum(outputs_mag ** 2)
            targets_energy = torch.sum(targets_mag ** 2)
            energy_ratio = outputs_energy / (targets_energy + 1e-8)
            metrics["spectral_energy_ratio"] = energy_ratio.item()
            
        except Exception as e:
            logger.warning(f"Error computing spectral metrics: {e}")
        
        return metrics

    def log_masking_statistics(self, mask: torch.Tensor, reconstruction_error: torch.Tensor):
        """Log statistics about masking effectiveness."""
        if mask is None:
            return
        
        with torch.no_grad():
            # Mask ratio
            if mask.dtype == torch.bool:
                mask_ratio = mask.float().mean().item()
            else:
                mask_ratio = (mask > 0.5).float().mean().item()
            
            self.mask_statistics["mask_ratios"].append(mask_ratio)
            
            # Reconstruction quality by mask ratio (binned)
            mask_bin = round(mask_ratio, 1)  # Bin to 0.1 precision
            if mask_bin not in self.mask_statistics["reconstruction_quality_by_mask"]:
                self.mask_statistics["reconstruction_quality_by_mask"][mask_bin] = []
            
            self.mask_statistics["reconstruction_quality_by_mask"][mask_bin].append(
                reconstruction_error.mean().item()
            )

    def compute_patch_level_metrics(self, outputs: torch.Tensor, targets: torch.Tensor,
                                   patch_size: int = 16) -> Dict[str, float]:
        """
        Compute patch-level reconstruction metrics.
        
        Args:
            outputs: Reconstructed data
            targets: Original data  
            patch_size: Size of patches for analysis
            
        Returns:
            Dictionary of patch-level metrics
        """
        metrics = {}
        
        if outputs.dim() == 4:  # [B, C, H, W]
            B, C, H, W = outputs.shape
            
            # Extract patches
            patches_out = outputs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            patches_tgt = targets.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            
            # Reshape to [B, num_patches, C, patch_size, patch_size]
            patches_out = patches_out.permute(0, 2, 3, 1, 4, 5).contiguous()
            patches_tgt = patches_tgt.permute(0, 2, 3, 1, 4, 5).contiguous()
            
            num_patches_h, num_patches_w = patches_out.shape[1], patches_out.shape[2]
            patches_out = patches_out.view(B, num_patches_h * num_patches_w, -1)
            patches_tgt = patches_tgt.view(B, num_patches_h * num_patches_w, -1)
            
            # Compute per-patch errors
            patch_errors = F.mse_loss(patches_out, patches_tgt, reduction='none').mean(dim=-1)
            
            # Patch-level statistics
            metrics["patch_error_mean"] = patch_errors.mean().item()
            metrics["patch_error_std"] = patch_errors.std().item()
            metrics["patch_error_max"] = patch_errors.max().item()
            metrics["patch_error_min"] = patch_errors.min().item()
            
            # Percentage of well-reconstructed patches (error < threshold)
            threshold = patch_errors.mean() + patch_errors.std()
            well_reconstructed = (patch_errors < threshold).float().mean()
            metrics["well_reconstructed_patches"] = well_reconstructed.item()
        
        return metrics

    def get_mae_summary(self) -> Dict[str, Any]:
        """Get comprehensive MAE pretraining summary."""
        summary = super().get_pretraining_summary()
        
        # Add MAE-specific summaries
        if self.mask_statistics["mask_ratios"]:
            summary["avg_mask_ratio"] = np.mean(self.mask_statistics["mask_ratios"])
            summary["mask_ratio_std"] = np.std(self.mask_statistics["mask_ratios"])
        
        # Reconstruction quality by mask ratio
        if self.mask_statistics["reconstruction_quality_by_mask"]:
            for ratio, errors in self.mask_statistics["reconstruction_quality_by_mask"].items():
                summary[f"reconstruction_quality_mask_{ratio}"] = np.mean(errors)
        
        return summary

    def visualize_reconstruction(self, outputs: torch.Tensor, targets: torch.Tensor,
                               mask: Optional[torch.Tensor] = None, 
                               save_path: Optional[str] = None) -> Optional[object]:
        """
        Create visualization of reconstruction results using Plotly with dark theme.
        
        Args:
            outputs: Reconstructed data
            targets: Original data
            mask: Masking pattern (optional)
            save_path: Path to save visualization
            
        Returns:
            plotly figure or None
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.io as pio
            
            # Take first sample for visualization
            if outputs.dim() >= 3:
                out_sample = outputs[0].cpu().numpy()
                tgt_sample = targets[0].cpu().numpy()
                
                if mask is not None:
                    mask_sample = mask[0].cpu().numpy()
                else:
                    mask_sample = None
            else:
                out_sample = outputs.cpu().numpy()
                tgt_sample = targets.cpu().numpy()
                mask_sample = mask.cpu().numpy() if mask is not None else None
            
            # Create subplot layout
            n_plots = 3 if mask_sample is not None else 2
            subplot_titles = ['Original', 'Reconstructed', 'Mask'] if mask_sample is not None else ['Original', 'Reconstructed']
            
            fig = make_subplots(
                rows=1, cols=n_plots,
                shared_yaxes=True,
                horizontal_spacing=0.1,
                subplot_titles=subplot_titles
            )
            
            # Original data
            if out_sample.ndim == 3:  # Multi-channel 2D
                fig.add_trace(
                    go.Heatmap(
                        z=out_sample.mean(axis=0),
                        colorscale='Viridis',
                        showscale=True,
                        name='Original',
                        colorbar=dict(x=0.3, len=0.8)
                    ),
                    row=1, col=1
                )
            else:  # 1D signal
                fig.add_trace(
                    go.Scatter(
                        y=out_sample,
                        mode='lines',
                        name='Original',
                        line=dict(color='#636EFA')
                    ),
                    row=1, col=1
                )
            
            # Reconstructed data
            if tgt_sample.ndim == 3:  # Multi-channel 2D
                fig.add_trace(
                    go.Heatmap(
                        z=tgt_sample.mean(axis=0),
                        colorscale='Viridis',
                        showscale=True,
                        name='Reconstructed',
                        colorbar=dict(x=0.65, len=0.8)
                    ),
                    row=1, col=2
                )
            else:  # 1D signal
                fig.add_trace(
                    go.Scatter(
                        y=tgt_sample,
                        mode='lines',
                        name='Reconstructed',
                        line=dict(color='#EF553B')
                    ),
                    row=1, col=2
                )
            
            # Mask (if available)
            if mask_sample is not None and n_plots > 2:
                if mask_sample.ndim == 2:
                    fig.add_trace(
                        go.Heatmap(
                            z=mask_sample,
                            colorscale='RdYlBu',
                            showscale=True,
                            name='Mask',
                            colorbar=dict(x=1.0, len=0.8)
                        ),
                        row=1, col=3
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            y=mask_sample,
                            mode='lines',
                            name='Mask',
                            line=dict(color='#00CC96')
                        ),
                        row=1, col=3
                    )
            
            # Update layout with dark theme
            fig.update_layout(
                template="plotly_dark",
                title_text="MAE Reconstruction Visualization",
                showlegend=False,
                height=400,
                margin=dict(t=60, b=40, l=40, r=40)
            )
            
            # Update axes
            if out_sample.ndim == 1:  # 1D signal
                fig.update_xaxes(title_text="Time Steps")
                fig.update_yaxes(title_text="Amplitude")
            else:  # 2D spectral
                fig.update_xaxes(title_text="Time")
                fig.update_yaxes(title_text="Frequency")
            
            if save_path:
                pio.write_image(fig, save_path, width=1200, height=400)
            
            return fig
            
        except ImportError:
            logger.warning("plotly not available for visualization")
            return None
        except Exception as e:
            logger.warning(f"Error creating reconstruction visualization: {e}")
            return None
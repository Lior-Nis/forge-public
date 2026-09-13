"""
SimCLR (Simple Framework for Contrastive Learning) metrics management.
Handles contrastive learning metrics, representation quality, and pair analysis.
"""

import logging
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import adjusted_rand_score

from .pretrain_base import PretrainingMetricsManager

logger = logging.getLogger(__name__)


class SimCLRMetricsManager(PretrainingMetricsManager):
    """Metrics manager specialized for SimCLR contrastive pretraining tasks."""

    def __init__(self, device: torch.device, track_gradients: bool = True,
                 temperature_range: Tuple[float, float] = (0.01, 1.0)):
        """
        Initialize SimCLR metrics manager.

        Args:
            device: Device to place metrics on
            track_gradients: Whether to track gradient statistics
            temperature_range: Range of temperatures to analyze
        """
        self.device = device
        self.temperature_range = temperature_range
        self.contrastive_statistics = {
            "positive_similarities": [],
            "negative_similarities": [],
            "temperature_analysis": {},
            "pair_quality_scores": []
        }

        super().__init__(device, track_gradients)

    def to(self, device: Optional[torch.device] = None):
        """Move all metrics to the specified device."""
        if device is not None:
            self.device = device
        # Most metrics are computed on-demand or stored as lists/numpy arrays
        # No persistent torch tensors to move
        return self

    def reset_metrics(self):
        """Reset all metrics to their initial state."""
        self.contrastive_statistics = {
            "positive_similarities": [],
            "negative_similarities": [],
            "temperature_analysis": {},
            "pair_quality_scores": []
        }
        # Call parent reset if it exists
        if hasattr(super(), 'reset'):
            super().reset()

    def update_metrics(self, probas: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new batch predictions and targets.
        For SimCLR, this is typically not used as metrics are computed
        in compute_reconstruction_metrics.
        """
        # SimCLR doesn't use the standard probas/targets paradigm
        # Metrics are computed via compute_reconstruction_metrics instead
        pass

    def compute_reconstruction_metrics(self, outputs: torch.Tensor, 
                                     targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute SimCLR-specific contrastive metrics.
        Note: For SimCLR, 'outputs' are embeddings and 'targets' are labels/indices.
        
        Args:
            outputs: Learned embeddings [N, D]
            targets: Sample indices or labels for positive pair identification
            
        Returns:
            Dictionary of SimCLR contrastive metrics
        """
        metrics = {}
        
        with torch.no_grad():
            # Basic embedding statistics
            metrics.update(self.compute_representation_quality(outputs))
            
            # Contrastive-specific metrics
            if outputs.dim() == 2 and outputs.shape[0] % 2 == 0:
                # Assuming pairs are organized as [anchor1, positive1, anchor2, positive2, ...]
                metrics.update(self._compute_contrastive_metrics(outputs))
            
            # Temperature analysis
            if len(self.contrastive_statistics["positive_similarities"]) > 0:
                metrics.update(self._analyze_temperature_effects())
        
        return metrics

    def _compute_contrastive_metrics(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """Compute contrastive learning specific metrics."""
        metrics = {}
        N = embeddings.shape[0]
        
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
        
        # Assuming pairs: (0,1), (2,3), (4,5), etc.
        positive_pairs = []
        all_negative_pairs = []
        
        for i in range(0, N, 2):
            if i + 1 < N:
                # Positive pair similarity
                pos_sim = similarity_matrix[i, i+1].item()
                positive_pairs.append(pos_sim)
                
                # Negative pairs for anchor i
                negative_indices = [j for j in range(N) if j != i and j != i+1]
                neg_sims = [similarity_matrix[i, j].item() for j in negative_indices]
                all_negative_pairs.extend(neg_sims)
        
        if positive_pairs and all_negative_pairs:
            # Store for temperature analysis
            self.contrastive_statistics["positive_similarities"].extend(positive_pairs)
            self.contrastive_statistics["negative_similarities"].extend(all_negative_pairs)
            
            # Basic contrastive metrics
            metrics["positive_sim_mean"] = np.mean(positive_pairs)
            metrics["positive_sim_std"] = np.std(positive_pairs)
            metrics["negative_sim_mean"] = np.mean(all_negative_pairs)
            metrics["negative_sim_std"] = np.std(all_negative_pairs)
            
            # Separation quality
            separation = np.mean(positive_pairs) - np.mean(all_negative_pairs)
            metrics["positive_negative_separation"] = separation
            
            # Pair quality score (how well separated are positive from negatives)
            pos_above_neg_threshold = np.mean([p > np.percentile(all_negative_pairs, 90) 
                                             for p in positive_pairs])
            metrics["pair_quality_score"] = pos_above_neg_threshold
            self.contrastive_statistics["pair_quality_scores"].append(pos_above_neg_threshold)
        
        return metrics

    def compute_contrastive_loss_components(self, embeddings: torch.Tensor, 
                                         temperature: float = 0.1) -> Dict[str, float]:
        """
        Compute InfoNCE loss components for analysis.
        
        Args:
            embeddings: Normalized embeddings
            temperature: Temperature parameter
            
        Returns:
            Dictionary with loss components
        """
        metrics = {}
        N = embeddings.shape[0]
        
        if N % 2 != 0:
            return metrics
        
        with torch.no_grad():
            # Normalize embeddings
            embeddings_norm = F.normalize(embeddings, dim=1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T) / temperature
            
            # Create positive pair mask
            positive_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
            for i in range(0, N, 2):
                if i + 1 < N:
                    positive_mask[i, i+1] = True
                    positive_mask[i+1, i] = True
            
            # InfoNCE loss computation
            # For each sample, positive is its pair, negatives are all others
            total_loss = 0.0
            num_pairs = 0
            
            for i in range(N):
                # Find positive samples for anchor i
                pos_indices = torch.where(positive_mask[i])[0]
                
                if len(pos_indices) > 0:
                    # Compute loss for this anchor
                    pos_sim = similarity_matrix[i, pos_indices[0]]  # Take first positive
                    
                    # All samples except self
                    neg_mask = torch.ones(N, dtype=torch.bool)
                    neg_mask[i] = False
                    all_sims = similarity_matrix[i, neg_mask]
                    
                    # InfoNCE denominator: sum of exp(sim) for all samples except self
                    denominator = torch.logsumexp(all_sims, dim=0)
                    loss = -pos_sim + denominator
                    
                    total_loss += loss.item()
                    num_pairs += 1
            
            if num_pairs > 0:
                metrics["infonce_loss"] = total_loss / num_pairs
                metrics["temperature_used"] = temperature
        
        return metrics

    def _analyze_temperature_effects(self) -> Dict[str, float]:
        """Analyze the effect of different temperature values."""
        metrics = {}
        
        if (len(self.contrastive_statistics["positive_similarities"]) == 0 or 
            len(self.contrastive_statistics["negative_similarities"]) == 0):
            return metrics
        
        pos_sims = np.array(self.contrastive_statistics["positive_similarities"][-100:])  # Last 100
        neg_sims = np.array(self.contrastive_statistics["negative_similarities"][-500:])  # Last 500
        
        # Test different temperature values
        temperatures = np.linspace(self.temperature_range[0], self.temperature_range[1], 10)
        separations = []
        
        for temp in temperatures:
            pos_scaled = pos_sims / temp
            neg_scaled = neg_sims / temp

            # Log-space ratio: log(mean(exp(pos/t))) - log(mean(exp(neg/t)))
            log_pos_mean = np.logaddexp.reduce(pos_scaled) - np.log(len(pos_scaled))
            log_neg_mean = np.logaddexp.reduce(neg_scaled) - np.log(len(neg_scaled))
            separations.append(log_pos_mean - log_neg_mean)

        # Find optimal temperature
        best_temp_idx = np.argmax(separations)
        metrics["optimal_temperature"] = temperatures[best_temp_idx]
        metrics["max_log_separation_ratio"] = separations[best_temp_idx]
        
        return metrics

    def compute_clustering_metrics(self, embeddings: torch.Tensor, 
                                 true_labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute clustering quality metrics for learned representations.
        
        Args:
            embeddings: Learned embeddings
            true_labels: Ground truth labels (if available)
            
        Returns:
            Dictionary of clustering metrics
        """
        metrics = {}
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            embeddings_np = embeddings.cpu().numpy()
            
            # Try different numbers of clusters
            n_clusters_range = [2, 4, 8, 16] if embeddings.shape[0] > 32 else [2, 4]
            
            best_silhouette = -1
            best_n_clusters = 2
            
            for n_clusters in n_clusters_range:
                if n_clusters >= embeddings.shape[0]:
                    continue
                    
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings_np)
                    
                    # Silhouette score
                    sil_score = silhouette_score(embeddings_np, cluster_labels)
                    metrics[f"silhouette_k{n_clusters}"] = sil_score
                    
                    if sil_score > best_silhouette:
                        best_silhouette = sil_score
                        best_n_clusters = n_clusters
                    
                    # Calinski-Harabasz score (variance ratio)
                    ch_score = calinski_harabasz_score(embeddings_np, cluster_labels)
                    metrics[f"calinski_harabasz_k{n_clusters}"] = ch_score
                    
                    # If true labels available, compute ARI
                    if true_labels is not None:
                        true_labels_np = true_labels.cpu().numpy()
                        if len(np.unique(true_labels_np)) > 1:
                            ari = adjusted_rand_score(true_labels_np, cluster_labels)
                            metrics[f"ari_k{n_clusters}"] = ari
                            
                except Exception as e:
                    logger.warning(f"Error computing clustering metrics for k={n_clusters}: {e}")
                    continue
            
            metrics["best_silhouette_score"] = best_silhouette
            metrics["best_n_clusters"] = best_n_clusters
            
        except ImportError:
            logger.warning("sklearn not available for clustering metrics")
        except Exception as e:
            logger.warning(f"Error computing clustering metrics: {e}")
        
        return metrics

    def compute_top_k_accuracy(self, embeddings: torch.Tensor, k: int = 5) -> Dict[str, float]:
        """
        Compute top-k accuracy for positive pair retrieval.

        Args:
            embeddings: Learned embeddings
            k: Top-k value

        Returns:
            Dictionary with top-k metrics
        """
        metrics = {}
        N = embeddings.shape[0]

        if N % 2 != 0 or N < 4:
            return metrics

        with torch.no_grad():
            # Normalize embeddings
            embeddings_norm = F.normalize(embeddings, dim=1)

            # Compute similarity matrix
            similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)

            correct_retrievals = 0
            total_queries = 0

            for i in range(0, N, 2):
                if i + 1 < N:
                    # Query with first element, target is second element
                    query_sims = similarity_matrix[i].clone()
                    target_idx = i + 1

                    # Remove self-similarity
                    query_sims[i] = -float('inf')

                    # Get top-k most similar
                    top_k_indices = torch.topk(query_sims, k=min(k, N-1)).indices

                    # Check if target is in top-k
                    if target_idx in top_k_indices:
                        correct_retrievals += 1

                    total_queries += 1

                    # Do the same for the reverse query
                    query_sims = similarity_matrix[i+1].clone()
                    target_idx = i
                    query_sims[i+1] = -float('inf')

                    top_k_indices = torch.topk(query_sims, k=min(k, N-1)).indices

                    if target_idx in top_k_indices:
                        correct_retrievals += 1

                    total_queries += 1

            if total_queries > 0:
                metrics[f"top_{k}_accuracy"] = correct_retrievals / total_queries

        return metrics

    def compute_batch_contrastive_metrics(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float,
        stage: str,
        batch_size: int
    ) -> Dict[str, float]:
        """
        Compute batch-level contrastive metrics for real-time monitoring.

        Args:
            z1: First view representations [B, D]
            z2: Second view representations [B, D]
            temperature: Temperature parameter for scaling
            stage: Current stage ("train", "val", "test")
            batch_size: Batch size for proper logging

        Returns:
            Dictionary of metrics ready to be logged by pipeline
        """
        metrics = {}

        with torch.no_grad():
            # Basic representation statistics
            cosine_sim = F.cosine_similarity(z1, z2, dim=-1).mean()
            metrics[f"representations/{stage}_positive_cosine_sim"] = cosine_sim.item()

            # Representation norms
            z1_norm = torch.norm(z1, dim=-1).mean()
            z2_norm = torch.norm(z2, dim=-1).mean()
            metrics[f"representations/{stage}_z1_norm"] = z1_norm.item()
            metrics[f"representations/{stage}_z2_norm"] = z2_norm.item()

            # Representation diversity
            z1_pairwise_dist = torch.cdist(z1, z1).mean()
            metrics[f"representations/{stage}_representation_diversity"] = z1_pairwise_dist.item()

            # Alignment and uniformity metrics
            alignment = F.cosine_similarity(z1, z2, dim=-1).mean()
            metrics[f"contrastive/{stage}_alignment"] = alignment.item()

            # Uniformity: measure how uniformly distributed representations are
            z1_normalized = F.normalize(z1, dim=-1)
            z2_normalized = F.normalize(z2, dim=-1)
            z_combined = torch.cat([z1_normalized, z2_normalized], dim=0)
            pairwise_dist = torch.pdist(z_combined, p=2)
            uniformity = torch.log(torch.exp(-2 * pairwise_dist).mean())
            metrics[f"contrastive/{stage}_uniformity"] = uniformity.item()

            # Temperature-scaled similarity statistics
            sim_matrix = torch.matmul(z1_normalized, z2_normalized.T) / temperature
            pos_sim = torch.diagonal(sim_matrix).mean()
            neg_sim = (sim_matrix.sum() - torch.diagonal(sim_matrix).sum()) / (batch_size * (batch_size - 1))

            metrics[f"contrastive/{stage}_positive_similarity"] = pos_sim.item()
            metrics[f"contrastive/{stage}_negative_similarity"] = neg_sim.item()
            metrics[f"contrastive/{stage}_similarity_ratio"] = (pos_sim / (neg_sim + 1e-8)).item()

            # Representation collapse detection
            z1_var = z1.var(dim=0).mean()
            z2_var = z2.var(dim=0).mean()
            metrics[f"contrastive/{stage}_z1_dimension_variance"] = z1_var.item()
            metrics[f"contrastive/{stage}_z2_dimension_variance"] = z2_var.item()

            # Effective rank (eigvals requires float32)
            try:
                z1_f32 = z1.float()
                z1_cov = torch.cov(z1_f32.T)
                z1_eigenvals = torch.linalg.eigvals(z1_cov).real
                z1_eigenvals = z1_eigenvals[z1_eigenvals > 1e-8]
                if len(z1_eigenvals) > 0:
                    p = z1_eigenvals / z1_eigenvals.sum()
                    z1_effective_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-8)))
                    metrics[f"contrastive/{stage}_effective_rank"] = z1_effective_rank.item()
            except Exception as e:
                logger.warning(f"Failed to compute effective rank: {e}")

            # Temperature
            metrics[f"contrastive/{stage}_temperature"] = temperature

        return metrics

    def get_simclr_summary(self) -> Dict[str, Any]:
        """Get comprehensive SimCLR pretraining summary."""
        summary = super().get_pretraining_summary()
        
        # Add SimCLR-specific summaries
        if self.contrastive_statistics["positive_similarities"]:
            pos_sims = self.contrastive_statistics["positive_similarities"]
            summary["avg_positive_similarity"] = np.mean(pos_sims)
            summary["positive_similarity_std"] = np.std(pos_sims)
        
        if self.contrastive_statistics["negative_similarities"]:
            neg_sims = self.contrastive_statistics["negative_similarities"]
            summary["avg_negative_similarity"] = np.mean(neg_sims)
            summary["negative_similarity_std"] = np.std(neg_sims)
        
        if self.contrastive_statistics["pair_quality_scores"]:
            summary["avg_pair_quality"] = np.mean(self.contrastive_statistics["pair_quality_scores"])
            summary["final_pair_quality"] = self.contrastive_statistics["pair_quality_scores"][-1]
        
        return summary

    def visualize_embedding_space(self, embeddings: torch.Tensor, 
                                 labels: Optional[torch.Tensor] = None,
                                 save_path: Optional[str] = None) -> Optional[object]:
        """
        Create 2D visualization of embedding space using t-SNE or PCA.
        
        Args:
            embeddings: Learned embeddings
            labels: Optional labels for coloring
            save_path: Path to save visualization
            
        Returns:
            matplotlib figure or None
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            
            embeddings_np = embeddings.cpu().numpy()
            
            # Use PCA for large datasets, t-SNE for smaller ones
            if embeddings_np.shape[0] > 1000:
                reducer = PCA(n_components=2)
                embeddings_2d = reducer.fit_transform(embeddings_np)
                method = "PCA"
            else:
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings_np.shape[0]-1))
                embeddings_2d = reducer.fit_transform(embeddings_np)
                method = "t-SNE"
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if labels is not None:
                labels_np = labels.cpu().numpy()
                scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   c=labels_np, cmap='tab10', alpha=0.7)
                plt.colorbar(scatter)
            else:
                ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
            
            ax.set_title(f'Embedding Space Visualization ({method})')
            ax.set_xlabel(f'{method} Component 1')
            ax.set_ylabel(f'{method} Component 2')
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except ImportError:
            logger.warning("Required packages (matplotlib, sklearn) not available for visualization")
            return None
        except Exception as e:
            logger.warning(f"Error creating embedding visualization: {e}")
            return None
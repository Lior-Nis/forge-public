"""
WandB Model Registry for comprehensive model lifecycle management.

Handles model versioning, metadata tracking, and artifact management
for the FoG detection research project.
"""

import logging
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import wandb
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class WandBModelRegistry:
    """
    Central registry for managing models as WandB artifacts.
    
    Provides comprehensive model lifecycle management including:
    - Automated model artifact creation with rich metadata
    - Semantic versioning and alias management
    - Model discovery and querying capabilities
    - Integration with PyTorch Lightning workflows
    """
    
    def __init__(self, project: str = "fog", entity: Optional[str] = None):
        """
        Initialize the model registry.
        
        Args:
            project: WandB project name
            entity: WandB entity (username or team)
        """
        self.project = project
        self.entity = entity
        self.api = wandb.Api()
        
    def save_model(
        self,
        model_path: str,
        artifact_name: str,
        metadata: Dict[str, Any],
        aliases: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> wandb.Artifact:
        """
        Save a model as a WandB artifact with comprehensive metadata.
        
        Args:
            model_path: Path to the model checkpoint file
            artifact_name: Name for the artifact (should be unique within project)
            metadata: Dictionary containing model metadata
            aliases: List of aliases to assign to this version
            description: Human-readable description of the model
            
        Returns:
            WandB artifact object
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Ensure we have an active WandB run
        if wandb.run is None:
            raise RuntimeError("No active WandB run. Call wandb.init() first.")
            
        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=description or f"FoG detection model: {artifact_name}",
            metadata=self._prepare_metadata(metadata)
        )
        
        # Add model file to artifact
        artifact.add_file(model_path, name="model.ckpt")
        
        # Log the artifact with aliases
        logged_artifact = wandb.log_artifact(artifact, aliases=aliases or ["latest"])
        
        logger.info(f"Saved model artifact: {artifact_name} with aliases: {aliases}")
        return logged_artifact
        
    def load_model(
        self,
        artifact_reference: str,
        download_dir: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any], Dict[str, str]]:
        """
        Load a model from the registry with artifact lineage tracking.

        Args:
            artifact_reference: Reference to the artifact (name:version or name:alias)
            download_dir: Directory to download the artifact (uses temp dir if None)

        Returns:
            Tuple of (model_path, metadata, parent_info)
        """
        # Format artifact reference with project/entity if not provided
        if "/" not in artifact_reference:
            if self.entity:
                artifact_reference = f"{self.entity}/{self.project}/{artifact_reference}"
            else:
                artifact_reference = f"{self.project}/{artifact_reference}"

        try:
            # Use wandb.use_artifact() if active run exists for automatic lineage tracking
            if wandb.run is not None:
                logger.info(f"Using artifact lineage tracking for {artifact_reference}")
                artifact = wandb.use_artifact(artifact_reference, type='model')
            else:
                # Fallback to API for non-run context
                artifact = self.api.artifact(artifact_reference)
        except Exception as e:
            raise ValueError(f"Failed to retrieve artifact {artifact_reference}: {e}")

        # Download artifact
        download_path = artifact.download(root=download_dir)
        model_path = os.path.join(download_path, "model.ckpt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found in artifact: {model_path}")

        # Extract parent run information for lineage tracking
        parent_info = self.get_artifact_parent_info(artifact_reference)

        logger.info(f"Loaded model from registry: {artifact_reference}")
        return model_path, artifact.metadata, parent_info
        
    def find_models(
        self,
        task_type: Optional[str] = None,
        architecture: Optional[str] = None,
        min_performance: Optional[Dict[str, float]] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query models in the registry based on criteria.
        
        Args:
            task_type: Filter by task type (classification, mae, simclr)
            architecture: Filter by model architecture
            min_performance: Minimum performance thresholds (e.g., {"val_f1": 0.8})
            max_results: Maximum number of results to return
            
        Returns:
            List of model information dictionaries
        """
        artifact_collection = f"{self.project}/model"
        if self.entity:
            artifact_collection = f"{self.entity}/{artifact_collection}"
            
        try:
            artifacts = self.api.artifacts(artifact_collection, per_page=max_results)
        except Exception as e:
            logger.warning(f"Failed to query artifacts: {e}")
            return []
            
        results = []
        for artifact in artifacts:
            metadata = artifact.metadata or {}
            
            # Apply filters
            if task_type and metadata.get("task_type") != task_type:
                continue
                
            if architecture and metadata.get("architecture") != architecture:
                continue
                
            if min_performance:
                performance = metadata.get("performance", {})
                if not all(
                    performance.get(metric, 0) >= threshold
                    for metric, threshold in min_performance.items()
                ):
                    continue
                    
            results.append({
                "name": artifact.name,
                "version": artifact.version,
                "aliases": list(artifact.aliases),
                "created_at": artifact.created_at,
                "metadata": metadata,
                "size": artifact.size
            })
            
        return results
        
    def get_best_model(
        self,
        task_type: str,
        metric: str = "val_f1",
        architecture: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the best performing model for a given task.
        
        Args:
            task_type: Task type to filter by
            metric: Performance metric to optimize
            architecture: Optional architecture filter
            
        Returns:
            Artifact reference for the best model or None if not found
        """
        models = self.find_models(
            task_type=task_type,
            architecture=architecture,
            max_results=50
        )
        
        if not models:
            return None
            
        # Find model with best performance on given metric
        best_model = None
        best_score = -1
        
        for model in models:
            performance = model["metadata"].get("performance", {})
            score = performance.get(metric, 0)
            
            if score > best_score:
                best_score = score
                best_model = model
                
        if best_model:
            return f"{best_model['name']}:{best_model['version']}"
            
        return None
        
    def update_aliases(
        self,
        artifact_reference: str,
        aliases: List[str]
    ) -> None:
        """
        Update aliases for an existing artifact.
        
        Args:
            artifact_reference: Reference to the artifact
            aliases: New list of aliases to assign
        """
        if "/" not in artifact_reference:
            if self.entity:
                artifact_reference = f"{self.entity}/{self.project}/{artifact_reference}"
            else:
                artifact_reference = f"{self.project}/{artifact_reference}"
                
        try:
            artifact = self.api.artifact(artifact_reference)
            artifact.aliases = aliases
            artifact.save()
            logger.info(f"Updated aliases for {artifact_reference}: {aliases}")
        except Exception as e:
            logger.error(f"Failed to update aliases: {e}")
            
    def delete_model(self, artifact_reference: str) -> None:
        """
        Delete a model artifact from the registry.
        
        Args:
            artifact_reference: Reference to the artifact to delete
        """
        if "/" not in artifact_reference:
            if self.entity:
                artifact_reference = f"{self.entity}/{self.project}/{artifact_reference}"
            else:
                artifact_reference = f"{self.project}/{artifact_reference}"
                
        try:
            artifact = self.api.artifact(artifact_reference)
            artifact.delete()
            logger.info(f"Deleted artifact: {artifact_reference}")
        except Exception as e:
            logger.error(f"Failed to delete artifact: {e}")
            
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare and enrich metadata for the artifact.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Enhanced metadata dictionary
        """
        enriched_metadata = metadata.copy()
        
        # Add timestamp
        enriched_metadata["created_at"] = datetime.now().isoformat()
        
        # Add git information if available
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            enriched_metadata["git_hash"] = git_hash
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("Git hash not available")
            
        # Ensure config is serializable
        if "config" in enriched_metadata:
            config = enriched_metadata["config"]
            if isinstance(config, DictConfig):
                enriched_metadata["config"] = OmegaConf.to_container(config, resolve=True)
                
        # Add model size if model path is provided
        if "model_path" in metadata:
            try:
                model_size = os.path.getsize(metadata["model_path"])
                enriched_metadata["model_size_bytes"] = model_size
            except OSError:
                pass
                
        return enriched_metadata
        
    def create_artifact_name(
        self,
        task_type: str,
        architecture: str,
        transform: Optional[str] = None,
        dataset: Optional[str] = None,
        suffix: Optional[str] = None
    ) -> str:
        """
        Create a standardized artifact name.
        
        Args:
            task_type: Type of task (classification, mae, simclr)
            architecture: Model architecture name
            transform: Data transform type
            dataset: Dataset identifier
            suffix: Optional suffix for uniqueness
            
        Returns:
            Standardized artifact name
        """
        components = [task_type, architecture.lower()]
        
        if transform:
            components.append(transform.lower())
            
        if dataset:
            components.append(dataset.lower())
            
        if suffix:
            components.append(suffix)
            
        return "-".join(components)
        
    def get_model_lineage(self, artifact_reference: str) -> List[Dict[str, Any]]:
        """
        Get the version history/lineage of a model.

        Args:
            artifact_reference: Reference to the artifact

        Returns:
            List of version information dictionaries
        """
        if "/" not in artifact_reference:
            if self.entity:
                artifact_reference = f"{self.entity}/{self.project}/{artifact_reference}"
            else:
                artifact_reference = f"{self.project}/{artifact_reference}"

        # Extract artifact name without version
        artifact_name = artifact_reference.split(":")[0]

        try:
            artifact_versions = self.api.artifact_versions("model", name=artifact_name.split("/")[-1])

            lineage = []
            for version in artifact_versions:
                lineage.append({
                    "version": version.version,
                    "aliases": list(version.aliases),
                    "created_at": version.created_at,
                    "metadata": version.metadata or {},
                    "size": version.size
                })

            return sorted(lineage, key=lambda x: x["created_at"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to get model lineage: {e}")
            return []

    def get_artifact_parent_info(self, artifact_reference: str) -> Dict[str, str]:
        """
        Get parent run information from an artifact for lineage tracking.

        Args:
            artifact_reference: Reference to the artifact (name:version or name:alias)

        Returns:
            Dictionary containing parent run information:
            {
                "parent_run_id": "...",
                "parent_run_name": "...",
                "parent_project": "...",
                "parent_entity": "..."
            }
        """
        # Format artifact reference with project/entity if not provided
        if "/" not in artifact_reference:
            if self.entity:
                artifact_reference = f"{self.entity}/{self.project}/{artifact_reference}"
            else:
                artifact_reference = f"{self.project}/{artifact_reference}"

        try:
            artifact = self.api.artifact(artifact_reference)

            # Extract parent run info from artifact metadata
            metadata = artifact.metadata or {}
            parent_info = {
                "parent_run_id": metadata.get("wandb_run_id"),
                "parent_run_name": metadata.get("wandb_run_name"),
                "parent_project": metadata.get("wandb_project"),
                "parent_entity": metadata.get("wandb_entity")
            }

            # Also get info from artifact's source run if metadata is missing
            if not parent_info["parent_run_id"]:
                try:
                    source_run = artifact.logged_by()
                    if source_run:
                        parent_info["parent_run_id"] = source_run.id
                        parent_info["parent_run_name"] = source_run.name
                        parent_info["parent_project"] = source_run.project
                        parent_info["parent_entity"] = source_run.entity
                except Exception as e:
                    logger.debug(f"Could not retrieve source run info: {e}")

            logger.info(f"Extracted parent info from artifact {artifact_reference}")
            return parent_info

        except Exception as e:
            logger.warning(f"Failed to get parent info from artifact {artifact_reference}: {e}")
            return {
                "parent_run_id": None,
                "parent_run_name": None,
                "parent_project": None,
                "parent_entity": None
            }
"""
WandB callbacks for PyTorch Lightning integration.

Provides automatic model registration and artifact management
integrated with the training lifecycle.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from utils.wandb_model_registry import WandBModelRegistry

logger = logging.getLogger(__name__)


class WandBModelRegistryCallback(pl.Callback):
    """
    PyTorch Lightning callback for automatic model registration to WandB.
    
    Monitors training progress and registers models when:
    - Training completes successfully
    - Validation metric improves (configurable)
    - At specified intervals
    """
    
    def __init__(
        self,
        registry_config: Optional[Dict[str, Any]] = None,
        monitor: str = "val_f1",
        mode: str = "max",
        register_on_train_end: bool = True,
        register_on_improvement: bool = True,
        register_interval: Optional[int] = None,
        min_improvement: float = 0.001,
        artifact_name_template: Optional[str] = None
    ):
        """
        Initialize the model registry callback.
        
        Args:
            registry_config: Configuration for the model registry
            monitor: Metric to monitor for improvements
            mode: "min" or "max" for metric optimization
            register_on_train_end: Register model when training ends
            register_on_improvement: Register model when monitored metric improves
            register_interval: Register model every N epochs (None to disable)
            min_improvement: Minimum improvement to trigger registration
            artifact_name_template: Template for artifact names
        """
        super().__init__()
        
        self.registry_config = registry_config or {}
        self.monitor = monitor
        self.mode = mode
        self.register_on_train_end = register_on_train_end
        self.register_on_improvement = register_on_improvement
        self.register_interval = register_interval
        self.min_improvement = min_improvement
        self.artifact_name_template = artifact_name_template
        
        # Track best metric value
        self.best_metric_value = float('inf') if mode == 'min' else float('-inf')
        self.registry = None
        
        # Performance tracking
        self.epoch_metrics = {}
        
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Setup callback after trainer initialization."""
        if wandb.run is None:
            logger.warning("WandB run not initialized. Model registry callback will be disabled.")
            return

        # Skip registry initialization if in offline mode
        if wandb.run.settings.mode == "offline":
            logger.info("WandB in offline mode. Model registry callback will be disabled.")
            return

        # Initialize registry
        project = wandb.run.project
        entity = wandb.run.entity
        self.registry = WandBModelRegistry(project=project, entity=entity)

        logger.info(f"Initialized WandB model registry callback monitoring '{self.monitor}'")
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the end of validation epoch."""
        if self.registry is None:
            return
            
        # Get current metrics
        current_metrics = trainer.logged_metrics
        self.epoch_metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in current_metrics.items()}
        
        # Check for improvement
        if self.register_on_improvement and self.monitor in current_metrics:
            current_value = current_metrics[self.monitor].item() if torch.is_tensor(current_metrics[self.monitor]) else current_metrics[self.monitor]
            
            improved = False
            if self.mode == 'max':
                improved = current_value > (self.best_metric_value + self.min_improvement)
            else:
                improved = current_value < (self.best_metric_value - self.min_improvement)
                
            if improved:
                self.best_metric_value = current_value
                self._register_model(trainer, pl_module, reason="improvement", aliases=["best-performing"])
                logger.info(f"Registered improved model with {self.monitor}={current_value:.4f}")
                
        # Check for interval registration
        if (self.register_interval and 
            trainer.current_epoch > 0 and 
            trainer.current_epoch % self.register_interval == 0):
            self._register_model(trainer, pl_module, reason="interval")
            
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training ends."""
        if self.registry is None or not self.register_on_train_end:
            return
            
        self._register_model(trainer, pl_module, reason="train_end", aliases=["final", "latest"])
        logger.info("Registered final model at training end")
        
    def _register_model(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        reason: str,
        aliases: Optional[List[str]] = None
    ) -> None:
        """
        Register the current model to WandB registry.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The Lightning module
            reason: Reason for registration (improvement, interval, train_end)
            aliases: List of aliases to assign
        """
        try:
            # Find the latest checkpoint
            checkpoint_path = self._get_latest_checkpoint(trainer)
            if not checkpoint_path:
                logger.warning("No checkpoint found for model registration")
                return
                
            # Generate artifact name
            artifact_name = self._generate_artifact_name(pl_module, reason)
            
            # Prepare metadata
            metadata = self._prepare_model_metadata(trainer, pl_module, reason)
            
            # Generate description
            description = self._generate_description(pl_module, reason, metadata)
            
            # Register the model
            artifact = self.registry.save_model(
                model_path=checkpoint_path,
                artifact_name=artifact_name,
                metadata=metadata,
                aliases=aliases or ["latest"],
                description=description
            )
            
            # Log registration to WandB run
            wandb.log({
                "model_registry/registered": 1,
                "model_registry/artifact_name": artifact_name,
                "model_registry/reason": reason
            })
            
            logger.info(f"Successfully registered model: {artifact_name} (reason: {reason})")
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            # Log failure to W&B for monitoring
            if wandb.run:
                wandb.log({
                    "model_registry/registration_failed": 1,
                    "model_registry/error": str(e)
                })
            
    def _get_latest_checkpoint(self, trainer: pl.Trainer) -> Optional[str]:
        """Get the path to the latest checkpoint."""
        # Check if ModelCheckpoint callback exists
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                if callback.last_model_path and os.path.exists(callback.last_model_path):
                    return callback.last_model_path
                elif callback.best_model_path and os.path.exists(callback.best_model_path):
                    return callback.best_model_path
                    
        # Fallback to trainer's checkpoint
        if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
            return trainer.checkpoint_callback.last_model_path
            
        return None
        
    def _generate_artifact_name(self, pl_module: pl.LightningModule, reason: str) -> str:
        """Generate a standardized artifact name."""
        if self.artifact_name_template:
            return self.artifact_name_template.format(
                reason=reason,
                epoch=pl_module.current_epoch,
                timestamp=wandb.run.start_time
            )
            
        # Extract model information
        task_type = self._extract_task_type(pl_module)
        architecture = self._extract_architecture(pl_module)
        transform = self._extract_transform(pl_module)
        
        # Use registry to generate standard name
        return self.registry.create_artifact_name(
            task_type=task_type,
            architecture=architecture,
            transform=transform,
            suffix=f"epoch{pl_module.current_epoch}"
        )
        
    def _prepare_model_metadata(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        reason: str
    ) -> Dict[str, Any]:
        """Prepare comprehensive metadata for the model artifact."""
        metadata = {
            "task_type": self._extract_task_type(pl_module),
            "architecture": self._extract_architecture(pl_module),
            "transform": self._extract_transform(pl_module),
            "registration_reason": reason,
            "epoch": pl_module.current_epoch,
            "global_step": trainer.global_step,
            "performance": self.epoch_metrics.copy(),
            "training_info": {
                "total_epochs": trainer.max_epochs,
                "batch_size": self._extract_batch_size(pl_module),
                "learning_rate": self._extract_learning_rate(pl_module),
                "optimizer": self._extract_optimizer_info(pl_module),
                "loss_function": self._extract_loss_info(pl_module),
            },
            "model_info": {
                "num_parameters": sum(p.numel() for p in pl_module.parameters()),
                "trainable_parameters": sum(p.numel() for p in pl_module.parameters() if p.requires_grad),
            }
        }

        # Add parent run information for lineage tracking
        if wandb.run:
            metadata["wandb_run_id"] = wandb.run.id
            metadata["wandb_run_name"] = wandb.run.name
            metadata["wandb_project"] = wandb.run.project
            metadata["wandb_entity"] = wandb.run.entity
        
        # Add full configuration if available
        if hasattr(pl_module, 'hparams') and pl_module.hparams:
            hparams = pl_module.hparams
            if isinstance(hparams, dict):
                metadata["config"] = hparams
            else:
                # Convert to dict if it's a namespace or other object
                try:
                    metadata["config"] = vars(hparams)
                except:
                    logger.debug("Could not extract hparams as dictionary")
                    
        # Add dataset information
        metadata["dataset_info"] = self._extract_dataset_info(pl_module)
        
        return metadata
        
    def _generate_description(self, pl_module: pl.LightningModule, reason: str, metadata: Dict[str, Any]) -> str:
        """Generate a human-readable description for the model."""
        task_type = metadata.get("task_type", "unknown")
        architecture = metadata.get("architecture", "unknown")
        performance = metadata.get("performance", {})
        
        # Get key performance metric
        key_metric = self.monitor
        metric_value = performance.get(key_metric, 0)
        
        description_parts = [
            f"FoG Detection {task_type.title()} Model",
            f"Architecture: {architecture}",
            f"Registered: {reason} (epoch {metadata.get('epoch', 0)})",
        ]
        
        if metric_value:
            description_parts.append(f"{key_metric}: {metric_value:.4f}")
            
        return " | ".join(description_parts)
        
    def _extract_task_type(self, pl_module: pl.LightningModule) -> str:
        """Extract task type from the module."""
        class_name = pl_module.__class__.__name__.lower()
        if "classification" in class_name:
            return "classification"
        elif "mae" in class_name:
            return "mae"
        elif "simclr" in class_name:
            return "simclr"
        else:
            return pl_module.train_cfg.task_manager.task_type
            
    def _extract_architecture(self, pl_module: pl.LightningModule) -> str:
        """Extract model architecture name."""
        backbone_name = pl_module.backbone.__class__.__name__
        return backbone_name.replace('Backbone', '').replace('_', '')
        
    def _extract_transform(self, pl_module: pl.LightningModule) -> str:
        """Extract transform type."""
        transform_name = pl_module.transform.__class__.__name__
        return transform_name.replace('Transform', '').lower()
        
    def _extract_batch_size(self, pl_module: pl.LightningModule) -> int:
        """Extract batch size from configuration."""
        return pl_module.data_cfg.dataloader.batch_size
        
    def _extract_learning_rate(self, pl_module: pl.LightningModule) -> float:
        """Extract learning rate from optimizer."""
        return pl_module.train_cfg.optimizer.lr
        
    def _extract_optimizer_info(self, pl_module: pl.LightningModule) -> str:
        """Extract optimizer information."""
        return getattr(pl_module.train_cfg.optimizer, '_target_', 'AdamW')
        
    def _extract_loss_info(self, pl_module: pl.LightningModule) -> str:
        """Extract loss function information."""
        return pl_module.loss.__class__.__name__
        
    def _extract_dataset_info(self, pl_module: pl.LightningModule) -> Dict[str, Any]:
        """Extract dataset information."""
        dataset_cfg = pl_module.data_cfg.dataset
        info = {
            "dataset_type": getattr(dataset_cfg, '_target_', 'AccelerometerDataset'),
            "classification_strategy": getattr(dataset_cfg, 'classification_strategy', 'dual_threshold'),
        }
        if hasattr(pl_module, 'num_classes'):
            info["num_classes"] = pl_module.num_classes
        return info


class WandBCheckpointDirCallback(pl.Callback):
    """
    Callback to dynamically set ModelCheckpoint directory based on WandB run name.

    Appends the WandB run name to the existing ModelCheckpoint dirpath.
    Example: ./checkpoints/classification → ./checkpoints/classification/stellar-mountain-42

    This makes it easy to correlate checkpoints with WandB experiments.
    """

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Update ModelCheckpoint dirpath after logger is initialized."""
        if stage != "fit":
            return

        if wandb.run is None:
            logger.warning("WandB run not initialized. Checkpoint directory will not be updated.")
            return

        # Get WandB run name
        run_name = wandb.run.name

        # Find ModelCheckpoint callback and update its dirpath
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                # Append run name to existing dirpath
                base_dirpath = callback.dirpath or "./checkpoints"
                new_dirpath = os.path.join(base_dirpath, run_name)
                callback.dirpath = new_dirpath

                # Create directory if it doesn't exist
                os.makedirs(new_dirpath, exist_ok=True)

                logger.info(f"Updated checkpoint directory to: {new_dirpath}")
                logger.info(f"WandB run: {run_name} (ID: {wandb.run.id})")
                break


class WandBModelComparisonCallback(pl.Callback):
    """
    Callback for comparing models and updating registry aliases based on performance.

    Automatically maintains "best-*" aliases pointing to the best performing models
    across different metrics.
    """
    
    def __init__(
        self,
        metrics_to_track: List[str] = None,
        update_aliases: bool = True,
        comparison_interval: int = 5
    ):
        """
        Initialize the model comparison callback.
        
        Args:
            metrics_to_track: List of metrics to track for best model identification
            update_aliases: Whether to automatically update registry aliases
            comparison_interval: How often (in epochs) to run comparisons
        """
        super().__init__()
        self.metrics_to_track = metrics_to_track or ["val_f1", "val_sensitivity", "val_specificity"]
        self.update_aliases = update_aliases
        self.comparison_interval = comparison_interval
        
        self.registry = None
        self.best_models = {}  # metric -> (artifact_reference, value)
        
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Setup callback after trainer initialization."""
        if wandb.run is None:
            return
            
        project = wandb.run.project
        entity = wandb.run.entity
        self.registry = WandBModelRegistry(project=project, entity=entity)
        
        # Initialize best models tracking
        for metric in self.metrics_to_track:
            self.best_models[metric] = (None, float('-inf'))
            
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check for new best models and update aliases."""
        if (self.registry is None or 
            trainer.current_epoch % self.comparison_interval != 0):
            return
            
        current_metrics = trainer.logged_metrics
        task_type = self._extract_task_type(pl_module)
        
        # Find models of the same task type
        models = self.registry.find_models(task_type=task_type, max_results=50)
        
        for metric in self.metrics_to_track:
            if metric not in current_metrics:
                continue
                
            # Find best model for this metric
            best_model = None
            best_value = float('-inf')
            
            for model in models:
                performance = model["metadata"].get("performance", {})
                if metric in performance and performance[metric] > best_value:
                    best_value = performance[metric]
                    best_model = f"{model['name']}:{model['version']}"
                    
            # Update alias if we have a new best
            if (best_model and 
                (self.best_models[metric][0] != best_model or 
                 best_value > self.best_models[metric][1])):
                
                self.best_models[metric] = (best_model, best_value)
                
                if self.update_aliases:
                    alias = f"best-{metric.replace('val_', '').replace('_', '-')}"
                    try:
                        self.registry.update_aliases(best_model, [alias])
                        logger.info(f"Updated {alias} alias to point to {best_model}")
                    except Exception as e:
                        logger.warning(f"Failed to update alias {alias}: {e}")
                        
    def _extract_task_type(self, pl_module: pl.LightningModule) -> str:
        """Extract task type from the module."""
        class_name = pl_module.__class__.__name__.lower()
        if "classification" in class_name:
            return "classification"
        elif "mae" in class_name:
            return "mae"
        elif "simclr" in class_name:
            return "simclr"
        return "unknown"
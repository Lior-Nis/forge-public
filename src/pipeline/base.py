import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from pipeline.config import Config
from model.preprocessors import SignalPreprocessor, PatientNormalizationPreprocessor

logger = logging.getLogger(__name__)


class BasePipeline(pl.LightningModule, ABC):
    """
    Base pipeline containing shared functionality for all training paradigms.

    COMPONENT GUARANTEES
    --------------------
    The following components are ALWAYS present (never None):
    - self.transform: Signal transform (identity if not specified)
    - self.backbone: Feature extraction backbone
    - self.head: Task-specific head (classification/reconstruction/projection)
    - self.loss: Loss function

    The following components are OPTIONAL (may be None):
    - self.preprocessors: Signal preprocessing
    - self.signal_augmentor: Time-domain augmentation
    - self.spectral_augmentor: Frequency-domain augmentation

    These guarantees are enforced by:
    1. Pydantic ModelConfig requiring backbone/head/transform
    2. _setup_model() always instantiating required components (line 58-64)
    3. Type hints declaring Optional only where truly optional

    ARCHITECTURE
    ------------
    This class provides:
    - Model loading and weight management
    - Optimizer and scheduler setup
    - Logging and metrics infrastructure
    - Checkpoint handling
    - Common validation logic
    """

    # Required components (ALWAYS present, never None)
    transform: torch.nn.Module
    backbone: torch.nn.Module
    head: torch.nn.Module
    loss: torch.nn.Module  # BaseLoss

    # Optional components (may be None)
    preprocessors: Optional[SignalPreprocessor]
    signal_augmentor: Optional[torch.nn.Module]
    spectral_augmentor: Optional[torch.nn.Module]

    def __init__(self, config: Config):
        """
        Initialize the base pipeline with unified configuration.

        Args:
            config: Unified configuration containing model, train, and data settings
        """
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        # Expose sub-configs for convenience
        self.model_cfg = config.model
        self.train_cfg = config.train
        self.data_cfg = config.data

        # Setup model components and managers
        self._setup_model()
        self._setup_managers()

    def _setup_model(self) -> None:
        """Setup model components using hydra."""
        # Convert Pydantic models to dicts for Hydra instantiation
        transform_dict = self.model_cfg.transform.model_dump()
        backbone_dict = self.model_cfg.backbone.model_dump()
        head_dict = self.model_cfg.head.model_dump()

        self.transform = hydra.utils.instantiate(transform_dict)
        self.backbone = hydra.utils.instantiate(backbone_dict)
        self.head = hydra.utils.instantiate(head_dict)

        optional_components = ['preprocessors',
                               'signal_augmentor',
                               'spectral_augmentor']

        # Initialize optional components to None, then instantiate if configured
        for component in optional_components:
            setattr(self, component, None)
            component_cfg = getattr(self.model_cfg, component, None)
            if component_cfg is not None:
                component_dict = component_cfg.model_dump()
                setattr(self, component, hydra.utils.instantiate(component_dict))

        # Instantiate loss (masking is handled internally by loss classes)
        loss_dict = self.train_cfg.loss.model_dump()
        self.loss = hydra.utils.instantiate(loss_dict)

    def _setup_managers(self) -> None:
        """Setup metric, logging, and weight managers."""
        device = "cuda" if self.train_cfg.trainer.accelerator == "gpu" else "cpu"

        self.metrics_manager = self._create_metrics_manager(self.config, device)
        # Logging manager created in setup() when datamodule is available
        self.logging_manager = None
        self.weight_manager = self._create_weight_manager(self.config, device)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Called by Lightning when trainer is attached - datamodule and trainer now available.

        Creates the logging manager here instead of in __init__ so we can pass
        the trainer (for global_step) and datamodule directly.
        """
        super().setup(stage)
        if self.logging_manager is None:
            device = "cuda" if self.train_cfg.trainer.accelerator == "gpu" else "cpu"
            self.logging_manager = self._create_logging_manager(
                config=self.config,
                device=device,
                trainer=self.trainer,
                datamodule=self.datamodule
            )


    def log_config_to_wandb(self, pl_logger) -> None: # TODO: simplify
        """Log the complete configuration and entire config directory to W&B."""
        if pl_logger and hasattr(pl_logger, "experiment"):
            full_config = self.config.to_dict()

            # Log resolved config as artifact
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write(OmegaConf.to_yaml(OmegaConf.create(full_config)))
                temp_config_path = f.name

            try:
                artifact = pl_logger.experiment.log_artifact(
                    temp_config_path,
                    name=f"resolved_config_{self.train_cfg.run_name or 'default'}",
                    type="config",
                )
            finally:
                os.unlink(temp_config_path)

            # Log entire configs directory as artifact for complete snapshot
            configs_dir = os.path.join(os.getcwd(), "configs")
            if os.path.exists(configs_dir):
                try:
                    configs_artifact = pl_logger.experiment.log_artifact(
                        configs_dir,
                        name=f"configs_snapshot_{self.train_cfg.run_name or 'default'}",
                        type="config_directory",
                    )
                    logging.info(f"Logged configs directory snapshot to wandb: {configs_artifact.name}")
                except Exception as e:
                    logging.warning(f"Failed to log configs directory: {e}")
            else:
                logging.warning(f"Configs directory not found at {configs_dir}")

    def on_train_start(self):
        """Called when training starts - update device for managers and inject normalization stats."""
        self.metrics_manager.to(self.device)

        # Inject normalization stats from datamodule into preprocessors
        if self.preprocessors is not None and self.datamodule is not None:
            stats = getattr(self.datamodule, 'train_normalization_stats', None)
            if stats is not None:
                for module in self.preprocessors.preprocessors:
                    if isinstance(module, PatientNormalizationPreprocessor):
                        module.set_stats(stats)
                        logger.info("Injected training fold normalization stats into PatientNormalizationPreprocessor")
                        break

        # Log configuration to W&B
        if self.trainer.logger is not None:
            self.log_config_to_wandb(self.trainer.logger)

    def on_train_epoch_start(self):
        """Called at the start of each training epoch - handle unfreezing schedule."""
        # Check if we should unfreeze backbone at this epoch
        unfroze = self.weight_manager.check_unfreeze_schedule(self.current_epoch)
        if unfroze:
            # Need to reconfigure optimizer to include newly unfrozen parameters
            self.trainer.strategy.setup_optimizers(self.trainer)

    def on_validation_start(self):
        """Called when validation starts - override in subclasses if needed."""
        pass

    @property
    def datamodule(self):
        """Access datamodule via trainer (standard Lightning pattern)."""
        if self.trainer is None:
            return None
        return getattr(self.trainer, 'datamodule', None)

    def configure_optimizers(self): # TODO: simplify
        """Configure optimizer and scheduler with support for frozen parameters and component-aware parameter groups."""
        # Check if parameter groups are specified in config
        # Convert Pydantic model to dict for Hydra instantiation
        optimizer_dict = self.train_cfg.optimizer.model_dump()
        # parameter_groups is consumed here — never forwarded to the optimizer constructor
        optimizer_dict.pop('parameter_groups', None)

        if self.train_cfg.optimizer.parameter_groups is not None:
            # Component-aware parameter groups.
            # Bypass hydra.utils.instantiate here — it recursively converts
            # param_groups (which contain Tensors) into OmegaConf objects,
            # causing "optimizer can only optimize Tensors" errors.
            param_groups = self._create_parameter_groups(self.train_cfg.optimizer.parameter_groups)
            target = optimizer_dict.pop('_target_')
            from hydra._internal.utils import _locate
            optimizer_cls = _locate(target)
            optimizer = optimizer_cls(param_groups, **optimizer_dict)
        else:
            # Standard optimizer with only trainable parameters
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            optimizer = hydra.utils.instantiate(optimizer_dict, params=trainable_params)

        if self.train_cfg.scheduler is not None:
            # Convert Pydantic model to dict for Hydra instantiation
            scheduler_dict = self.train_cfg.scheduler.model_dump()

            # Special handling for OneCycleLR to use trainer.estimated_stepping_batches
            if scheduler_dict.get('_target_') == "torch.optim.lr_scheduler.OneCycleLR":
                # Remove steps_per_epoch and epochs from config, use total_steps from trainer
                scheduler_cfg = {k: v for k, v in scheduler_dict.items() if k not in ['steps_per_epoch', 'epochs']}
                scheduler = hydra.utils.instantiate(
                    scheduler_cfg,
                    optimizer=optimizer,
                    total_steps=self.trainer.estimated_stepping_batches
                )
                logger.info(f"OneCycleLR configured with total_steps={self.trainer.estimated_stepping_batches}")
            else:
                scheduler = hydra.utils.instantiate(
                    scheduler_dict, optimizer=optimizer
                )

            # Configure scheduler based on its type for robust handling
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # This scheduler requires a monitored value and runs per-epoch
                scheduler_config = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "losses/val_loss",
                }
            elif isinstance(scheduler, (
                torch.optim.lr_scheduler.OneCycleLR,
                torch.optim.lr_scheduler.CyclicLR,
            )):
                # These schedulers are designed to step per batch
                scheduler_config = {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            else:
                # Default for epoch-level schedulers (CosineAnnealingLR, StepLR, etc.)
                scheduler_config = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }

            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        return optimizer

    def _create_parameter_groups(self, parameter_groups_config): # TODO: simplify
        """Create component-aware parameter groups for optimizer."""
        param_groups = []
        base_lr = self.train_cfg.optimizer.lr
        
        # Create mapping of component names to their parameters
        component_params = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters
                
            component = name.split('.')[0]  # e.g., 'backbone.layer1.conv' -> 'backbone'
            if component not in component_params:
                component_params[component] = []
            component_params[component].append(param)
        
        # Create parameter groups based on config
        assigned_params = set()
        for group_config in parameter_groups_config:
            component = group_config['component']
            if 'lr' in group_config:
                group_lr = group_config['lr']
            else:
                group_lr = base_lr * group_config.get('lr_multiplier', 1.0)

            if component in component_params:
                param_group = {
                    'params': component_params[component],
                    'lr': group_lr
                }
                param_groups.append(param_group)
                assigned_params.update(id(p) for p in component_params[component])
                logger.info(f"Created parameter group for '{component}' with lr={group_lr:.6f} "
                           f"({len(component_params[component])} parameters)")
        
        # Add any remaining trainable parameters to default group
        remaining_params = [p for p in self.parameters() 
                          if p.requires_grad and id(p) not in assigned_params]
        
        if remaining_params:
            param_groups.append({'params': remaining_params, 'lr': base_lr})
            logger.info(f"Created default parameter group with lr={base_lr:.6f} "
                       f"({len(remaining_params)} parameters)")
        
        return param_groups

    

    def _log_loss(self, loss: torch.Tensor, stage: str, batch_size: int) -> None:
        """
        Log loss with standardized stage-specific configuration.
        
        Args:
            loss: The loss tensor to log
            stage: The current stage ('train', 'val', 'test')
            batch_size: Batch size for weighted averaging
        """
        log_config = {
            "train": {"prog_bar": True, "on_step": True, "on_epoch": False},
            "val": {"prog_bar": True, "on_step": False, "on_epoch": True},
            "test": {"prog_bar": True, "on_step": False, "on_epoch": True}
        }
        
        config = log_config.get(stage, log_config["train"])
        self.log(
            f"losses/{stage}_loss",
            loss,
            batch_size=batch_size,
            logger=True,
            **config
        )

    def log_metrics(self, metrics: Dict[str, Any], stage: str):
        """
        Log metrics dictionary to the logger.
        
        Args:
            metrics: Dictionary of metric names and values
            stage: Current stage ('train', 'val', 'test')
        """
        # Prefix metrics with stage
        metrics_to_log = {f"metrics/{stage}_{k}": v for k, v in metrics.items()}
        
        self.log_dict(
            metrics_to_log,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

    def on_train_epoch_end(self):
        self._process_epoch_end("train")

    def on_validation_epoch_end(self):
        """Process validation epoch end with logging."""
        self._process_epoch_end("val")

    def on_test_epoch_end(self):
        """Process test epoch end with logging."""
        self._process_epoch_end("test")

    @abstractmethod
    def _create_metrics_manager(self, device):
        """Create appropriate metrics manager based on task type."""
        raise NotImplementedError

    @abstractmethod
    def _create_logging_manager(self, config: Config, device: str, trainer, datamodule=None):
        """Create appropriate logging manager based on task type."""
        raise NotImplementedError

    @abstractmethod
    def _create_weight_manager(self, configs, device):
        """Create appropriate weight manager based on task type."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Abstract forward pass - must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch, batch_idx: int):
        """Abstract training step - must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx: int):
        """Abstract validation step - must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def test_step(self, batch, batch_idx: int):
        """Abstract test step - must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _process_epoch_end(self, stage: str) -> None:
        """Abstract epoch end processing - must be implemented by subclasses."""
        raise NotImplementedError

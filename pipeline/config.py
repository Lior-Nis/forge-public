"""Type-safe configuration for training pipeline."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List

from model.config import ModelConfig
from data.datamodule.config import DataConfig


class WeightsConfig(BaseModel):
    """Weight loading configuration."""
    load_from: Optional[str] = Field(
        default=None,
        description="Path to checkpoint file"
    )
    load_from_registry: Optional[str] = Field(
        default=None,
        description="WandB artifact reference"
    )
    load_strategy: str = Field(
        default="full",
        description="How to load weights (full, backbone_only, custom)"
    )
    freeze_backbone: bool = Field(
        default=False,
        description="Whether to freeze backbone after loading"
    )
    validation: Dict[str, bool] = Field(
        default_factory=lambda: {
            "warn_on_mismatch": True,
            "strict_loading": False
        }
    )
    unfreeze_schedule: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Unfreezing schedule config"
    )

    model_config = {"frozen": True, "extra": "forbid"}


class RegistryConfig(BaseModel):
    """WandB Model Registry configuration."""
    project: str = Field(default="fog")
    entity: Optional[str] = None
    auto_register: bool = True
    register_on_improvement: bool = True
    register_on_train_end: bool = True
    register_interval: Optional[int] = None
    monitor_metric: str = "val_f1"
    monitor_mode: str = "max"
    min_improvement: float = 0.001
    artifact_name_template: Optional[str] = None
    artifact_description_template: Optional[str] = None
    preferred_aliases: List[str] = Field(
        default_factory=lambda: ["latest", "best-performing", "best-map"]
    )
    fallback_to_files: bool = True

    model_config = {"frozen": True, "extra": "forbid"}


class LoggerConfig(BaseModel):
    """Logging configuration."""
    _target_: str = "pytorch_lightning.loggers.WandbLogger"
    project: str = "fog-classification"
    save_dir: str = "./logs"
    tags: List[str] = Field(default_factory=list)
    log_model: bool = True

    model_config = {"extra": "allow"}  # Allow logger-specific params


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""
    _target_: str = "torch.optim.AdamW"
    lr: float = Field(gt=0, default=1e-4)
    weight_decay: float = Field(ge=0, default=0.01)
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    amsgrad: bool = False
    parameter_groups: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Component-aware parameter groups for differential learning rates"
    )

    model_config = {"frozen": True, "extra": "allow"}  # Allow optimizer-specific params (momentum, etc.)


class SchedulerConfig(BaseModel):
    """Scheduler configuration."""
    _target_: str

    model_config = {"extra": "allow"}  # Allow scheduler-specific params


class TrainerConfig(BaseModel):
    """PyTorch Lightning Trainer configuration."""
    _target_: str = "pytorch_lightning.Trainer"
    accelerator: str = "gpu"
    devices: List[int] = Field(default_factory=lambda: [0])
    max_epochs: int = Field(gt=0, default=50)
    log_every_n_steps: int = Field(gt=0, default=100)
    gradient_clip_val: float = Field(ge=0, default=1.0)
    accumulate_grad_batches: int = Field(gt=0, default=1)
    limit_train_batches: Optional[int] = None
    fast_dev_run: bool = False
    precision: str = Field(
        default="32-true",
        pattern="^(16-mixed|bf16-mixed|32-true)$",
        description="Training precision: 16-mixed, bf16-mixed, or 32-true"
    )

    model_config = {"frozen": True, "extra": "allow"}  # Allow trainer-specific params


class LossConfig(BaseModel):
    """Loss function configuration."""
    _target_: str

    model_config = {"extra": "allow"}  # Allow loss-specific params


class TaskManagerConfig(BaseModel):
    """Task manager configuration."""
    _target_: str
    task_type: str = Field(
        pattern="^(classification|mae|simclr|jepa|ibot|segmentation|patient_contrastive|multitask_ssl)$",
        default="classification"
    )

    model_config = {"extra": "allow"}  # Allow task-specific params


class CallbackConfig(BaseModel):
    """Callback configuration."""

    model_config = {"extra": "allow"}  # Allow all callback params


class AdversarialConfig(BaseModel):
    """DANN adversarial training configuration."""
    lambda_max: float = Field(default=0.1, gt=0, description="Maximum GRL lambda")
    rampup_epochs: int = Field(default=10, gt=0, description="Epochs to ramp lambda from 0 to lambda_max")
    hidden_dim: int = Field(default=256, gt=0, description="Patient classifier hidden dim")
    dropout: float = Field(default=0.3, ge=0, le=1, description="Patient classifier dropout")

    model_config = {"frozen": True, "extra": "forbid"}


class LoggingConfig(BaseModel):
    """Logging configuration for training visualization and metrics."""
    spectral_log_interval: int = Field(default=10, gt=0, description="Interval for logging spectral visualizations")
    reconstruction_interval: int = Field(default=5, gt=0, description="Interval for logging reconstruction visualizations (MAE/pretraining)")
    accumulation_intervals: Dict[str, int] = Field(
        default_factory=lambda: {
            "spectrals": 10,
            "patches": 10,  # Expensive with large validation sets
            "timestamps": 5
        },
        description="Intervals for accumulating different types of logging data"
    )
    benchmark_session_ids: List[str] = Field(
        default_factory=lambda: ["auto"],
        description="Session IDs to visualize (auto = select diverse examples)"
    )
    reconstruction_interval: int = Field(
        default=1, gt=0,
        description="Interval (epochs) for MAE reconstruction logging"
    )
    representation_interval: int = Field(
        default=1, gt=0,
        description="Interval (epochs) for SimCLR representation logging"
    )
    vis_log_every_n_epochs: int = Field(
        default=5, gt=0,
        description="Interval (epochs) for classification visualizations (confusion matrix, PR curve, etc). Scalar metrics are always logged every epoch."
    )
    model_config = {"frozen": True, "extra": "forbid"}


class TrainConfig(BaseModel):
    """Top-level training configuration."""
    pipeline_type: str = Field(
        pattern="^(classification|mae|simclr|jepa|ibot|segmentation|patient_contrastive|multitask_ssl)$",
        default="classification"
    )
    run_name: Optional[str] = None
    use_masked_loss: bool = Field(
        default=False,
        description="Wrap loss function with MaskedLossWrapper for handling padded sequences"
    )

    # Sub-configs
    optimizer: OptimizerConfig
    loss: LossConfig
    trainer: TrainerConfig
    logger: LoggerConfig
    task_manager: TaskManagerConfig

    # Optional sub-configs
    scheduler: Optional[SchedulerConfig] = None
    weights: Optional[WeightsConfig] = None
    registry: Optional[RegistryConfig] = None
    callbacks: Optional[CallbackConfig] = None
    logging: Optional[LoggingConfig] = None
    adversarial: Optional[AdversarialConfig] = None

    model_config = {"frozen": True, "extra": "forbid"}


class Config(BaseModel):
    """
    Unified configuration containing all pipeline settings.

    This provides a single config object that can be passed through the pipeline,
    eliminating the need to reassemble configs in different places.
    """
    model: ModelConfig
    train: TrainConfig
    data: DataConfig

    model_config = {"frozen": True, "extra": "forbid"}

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        return {
            "model": self.model.model_dump(),
            "train": self.train.model_dump(),
            "data": self.data.model_dump(),
        }

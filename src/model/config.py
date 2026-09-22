"""Type-safe configuration for model components."""
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any


class BackboneConfig(BaseModel):
    """Backbone network configuration."""
    _target_: str
    output_dim: int = Field(gt=0, description="Output dimension of backbone")

    model_config = {"extra": "allow"}  # Allow backbone-specific params


class HeadConfig(BaseModel):
    """Classification/reconstruction head configuration."""
    _target_: str

    model_config = {"extra": "allow"}  # Allow head-specific params


class TransformConfig(BaseModel):
    """Transform configuration."""
    _target_: str

    model_config = {"extra": "allow"}  # Allow transform-specific params


class PreprocessorsConfig(BaseModel):
    """Preprocessor configuration."""
    _target_: str

    model_config = {"extra": "allow"}  # Allow preprocessor-specific params


class AugmentorConfig(BaseModel):
    """Augmentor configuration."""
    _target_: str

    model_config = {"extra": "allow"}  # Allow augmentor-specific params


class SpectralAugmentorConfig(BaseModel):
    """Spectral augmentor configuration."""
    _target_: Optional[str] = None

    model_config = {"extra": "allow"}  # Allow augmentor-specific params


class BaseModelConfig(BaseModel):
    """Base model configuration shared across all task types."""
    _target_: str = "model.fog_model.FOGModel"

    # Required components
    backbone: BackboneConfig
    head: HeadConfig
    transform: TransformConfig

    # Optional components
    preprocessors: Optional[PreprocessorsConfig] = None
    signal_augmentor: Optional[AugmentorConfig] = None
    spectral_augmentor: Optional[SpectralAugmentorConfig] = None

    @model_validator(mode='after')
    def validate_dimensions(self) -> 'BaseModelConfig':
        """Validate that backbone output matches head input dimension."""
        backbone_dim = getattr(self.backbone, 'output_dim', None)

        # Head often uses 'input_dim' or 'embed_dim'
        head_dim = (
            getattr(self.head, 'input_dim', None) or
            getattr(self.head, 'embed_dim', None) or
            self.head.__dict__.get('input_dim') or
            self.head.__dict__.get('embed_dim')
        )

        if backbone_dim and head_dim and backbone_dim != head_dim:
            raise ValueError(
                f"Dimension mismatch: Backbone output_dim ({backbone_dim}) "
                f"!= Head input dimension ({head_dim})"
            )
        return self

    model_config = {"frozen": True, "extra": "allow"}  # Allow model-level params


class ClassificationModelConfig(BaseModelConfig):
    """Model configuration for supervised classification tasks."""
    num_classes: int = Field(
        default=2,
        ge=2,
        description="Number of classification classes (binary=2, multi-class>2)"
    )


class MAEModelConfig(BaseModelConfig):
    """Model configuration for MAE (Masked Autoencoder) pretraining."""
    mask_ratio: float = Field(
        default=0.4,
        gt=0.0,
        le=1.0,
        description="Ratio of patches to mask during reconstruction (0.0-1.0). Must be >0 for MAE."
    )
    mask_mode: str = Field(
        default="temporal",
        pattern="^(temporal|frequency|2d_patch)$",
        description="Masking strategy: temporal (time patches), frequency (freq bands), or 2d_patch (2D time-frequency)"
    )


class SimCLRModelConfig(BaseModelConfig):
    """Model configuration for SimCLR contrastive learning."""
    # Note: temperature is defined in loss config as single source of truth
    transform_alt: Optional[TransformConfig] = None
    backbone_alt: Optional[BackboneConfig] = None


class JEPAModelConfig(BaseModelConfig):
    """Model configuration for JEPA (Joint Embedding Predictive Architecture) pretraining."""
    jepa_mode: str = Field(
        default="ijepa",
        pattern="^(ijepa|lejepa)$",
        description="JEPA variant: 'ijepa' (EMA target encoder) or 'lejepa' (SIGReg, no EMA)"
    )
    mask_ratio: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Ratio of patches to mask for prediction (0.0-1.0)"
    )
    ema_decay: float = Field(
        default=0.996,
        ge=0.0,
        le=1.0,
        description="Initial EMA decay for target encoder (ijepa mode only)"
    )
    ema_decay_end: float = Field(
        default=0.999,
        ge=0.0,
        le=1.0,
        description="Final EMA decay for target encoder (ijepa mode only, cosine schedule)"
    )


class IBOTModelConfig(BaseModelConfig):
    """Model configuration for iBOT (Image BERT Pre-Training with Online Tokenizer) pretraining."""
    mask_ratio: float = Field(
        default=0.75,
        gt=0.0,
        le=1.0,
        description="Ratio of temporal patches to mask (0.0-1.0)"
    )
    mask_mode: str = Field(
        default="temporal",
        pattern="^(temporal|2d_patch)$",
        description="Masking strategy: temporal (time patches) or 2d_patch (time-frequency)"
    )
    ema_decay: float = Field(
        default=0.996,
        ge=0.0,
        le=1.0,
        description="Initial EMA decay for target encoder (cosine schedule start)"
    )
    ema_decay_end: float = Field(
        default=0.9999,
        ge=0.0,
        le=1.0,
        description="Final EMA decay for target encoder (cosine schedule end)"
    )


# Type alias for backward compatibility - defaults to base config
ModelConfig = BaseModelConfig
"""
WandB Registry-First Weight Management.

Modern weight management system that prioritizes loading models from WandB
model registry while maintaining component freezing/unfreezing capabilities
and fallback support for direct file loading.
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import wandb

from utils.config_loaders import ensure_dict
from utils.wandb_model_registry import WandBModelRegistry

logger = logging.getLogger(__name__)


class WeightManager:
    """
    Registry-first weight management for model loading and component control.
    
    Supports multiple loading strategies:
    - Registry-based: Load models from WandB artifacts (primary method)
    - File-based: Direct file loading (fallback)
    - Component control: Freezing/unfreezing with scheduling
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: str = "cpu", 
                 registry_config: Optional[Dict[str, Any]] = None,
                 weights_config: Optional[Dict[str, Any]] = None):
        """
        Initialize registry-first WeightManager.
        
        Args:
            model: PyTorch Lightning model to manage
            device: Device for loading weights
            registry_config: WandB registry configuration
            weights_config: Weight loading and component configuration
        """
        self.model = model
        self.device = device
        self.frozen_components = set()
        self.unfreeze_at_epoch = None

        # Convert Pydantic configs to dicts for uniform .get() access
        registry_config = ensure_dict(registry_config)
        weights_config = ensure_dict(weights_config)

        # Initialize registry if WandB is available
        self.registry = None
        if registry_config and wandb.run:
            try:
                project = registry_config.get('project', wandb.run.project)
                entity = registry_config.get('entity', wandb.run.entity)
                self.registry = WandBModelRegistry(project=project, entity=entity)
                logger.info("Initialized WandB model registry for weight loading")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB registry: {e}")
                
        # Auto-setup from config if provided
        if weights_config:
            self._setup_from_config(weights_config)
            
    def load_from_registry(
        self,
        artifact_reference: str,
        load_strategy: str = "full",
        validation: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Load model weights from WandB model registry with lineage tracking.

        Args:
            artifact_reference: WandB artifact reference (name:version or name:alias)
            load_strategy: Loading strategy ('full', 'backbone_only', 'custom')
            validation: Validation options (warn_on_mismatch, strict_loading)

        Returns:
            Tuple of (missing_keys, unexpected_keys, parent_info)
        """
        if not self.registry:
            raise RuntimeError("WandB registry not initialized. Use load_from_path() instead.")

        logger.info(f"Loading model from registry: {artifact_reference}")

        try:
            # Download model from registry and get parent info
            model_path, metadata, parent_info = self.registry.load_model(artifact_reference)

            # Load the model using standard loading logic
            missing_keys, unexpected_keys = self.load_from_path(
                model_path, load_strategy, validation
            )

            # Log registry loading info
            logger.info(f"Successfully loaded from registry with metadata: {list(metadata.keys())}")
            if parent_info.get("parent_run_id"):
                logger.info(f"Parent run: {parent_info['parent_run_name']} (ID: {parent_info['parent_run_id']})")

            return missing_keys, unexpected_keys, parent_info

        except Exception as e:
            logger.error(f"Failed to load from registry: {e}")
            raise
            
    def load_from_path(
        self,
        checkpoint_path: str,
        load_strategy: str = "full",
        validation: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Load weights directly from file path or artifact reference.

        Auto-detects artifact references and delegates to load_from_registry()
        for automatic lineage tracking.

        Args:
            checkpoint_path: Path to checkpoint file or artifact reference
            load_strategy: Loading strategy ('full', 'backbone_only', 'custom')
            validation: Validation options (warn_on_mismatch, strict_loading)

        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        # Auto-detect artifact references and delegate to registry loading
        is_artifact_reference = (
            checkpoint_path.startswith('wandb://') or
            (':' in checkpoint_path and not os.path.exists(checkpoint_path))
        )

        if is_artifact_reference and self.registry:
            logger.info(f"Detected artifact reference, using registry loading: {checkpoint_path}")
            # Strip wandb:// prefix if present
            artifact_ref = checkpoint_path.replace('wandb://', '')
            missing_keys, unexpected_keys, _ = self.load_from_registry(
                artifact_ref, load_strategy, validation
            )
            return missing_keys, unexpected_keys

        # Set validation defaults
        validation = validation or {}
        warn_on_mismatch = validation.get("warn_on_mismatch", True)
        strict_loading = validation.get("strict_loading", False)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading weights from path {checkpoint_path} with strategy '{load_strategy}'")

        # Load checkpoint. Accepts both release formats: a released `.safetensors`
        # tensor set, or a locally trained Lightning `.ckpt` (read with
        # weights_only=False, as it also carries hyperparams).
        from utils.released_weights import load_encoder_state_dict

        pretrained_state = load_encoder_state_dict(checkpoint_path)
        checkpoint = {'state_dict': pretrained_state}

        # Apply loading strategy
        if load_strategy == "full":
            missing_keys, unexpected_keys = self._load_full_model(pretrained_state, strict_loading)

        elif load_strategy == "backbone_only":
            missing_keys, unexpected_keys = self._load_backbone_only(pretrained_state)

        elif load_strategy == "backbone_and_head":
            missing_keys, unexpected_keys = self._load_backbone_and_head(pretrained_state)

        elif load_strategy == "custom":
            missing_keys, unexpected_keys = self._load_custom_weights(checkpoint, validation)

        else:
            raise ValueError(f"Unknown load_strategy: {load_strategy}")

        # Log loading statistics
        self._log_loading_stats(load_strategy, missing_keys, unexpected_keys, warn_on_mismatch)

        logger.info(f"Successfully loaded weights with strategy '{load_strategy}'")
        return missing_keys, unexpected_keys

    def find_best_model(
        self,
        task_type: str,
        metric: str = "val_f1",
        architecture: Optional[str] = None
    ) -> Optional[str]:
        """
        Find the best performing model in the registry.
        
        Args:
            task_type: Task type to filter by
            metric: Performance metric to optimize
            architecture: Optional architecture filter
            
        Returns:
            Artifact reference for the best model or None if not found
        """
        if not self.registry:
            logger.warning("Registry not available for model search")
            return None
            
        return self.registry.get_best_model(task_type, metric, architecture)
        
    def query_models(
        self,
        task_type: Optional[str] = None,
        architecture: Optional[str] = None,
        min_performance: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query models in the registry based on criteria.
        
        Args:
            task_type: Filter by task type
            architecture: Filter by architecture
            min_performance: Minimum performance thresholds
            
        Returns:
            List of matching models
        """
        if not self.registry:
            logger.warning("Registry not available for model query")
            return []
            
        return self.registry.find_models(task_type, architecture, min_performance)
        
    def _load_full_model(self, state_dict: Dict[str, torch.Tensor], strict: bool) -> Tuple[List[str], List[str]]:
        """Load complete model state."""
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=strict)
        return list(missing_keys), list(unexpected_keys)
        
    def _load_backbone_only(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
        """Load only backbone weights."""
        backbone_state = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                backbone_state[key] = value

        if not backbone_state:
            logger.warning("No backbone weights found in checkpoint")
            return [], []

        # FogFormer lazily creates input_projection on first forward() — it is None
        # at init time, so load_state_dict would silently skip those keys.
        # Reconstruct the module from the checkpoint weight shape before loading.
        if (hasattr(self.model, 'backbone') and
                getattr(self.model.backbone, 'input_projection', None) is None and
                'backbone.input_projection.weight' in backbone_state):
            w = backbone_state['backbone.input_projection.weight']
            # weight shape: [out_channels, in_channels, kH, kW]
            self.model.backbone.input_projection = nn.Conv2d(
                w.shape[1], w.shape[0],
                kernel_size=(w.shape[2], w.shape[3]),
                stride=(w.shape[2], w.shape[3]),
            )
            logger.info(
                f"Reconstructed lazy input_projection from checkpoint: "
                f"in={w.shape[1]}, out={w.shape[0]}, kernel=({w.shape[2]},{w.shape[3]})"
            )

        missing_keys, unexpected_keys = self.model.load_state_dict(backbone_state, strict=False)
        return list(missing_keys), list(unexpected_keys)

    def _load_backbone_and_head(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
        """Load backbone + head + transform, but skip cohort-specific preprocessor
        normalizer buffers.

        Use this to FINE-TUNE a trained model on a new cohort: the head (the part
        you want to adapt) is loaded from the source, while the patient/session
        normalizer stats — whose shape is patient-count-dependent and which are
        re-injected per training fold in on_train_start — are left fresh. Loading
        them (as 'full' does) raises a shape mismatch and is meaningless anyway.
        """
        filtered = {
            k: v for k, v in state_dict.items()
            if not ("preprocessors" in k and "normalizer" in k)
        }
        # Reconstruct FogFormer's lazy input_projection from the checkpoint, same as
        # _load_backbone_only, so its weights actually load.
        if (hasattr(self.model, "backbone") and
                getattr(self.model.backbone, "input_projection", None) is None and
                "backbone.input_projection.weight" in filtered):
            w = filtered["backbone.input_projection.weight"]
            self.model.backbone.input_projection = nn.Conv2d(
                w.shape[1], w.shape[0],
                kernel_size=(w.shape[2], w.shape[3]),
                stride=(w.shape[2], w.shape[3]),
            )
        missing_keys, unexpected_keys = self.model.load_state_dict(filtered, strict=False)
        return list(missing_keys), list(unexpected_keys)

    def _load_custom_weights(self, checkpoint: Dict[str, Any], validation: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Hook for custom loading strategies - can be extended."""
        logger.warning("Custom weight loading not implemented - falling back to full loading")
        state_dict = checkpoint.get('state_dict', checkpoint)
        strict = validation.get("strict_loading", False)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=strict)
        return list(missing_keys), list(unexpected_keys)
        
    def _log_loading_stats(self, load_type: str, missing_keys: List[str], unexpected_keys: List[str], warn: bool):
        """Log detailed statistics about weight loading."""
        total_params = len(list(self.model.state_dict().keys()))
        loaded_params = total_params - len(missing_keys)
        
        logger.info(f"Loaded {loaded_params}/{total_params} parameters ({load_type})")
        
        if warn:
            if missing_keys:
                logger.warning(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
                
    def freeze_component(self, component_name: str) -> int:
        """
        Freeze all parameters of a model component.
        
        Args:
            component_name: Name of component to freeze (e.g., 'backbone', 'head')
            
        Returns:
            Number of parameters frozen
        """
        if not hasattr(self.model, component_name):
            logger.warning(f"Component '{component_name}' not found - cannot freeze")
            return 0

        component = getattr(self.model, component_name)
        frozen_count = 0

        for param in component.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
                
        self.frozen_components.add(component_name)
        logger.info(f"Froze {frozen_count} parameters in '{component_name}'")
        return frozen_count
        
    def unfreeze_component(self, component_name: str) -> int:
        """
        Unfreeze all parameters of a model component.
        
        Args:
            component_name: Name of component to unfreeze
            
        Returns:
            Number of parameters unfrozen
        """
        if not hasattr(self.model, component_name):
            logger.warning(f"Component '{component_name}' not found - cannot unfreeze")
            return 0

        component = getattr(self.model, component_name)
        unfrozen_count = 0

        for param in component.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1
                
        self.frozen_components.discard(component_name)
        logger.info(f"Unfroze {unfrozen_count} parameters in '{component_name}'")
        return unfrozen_count
        
    def freeze_backbone(self) -> int:
        """Convenience method to freeze backbone."""
        return self.freeze_component('backbone')
        
    def unfreeze_backbone(self) -> int:
        """Convenience method to unfreeze backbone."""
        return self.unfreeze_component('backbone')
        
    def is_frozen(self, component_name: str) -> bool:
        """Check if a component is frozen."""
        return component_name in self.frozen_components
        
    def get_frozen_components(self) -> List[str]:
        """Get list of currently frozen components."""
        return list(self.frozen_components)
        
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of currently trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]
        
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": frozen_params
        }
        
    def log_parameter_status(self):
        """Log current parameter status for debugging."""
        counts = self.get_parameter_count()
        logger.info(f"Parameter status - Total: {counts['total']}, "
                   f"Trainable: {counts['trainable']}, Frozen: {counts['frozen']}")
        
        if self.frozen_components:
            logger.info(f"Frozen components: {', '.join(self.frozen_components)}")
            
    def _setup_from_config(self, weights_config: Dict[str, Any]):
        """
        Setup weight loading and freezing from configuration.
        
        Enhanced config structure:
        {
            # Registry-based loading (preferred)
            'load_from_registry': str,           # Artifact reference
            
            # File-based loading (fallback)
            'load_from': str,                    # Checkpoint path
            
            # Common options
            'load_strategy': str,                # 'full', 'backbone_only', 'custom'
            'freeze_backbone': bool,             # Whether to freeze after loading
            'validation': {...},                 # Validation options
            'unfreeze_schedule': {               # Optional unfreezing schedule
                'enabled': bool,
                'unfreeze_at_epoch': int
            }
        }
        """
        # Priority 1: Load from registry if specified and available
        if weights_config.get('load_from_registry') and self.registry:
            try:
                self.load_from_registry(
                    artifact_reference=weights_config['load_from_registry'],
                    load_strategy=weights_config.get('load_strategy', 'full'),
                    validation=weights_config.get('validation', {})
                )
            except Exception as e:
                logger.warning(f"Registry loading failed, falling back to file: {e}")
                # Fall through to file loading
            else:
                # Registry loading successful, skip file loading
                weights_config = weights_config.copy()  # Don't modify original
                weights_config.pop('load_from', None)  # Remove file path to skip file loading
        
        # Priority 2: Load from file path if specified
        if weights_config.get('load_from'):
            self.load_from_path(
                checkpoint_path=weights_config['load_from'],
                load_strategy=weights_config.get('load_strategy', 'full'),
                validation=weights_config.get('validation', {})
            )
            
        # Apply freezing if requested
        if weights_config.get('freeze_backbone', False):
            self.freeze_backbone()
            
        # Setup unfreeze schedule
        unfreeze_cfg = weights_config.get('unfreeze_schedule') or {}
        if unfreeze_cfg.get('enabled', False):
            self.unfreeze_at_epoch = unfreeze_cfg.get('unfreeze_at_epoch')
            logger.info(f"Scheduled backbone unfreezing at epoch {self.unfreeze_at_epoch}")
            
    def check_unfreeze_schedule(self, current_epoch: int) -> bool:
        """
        Check if we should unfreeze backbone at current epoch.
        
        Args:
            current_epoch: Current training epoch
            
        Returns:
            True if backbone was unfrozen, False otherwise
        """
        if (self.unfreeze_at_epoch is not None and 
            current_epoch == self.unfreeze_at_epoch and 
            self.is_frozen('backbone')):
            
            self.unfreeze_backbone()
            logger.info(f"Automatically unfroze backbone at epoch {current_epoch}")
            return True
            
        return False
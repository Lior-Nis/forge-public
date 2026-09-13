import logging
from typing import Dict, Literal

import torch

from pipeline.base import BasePipeline
from pipeline.config import Config
from pipeline.schemas import ClassificationBatchLogData
from managers.logging.classification import ClassificationLoggingManager
from managers.metrics.classification import ClassificationMetricsManager
from managers.weight import WeightManager

logger = logging.getLogger(__name__)


class ClassificationPipeline(BasePipeline):

    def __init__(self, config: Config):
        # Set num_classes before calling super().__init__() as it's needed by managers
        self.num_classes = config.model.head.num_classes

        # Validate alignment between dataset classification_strategy and model num_classes
        strategy = config.data.dataset.classification_strategy
        if strategy == "binary_any_fog" and self.num_classes != 2:
            logger.warning(
                f"Configuration mismatch: classification_strategy='{strategy}' "
                f"but num_classes={self.num_classes}. Expected num_classes=2."
            )
        elif strategy == "multiclass_with_background" and self.num_classes != 4:
            logger.warning(
                f"Configuration mismatch: classification_strategy='{strategy}' "
                f"but num_classes={self.num_classes}. Expected num_classes=4."
            )

        super().__init__(config)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.preprocessors is not None:
            x = self.preprocessors(x)

        if self.training:
            if self.signal_augmentor is not None:
                x = self.signal_augmentor(x)

        x = self.transform(x)

        if self.training:
            if self.spectral_augmentor is not None:
                x = self.spectral_augmentor(x)

        x = self.backbone(x)

        # SpectralPatchEncoder outputs tokens in freq-major order [B, nH*nW, D].
        # For temporal classification the GRU needs to see the joint spectral
        # signature at each time step, not individual frequency bands over time.
        # Reshape to [B, nH, nW, D] → mean over freq dim → [B, nW, D]:
        # the GRU receives nW=20 clean temporal steps instead of 100 mixed ones.
        if hasattr(self.backbone, '_last_nH'):
            nH = self.backbone._last_nH
            nW = self.backbone._last_nW
            B_t, N_t, D_t = x.shape
            x = x.view(B_t, nH, nW, D_t).mean(dim=1)  # [B, nW, D]

        x = self.head(x)
        return x

    def on_train_start(self):
        # Initialize accumulation flags (set per-stage in on_validation_start / on_test_start)
        self.accumulate_spectrals = False
        self.accumulate_timestamps = False
        super().on_train_start()
        # Log metadata tables once at training start (logger is now initialized)
        self.logging_manager.log_patch_metadata_table(stage="train")
        self.logging_manager.log_patient_histogram(stage="train")
        self.logging_manager.log_patch_metadata_table(stage="val")
        self.logging_manager.log_patient_histogram(stage="val")

    def on_validation_start(self):
        super().on_validation_start()
        lm = self.logging_manager
        epoch = self.current_epoch
        self.accumulate_spectrals = (epoch % lm.spectrals_interval == 0)
        self.accumulate_timestamps = (epoch % lm.timestamps_interval == 0)

    def on_test_start(self):
        super().on_test_start()
        lm = self.logging_manager
        self.accumulate_spectrals = True
        self.accumulate_timestamps = True

        # Log test metadata (loaded on-demand from datamodule)
        lm.log_patch_metadata_table(stage="test")
        lm.log_patient_histogram(stage="test")

    def _common_step(
        self,
        batch,
        stage: Literal["train", "val", "test"],
        compute_metrics: bool = False,
        include_x: bool = False,
    ) -> torch.Tensor:
        """
        Common step logic shared across training, validation, and test steps.

        Args:
            batch: Input batch from dataloader
            stage: Current stage ("train", "val", or "test")
            compute_metrics: Whether to update metrics (False for train, True for val/test)
            include_x: Whether to include input tensor x in BatchLogData (for spectral logging)

        Returns:
            Mean loss tensor for this batch
        """
        # Parse batch
        x = batch['x']
        y = batch['y']
        patch_y = batch['patch_y']
        valid_masks = batch['valid_mask']
        patches_metadata = batch['metadata']
        batch_size = x.shape[0]

        # Forward pass
        logits = self(x)

        # Determine targets based on output shape
        targets = y if logits.dim() == 3 else patch_y

        # Handle valid masks
        if valid_masks is None:
            valid_masks = torch.ones_like(targets, dtype=torch.bool)
        elif logits.dim() == 2 and valid_masks.dim() > 1:
            valid_masks = valid_masks.all(dim=-1)

        # Compute and log loss
        losses = self.loss(logits, targets, valid_masks)
        mean_loss = losses.mean()
        self._log_loss(mean_loss, stage, batch_size)

        # Update metrics and compute probabilities (val/test only)
        probabilities = None
        if compute_metrics:
            probabilities = torch.softmax(logits, dim=-1)
            # Soft labels (fog_ratio) must be binarized for torchmetrics
            targets_for_metrics = (targets > 0.5).long() if targets.is_floating_point() else targets
            self.metrics_manager.update_metrics(
                probabilities=probabilities,
                targets=targets_for_metrics,
                stage=stage,
                valid_masks=valid_masks if logits.dim() == 3 else None,
            )

        # Logging managers expect integer labels; binarize soft labels
        labels_for_logging = (targets > 0.5).long() if targets.is_floating_point() else targets

        # Create batch log data and run accumulators
        batch_data = ClassificationBatchLogData(
            logits=logits,
            labels=labels_for_logging,
            losses=losses,
            valid_masks=valid_masks,
            patches_metadata=patches_metadata,
            probabilities=probabilities,
            x=x if include_x else None
        )
        self.run_accumulators(stage=stage, data=batch_data)

        return mean_loss

    def training_step(self, batch, batch_idx: int):
        return self._common_step(batch, stage="train", 
                                 compute_metrics=False, 
                                 include_x=False)

    def validation_step(self, batch, batch_idx: int):
        """Validation step for classification."""
        return self._common_step(batch, stage="val", 
                                 compute_metrics=True, 
                                 include_x=True)

    def test_step(self, batch, batch_idx: int):
        return self._common_step(batch, stage="test",
                                 compute_metrics=True,
                                 include_x=True)

    def run_accumulators(
        self,
        stage: str,
        data: ClassificationBatchLogData,
    ) -> None:
        """
        Run conditional accumulation of outputs, spectrals, and patches based on epoch intervals.
        
        Args:
            stage: Current stage ("train", "val", "test")
            data: Batch data context object
        """
        # Offload detach/cpu transfer here to avoid repetition in steps
        data.detach_cpu()

        self.logging_manager.accumulate_patches(data, stage)

        if self.accumulate_spectrals:
            self.logging_manager.accumulate_spectrals(data, stage)

    def _process_epoch_end(self, stage: Literal["train", "val", "test"]) -> None:
        """Process epoch end for classification."""
        if not self.trainer.sanity_checking:
            self.logging_manager.log_patch_performance_table(stage=stage)

            if stage != "train":
                # Log enhanced classification visualizations (throttled to reduce CPU overhead)
                vis_interval = self.config.train.logging.vis_log_every_n_epochs if self.config.train.logging else 5
                is_last_epoch = self.trainer.current_epoch + 1 >= self.trainer.max_epochs
                if self.trainer.current_epoch % vis_interval == 0 or stage == "test" or is_last_epoch:
                    self.logging_manager.log_classification_visualizations(stage=stage)

                if self.accumulate_spectrals:
                    self.logging_manager.log_spectrals(stage=stage)
                if self.accumulate_timestamps:
                    self.logging_manager.aggregate_timestamps(stage=stage)

        if stage != "train":
            metrics = self.metrics_manager.compute_metrics(stage=stage)
            self.log_metrics(metrics, stage=stage)

            # Log metrics to stdout for easy monitoring
            self._log_epoch_metrics(stage, metrics)

    def _log_epoch_metrics(self, stage: str, metrics: Dict[str, float]) -> None:
        """Log metrics in a structured, easily parseable format."""
        import json
        import torch

        epoch = self.current_epoch

        # Convert tensors to Python floats
        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            clean_metrics[key] = value

        # Log header
        logger.info("="*80)
        logger.info(f"EPOCH {epoch} {stage.upper()} METRICS")
        logger.info("="*80)

        # Log each metric in key=value format for easy grepping
        for key, value in sorted(clean_metrics.items()):
            if isinstance(value, float):
                logger.info(f"{key}={value:.6f}")
            else:
                logger.info(f"{key}={value}")

        # Log as single-line JSON for machine parsing
        metrics_json = {"epoch": epoch, "stage": stage, **clean_metrics}
        logger.info(f"METRICS_JSON: {json.dumps(metrics_json)}")
        logger.info("="*80)

    def _create_logging_manager(self, config: Config, device: str, trainer, datamodule=None):
        return ClassificationLoggingManager(
            config=config,
            device=device,
            num_classes=self.num_classes,
            trainer=trainer,
            datamodule=datamodule,
        )

    def _create_metrics_manager(self, config: Config, device: str):
        return ClassificationMetricsManager(num_classes=self.num_classes)

    def _create_weight_manager(self, config: Config, device: str):
        return WeightManager(
            model=self,
            device=device,
            registry_config=config.train.registry,
            weights_config=config.train.weights
        )
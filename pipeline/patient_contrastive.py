"""Patient-Contrastive Pipeline for pretraining with patient_id supervision."""

import logging
from typing import Literal

import torch

from managers.logging.simclr import SimCLRLoggingManager
from managers.metrics.simclr import SimCLRMetricsManager
from managers.weight import WeightManager
from pipeline.base import BasePipeline
from pipeline.config import Config
from pipeline.schemas import SimCLRBatchLogData

logger = logging.getLogger(__name__)


class PatientContrastivePipeline(BasePipeline):
    """
    Pipeline for patient-contrastive pretraining.

    Single-view supervised contrastive learning using patient_id as labels.
    Same patient = positive pair, different patient = negative pair.
    Reuses SimCLR infrastructure (logging, metrics, schemas).
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.temperature = getattr(self.loss, 'temperature', 0.1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Single-view forward pass: preprocess → augment → transform → backbone → head."""
        if self.preprocessors is not None:
            x = self.preprocessors(x)

        if self.training and self.signal_augmentor is not None:
            x = self.signal_augmentor(x)

        x = self.transform(x)

        if self.training and self.spectral_augmentor is not None:
            x = self.spectral_augmentor(x)

        x = self.backbone(x)
        x = self.head(x)
        return x

    def on_validation_start(self):
        super().on_validation_start()
        lm = self.logging_manager
        epoch = self.current_epoch
        self.accumulate_representations = (epoch % lm.representation_interval == 0)

    def on_test_start(self):
        super().on_test_start()
        self.accumulate_representations = True

    def _common_step(
        self,
        batch,
        stage: Literal["train", "val", "test"],
        include_x: bool = False,
        log_metrics: bool = False,
    ) -> torch.Tensor:
        """
        Common step: single forward pass, patient-contrastive loss.

        Uses patient_id from metadata as supervision signal.
        """
        x_raw = batch['input']
        metadata = batch['metadata']
        batch_size = x_raw.shape[0]

        # Extract patient IDs from metadata
        patient_ids = [m['patient_id'] for m in metadata]

        # Single-view forward pass with augmentation
        if self.preprocessors is not None:
            x = self.preprocessors(x_raw)
        else:
            x = x_raw

        if self.signal_augmentor is not None:
            x = self.signal_augmentor(x)

        x = self.transform(x)

        if self.spectral_augmentor is not None:
            x = self.spectral_augmentor(x)

        embeddings = self.backbone(x)
        embeddings = self.head(embeddings)

        if embeddings.dim() != 2:
            raise ValueError(
                f"Expected 2D embeddings [B, D], got {embeddings.shape}. "
                f"Check projection head output."
            )

        # Compute patient-contrastive loss
        loss = self.loss(embeddings, patient_ids)
        per_sample_losses = self.loss.compute_per_sample(embeddings, patient_ids)

        # Log loss
        self._log_loss(loss, stage, batch_size)

        # Log contrastive metrics (reuse SimCLR metrics with z1=z2=embeddings)
        if log_metrics:
            metrics_dict = self.metrics_manager.compute_batch_contrastive_metrics(
                z1=embeddings,
                z2=embeddings,
                temperature=self.temperature,
                stage=stage,
                batch_size=batch_size,
            )
            for metric_name, metric_value in metrics_dict.items():
                self.log(metric_name, metric_value, batch_size=batch_size)

        # Create batch log data (z1=z2=embeddings for SimCLR logging compatibility)
        batch_data = SimCLRBatchLogData(
            z1=embeddings,
            z2=embeddings,
            losses=per_sample_losses,
            metadata=metadata,
            x_raw=x_raw if include_x else None,
        )
        self.run_accumulators(stage=stage, data=batch_data)

        return loss

    def training_step(self, batch, batch_idx: int):
        return self._common_step(batch, stage="train", include_x=False, log_metrics=(batch_idx % 100 == 0))

    def validation_step(self, batch, batch_idx: int):
        return self._common_step(batch, stage="val", include_x=True, log_metrics=(batch_idx % 100 == 0))

    def test_step(self, batch, batch_idx: int):
        return self._common_step(batch, stage="test", include_x=True, log_metrics=(batch_idx % 100 == 0))

    def run_accumulators(self, stage: str, data: SimCLRBatchLogData) -> None:
        data.detach_cpu()
        self.logging_manager.accumulate_patches(data, stage)
        if stage != "train" and self.accumulate_representations:
            self.logging_manager.log_representation_samples(data, stage)

    def _process_epoch_end(self, stage: Literal["train", "val", "test"]) -> None:
        self.logging_manager.log_patch_performance_table(stage=stage)
        if stage != "train" and self.accumulate_representations:
            self.logging_manager.log_representation_visualizations(stage=stage)
        if stage == "val":
            self.logging_manager.clear_validation_data()
        elif stage == "test":
            self.logging_manager.clear_test_data()
        logger.info(f"Patient-contrastive {stage} epoch completed")

    def _create_logging_manager(self, config: Config, device: str, trainer, datamodule=None):
        return SimCLRLoggingManager(
            config=config,
            device=device,
            trainer=trainer,
            datamodule=datamodule,
        )

    def _create_metrics_manager(self, config: Config, device: str):
        return SimCLRMetricsManager(device=device)

    def _create_weight_manager(self, config: Config, device: str):
        return WeightManager(
            model=self,
            device=device,
            registry_config=config.train.registry,
            weights_config=config.train.weights,
        )

"""Time-Frequency Cross-View Contrastive (TFC) Pipeline.

Subclasses SimCLRPipeline to use different transforms per view instead of
different augmentations on the same transform. View 1 = time-domain (raw),
View 2 = frequency-domain (wavelet). Same-sample cross-domain pairs are
positives, different samples are negatives.

Uses dual encoders (separate backbones per domain) to prevent embedding
collapse from shared-weight gradient conflicts.
"""

import logging
from typing import Literal, Tuple

import hydra
import torch

from pipeline.config import Config
from pipeline.schemas import SimCLRBatchLogData
from pipeline.simclr import SimCLRPipeline

logger = logging.getLogger(__name__)


class TFContrastivePipeline(SimCLRPipeline):
    """
    Time-Frequency Cross-View Contrastive Learning Pipeline.

    Uses dual encoders: separate backbones for time and frequency domains,
    with a shared projection head for contrastive alignment.
    """

    def __init__(self, config: Config, temperature: float = 0.07):
        super().__init__(config, temperature)

        # Second transform (required)
        transform_alt_cfg = getattr(self.model_cfg, 'transform_alt', None)
        if transform_alt_cfg is None:
            raise ValueError(
                "TFContrastivePipeline requires 'transform_alt' in model config. "
                "Set model.transform_alt with a second transform (e.g., wavelet)."
            )
        if hasattr(transform_alt_cfg, 'model_dump'):
            transform_alt_dict = transform_alt_cfg.model_dump()
        else:
            transform_alt_dict = dict(transform_alt_cfg)
        self.transform_alt = hydra.utils.instantiate(transform_alt_dict)

        # Second backbone (required for cross-domain contrastive)
        backbone_alt_cfg = getattr(self.model_cfg, 'backbone_alt', None)
        if backbone_alt_cfg is None:
            raise ValueError(
                "TFContrastivePipeline requires 'backbone_alt' in model config. "
                "Set model.backbone_alt with a second backbone for the frequency domain."
            )
        if hasattr(backbone_alt_cfg, 'model_dump'):
            backbone_alt_dict = backbone_alt_cfg.model_dump()
        else:
            backbone_alt_dict = dict(backbone_alt_cfg)
        self.backbone_alt = hydra.utils.instantiate(backbone_alt_dict)

        logger.info(
            f"TFC initialized: view1={type(self.transform).__name__}, "
            f"view2={type(self.transform_alt).__name__}, "
            f"backbone={type(self.backbone).__name__}, "
            f"backbone_alt={type(self.backbone_alt).__name__}"
        )

    def _create_dual_views(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create two views using different transforms for cross-domain contrastive learning.

        View 1: preprocessors -> signal_aug -> transform (primary, e.g., raw)
        View 2: preprocessors -> signal_aug -> transform_alt (alternative, e.g., wavelet)
        """
        if self.preprocessors is not None:
            x = self.preprocessors(x)

        # View 1: primary transform
        x1 = x.clone()
        if self.signal_augmentor is not None:
            x1 = self.signal_augmentor(x1)
        x1 = self.transform(x1)
        if self.spectral_augmentor is not None:
            x1 = self.spectral_augmentor(x1)

        # View 2: alternative transform
        x2 = x.clone()
        if self.signal_augmentor is not None:
            x2 = self.signal_augmentor(x2)
        x2 = self.transform_alt(x2)
        if self.spectral_augmentor is not None:
            x2 = self.spectral_augmentor(x2)

        return x1, x2

    def _common_step(
        self,
        batch,
        stage: Literal["train", "val", "test"],
        include_x: bool = False,
        log_metrics: bool = False
    ) -> torch.Tensor:
        """Override to use dual backbones (one per domain)."""
        x_raw = batch['input']
        metadata = batch['metadata']
        batch_size = x_raw.shape[0]

        x1_transformed, x2_transformed = self._create_dual_views(x_raw)

        # Dual encoders: each view through its own backbone, shared head
        z1 = self.backbone(x1_transformed)
        z1 = self.head(z1)

        z2 = self.backbone_alt(x2_transformed)
        z2 = self.head(z2)

        if z1.shape != z2.shape:
            raise RuntimeError(
                f"Representation shape mismatch: z1 {z1.shape} vs z2 {z2.shape}. "
                f"Ensure both backbones have the same output_dim."
            )
        if z1.dim() != 2:
            raise ValueError(
                f"Expected 2D representations [batch, features], got z1.shape={z1.shape}."
            )

        loss = self.loss(z1, z2)
        per_sample_losses = self.loss.compute_per_sample(z1, z2)
        self._log_loss(loss, stage, batch_size)

        if log_metrics:
            metrics_dict = self.metrics_manager.compute_batch_contrastive_metrics(
                z1=z1, z2=z2, temperature=self.temperature,
                stage=stage, batch_size=batch_size,
            )
            for metric_name, metric_value in metrics_dict.items():
                self.log(metric_name, metric_value, batch_size=batch_size)

        batch_data = SimCLRBatchLogData(
            z1=z1, z2=z2, losses=per_sample_losses,
            metadata=metadata, x_raw=x_raw if include_x else None,
        )
        self.run_accumulators(stage=stage, data=batch_data)

        return loss

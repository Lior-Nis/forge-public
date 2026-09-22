"""Adversarial classification pipeline (DANN).

Adds a gradient-reversed patient classifier to force the backbone
to learn patient-invariant features.
"""

import logging
import math
from typing import Literal

import torch
import torch.nn.functional as F

from model.adversarial import GradientReversalLayer, PatientClassifierHead
from pipeline.classification import ClassificationPipeline
from pipeline.config import Config

logger = logging.getLogger(__name__)


class AdversarialClassificationPipeline(ClassificationPipeline):

    def __init__(self, config: Config):
        super().__init__(config)
        adv = config.train.adversarial
        self.adv_lambda_max = adv.lambda_max
        self.adv_rampup_epochs = adv.rampup_epochs
        self.adv_hidden_dim = adv.hidden_dim
        self.adv_dropout = adv.dropout

        # Created in on_train_start once we know num_patients
        self.grl = None
        self.patient_head = None
        self.patient_id_to_idx = None

    # ------------------------------------------------------------------
    # Forward: optionally return backbone features for adversarial branch
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, return_backbone_features: bool = False, **kwargs):
        if self.preprocessors is not None:
            x = self.preprocessors(x)

        if self.training and self.signal_augmentor is not None:
            x = self.signal_augmentor(x)

        x = self.transform(x)

        if self.training and self.spectral_augmentor is not None:
            x = self.spectral_augmentor(x)

        backbone_features = self.backbone(x)
        logits = self.head(backbone_features)

        if return_backbone_features:
            return logits, backbone_features
        return logits

    # ------------------------------------------------------------------
    # Setup: build patient mapping + adversarial modules
    # ------------------------------------------------------------------

    def on_train_start(self):
        super().on_train_start()
        self._setup_adversarial()

    def _setup_adversarial(self):
        """Build patient_id→int mapping from training data and create adversarial modules."""
        train_dataset = self.datamodule.train_dataset
        patient_ids = sorted(train_dataset.metadata_df['patient_id'].unique().tolist())
        self.patient_id_to_idx = {pid: i for i, pid in enumerate(patient_ids)}
        num_patients = len(patient_ids)
        logger.info(f"DANN: {num_patients} patients in training set")

        # Create adversarial modules on the correct device
        self.grl = GradientReversalLayer(lambda_=0.0).to(self.device)
        self.patient_head = PatientClassifierHead(
            input_dim=self.backbone.output_dim,
            num_patients=num_patients,
            hidden_dim=self.adv_hidden_dim,
            dropout=self.adv_dropout,
        ).to(self.device)

        # Add patient head params to existing optimizer
        optimizer = self.optimizers()
        optimizer.add_param_group({
            'params': list(self.patient_head.parameters()),
            'lr': self.train_cfg.optimizer.lr,
        })
        logger.info(
            f"DANN: added {sum(p.numel() for p in self.patient_head.parameters())} "
            f"patient-head params to optimizer"
        )

    # ------------------------------------------------------------------
    # Lambda schedule (DANN sigmoid ramp)
    # ------------------------------------------------------------------

    def _compute_lambda(self) -> float:
        """DANN sigmoid schedule: 0 → lambda_max over rampup_epochs."""
        p = min(self.current_epoch / self.adv_rampup_epochs, 1.0)
        return float(self.adv_lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0))

    # ------------------------------------------------------------------
    # Training step: FoG loss + adversarial loss
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int):
        # Update GRL lambda
        current_lambda = self._compute_lambda()
        self.grl.lambda_ = current_lambda

        # --- FoG classification (parent logic inlined for backbone access) ---
        x = batch['x']
        y = batch['y']
        patch_y = batch['patch_y']
        valid_masks = batch['valid_mask']
        patches_metadata = batch['metadata']
        batch_size = x.shape[0]

        logits, backbone_features = self(x, return_backbone_features=True)

        targets = y if logits.dim() == 3 else patch_y
        if valid_masks is None:
            valid_masks = torch.ones_like(targets, dtype=torch.bool)
        elif logits.dim() == 2 and valid_masks.dim() > 1:
            valid_masks = valid_masks.all(dim=-1)

        fog_losses = self.loss(logits, targets, valid_masks)
        fog_loss = fog_losses.mean()

        # --- Adversarial branch ---
        reversed_features = self.grl(backbone_features)
        patient_logits = self.patient_head(reversed_features)

        # Build patient targets from batch metadata
        patient_targets = torch.tensor(
            [self.patient_id_to_idx[m['patient_id']] for m in patches_metadata],
            device=self.device,
            dtype=torch.long,
        )
        adv_loss = F.cross_entropy(patient_logits, patient_targets)

        # Combined loss (GRL handles gradient reversal, so we just add)
        total_loss = fog_loss + adv_loss

        # --- Logging ---
        self._log_loss(fog_loss, "train", batch_size)
        self.log("train_adv_loss", adv_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_adv_lambda", current_lambda, on_step=True, on_epoch=False)

        # Patient accuracy
        with torch.no_grad():
            patient_acc = (patient_logits.argmax(dim=-1) == patient_targets).float().mean()
        self.log("train_patient_acc", patient_acc, prog_bar=True, on_step=True, on_epoch=True)

        # Run accumulators (same as parent but we already have all the data)
        from pipeline.schemas import ClassificationBatchLogData
        batch_data = ClassificationBatchLogData(
            logits=logits,
            labels=targets,
            losses=fog_losses,
            valid_masks=valid_masks,
            patches_metadata=patches_metadata,
            probabilities=None,
            x=None,
        )
        self.run_accumulators(stage="train", data=batch_data)

        return total_loss

    # Val/test: delegate entirely to parent (no adversarial branch)

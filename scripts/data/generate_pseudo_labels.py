"""Generate pseudo-labels for unlabeled data using a trained classification model.

Runs batched inference on the unlabeled dataset and saves per-patch predictions
(FOG probability + predicted label) for use in semi-supervised training.

Usage:
    python scripts/generate_pseudo_labels.py \
        experiment=classification/best_fixed \
        data/paths=kaggle_daily \
        data/process=kaggle_longcontext \
        ckpt_path=checkpoints/classification/blooming-totem-265/last.ckpt
"""

import logging
import os
from datetime import datetime

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset.mae import FOGMAEDataset
from pipeline.classification import ClassificationPipeline
from utils.config_loaders import load_config
from utils.paths import normalize_data_paths

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


def pseudo_label_collate(batch):
    """Collate for pseudo-labeling: returns signals + metadata (no labels needed)."""
    signals = [item.signal for item in batch]
    metadata = [item.metadata for item in batch]
    return {
        "x": torch.stack(signals, dim=0),
        "metadata": metadata,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(hydra_config: DictConfig) -> None:
    ckpt_path = hydra_config.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("ckpt_path is required. Pass it as: ckpt_path=/path/to/checkpoint.ckpt")

    output_dir = hydra_config.get("output_dir", "predictions")
    batch_size = hydra_config.get("pseudo_batch_size", 512)

    pl.seed_everything(hydra_config["global"]["seed"], workers=True)
    normalize_data_paths(hydra_config.data.paths)
    config = load_config(hydra_config)
    logger.info("Configuration validation passed")

    # Build model and load checkpoint
    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = ClassificationPipeline(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["state_dict"]
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k not in model_state or v.shape == model_state[k].shape
    }
    skipped = set(state_dict.keys()) - set(filtered.keys())
    if skipped:
        logger.info(f"Skipped {len(skipped)} keys with shape mismatch: {skipped}")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
    logger.info("Checkpoint loaded successfully")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create dataset over ALL unlabeled patients
    # Gather all patient IDs from splits (train + val + test) to cover everything
    all_patients = list(set(
        config.data.splits.train + config.data.splits.val + config.data.splits.test
    ))
    logger.info(f"Using {len(all_patients)} patients from splits")

    dataset = FOGMAEDataset(
        paths_cfg=config.data.paths,
        dataset_cfg=config.data.dataset,
        stage="train",
        patient_ids=all_patients,
    )
    logger.info(f"Unlabeled dataset: {len(dataset)} patches")

    # Inject normalization stats if available
    from data.process.stats.aggregator import StatsAggregator
    processed_path = config.data.paths.processed_dataset_path
    aggregator = StatsAggregator(processed_path)
    stats = aggregator.compute_split_stats(all_patients)
    if stats is not None and model.preprocessors is not None:
        from model.preprocessors import PatientNormalizationPreprocessor
        for module in model.preprocessors.preprocessors:
            if isinstance(module, PatientNormalizationPreprocessor):
                module.set_stats(stats)
                logger.info("Injected normalization stats from datamodule")
                break

    num_workers = min(config.data.dataloader.num_workers, 8)
    dl_kwargs = {
        "collate_fn": pseudo_label_collate,
        "pin_memory": True,
        "num_workers": num_workers,
        "shuffle": False,
        "drop_last": False,
        "batch_size": batch_size,
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = False
    dataloader = DataLoader(dataset, **dl_kwargs)

    # Run inference
    all_global_indices = []
    all_probs = []
    all_patient_ids = []
    all_session_ids = []

    logger.info(f"Running inference on {len(dataset)} patches (batch_size={batch_size})...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating pseudo-labels"):
            x = batch["x"].to(device)
            metadata = batch["metadata"]

            logits = model(x)

            # Handle both patch-level [B, C] and sequence [B, T, C] outputs
            if logits.dim() == 2:
                probs = F.softmax(logits, dim=-1)[:, 1]  # [B] FOG probability
            else:
                # Sequence output: average over time for patch-level pseudo-label
                probs = F.softmax(logits, dim=-1)[..., 1].mean(dim=-1)  # [B]

            all_probs.append(probs.cpu())
            for m in metadata:
                all_global_indices.append(m["global_idx"])
                all_patient_ids.append(m["patient_id"])
                all_session_ids.append(m["session_id"])

    # Concatenate results
    all_probs = torch.cat(all_probs, dim=0)
    all_global_indices = np.array(all_global_indices)
    all_patient_ids = np.array(all_patient_ids)
    all_session_ids = np.array(all_session_ids)

    # Compute predicted labels
    predicted_labels = (all_probs >= 0.5).long()

    # Summary statistics
    n_total = len(all_probs)
    n_fog = predicted_labels.sum().item()
    n_no_fog = n_total - n_fog
    logger.info(f"Pseudo-label summary: {n_total} total, {n_fog} FOG ({100*n_fog/n_total:.1f}%), {n_no_fog} No-FOG")

    # Confidence distribution
    for threshold in [0.7, 0.8, 0.9, 0.95, 0.99]:
        confident = ((all_probs >= threshold) | (all_probs <= (1 - threshold))).sum().item()
        logger.info(f"  Confidence >= {threshold}: {confident} ({100*confident/n_total:.1f}%)")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    run_name = os.path.basename(os.path.dirname(ckpt_path))
    output_path = os.path.join(output_dir, f"pseudo_labels_{run_name}.pt")

    output = {
        "global_indices": all_global_indices,
        "fog_probability": all_probs,
        "predicted_label": predicted_labels,
        "patient_ids": all_patient_ids,
        "session_ids": all_session_ids,
        "metadata": {
            "checkpoint": str(ckpt_path),
            "n_total": n_total,
            "n_fog": n_fog,
            "n_no_fog": n_no_fog,
            "timestamp": datetime.now().isoformat(),
        },
    }
    torch.save(output, output_path)
    logger.info(f"Pseudo-labels saved to {output_path}")
    print(f"\nSaved: {output_path}")
    print(f"Total: {n_total}, FOG: {n_fog} ({100*n_fog/n_total:.1f}%), No-FOG: {n_no_fog}")


if __name__ == "__main__":
    main()

"""Generate timestamp-level FOG predictions by reconstructing overlapping patches.

Loads a trained classification checkpoint and runs inference on val and test splits.
Patches (block_len with stride_len overlap) are averaged to produce per-timestep
FOG probabilities for each session.

Usage:
    python scripts/predict_classification.py \
        experiment=classification/best_fixed \
        data/paths=kaggle_pure_valid_no_notype_longcontext \
        data/process=kaggle_longcontext \
        ckpt_path=checkpoints/classification/true-terrain-236/last.ckpt
"""

import logging
import os
from collections import defaultdict
from datetime import datetime

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.datamodule import collate
from data.datamodule.datamodule import FOGDataModule
from pipeline.classification import ClassificationPipeline
from utils.config_loaders import load_config
from utils.paths import normalize_data_paths

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


def run_inference(model, dataloader, device):
    """Run inference on a dataloader, collecting per-patch predictions and metadata."""
    results = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            y = batch["y"]              # [B, seq_len]
            valid_mask = batch["valid_mask"]  # [B, seq_len]
            metadata = batch["metadata"]      # list of dicts

            logits = model(x)  # [B, num_classes] or [B, seq_len, num_classes]
            # Handle both patch-level ([B, C]) and sequence ([B, T, C]) outputs
            if logits.dim() == 2:
                probs = F.softmax(logits, dim=-1)[:, 1]  # [B] FOG probability
            else:
                probs = F.softmax(logits, dim=-1)[..., 1]  # [B, T]

            results.append({
                "probs": probs.cpu(),
                "y": y,
                "valid_mask": valid_mask,
                "metadata": metadata,
            })
    return results


def reconstruct_sessions(batch_results, block_len, stride_len):
    """Reconstruct per-timestep predictions by averaging overlapping patches.

    Args:
        batch_results: List of dicts from run_inference
        block_len: Patch length in timesteps
        stride_len: Stride between patches

    Returns:
        Dict mapping session_id to session data dict
    """
    # Group patches by session
    session_patches = defaultdict(list)
    for batch in batch_results:
        probs = batch["probs"]
        y = batch["y"]
        valid_mask = batch["valid_mask"]
        metadata = batch["metadata"]

        for i in range(len(metadata)):
            meta = metadata[i]
            session_patches[meta["session_id"]].append({
                "session_idx": meta["session_idx"],
                "prob": probs[i],         # scalar or [T]
                "y": y[i],                # [seq_len]
                "valid_mask": valid_mask[i],  # [seq_len]
                "patient_id": meta["patient_id"],
                "protocol": meta["protocol"],
            })

    # Reconstruct each session
    sessions = {}
    for session_id, patches in session_patches.items():
        # Sort by session_idx for deterministic reconstruction
        patches.sort(key=lambda p: p["session_idx"])

        max_idx = max(p["session_idx"] for p in patches)
        session_len = max_idx * stride_len + block_len

        prob_sum = torch.zeros(session_len)
        coverage = torch.zeros(session_len)
        labels = torch.zeros(session_len, dtype=torch.long)
        valid_mask = torch.zeros(session_len, dtype=torch.bool)

        for p in patches:
            start = p["session_idx"] * stride_len
            end = start + block_len

            if p["prob"].dim() == 0:
                # Scalar prediction per patch — broadcast to window
                prob_sum[start:end] += p["prob"].item()
            else:
                # Sequence prediction — align timesteps
                seq_len = min(p["prob"].shape[0], end - start)
                prob_sum[start:start + seq_len] += p["prob"][:seq_len]

            coverage[start:end] += 1
            labels[start:end] = p["y"][:block_len]
            valid_mask[start:end] |= p["valid_mask"][:block_len]

        # Average overlapping predictions (covered timesteps only)
        covered = coverage > 0
        probabilities = torch.full((session_len,), -1.0)
        probabilities[covered] = prob_sum[covered] / coverage[covered]

        # Fill uncovered gaps with nearest covered timestep's prediction
        if covered.any() and not covered.all():
            # Find indices of covered timesteps
            covered_idx = covered.nonzero(as_tuple=True)[0]
            uncovered_idx = (~covered).nonzero(as_tuple=True)[0]
            # For each uncovered timestep, find nearest covered one
            # Use searchsorted for efficient nearest-neighbor lookup
            insert_pos = torch.searchsorted(covered_idx, uncovered_idx)
            insert_pos = insert_pos.clamp(0, len(covered_idx) - 1)
            # Check left neighbor too and pick closer one
            left_pos = (insert_pos - 1).clamp(0)
            right_dist = (covered_idx[insert_pos] - uncovered_idx).abs()
            left_dist = (covered_idx[left_pos] - uncovered_idx).abs()
            nearest = torch.where(left_dist <= right_dist, left_pos, insert_pos)
            probabilities[uncovered_idx] = probabilities[covered_idx[nearest]]

        # Mark invalid timesteps as -1
        probabilities[~valid_mask] = -1.0

        sessions[session_id] = {
            "probabilities": probabilities,
            "labels": labels,
            "valid_mask": valid_mask,
            "coverage": coverage,
            "patient_id": patches[0]["patient_id"],
            "protocol": patches[0]["protocol"],
            "num_patches": len(patches),
        }

    return sessions


def print_summary(split_name, sessions):
    """Print a summary of reconstructed sessions."""
    total_timesteps = 0
    total_fog = 0
    total_valid = 0

    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} Split: {len(sessions)} sessions")
    print(f"{'='*60}")
    print(f"  {'Session':<35} {'Length':>8} {'Coverage':>10} {'FOG%':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*8}")

    for sid, data in sorted(sessions.items()):
        length = data["probabilities"].shape[0]
        valid = data["valid_mask"].sum().item()
        fog_count = (data["labels"][data["valid_mask"]] > 0).sum().item() if valid > 0 else 0
        fog_pct = 100 * fog_count / valid if valid > 0 else 0
        mean_cov = data["coverage"][data["coverage"] > 0].mean().item()

        total_timesteps += length
        total_fog += fog_count
        total_valid += valid

        print(f"  {sid:<35} {length:>8} {mean_cov:>10.1f} {fog_pct:>7.1f}%")

    overall_fog = 100 * total_fog / total_valid if total_valid > 0 else 0
    print(f"\n  Total timesteps: {total_timesteps:,}")
    print(f"  FOG prevalence:  {overall_fog:.1f}%")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(hydra_config: DictConfig) -> None:
    ckpt_path = hydra_config.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("ckpt_path is required. Pass it as: ckpt_path=/path/to/checkpoint.ckpt")

    output_dir = hydra_config.get("output_dir", "predictions")

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

    # Read block_len and stride_len from config
    block_len = config.data.process.lengths.block_len
    stride_len = config.data.process.lengths.stride_len
    logger.info(f"Patch config: block_len={block_len}, stride_len={stride_len}")

    # Setup data module for both val and test
    data_module = FOGDataModule(data_cfg=config.data, task_type="classification")

    # Build dataloaders
    batch_size = config.data.dataloader.batch_size
    dl_kwargs = {
        "collate_fn": collate.classification,
        "pin_memory": True,
        "num_workers": config.data.dataloader.num_workers,
        "shuffle": False,
        "drop_last": False,
        "batch_size": batch_size,
    }
    if config.data.dataloader.num_workers > 0:
        dl_kwargs["persistent_workers"] = False

    # Val split (from stage="fit")
    data_module.setup(stage="fit")

    # Inject normalization stats into model (checkpoint may have identity defaults)
    stats = getattr(data_module, "train_normalization_stats", None)
    if stats is not None and model.preprocessors is not None:
        from model.preprocessors import PatientNormalizationPreprocessor
        for module in model.preprocessors.preprocessors:
            if isinstance(module, PatientNormalizationPreprocessor):
                module.set_stats(stats)
                logger.info("Injected normalization stats from datamodule")
                break

    val_loader = DataLoader(data_module.val_dataset, **dl_kwargs)
    logger.info(f"Val dataset: {len(data_module.val_dataset)} patches")

    # Test split
    data_module.setup(stage="test")
    test_loader = DataLoader(data_module.test_dataset, **dl_kwargs)
    logger.info(f"Test dataset: {len(data_module.test_dataset)} patches")

    # Run inference
    logger.info("Running val inference...")
    val_results = run_inference(model, val_loader, device)
    val_sessions = reconstruct_sessions(val_results, block_len, stride_len)

    logger.info("Running test inference...")
    test_results = run_inference(model, test_loader, device)
    test_sessions = reconstruct_sessions(test_results, block_len, stride_len)

    # Print summaries
    print_summary("val", val_sessions)
    print_summary("test", test_sessions)

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    run_name = os.path.basename(os.path.dirname(ckpt_path))
    output_path = os.path.join(output_dir, f"{run_name}_predictions.pt")

    output = {
        "val": val_sessions,
        "test": test_sessions,
        "metadata": {
            "checkpoint": str(ckpt_path),
            "block_len": block_len,
            "stride_len": stride_len,
            "sampling_rate": config.data.process.sampling_rate,
            "timestamp": datetime.now().isoformat(),
        },
    }
    torch.save(output, output_path)
    logger.info(f"Predictions saved to {output_path}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

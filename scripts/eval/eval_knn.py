"""Evaluate MAE pretrained representations via kNN classification.

Loads a pretrained MAE checkpoint, extracts backbone embeddings from labeled
datasets, fits a kNN classifier on kaggle train embeddings, and evaluates on
kaggle test + external datasets (fogstar, fogathome).

Usage:
    # Evaluate on all datasets with default k values
    python scripts/eval_knn.py \
        experiment=pretraining/mae_daily \
        data.process.lengths.block_len=1000 \
        data.process.lengths.stride_len=200 \
        ckpt_path=checkpoints/mae/<run_name>/last.ckpt

    # Custom k values
    python scripts/eval_knn.py \
        experiment=pretraining/mae_daily \
        data.process.lengths.block_len=1000 \
        data.process.lengths.stride_len=200 \
        ckpt_path=checkpoints/mae/<run_name>/last.ckpt \
        '+eval.k_values=[5,10,20,50]'
"""

import logging
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from data.datamodule.collate import classification as classification_collate
from data.datamodule.datamodule import FOGDataModule
from pipeline.mae import MAEPipeline
from utils.config_loaders import load_config
from utils.paths import normalize_data_paths

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)

# Datasets to evaluate on: (name, zarr_suffix, splits_path)
# zarr_suffix is appended to "len{block_len}_stride{stride_len}_" to form the zarr name
DATASETS = {
    "kaggle": {"zarr_suffix": "kaggle", "splits": "kaggle_labeled/dedup"},
    "fogstar": {"zarr_suffix": "fogstar", "splits": "fogstar"},
    "fogathome": {"zarr_suffix": "fogathome", "splits": "fogathome"},
}


def extract_embeddings(
    model: MAEPipeline,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract backbone embeddings and labels from a classification dataloader.

    Args:
        model: MAE pipeline (only preprocessors, transform, backbone are used)
        dataloader: DataLoader yielding classification batches with 'x' and 'patch_y'
        device: Device to run inference on

    Returns:
        Tuple of (embeddings [N, embed_dim], labels [N])
    """
    all_embeddings = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            patch_y = batch["patch_y"]

            # preprocessors → transform → backbone (same as MAE forward, minus masking/head)
            if model.preprocessors is not None:
                x = model.preprocessors(x)
            x = model.transform(x)
            encoded = model.backbone(x)  # [B, num_patches, embed_dim]

            # Mean pool over patches → [B, embed_dim]
            pooled = encoded.mean(dim=1)
            all_embeddings.append(pooled.cpu().numpy())

            # Binary label per sample: patch_y is already per-sample (scalar per patch)
            # For binary strategy it's already 0/1; for multiclass, binarize
            if patch_y.dim() == 1:
                labels = (patch_y > 0).long().numpy()
            else:
                labels = (patch_y.max(dim=-1).values > 0).long().numpy()
            all_labels.append(labels)

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def evaluate_knn(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    k_values: List[int],
) -> Dict[str, Dict[str, float]]:
    """Fit kNN on train embeddings and evaluate on test.

    Args:
        train_emb: Training embeddings [N_train, D]
        train_labels: Training labels [N_train]
        test_emb: Test embeddings [N_test, D]
        test_labels: Test labels [N_test]
        k_values: List of k values to evaluate

    Returns:
        Dict mapping "k={k}" to {"ap": float, "f1": float}
    """
    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
        knn.fit(train_emb, train_labels)

        proba = knn.predict_proba(test_emb)
        preds = knn.predict(test_emb)

        # AP requires probabilities for the positive class
        if proba.shape[1] == 2:
            ap = average_precision_score(test_labels, proba[:, 1])
        else:
            ap = average_precision_score(test_labels, proba[:, 1] if 1 < proba.shape[1] else preds)

        f1 = f1_score(test_labels, preds, zero_division=0)

        results[f"k={k}"] = {"ap": ap, "f1": f1}
        logger.info(f"  k={k}: AP={ap:.4f}, F1={f1:.4f}")

    return results


def build_dataset(
    base_data_cfg: "DataConfig",
    zarr_suffix: str,
    splits_yaml: str,
    block_len: int,
    stride_len: int,
    stage: str = "test",
) -> FOGDataModule:
    """Build a FOGDataModule for a labeled dataset.

    Constructs PathsConfig and SplitsConfig from scratch, reusing shared
    settings (dataloader, dataset, process) from the base MAE config.

    Args:
        base_data_cfg: Base DataConfig (from MAE experiment) for shared settings
        zarr_suffix: Suffix for zarr name (e.g. "kaggle", "fogstar")
        splits_yaml: Path to splits YAML relative to configs/data/splits/
        block_len: Block length used for processing
        stride_len: Stride length used for processing
        stage: Lightning stage to setup ("fit", "test", or None for both)
    """
    import yaml
    from hydra.utils import to_absolute_path
    from data.config import PathsConfig
    from data.datamodule.config import DataConfig, SplitsConfig

    # Load splits from YAML (plain patient ID lists, no Hydra interpolation)
    splits_path = to_absolute_path(f"configs/data/splits/{splits_yaml}.yaml")
    with open(splits_path) as f:
        splits_data = yaml.safe_load(f)
    splits = SplitsConfig(**splits_data)

    # Construct paths: same processed_dir, dataset-specific zarr
    base_paths = base_data_cfg.paths
    dataset_zarr = f"len{block_len}_stride{stride_len}_{zarr_suffix}.zarr"
    paths = PathsConfig(
        input_dir=base_paths.input_dir,
        processed_dir=base_paths.processed_dir,
        dataset_path=dataset_zarr,
    )

    # Classification dataset config (binary)
    from data.dataset.config import DatasetConfig
    dataset_cfg = DatasetConfig(
        classification_strategy="binary_any_fog",
    )

    ext_data_cfg = DataConfig(
        dataloader=base_data_cfg.dataloader,
        dataset=dataset_cfg,
        splits=splits,
        paths=paths,
        process=base_data_cfg.process,
        hierarchical_norm=base_data_cfg.hierarchical_norm,
    )

    dm = FOGDataModule(data_cfg=ext_data_cfg, task_type="classification")
    dm.setup(stage=stage)
    return dm


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(hydra_config: DictConfig) -> None:
    ckpt_path = hydra_config.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("ckpt_path is required. Pass it as: ckpt_path=/path/to/checkpoint.ckpt")

    # Config
    pl.seed_everything(hydra_config["global"]["seed"], workers=True)
    normalize_data_paths(hydra_config.data.paths)
    config = load_config(hydra_config)
    logger.info("Configuration validation passed")

    # kNN settings
    eval_raw = hydra_config.get("eval")
    if eval_raw is not None:
        eval_cfg = OmegaConf.to_container(eval_raw, resolve=True)
    else:
        eval_cfg = {}
    k_values = eval_cfg.get("k_values", [5, 10, 20])

    # Build MAE model and load checkpoint
    logger.info(f"Loading MAE checkpoint: {ckpt_path}")
    model = MAEPipeline(config)
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
        logger.warning(f"Unexpected keys: {unexpected}")
    logger.info("Checkpoint loaded successfully")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    block_len = config.data.process.lengths.block_len
    stride_len = config.data.process.lengths.stride_len

    # Build dataloaders (no balanced sampling, no drop_last for complete eval)
    dl_kwargs = {
        "batch_size": 512,
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": classification_collate,
    }

    # Load kaggle labeled train data for kNN fitting
    kaggle_info = DATASETS["kaggle"]
    logger.info("Loading kaggle labeled dataset...")
    kaggle_dm = build_dataset(
        config.data, kaggle_info["zarr_suffix"], kaggle_info["splits"],
        block_len, stride_len, stage="fit",
    )
    train_dl = DataLoader(kaggle_dm.train_dataset, **dl_kwargs)

    logger.info("Extracting train embeddings...")
    train_emb, train_labels = extract_embeddings(model, train_dl, device)
    logger.info(f"Train: {train_emb.shape[0]} samples, {train_labels.sum()} positive ({train_labels.mean()*100:.1f}%)")

    # Evaluate kNN on each dataset's test split
    all_results = {}
    for dataset_name, dataset_info in DATASETS.items():
        logger.info(f"=== {dataset_name} ===")
        try:
            dm = build_dataset(
                config.data, dataset_info["zarr_suffix"], dataset_info["splits"],
                block_len, stride_len, stage="test",
            )
            test_dl = DataLoader(dm.test_dataset, **dl_kwargs)
            test_emb, test_labels = extract_embeddings(model, test_dl, device)
            logger.info(f"{dataset_name}: {test_emb.shape[0]} samples, {test_labels.sum()} positive ({test_labels.mean()*100:.1f}%)")
            all_results[dataset_name] = evaluate_knn(train_emb, train_labels, test_emb, test_labels, k_values)
        except Exception as e:
            logger.error(f"Failed to evaluate {dataset_name}: {e}")
            all_results[dataset_name] = {"error": str(e)}

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("kNN EVALUATION SUMMARY")
    logger.info("=" * 70)
    header = f"{'Dataset':<15} {'k':<5} {'AP':>8} {'F1':>8}"
    logger.info(header)
    logger.info("-" * 40)
    for dataset_name, results in all_results.items():
        if "error" in results:
            logger.info(f"{dataset_name:<15} ERROR: {results['error']}")
            continue
        for k_key, metrics in results.items():
            logger.info(f"{dataset_name:<15} {k_key:<5} {metrics['ap']:>8.4f} {metrics['f1']:>8.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

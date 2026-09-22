import hashlib
import logging
import os
import pickle
import random
import time
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple
import zarr
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from tqdm import tqdm
from data.config import PathsConfig
from data.dataset.config import DatasetConfig
from data.datamodule.config import DataConfig
from data.datamodule.samplers import ProbabilisticBalancedSampler
from data.datamodule.cache_manager import get_cache_key
from data.dataset import create_dataset
from data.dataset.classification import FOGClassificationDataset
from data.dataset.mae import FOGMAEDataset
from data.dataset.simclr import FOGSimCLRDataset
import data.datamodule.collate as collate
logger = logging.getLogger(__name__)


class SamplableConcatDataset(ConcatDataset):
    """ConcatDataset that supports get_sampling_metadata() for balanced sampling.

    Concatenates metadata DataFrames from all sub-datasets, re-indexing so that
    metadata[i] corresponds to dataset[i] across the combined dataset.
    """

    @property
    def metadata_df(self) -> pd.DataFrame:
        """Combined metadata_df from all sub-datasets (for logging manager)."""
        dfs = []
        for ds in self.datasets:
            if hasattr(ds, "metadata_df"):
                dfs.append(ds.metadata_df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def get_sampling_metadata(self) -> pd.DataFrame:
        dfs = []
        for ds in self.datasets:
            if hasattr(ds, "get_sampling_metadata"):
                dfs.append(ds.get_sampling_metadata())
            else:
                raise TypeError(
                    f"Sub-dataset {type(ds).__name__} does not implement "
                    f"get_sampling_metadata()"
                )
        combined = pd.concat(dfs, ignore_index=True)
        return combined


class FOGDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for FoG detection with cross-validation support.

    Handles data loading for supervised classification and self-supervised pretraining
    with hierarchical balanced sampling across classes, protocols, and patients.

    Supported Tasks:
        - 'classification': Supervised FoG detection
        - 'mae': Masked autoencoder pretraining
        - 'simclr': Contrastive learning

    Args:
        data_cfg: Hydra config with paths, processing params, and dataloader settings

    See docs/datamodule.md for detailed usage examples and configuration.
    """

    def __init__(self, data_cfg: DataConfig, task_type: str = "classification") -> None:
        """
        Initialize FOG DataModule with validated configuration.

        Args:
            data_cfg: Validated DataConfig (Pydantic model)
            task_type: Task type (classification, mae, simclr)
        """
        super().__init__()

        self.config: DataConfig = data_cfg

        # Backward compatibility for existing methods
        self.paths_cfg = self.config.paths
        self.dataset_cfg = self.config.dataset
        self.datamodule_cfg = self.config # For dataloader and splits access

        self.task_type = task_type

    def _resolve_dataset_path(self, dataset_path: str) -> str:
        """
        Resolve dataset path, handling views.

        If the path points to a view (.yaml file), load the view config
        and return the physical dataset path. Otherwise, return the path as-is.

        Args:
            dataset_path: Path to dataset (physical .zarr or view .yaml)

        Returns:
            Path to physical Zarr dataset
        """
        if dataset_path.endswith('.yaml'):
            # This is a view - load it and get the physical dataset path
            import yaml
            from data.views.schemas import ViewConfig

            logger.info(f"Loading view from: {dataset_path}")
            with open(dataset_path, 'r') as f:
                view_config_dict = yaml.safe_load(f)

            view_config = ViewConfig(**view_config_dict)
            physical_path = view_config.source.dataset_path
            logger.info(f"Resolved view to physical dataset: {physical_path}")
            return physical_path
        else:
            # Already a physical dataset
            return dataset_path

    def setup(self, stage: Optional[str] = None) -> None:
        # Load splits from validated config
        train_patients = self.config.splits.train
        val_patients = self.config.splits.val
        test_patients = self.config.splits.test

        # Load and aggregate normalization stats from Zarr (training fold only)
        if stage == "fit" or stage is None:
            from data.process.stats.aggregator import StatsAggregator

            # Resolve views to physical datasets
            processed_path = self._resolve_dataset_path(self.config.paths.processed_dataset_path)
            aggregator = StatsAggregator(processed_path)
            
            self.train_normalization_stats = aggregator.compute_split_stats(train_patients)
            logger.debug(f"Training stats: mean={self.train_normalization_stats['mean'].numpy()}, "
                        f"std={self.train_normalization_stats['std'].numpy()}")

            self.train_dataset = self._create_dataset("train", train_patients)
            self.val_dataset = self._create_dataset("val", val_patients)

            # Add pseudo-labeled data if configured
            pl_cfg = self.config.pseudo_labels
            if pl_cfg.enabled:
                self.train_dataset = self._create_pseudo_labeled_train(
                    self.train_dataset, pl_cfg
                )

            # Setup probabilistic sampling
            dl_cfg = self.config.dataloader
            if dl_cfg.balanced_sampling:
                dataset_description_hash = get_cache_key(self.config, f"{dl_cfg.sampler_type}_probabilistic")
                self.weights_cache_path = os.path.join(
                    dl_cfg.cache_data_dir,
                    f"{dl_cfg.sampler_type}_weights_cache_{dataset_description_hash}.pkl",
                )
                # TODO: make sure that the weights are statistically stable
                self.balanced_sampler = ProbabilisticBalancedSampler(
                    self.train_dataset,
                    dl_cfg.batch_size,
                    weights_cache_path=self.weights_cache_path,
                    balance_config=dl_cfg.balance_weights
                )
            else:
                self.balanced_sampler = None

        if stage == "test" or stage is None:
            self.test_dataset = self._create_dataset("test", test_patients)

    def _create_dataset(self, stage: str, patient_ids: List[str]):
        """
        Factory method to create task-specific datasets using registry pattern.

        Args:
            stage: Dataset stage (train/val/test)
            patient_ids: Patient IDs for this stage

        Returns:
            Task-specific dataset instance
        """
        # Use registry pattern for dataset creation
        return create_dataset(
            task_type=self.task_type,
            paths_cfg=self.paths_cfg,
            dataset_cfg=self.dataset_cfg,
            stage=stage,
            patient_ids=patient_ids
        )

    def _create_pseudo_labeled_train(self, real_train_dataset, pl_cfg):
        """
        Create a combined dataset mixing real labeled + pseudo-labeled data.

        Args:
            real_train_dataset: The original labeled training dataset
            pl_cfg: PseudoLabelConfig with file_path, confidence_threshold, max_samples

        Returns:
            SamplableConcatDataset combining real and pseudo-labeled data
        """
        from data.dataset.pseudo_labeled import FOGPseudoLabeledDataset

        if not pl_cfg.file_path:
            raise ValueError(
                "pseudo_labels.enabled=true but pseudo_labels.file_path is not set. "
                "Generate pseudo-labels first with scripts/generate_pseudo_labels.py"
            )

        # Resolve paths for unlabeled data
        # Use the same paths config but pointing to unlabeled zarr
        unlabeled_paths_cfg = self._resolve_unlabeled_paths(pl_cfg)

        # Get all patient IDs from the pseudo-label file to use as filter
        pseudo_data = torch.load(pl_cfg.file_path, map_location="cpu", weights_only=False)
        all_pseudo_patients = list(set(pseudo_data["patient_ids"].tolist()))
        logger.info(f"Pseudo-label file has {len(all_pseudo_patients)} unique patients")

        # Create pseudo-labeled dataset
        pseudo_dataset = FOGPseudoLabeledDataset(
            paths_cfg=unlabeled_paths_cfg,
            dataset_cfg=self.dataset_cfg,
            stage="train",
            patient_ids=all_pseudo_patients,
            pseudo_label_path=pl_cfg.file_path,
            confidence_threshold=pl_cfg.confidence_threshold,
            max_samples=pl_cfg.max_samples,
        )

        if len(pseudo_dataset) == 0:
            logger.warning("No pseudo-labeled samples passed confidence threshold. Using labeled data only.")
            return real_train_dataset

        logger.info(
            f"Combined dataset: {len(real_train_dataset)} labeled + "
            f"{len(pseudo_dataset)} pseudo-labeled = "
            f"{len(real_train_dataset) + len(pseudo_dataset)} total"
        )
        return SamplableConcatDataset([real_train_dataset, pseudo_dataset])

    def _resolve_unlabeled_paths(self, pl_cfg):
        """Resolve paths config for unlabeled data."""
        if pl_cfg.unlabeled_paths:
            # Load a separate paths config for unlabeled data
            # Build a PathsConfig pointing to the unlabeled zarr
            from data.config import PathsConfig
            return PathsConfig(
                input_dir=self.paths_cfg.input_dir,
                processed_dir=self.paths_cfg.processed_dir,
                dataset_path=pl_cfg.unlabeled_paths,
            )
        else:
            # Default: construct unlabeled path from current config
            # Replace the dataset name pattern to point to unlabeled zarr
            process_cfg = self.config.process
            block_len = process_cfg.lengths.block_len
            stride_len = process_cfg.lengths.stride_len
            unlabeled_path = f"len{block_len}_stride{stride_len}_kaggle_daily_unlabeled.zarr"
            from data.config import PathsConfig
            return PathsConfig(
                input_dir=self.paths_cfg.input_dir,
                processed_dir=self.paths_cfg.processed_dir,
                dataset_path=unlabeled_path,
            )

    def _get_collate_fn(self):
        """Get task-specific collate function for maximum speed."""
        if self.task_type in ("classification", "segmentation"):
            return collate.classification
        if self.task_type in ["mae", "simclr", "jepa", "ibot", "patient_contrastive", "multitask_ssl"]:
            return collate.selfsupervised
        raise ValueError(f"Unknown task type '{self.task_type}'. No legacy fallback provided.")

    def _build_dataloader_kwargs(self, shuffle: bool = False, batch_size: Optional[int] = None) -> dict:
        """Build common DataLoader kwargs with worker configuration."""
        cfg = self.config.dataloader # Use validated config
        num_workers = cfg.num_workers

        kwargs = {
            'collate_fn': self._get_collate_fn(),
            'pin_memory': cfg.pin_memory,
            'num_workers': num_workers,
            'shuffle': shuffle,
        }

        if batch_size is not None:
            kwargs['batch_size'] = batch_size

        if num_workers > 0:
            kwargs['persistent_workers'] = cfg.persistent_workers
            kwargs['prefetch_factor'] = cfg.prefetch_factor

        return kwargs

    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for training with balanced sampling."""
        if self.config.dataloader.balanced_sampling:
            # Use pre-initialized sampler from setup()
            kwargs = self._build_dataloader_kwargs()
            kwargs['batch_sampler'] = self.balanced_sampler
            kwargs.pop('shuffle')  # Incompatible with batch_sampler
            kwargs.pop('batch_size', None)  # Incompatible with batch_sampler
            return DataLoader(self.train_dataset, **kwargs)
        else:
            # Default to standard DataLoader if no balancing is specified
            kwargs = self._build_dataloader_kwargs(
                shuffle=self.config.dataloader.shuffle,
                batch_size=self.config.dataloader.batch_size
            )
            return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader for validation."""
        cfg = self.config.dataloader
        kwargs = self._build_dataloader_kwargs(
            shuffle=False,
            batch_size=cfg.batch_size * 2  # 2x for validation
        )
        kwargs['drop_last'] = False  # Never drop samples during evaluation
        return DataLoader(self.val_dataset, **kwargs)

    def test_dataloader(self) -> DataLoader:
        """Return the DataLoader for testing."""
        cfg = self.config.dataloader
        kwargs = self._build_dataloader_kwargs(
            shuffle=False,
            batch_size=cfg.batch_size * 5  # 5x for testing
        )
        kwargs['drop_last'] = False  # Never drop samples during evaluation
        return DataLoader(self.test_dataset, **kwargs)
"""Dataset classes for FoG detection tasks."""

from typing import List, Optional

from data.config import PathsConfig
from data.dataset.config import DatasetConfig
from .base import BaseFOGDataset
from .classification import FOGClassificationDataset
from .mae import FOGMAEDataset
from .pseudo_labeled import FOGPseudoLabeledDataset
from .simclr import FOGSimCLRDataset


# Dataset registry mapping task types to dataset classes
DATASET_REGISTRY = {
    'classification': FOGClassificationDataset,
    'segmentation': FOGClassificationDataset,  # same dataset; pipeline handles seq labels
    'mae': FOGMAEDataset,
    'simclr': FOGSimCLRDataset,
    'jepa': FOGMAEDataset,
    'ibot': FOGMAEDataset,
    'patient_contrastive': FOGSimCLRDataset,
    'multitask_ssl': FOGMAEDataset,
}


def create_dataset(
    task_type: str,
    paths_cfg: PathsConfig,
    dataset_cfg: DatasetConfig,
    stage: str,
    patient_ids: Optional[List[str]] = None,
    **kwargs
) -> BaseFOGDataset:
    """
    Factory function for creating task-specific datasets.

    Args:
        task_type: Type of task ('classification', 'mae', 'simclr')
        paths_cfg: Path configuration (shared across layers)
        dataset_cfg: Dataset-specific configuration
        stage: Dataset stage (train/val/test)
        patient_ids: Patient IDs for this stage
        **kwargs: Task-specific parameters
            - mask_ratio (float): For MAE task, ratio of patches to mask (default: 0.4)

    Returns:
        Instantiated dataset instance

    Raises:
        ValueError: If task_type is not recognized

    Examples:
        >>> # Classification dataset
        >>> dataset = create_dataset('classification', paths_cfg, dataset_cfg, 'train', ['p1', 'p2'])

        >>> # MAE dataset with custom mask ratio
        >>> dataset = create_dataset('mae', paths_cfg, dataset_cfg, 'train', ['p1', 'p2'], mask_ratio=0.75)
    """
    if task_type not in DATASET_REGISTRY:
        available_tasks = list(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown task type '{task_type}'. Available tasks: {available_tasks}"
        )

    dataset_cls = DATASET_REGISTRY[task_type]

    # All datasets accept base parameters
    # Task-specific parameters (like mask_ratio) are passed via **kwargs
    return dataset_cls(
        paths_cfg=paths_cfg,
        dataset_cfg=dataset_cfg,
        stage=stage,
        patient_ids=patient_ids,
        **kwargs
    )


__all__ = [
    "BaseFOGDataset",
    "FOGClassificationDataset",
    "FOGMAEDataset",
    "FOGPseudoLabeledDataset",
    "FOGSimCLRDataset",
    "create_dataset",
    "DATASET_REGISTRY",
]
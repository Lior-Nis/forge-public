import torch


# Optimized task-specific collate functions
def classification(batch):
    """
    Fast collate function for classification datasets returning DatasetSample.

    Args:
        batch: List of DatasetSample instances from FOGClassificationDataset

    Returns:
        Dictionary with batched tensors and metadata list
    """
    # Extract fields from DatasetSample instances
    signals = [item.signal for item in batch]
    labels = [item.labels for item in batch]
    patch_labels = [item.patch_label for item in batch]
    valid_masks = [item.valid_mask for item in batch]
    metadata = [item.metadata for item in batch]

    return {
        'x': torch.stack(signals, dim=0),
        'y': torch.stack(labels, dim=0),
        'patch_y': torch.stack(patch_labels, dim=0),
        'valid_mask': torch.stack(valid_masks, dim=0),
        'metadata': metadata
    }


def selfsupervised(batch):
    """
    Fast collate function for self-supervised datasets (MAE/SimCLR).

    Both MAE and SimCLR now return raw signals from the dataset.
    Augmentation and dual-view creation happen in the model pipeline on GPU.

    Args:
        batch: List of DatasetSample instances from FOGMAEDataset or FOGSimCLRDataset.
               Each sample has: signal, metadata

    Returns:
        Dictionary with batched tensors:
        - 'input': Batched input signals [B, C, T]
        - 'target': Same reference as input (augmentation happens in model)
        - 'metadata': List of metadata dicts (one per sample)

    Note:
        For MAE: Masking happens in the model's forward pass.
        For SimCLR: Dual-view creation and augmentation happen in the model pipeline.
    """
    signals = [item.signal for item in batch]
    metadata = [item.metadata for item in batch]

    # Stack signals once
    stacked = torch.stack(signals, dim=0)

    return {
        'input': stacked,
        'target': stacked,  # Same reference - transformations done by pipline
        'metadata': metadata
    }

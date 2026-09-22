"""Simple dataset that loads precomputed backbone embeddings from disk."""

from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    Loads precomputed embeddings saved by scripts/precompute_mae_embeddings.py.

    Each sample is a dict matching the FOGClassificationDataset output format
    so it can drop into any existing training loop unchanged.
    """

    def __init__(self, pt_path: str):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.embeddings: torch.Tensor = data["embeddings"]   # [N, nW, D]
        self.labels: torch.Tensor = data["labels"]           # [N]  int64
        self._metadata: list = data.get("metadata", [{}] * len(self.labels))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        label = self.labels[idx].long()
        return {
            "x": self.embeddings[idx],          # [nW, D]  — fed directly to head
            "y": label,
            "patch_y": label,
            "valid_mask": torch.ones(1, dtype=torch.bool),
            "metadata": self._metadata[idx] if idx < len(self._metadata) else {},
        }

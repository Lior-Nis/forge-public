"""
Path configuration helpers and normalization utilities.
Centralizes path handling to keep configs environment-agnostic.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@dataclass
class PathConfig:
    """Configuration for all project paths."""

    # Base paths
    project_root: Path
    data_root: Path

    # Checkpoint paths
    checkpoint_dir: Path

    # Data dictionary paths
    dicts_dir: Path

    # Results paths
    results_dir: Path

    @classmethod
    def from_config(cls, config: Optional[Dict] = None) -> "PathConfig":
        """Create PathConfig from configuration dictionary or defaults."""

        # Get project root (directory containing this file's parent)
        project_root = Path(__file__).parent.parent.absolute()

        if config is None:
            config = {}

        # Use config values or defaults
        data_root = Path(config.get("data_root", project_root / "data"))
        checkpoint_dir = Path(
            config.get("checkpoint_dir", project_root / "checkpoints")
        )
        dicts_dir = Path(config.get("dicts_dir", data_root / "dicts"))
        results_dir = Path(config.get("results_dir", project_root / "results"))

        # Ensure directories exist
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            project_root=project_root,
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            dicts_dir=dicts_dir,
            results_dir=results_dir,
        )

    def get_checkpoint_path(self, experiment_name: str) -> Path:
        """Get the full checkpoint path for an experiment."""
        return self.checkpoint_dir / f"{experiment_name}.ckpt"

    def get_dict_path(self, dict_name: str) -> Path:
        """Get the full path for a data dictionary file."""
        return self.dicts_dir / dict_name

    def get_results_path(self, experiment_name: str) -> Path:
        """Get the results directory path for an experiment."""
        results_path = self.results_dir / experiment_name
        results_path.mkdir(parents=True, exist_ok=True)
        return results_path


def load_paths_from_config(config_dict: Dict) -> PathConfig:
    """Load path configuration from a config dictionary."""
    paths_config = config_dict.get("paths", {})
    return PathConfig.from_config(paths_config)


def get_default_paths() -> PathConfig:
    """Get default path configuration."""
    return PathConfig.from_config()


def _ensure_absolute(path_value: Optional[str]) -> Optional[str]:
    """Convert relative paths to absolute using Hydra's original cwd."""
    if not path_value or not isinstance(path_value, str):
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return to_absolute_path(path_value)


def normalize_data_paths(paths_cfg) -> None:
    """
    Normalize data path entries in-place so that downstream consumers
    always operate with absolute paths regardless of Hydra's run dir.
    """
    if paths_cfg is None:
        return

    path_keys = [
        "data_root",
        "raw_root",
        "processed_root",
        "cache_root",
        "input_dir",
        "processed_dir",
        "patient_norm_stats",
        "cv_split_path",
    ]

    for key in path_keys:
        try:
            if key in paths_cfg and paths_cfg[key]:
                paths_cfg[key] = _ensure_absolute(paths_cfg[key])
        except (AttributeError, TypeError):
            # DictConfig may not support `in`; fall back to hasattr check
            if hasattr(paths_cfg, key):
                value = paths_cfg.__dict__.get(key)
                if value:
                    setattr(paths_cfg, key, _ensure_absolute(value))


def build_processed_dataset_filename(
    block_len: int,
    stride_len: int,
) -> str:
    """
    Build filename for processed dataset.

    Format: 'len{block}_stride{stride}.zarr'
    """
    return f"len{block_len}_stride{stride_len}.zarr"


def build_processed_dataset_path(
    processed_dir: str,
    block_len: int,
    stride_len: int,
) -> str:
    """
    Build full path to processed dataset file.
    Pure function, no dependency on config objects.
    """
    filename = build_processed_dataset_filename(
        block_len=block_len,
        stride_len=stride_len,
    )
    return os.path.join(processed_dir, filename)
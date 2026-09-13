"""Shared fixtures and helpers for config tests."""

import glob
import os
from pathlib import Path
from typing import Optional

import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parents[2]
CONFIG_DIR = str(REPO_ROOT / "configs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_experiment_names() -> list[str]:
    # Derive the Hydra group-relative name (e.g. "classification/baseline") in an
    # OS-agnostic way. String-replacing an OS-native path breaks on Windows, where
    # glob returns backslash separators and the "experiment/" replace never matches.
    exp_dir = REPO_ROOT / "configs" / "experiment"
    return sorted(
        p.relative_to(exp_dir).with_suffix("").as_posix()
        for p in exp_dir.glob("**/*.yaml")
    )


def collect_component_yamls(subdir: str) -> list[tuple[str, str]]:
    """Return (name, path) pairs for all YAMLs in configs/<subdir>."""
    pattern = str(REPO_ROOT / "configs" / subdir / "*.yaml")
    return [
        (os.path.basename(f).replace(".yaml", ""), f)
        for f in sorted(glob.glob(pattern))
    ]


def load_yaml(path: str) -> Optional[dict]:
    with open(path) as fh:
        return yaml.safe_load(fh)


# Resolve OmegaConf-style ${...} interpolations with canned defaults so
# standalone component configs can be instantiated without a full Hydra context.
_INTERP_DEFAULTS = {
    "global": {"img_size": 224},
    "model": {"backbone": {"embed_dim": 512, "patch_size": 10, "output_dim": 512}, "num_classes": 4},
}


def resolve_component_cfg(raw: dict) -> dict:
    """
    Replace OmegaConf ${key.path} interpolations with known test defaults so
    component configs can be instantiated standalone.
    """
    import re

    def _resolve(obj):
        if isinstance(obj, str):
            match = re.fullmatch(r"\$\{([^}]+)\}", obj)
            if match:
                parts = match.group(1).split(".")
                val = _INTERP_DEFAULTS
                for p in parts:
                    val = val.get(p, obj) if isinstance(val, dict) else obj
                return val
            return obj
        if isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(v) for v in obj]
        return obj

    return _resolve(raw)


# ---------------------------------------------------------------------------
# Session-scoped Hydra context
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hydra_cfg_dir() -> str:
    return CONFIG_DIR


@pytest.fixture(scope="session")
def compose_experiment(hydra_cfg_dir):
    """Factory: compose a full config for the named experiment."""
    results: dict = {}

    def _compose(name: str):
        if name not in results:
            with initialize_config_dir(config_dir=hydra_cfg_dir, version_base="1.3"):
                results[name] = compose(
                    config_name="config",
                    overrides=[f"experiment={name}"],
                )
        return results[name]

    return _compose

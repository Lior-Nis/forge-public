"""
Unit tests for individual component config files.

Each YAML under configs/model/{backbone,transform,head}/ and
configs/train/loss/ is loaded and verified to:
- Reference an importable Python class (_target_)
- Instantiate without error (after resolving OmegaConf interpolations)
- Meet component-specific contracts (e.g. backbone.output_dim)

Component configs that use OmegaConf interpolations (${global.img_size},
${model.backbone.embed_dim}) are resolved with safe test defaults before
instantiation; they are also covered end-to-end via test_experiment_configs.py.
"""

import importlib
import os

import hydra
import pytest
import torch

from .conftest import (
    collect_component_yamls,
    load_yaml,
    resolve_component_cfg,
)

# ---------------------------------------------------------------------------
# Parametrize over component config files
# ---------------------------------------------------------------------------

BACKBONE_CFGS = collect_component_yamls("model/backbone")
TRANSFORM_CFGS = collect_component_yamls("model/transform")
HEAD_CFGS = collect_component_yamls("model/head")
LOSS_CFGS = collect_component_yamls("train/loss")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _target_class_exists(cfg_dict: dict) -> bool:
    """Return True if the _target_ class can be imported."""
    target = cfg_dict.get("_target_")
    if not target:
        return False
    parts = target.rsplit(".", 1)
    if len(parts) != 2:
        return False
    module_path, class_name = parts
    try:
        mod = importlib.import_module(module_path)
        return hasattr(mod, class_name)
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Backbone tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,path", BACKBONE_CFGS, ids=[n for n, _ in BACKBONE_CFGS])
def test_backbone_target_importable(name, path):
    """Backbone config references an existing Python class."""
    raw = load_yaml(path)
    assert raw is not None, f"{name}.yaml is empty"
    assert "_target_" in raw, f"{name}.yaml missing _target_"
    assert _target_class_exists(raw), (
        f"{name}.yaml _target_='{raw['_target_']}' cannot be imported"
    )


@pytest.mark.parametrize("name,path", BACKBONE_CFGS, ids=[n for n, _ in BACKBONE_CFGS])
def test_backbone_instantiates(name, path):
    """Backbone config instantiates and exposes output_dim."""
    raw = load_yaml(path)
    cfg = resolve_component_cfg(raw)
    try:
        backbone = hydra.utils.instantiate(cfg)
    except Exception as e:
        # Some backbones (e.g. MOMENT) need an optional dependency that is not part
        # of the base install; skip rather than fail when it is absent.
        msg = str(e)
        if "momentfm" in msg or "is required for" in msg:
            pytest.skip(f"{name}: optional backbone dependency not installed")
        raise
    assert hasattr(backbone, "output_dim"), (
        f"{name}: backbone must expose output_dim"
    )
    assert backbone.output_dim > 0


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,path", TRANSFORM_CFGS, ids=[n for n, _ in TRANSFORM_CFGS])
def test_transform_target_importable(name, path):
    """Transform config references an existing Python class."""
    raw = load_yaml(path)
    assert raw is not None, f"{name}.yaml is empty"
    assert "_target_" in raw, f"{name}.yaml missing _target_"
    assert _target_class_exists(raw), (
        f"{name}.yaml _target_='{raw['_target_']}' cannot be imported"
    )


@pytest.mark.parametrize("name,path", TRANSFORM_CFGS, ids=[n for n, _ in TRANSFORM_CFGS])
def test_transform_instantiates(name, path):
    """Transform config instantiates successfully."""
    raw = load_yaml(path)
    cfg = resolve_component_cfg(raw)
    transform = hydra.utils.instantiate(cfg)
    assert transform is not None
    assert hasattr(transform, "__call__")


@pytest.mark.parametrize("name,path", TRANSFORM_CFGS, ids=[n for n, _ in TRANSFORM_CFGS])
def test_transform_forward_shape(name, path):
    """Transform produces a 4-D or 3-D tensor from a [B, C, T] input."""
    raw = load_yaml(path)
    cfg = resolve_component_cfg(raw)
    transform = hydra.utils.instantiate(cfg)

    x = torch.randn(2, 3, 200)
    with torch.no_grad():
        out = transform(x)

    assert isinstance(out, torch.Tensor), f"{name}: output must be a tensor"
    assert out.ndim in (3, 4), f"{name}: expected 3-D or 4-D output, got {out.ndim}-D"
    assert out.shape[0] == 2, f"{name}: batch dimension must be preserved"
    assert not torch.isnan(out).any(), f"{name}: transform output contains NaN"


# ---------------------------------------------------------------------------
# Head tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,path", HEAD_CFGS, ids=[n for n, _ in HEAD_CFGS])
def test_head_target_importable(name, path):
    """Head config references an existing Python class."""
    raw = load_yaml(path)
    assert raw is not None, f"{name}.yaml is empty"
    assert "_target_" in raw, f"{name}.yaml missing _target_"
    assert _target_class_exists(raw), (
        f"{name}.yaml _target_='{raw['_target_']}' cannot be imported"
    )


@pytest.mark.parametrize("name,path", HEAD_CFGS, ids=[n for n, _ in HEAD_CFGS])
def test_head_instantiates(name, path):
    """Head config instantiates successfully (with interpolations resolved)."""
    raw = load_yaml(path)
    cfg = resolve_component_cfg(raw)
    head = hydra.utils.instantiate(cfg)
    assert head is not None
    assert hasattr(head, "__call__")


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,path", LOSS_CFGS, ids=[n for n, _ in LOSS_CFGS])
def test_loss_target_importable(name, path):
    """Loss config references an existing Python class."""
    raw = load_yaml(path)
    assert raw is not None, f"{name}.yaml is empty"
    assert "_target_" in raw, f"{name}.yaml missing _target_"
    assert _target_class_exists(raw), (
        f"{name}.yaml _target_='{raw['_target_']}' cannot be imported"
    )


@pytest.mark.parametrize("name,path", LOSS_CFGS, ids=[n for n, _ in LOSS_CFGS])
def test_loss_instantiates(name, path):
    """Loss config instantiates without error."""
    raw = load_yaml(path)
    cfg = resolve_component_cfg(raw)
    loss = hydra.utils.instantiate(cfg)
    assert loss is not None
    assert hasattr(loss, "__call__")

"""
Parametrized tests for every experiment config in configs/experiment/.

For each experiment the test suite checks:
1. Hydra composition succeeds (no missing config groups, no syntax errors)
2. Pydantic validation passes (types, dimension constraints, required fields)
3. Model components instantiate (transform, backbone, optional head)
4. transform → backbone forward pass produces a valid tensor
5. Loss function instantiates without error
"""

import hydra
import pytest
import torch
from omegaconf import OmegaConf

from utils.config_loaders import load_config
from .conftest import collect_experiment_names, CONFIG_DIR

# ---------------------------------------------------------------------------
# Parametrize over all experiment configs
# ---------------------------------------------------------------------------

ALL_EXPERIMENTS = collect_experiment_names()


# ---------------------------------------------------------------------------
# Helpers used inside individual tests (avoid session-fixture nesting issues)
# ---------------------------------------------------------------------------

def _compose(name: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=[f"experiment={name}"])


# ---------------------------------------------------------------------------
# Test 1 – Hydra composition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("experiment", ALL_EXPERIMENTS)
def test_experiment_composes(experiment):
    """Config composes without missing groups or interpolation errors."""
    cfg = _compose(experiment)
    assert cfg is not None
    assert hasattr(cfg, "model")
    assert hasattr(cfg, "train")
    assert hasattr(cfg, "data")


# ---------------------------------------------------------------------------
# Test 2 – Pydantic validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("experiment", ALL_EXPERIMENTS)
def test_experiment_pydantic_validates(experiment):
    """Composed config passes Pydantic schema validation."""
    cfg = _compose(experiment)
    validated = load_config(cfg)
    assert validated.model is not None
    assert validated.train is not None
    assert validated.data is not None


# ---------------------------------------------------------------------------
# Test 3 – Transform instantiation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("experiment", ALL_EXPERIMENTS)
def test_experiment_transform_instantiates(experiment):
    """Transform class referenced in config can be instantiated."""
    cfg = _compose(experiment)
    mc = OmegaConf.to_container(cfg.model, resolve=True)
    transform = hydra.utils.instantiate(mc["transform"])
    assert transform is not None
    assert hasattr(transform, "__call__")


# ---------------------------------------------------------------------------
# Test 4 – Backbone instantiation and output_dim
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("experiment", ALL_EXPERIMENTS)
def test_experiment_backbone_instantiates(experiment):
    """Backbone class instantiates and exposes output_dim."""
    cfg = _compose(experiment)
    mc = OmegaConf.to_container(cfg.model, resolve=True)
    backbone = hydra.utils.instantiate(mc["backbone"])
    assert backbone is not None
    assert hasattr(backbone, "output_dim"), (
        f"Backbone {type(backbone).__name__} must expose output_dim"
    )
    assert backbone.output_dim > 0


# ---------------------------------------------------------------------------
# Test 5 – transform → backbone forward pass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("experiment", ALL_EXPERIMENTS)
def test_experiment_forward_pass(experiment):
    """transform(x) → backbone(x_t) produces a valid non-NaN tensor."""
    cfg = _compose(experiment)
    mc = OmegaConf.to_container(cfg.model, resolve=True)

    seq_len = cfg.data.dataset.get("seq_len", 200)
    x = torch.randn(2, 3, seq_len)

    transform = hydra.utils.instantiate(mc["transform"])
    backbone = hydra.utils.instantiate(mc["backbone"])

    with torch.no_grad():
        x_t = transform(x)
        out = backbone(x_t)

    assert out is not None
    assert isinstance(out, torch.Tensor)
    assert out.ndim >= 2
    assert not torch.isnan(out).any(), "Backbone output contains NaN"
    assert out.shape[0] == 2, "Batch dimension must be preserved"


# ---------------------------------------------------------------------------
# Test 6 – Loss function instantiation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("experiment", ALL_EXPERIMENTS)
def test_experiment_loss_instantiates(experiment):
    """Loss function referenced in train.loss can be instantiated."""
    cfg = _compose(experiment)
    loss_cfg = OmegaConf.to_container(cfg.train.loss, resolve=True)
    loss = hydra.utils.instantiate(loss_cfg)
    assert loss is not None
    assert hasattr(loss, "__call__")

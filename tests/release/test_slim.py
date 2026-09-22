import torch
from scripts.release.slim import slim_checkpoint, DROP_KEYS

def _fake_ckpt():
    sd = {"transform.scales": torch.arange(5).float(), "head.weight": torch.randn(3, 4)}
    return {
        "epoch": 47, "global_step": 1000, "pytorch-lightning_version": "2.6.0",
        "state_dict": sd, "hparams_name": "config",
        "hyper_parameters": {"config": {"sentinel": True}},
        "optimizer_states": [{"big": torch.randn(1000)}],
        "lr_schedulers": [{"x": 1}], "callbacks": {"cb": 1}, "loops": {"l": 1},
    }

def test_slim_drops_optimizer_and_friends():
    slim = slim_checkpoint(_fake_ckpt())
    for k in DROP_KEYS:
        assert k not in slim

def test_slim_preserves_state_dict_bytes():
    ck = _fake_ckpt()
    slim = slim_checkpoint(ck)
    for k, v in ck["state_dict"].items():
        assert torch.equal(slim["state_dict"][k], v)

def test_slim_preserves_config_for_eval_loader():
    slim = slim_checkpoint(_fake_ckpt())
    # eval_comprehensive.py reads ckpt["hyper_parameters"]["config"]
    assert slim["hyper_parameters"]["config"] == {"sentinel": True}

def test_slim_adds_forge_meta_when_provided():
    slim = slim_checkpoint(_fake_ckpt(), forge_meta={"name": "mc_probe_fold0"})
    assert slim["forge_meta"]["name"] == "mc_probe_fold0"


from pathlib import Path
import torch as _torch
from scripts.release.slim import slim_checkpoint as _slim

REPO = Path(__file__).resolve().parents[2]

def test_real_checkpoint_round_trips_and_is_loadable(tmp_path):
    src = REPO / "checkpoints/classification/soft_probe_mc_all128_fold0/last.ckpt"
    if not src.is_file():
        import pytest; pytest.skip("real checkpoint not present")
    full = _torch.load(src, map_location="cpu", weights_only=False)
    slim = _slim(full, forge_meta={"name": "mc_probe_fold0"})
    out = tmp_path / "mc_probe_fold0.ckpt"
    _torch.save(slim, out)
    # Reload and confirm the eval loader's required keys survive
    reloaded = _torch.load(out, map_location="cpu", weights_only=False)
    assert "config" in reloaded["hyper_parameters"]
    assert set(reloaded["state_dict"].keys()) == set(full["state_dict"].keys())
    assert out.stat().st_size < src.stat().st_size  # slimmer

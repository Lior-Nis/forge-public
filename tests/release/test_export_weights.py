"""The weights-only release format: what travels with the tensors, and what must not."""
import torch

from utils.released_weights import (
    runtime_buffer_keys,
    save_released_weights,
    strip_runtime_buffers,
    load_encoder_state_dict,
)

# RevIN (session normaliser) keeps last-batch statistics in `mean`/`stdev`; the
# patient normaliser keeps trained-fold statistics in `mean`/`std`. Only the
# former is runtime state.
REVIN = "preprocessors.preprocessors.3.normalizer"
PATIENT = "preprocessors.preprocessors.2.normalizer"


def _state_dict():
    return {
        "backbone.weight": torch.randn(4, 3),
        f"{PATIENT}.mean": torch.randn(1, 3, 1),
        f"{PATIENT}.std": torch.randn(1, 3, 1),
        f"{REVIN}.mean": torch.randn(269, 3, 1),
        f"{REVIN}.stdev": torch.randn(269, 3, 1),
        f"{REVIN}.affine_weight": torch.randn(3),
    }


def test_runtime_buffers_are_the_revin_pair_only():
    assert runtime_buffer_keys(_state_dict()) == [f"{REVIN}.mean", f"{REVIN}.stdev"]


def test_patient_normalisation_stats_are_kept():
    """They are trained-fold statistics — dropping them would change predictions."""
    kept = strip_runtime_buffers(_state_dict())
    assert f"{PATIENT}.mean" in kept and f"{PATIENT}.std" in kept
    assert f"{REVIN}.affine_weight" in kept  # learned, despite living on RevIN


def test_saved_file_round_trips_bit_identically(tmp_path):
    sd = _state_dict()
    out = tmp_path / "m.safetensors"
    written = save_released_weights(sd, out, {"name": "mc_probe_fold0", "fold": 0})
    reloaded = load_encoder_state_dict(out)
    assert set(reloaded) == set(written)
    for k, v in reloaded.items():
        assert torch.equal(v, sd[k])


def test_metadata_travels_with_the_weights(tmp_path):
    from safetensors import safe_open

    out = tmp_path / "m.safetensors"
    save_released_weights(_state_dict(), out, {
        "name": "mc_probe_fold0", "experiment": "classification/x", "fold": 0})
    with safe_open(str(out), framework="pt") as f:
        meta = f.metadata()
    assert meta["name"] == "mc_probe_fold0"
    assert meta["experiment"] == "classification/x"


def test_released_file_carries_no_pickled_objects(tmp_path):
    """The point of the format: loading runs no arbitrary code and leaks no paths."""
    out = tmp_path / "m.safetensors"
    save_released_weights(_state_dict(), out, {"name": "x"})
    blob = out.read_bytes()
    assert b"pickle" not in blob
    assert b"/Users/" not in blob and b"/home/" not in blob

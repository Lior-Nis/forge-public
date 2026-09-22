from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.eval.eval_comprehensive import released_members, run_inference
from utils.released_weights import load_released_model


def test_released_member_selection_is_nine_external_and_three_defog():
    external = released_members("mc", "probe", "fogathome")
    defog = released_members("mc", "probe", "kaggle")

    assert {(m["seed"], m["fold"]) for m in external} == {
        (seed, fold) for seed in (42, 43, 44) for fold in (0, 1, 2)
    }
    assert {(m["seed"], m["fold"]) for m in defog} == {
        (42, fold) for fold in (0, 1, 2)
    }
    assert all(Path(m["path"]).suffix == ".safetensors" for m in external + defog)


def test_public_safetensors_metadata_rebuilds_model(tmp_path):
    """The real-artifact smoke test is run separately; this pins metadata handling."""
    metadata = {
        "name": "mc_probe_s43_fold1", "kind": "classification",
        "context": "mc", "phase": "probe", "fold": "1", "seed": "43",
        "experiment": "classification/spectral_patch_mae_mc_valid_defog_soft",
        "splits": "kaggle_labeled/kfold_defog_fogcount_valid_mc1",
    }
    path = Path(tmp_path) / "bad.safetensors"
    save_file({"dummy": torch.zeros(1)}, path, metadata=metadata)

    with pytest.raises(RuntimeError, match="incompatible state dict"):
        load_released_model(path)


def test_released_model_loader_rejects_legacy_checkpoints(tmp_path):
    path = Path(tmp_path) / "legacy.ckpt"
    path.write_bytes(b"not a safetensors file")

    with pytest.raises(ValueError, match="must use .safetensors"):
        load_released_model(path)


def test_inference_rejects_legacy_checkpoint_even_when_predictions_are_cached(tmp_path):
    path = Path(tmp_path) / "legacy.ckpt"
    path.write_bytes(b"not a safetensors file")
    (Path(tmp_path) / "fogathome_mc_probe_legacy_preds.csv").write_text(
        "global_idx,pred_prob_fog\n0,0.5\n"
    )

    member = {"fold": 0, "seed": 42, "path": path}
    with pytest.raises(ValueError, match="must use .safetensors"):
        run_inference("mc", "probe", "fogathome", member, Path(tmp_path))

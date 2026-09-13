from scripts.release.manifest import build_manifest
from scripts.release.hf_cards import model_card, dataset_card

def test_model_card_has_license_and_headline():
    txt = model_card(build_manifest())
    assert "license: mit" in txt.lower()
    assert "0.899" in txt           # headline ICC(%TF), FogAtHome-provoking
    assert "Liornis/fog-dataset" in txt
    assert "encoders/mc.safetensors" in txt

def test_model_card_lists_all_30_artifacts():
    txt = model_card(build_manifest())
    assert txt.count(".safetensors") >= 30

def test_dataset_card_documents_dailyliving():
    txt = dataset_card(build_manifest())
    assert "fogathome_dailyliving" in txt
    assert "1.09%" in txt or "0.0109" in txt

def test_dataset_card_gates_access_and_does_not_claim_mit():
    """Participant recordings from third-party studies: gated, and not MIT."""
    txt = dataset_card(build_manifest())
    assert "extra_gated_prompt" in txt and "extra_gated_fields" in txt
    assert "license: mit" not in txt.lower()
    assert "re-identify" in txt

def test_model_card_has_loadable_snippet():
    txt = model_card(build_manifest())
    assert "```python" in txt
    assert "load_released_model" in txt   # the weights-only loading contract
    assert "experiment=" in txt           # the config that rebuilds the model

def test_model_card_separates_the_released_detector_from_the_controlled_comparison():
    """Both sets report the same metrics on the same cohorts at different values;
    the card must show which is which."""
    txt = model_card(build_manifest())
    assert "Controlled comparison" in txt
    assert "0.861" in txt and "0.752" in txt        # matched arms, FogAtHome-provoking
    assert "0.887" in txt                            # released detector, same cohort
    assert "Supervised from scratch" in txt

def test_model_card_does_not_advertise_pickled_checkpoints():
    """The release is weights-only; nothing should tell users to torch.load it."""
    txt = model_card(build_manifest())
    assert "hyper_parameters" not in txt
    assert "weights_only=False" not in txt
    assert ".ckpt" not in txt


from pathlib import Path
import pytest
from scripts.release.upload_weights_hf import plan_upload, stale_files

def _populate_staging(staging: Path, m: dict):
    for e in m["encoders"].values():
        (staging / e["hf_path"]).parent.mkdir(parents=True, exist_ok=True)
        (staging / e["hf_path"]).write_text("x")
    for c in m["classification"]:
        (staging / c["hf_path"]).parent.mkdir(parents=True, exist_ok=True)
        (staging / c["hf_path"]).write_text("x")
    (staging / "checksums.json").write_text("[]")

def test_plan_upload_lists_33_files_when_staging_complete(tmp_path):
    m = build_manifest()
    _populate_staging(tmp_path, m)
    files = plan_upload(tmp_path, m)
    assert len(files) == 33  # 30 weight files + manifest.yaml + README.md + checksums.json

def test_plan_upload_raises_when_a_weight_file_is_missing(tmp_path):
    m = build_manifest()
    _populate_staging(tmp_path, m)
    (tmp_path / m["classification"][0]["hf_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        plan_upload(tmp_path, m)

def test_superseded_files_are_marked_for_deletion():
    """Publishing must remove the old .ckpt set, not leave it beside the weights."""
    m = build_manifest()
    keep = plan_upload_names(m)
    existing = keep + ["classification/mc_probe_fold0.ckpt", "encoders/mc.ckpt"]
    assert stale_files(existing, keep) == [
        "classification/mc_probe_fold0.ckpt", "encoders/mc.ckpt"]

def test_gitattributes_is_never_deleted():
    m = build_manifest()
    keep = plan_upload_names(m)
    assert stale_files(keep + [".gitattributes"], keep) == []

def plan_upload_names(m: dict) -> list[str]:
    return ([e["hf_path"] for e in m["encoders"].values()]
            + [c["hf_path"] for c in m["classification"]]
            + ["manifest.yaml", "README.md", "checksums.json"])

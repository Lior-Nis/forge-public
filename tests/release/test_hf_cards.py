from pathlib import Path

import pytest

from scripts.release.export_checkpoints import export_all
from scripts.release.hf_cards import dataset_card, model_card
from scripts.release.manifest import build_manifest
from scripts.release.upload_weights_hf import plan_upload


def test_model_card_has_license_and_headline():
    txt = model_card(build_manifest())
    assert "license: mit" in txt.lower()
    assert "0.899" in txt  # headline ICC(%TF), FogAtHome-provoking
    assert "Liornis/fog-dataset" in txt
    assert "encoders/mc.safetensors" in txt


def test_model_card_lists_all_36_artifacts():
    txt = model_card(build_manifest())
    assert txt.count(".safetensors") >= 36


def test_dataset_card_documents_dailyliving():
    txt = dataset_card(build_manifest())
    assert "fogathome_dailyliving" in txt
    assert "1.09%" in txt or "0.0109" in txt
    assert "license: other" in txt.lower()
    assert "does not relicense" in txt


def test_model_card_has_loadable_snippet():
    txt = model_card(build_manifest())
    assert "```python" in txt
    assert "load_released_model" in txt


def test_model_card_documents_head_seeds_and_research_only_use():
    txt = model_card(build_manifest())
    assert "| File | Context | Phase | Fold | Seed |" in txt
    assert "not a medical device" in txt


def _populate_staging(staging: Path, m: dict):
    for e in m["encoders"].values():
        (staging / e["hf_path"]).parent.mkdir(parents=True, exist_ok=True)
        (staging / e["hf_path"]).write_text("x")
    for c in m["classification"]:
        (staging / c["hf_path"]).parent.mkdir(parents=True, exist_ok=True)
        (staging / c["hf_path"]).write_text("x")
    (staging / "checksums.json").write_text("[]")


def test_plan_upload_lists_39_files_when_staging_complete(tmp_path):
    m = build_manifest()
    _populate_staging(tmp_path, m)
    files = plan_upload(tmp_path, m)
    assert len(files) == 39  # 36 artifacts + manifest.yaml + README.md + checksums.json


def test_plan_upload_raises_when_a_checkpoint_is_missing(tmp_path):
    m = build_manifest()
    _populate_staging(tmp_path, m)
    (tmp_path / m["classification"][0]["hf_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        plan_upload(tmp_path, m)


def test_export_checksums_cover_all_36_safetensors(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    manifest = build_manifest()
    artifacts = [*manifest["encoders"].values(), *manifest["classification"]]
    for artifact in artifacts:
        path = source / artifact["local"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(artifact["name"].encode())

    records = export_all(staging, repo_root=source)

    assert len(records) == 36
    assert {record["hf_path"] for record in records} == {artifact["hf_path"] for artifact in artifacts}

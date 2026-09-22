import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.eval import eval_comprehensive
from scripts.eval.eval_external_cohorts import restrict_daily_living, score_dataset
from scripts.release.manifest import build_manifest

REPO = Path(__file__).resolve().parents[2]
EXPECTED_DATASETS = ("fogathome", "tdcsfog", "dailyliving", "stanford")


def _reproduce_module():
    spec = importlib.util.spec_from_file_location("_external_reproduce", REPO / "reproduce.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_click_reproduction_targets_only_external_cohorts():
    reproduce = _reproduce_module()

    assert reproduce.DEFAULT_DATASETS == EXPECTED_DATASETS
    assert "kaggle" not in reproduce.DEFAULT_DATASETS


def test_one_click_contract_checks_all_twelve_external_metrics():
    reproduce = _reproduce_module()
    result_prefix = {"tdcsfog": "tdcs"}
    expected = {
        f"{result_prefix.get(cohort, cohort)}_{metric}"
        for cohort in EXPECTED_DATASETS
        for metric in ("auroc", "ap", "icc_tf")
    }

    assert set(reproduce.RESULT_CHECKS) == expected


def test_tdcs_targets_use_one_all_nine_head_assembly():
    results = {row["id"]: row for row in build_manifest()["results"]}

    assert results["tdcs_auroc"]["value"] == 0.917
    assert results["tdcs_ap"]["value"] == 0.866
    assert results["tdcs_icc_tf"]["value"] == 0.876
    assert "all-71 nine-head assembly" in results["tdcs_ap"]["basis"]


def test_external_frame_cache_excludes_invalid_padding():
    assert eval_comprehensive.CACHE_FORMAT == "safetensors_valid"


def test_external_scorer_uses_participant_icc_at_fixed_threshold():
    frame = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "native_label": [0, 0, 0, 1, 1, 1],
            "pred_prob_fog": [0.1, 0.2, 0.1, 0.9, 0.8, 0.9],
        }
    )

    result = score_dataset(frame, "fogathome")

    assert result["ICC_TF"] == 1.0
    assert result["threshold"] == 0.35


def test_daily_living_filter_requires_the_public_activity_sidecar(tmp_path):
    frame = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1"],
            "abs_frame": [0, 1, 2],
            "patient_id": ["p1", "p1", "p1"],
            "native_label": [0, 1, 0],
            "pred_prob_fog": [0.1, 0.9, 0.2],
        }
    )
    activity_path = tmp_path / "activity.parquet"
    pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1"],
            "start_frame": [0, 1, 2],
            "end_frame": [1, 2, 3],
            "Activity": [1, 4, 3],
        }
    ).to_parquet(activity_path, index=False)

    filtered = restrict_daily_living(frame, activity_path)

    assert filtered["abs_frame"].tolist() == [0, 1]


def test_frame_aggregation_streams_sessions_to_parquet(monkeypatch, tmp_path):
    class FakeGroup(dict):
        attrs = {}

    metadata = FakeGroup(
        global_idx=np.array([0, 1]),
        session_idx=np.array([0, 1]),
        patient_id=np.array(["p1", "p1"]),
        session_id=np.array(["s1", "s1"]),
        start_frame=np.array([0, 50]),
    )
    store = {
        "patch_labels": np.array([0.0, 1.0], dtype=np.float32),
        "labels": np.stack(
            [np.zeros(500, dtype=np.int8), np.ones(500, dtype=np.int8)]
        ),
        "valid_masks": np.ones((2, 500), dtype=np.int8),
        "metadata": metadata,
    }
    ensemble = pd.DataFrame(
        {
            "global_idx": [0, 1],
            "pred_prob_fog": [0.2, 0.8],
        }
    )
    cache = tmp_path / "frames.parquet"
    monkeypatch.setattr(eval_comprehensive, "_open_zarr", lambda _: store)

    frame = eval_comprehensive.build_frame_df(
        ensemble,
        "unused.zarr",
        "mc",
        cache,
        eligible_intervals={"s1": np.array([[25, 525]])},
    )

    assert len(frame) == 500
    assert frame.loc[0, "abs_frame"] == 25
    assert frame.loc[0, "pred_prob_fog"] == np.float32(0.2)
    assert frame.loc[25, "pred_prob_fog"] == np.float32(0.5)
    assert frame.loc[499, "pred_prob_fog"] == np.float32(0.8)
    assert str(frame["patient_id"].dtype) == "category"
    assert pq.ParquetFile(cache).num_row_groups == 1


def test_asset_preflight_checks_every_released_model_and_dataset(monkeypatch, tmp_path):
    reproduce = _reproduce_module()
    manifest = build_manifest()
    model_files = {
        artifact["hf_path"]
        for artifact in [*manifest["encoders"].values(), *manifest["classification"]]
    }
    dataset_files = {
        "fogathome/sessions/example.csv",
        "fogathome_dailyliving/sessions/example.csv",
        "fogathome_dailyliving/activity.parquet",
        "kaggle_labeled/tdcsfog/example.csv",
    }
    checksums = tmp_path / "checksums.json"
    checksums.write_text(
        json.dumps(
            [
                {"hf_path": path, "sha256": "a" * 64}
                for path in sorted(model_files)
            ]
        )
    )

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type, revision):
            return sorted(model_files if repo_type == "model" else dataset_files)

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(checksums))
    monkeypatch.setattr(reproduce.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    reproduce.verify_public_assets(manifest, list(EXPECTED_DATASETS))

import subprocess
import sys
from pathlib import Path

import data
import managers
import model
import pipeline
import utils
from utils.paths import PathConfig
from utils.released_weights import REPO_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_core_packages_are_loaded_from_src():
    for package in (data, managers, model, pipeline, utils):
        package_path = Path(package.__file__).resolve()
        assert package_path.is_relative_to(PROJECT_ROOT / "src")


def test_source_layout_helpers_resolve_the_repository_root(tmp_path):
    paths = PathConfig.from_config(
        {
            "data_root": tmp_path / "data",
            "checkpoint_dir": tmp_path / "checkpoints",
            "results_dir": tmp_path / "results",
        }
    )

    assert paths.project_root == PROJECT_ROOT
    assert REPO_ROOT == PROJECT_ROOT


def test_data_processing_cli_finds_repository_configs():
    result = subprocess.run(
        [sys.executable, "-m", "data.process", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "paths: base, fogathome" in result.stdout
    assert "process: fogathome" in result.stdout

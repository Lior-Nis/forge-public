"""Keep the Hydra surface free of dead and duplicate aliases."""

from collections import defaultdict
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_no_disabled_config_files_are_tracked():
    assert not list(CONFIGS.rglob("*.unused"))


def test_no_exact_duplicate_yaml_configs():
    by_content = defaultdict(list)
    for path in CONFIGS.rglob("*.yaml"):
        by_content[path.read_bytes()].append(path.relative_to(CONFIGS))

    duplicates = [paths for paths in by_content.values() if len(paths) > 1]
    assert not duplicates, f"consolidate exact config aliases: {duplicates}"

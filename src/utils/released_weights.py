"""Load released FORGE classification safetensors."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from safetensors import safe_open
from safetensors.torch import load_file

from pipeline.classification import ClassificationPipeline
from utils.config_loaders import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_METADATA = {"name", "kind", "context", "phase", "fold", "seed", "experiment", "splits"}
TRANSIENT_KEYS = {
    "preprocessors.preprocessors.3.normalizer.mean",
    "preprocessors.preprocessors.3.normalizer.stdev",
}


def load_released_model(path: str | Path):
    """Rebuild a released classifier from its embedded config metadata."""
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError(f"{path}: released models must use .safetensors")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    missing = REQUIRED_METADATA - metadata.keys()
    if missing:
        raise ValueError(f"{path}: missing safetensors metadata: {sorted(missing)}")
    if metadata["kind"] != "classification":
        raise ValueError(f"{path}: expected classification weights")

    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        hydra_config = compose(
            config_name="config",
            overrides=[
                f"experiment={metadata['experiment']}",
                f"data/splits={metadata['splits']}",
                f"global.seed={int(metadata['seed'])}",
            ],
        )
    config = load_config(hydra_config)
    model = ClassificationPipeline(config)
    incompatible = model.load_state_dict(load_file(path, device="cpu"), strict=False)
    if set(incompatible.missing_keys) != TRANSIENT_KEYS or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{path}: incompatible state dict; missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model, config

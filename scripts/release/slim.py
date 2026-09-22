"""Slim a Lightning checkpoint to a loadable release artifact."""
from __future__ import annotations

KEEP_KEYS = {
    "state_dict", "hyper_parameters", "hparams_name",
    "epoch", "global_step", "pytorch-lightning_version",
}
DROP_KEYS = {"optimizer_states", "lr_schedulers", "callbacks", "loops"}

def slim_checkpoint(ckpt: dict, forge_meta: dict | None = None) -> dict:
    """Return a copy with only release-relevant keys; state_dict untouched."""
    slim = {k: ckpt[k] for k in KEEP_KEYS if k in ckpt}
    if "state_dict" not in slim:
        raise KeyError("checkpoint has no state_dict; cannot slim")
    if forge_meta is not None:
        slim["forge_meta"] = forge_meta
    return slim

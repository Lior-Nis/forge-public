"""Export the release set as weights-only `.safetensors` files.

Reads the trained Lightning checkpoints and writes, for each of the 30 release
artifacts, a tensor file plus small string metadata (name, context, phase, fold,
seed, and the `experiment` config needed to rebuild the model). Training
configuration, optimiser state and local file paths are not carried over.

    python scripts/release/export_weights.py --source <dir of trained .ckpt> \
        --staging release/staging

`--source` accepts either a directory of checkpoints laid out like the released
tree (`classification/<name>.ckpt`, `encoders/<ctx>.ckpt`) or, with
`--from-local`, each entry's historical training path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.release.manifest import build_manifest, REPO_ROOT
from utils.released_weights import runtime_buffer_keys, save_released_weights

FORMAT_VERSION = "1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_artifacts(m: dict):
    for ctx, enc in m["encoders"].items():
        yield {**enc, "kind": "encoder", "context": ctx}
    for c in m["classification"]:
        yield {**c, "kind": "classification"}


def _source_path(art: dict, source: Path, from_local: bool) -> Path:
    if from_local:
        return REPO_ROOT / art["local"]
    # Released layout, with the pre-conversion extension.
    return source / Path(art["hf_path"]).with_suffix(".ckpt")


def export_all(staging_dir: Path, source: Path, from_local: bool = False) -> list[dict]:
    m = build_manifest()
    records = []
    for art in _iter_artifacts(m):
        src = _source_path(art, source, from_local)
        if not src.is_file():
            raise FileNotFoundError(f"source checkpoint not found: {src}")
        size_before = src.stat().st_size
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        assert src.stat().st_size == size_before, f"source mutated: {src}"  # read-only guard
        state_dict = ckpt.get("state_dict", ckpt)
        dropped = runtime_buffer_keys(state_dict)

        metadata = {
            "format": "forge-weights",
            "format_version": FORMAT_VERSION,
            "name": art["name"],
            "kind": art["kind"],
            "context": art.get("context") or "",
            "phase": art.get("phase") or "",
            "fold": art.get("fold") if art.get("fold") is not None else "",
            "seed": art.get("seed") or "",
            "experiment": art["experiment"],
            "splits": art.get("splits") or "",
            "paper": m["meta"]["paper"],
            "license": m["meta"]["license"],
        }
        dst = staging_dir / art["hf_path"]
        written = save_released_weights(state_dict, dst, metadata)
        records.append({
            "hf_path": art["hf_path"], "name": art["name"],
            "sha256": _sha256(dst),
            "tensors": len(written),
            "params": int(sum(v.numel() for v in written.values())),
            "dropped_runtime_buffers": dropped,
            "source_bytes": size_before, "weights_bytes": dst.stat().st_size,
        })
    (staging_dir / "checksums.json").write_text(json.dumps(records, indent=2))
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--staging", default=str(REPO_ROOT / "release" / "staging"))
    p.add_argument("--source", default=str(REPO_ROOT / "release" / "forge-fog"),
                   help="directory of trained checkpoints in released layout")
    p.add_argument("--from-local", action="store_true",
                   help="read each entry's `local` path instead of --source")
    args = p.parse_args()

    recs = export_all(Path(args.staging), Path(args.source), args.from_local)
    src = sum(r["source_bytes"] for r in recs)
    out = sum(r["weights_bytes"] for r in recs)
    print(f"Exported {len(recs)} weight files -> {args.staging}")
    print(f"  source {src / 1e9:.2f} GB -> weights {out / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

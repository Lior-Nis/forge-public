"""Stage the released safetensors files and write checksums.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from scripts.release.manifest import REPO_ROOT, build_manifest


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

def export_all(staging_dir: Path, repo_root: Path = REPO_ROOT) -> list[dict]:
    m = build_manifest()
    records = []
    for art in _iter_artifacts(m):
        src = repo_root / art["local"]
        if src.suffix != ".safetensors":
            raise ValueError(f"release artifact must be safetensors: {src}")
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = staging_dir / art["hf_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        records.append({
            "hf_path": art["hf_path"], "name": art["name"],
            "sha256": _sha256(dst),
            "bytes": dst.stat().st_size,
        })
    (staging_dir / "checksums.json").write_text(json.dumps(records, indent=2))
    return records

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--staging", default=str(REPO_ROOT / "release" / "staging"))
    args = p.parse_args()
    recs = export_all(Path(args.staging))
    total = sum(r["bytes"] for r in recs)
    print(f"Staged {len(recs)} safetensors files ({total / 1e9:.2f} GB) -> {args.staging}")

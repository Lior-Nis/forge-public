"""Upload slimmed FORGE weights to the public HF model repo Liornis/forge-fog.

Dry-run by default. Pass --execute to create the repo and upload.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from scripts.release.manifest import build_manifest, REPO_ROOT, HF_WEIGHTS_REPO
from scripts.release.hf_cards import model_card

def plan_upload(staging: Path, m: dict) -> list[str]:
    files = []
    for e in m["encoders"].values():
        files.append(e["hf_path"])
    for c in m["classification"]:
        files.append(c["hf_path"])
    files += ["manifest.yaml", "README.md", "checksums.json"]
    missing = [f for f in files
               if f not in ("README.md",) and not (staging / f).exists()
               and f != "manifest.yaml"]
    if missing:
        raise FileNotFoundError(f"staging missing {missing}; run export_checkpoints.py first")
    return files

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--staging", default=str(REPO_ROOT / "release" / "staging"))
    p.add_argument("--execute", action="store_true", help="actually create repo + upload")
    args = p.parse_args()
    staging = Path(args.staging)
    m = build_manifest()

    # Materialize README + manifest into staging for upload
    (staging / "README.md").write_text(model_card(m))
    if (REPO_ROOT / "release" / "manifest.yaml").exists():
        (staging / "manifest.yaml").write_text(
            (REPO_ROOT / "release" / "manifest.yaml").read_text())

    files = plan_upload(staging, m)
    print(f"Repo: {HF_WEIGHTS_REPO}\nWould upload {len(files)} files:")
    for f in files:
        print(f"  {f}")
    if not args.execute:
        print("\nDRY RUN — pass --execute to upload.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(HF_WEIGHTS_REPO, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(staging), repo_id=HF_WEIGHTS_REPO, repo_type="model")
    # Post-upload verification
    listed = set(api.list_repo_files(HF_WEIGHTS_REPO, repo_type="model"))
    missing = [f for f in files if f not in listed]
    if missing:
        raise RuntimeError(f"upload verification FAILED, missing on HF: {missing}")
    print(f"\nUploaded + verified {len(files)} files to {HF_WEIGHTS_REPO}")

if __name__ == "__main__":
    main()

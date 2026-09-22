"""Publish the FogAtHome DailyLiving files and Activity sidecar.

Dry-run by default. Build `activity.parquet` with
`build_dailyliving_activity.py` before executing the upload.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.release.hf_cards import dataset_card
from scripts.release.manifest import HF_DATASET_REPO, build_manifest

LOCAL_DL = Path(os.path.expanduser("~/Datasets/fog-dataset/fogathome_dailyliving"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    p.add_argument("--local", default=str(LOCAL_DL))
    args = p.parse_args()
    local = Path(args.local)
    if not local.is_dir():
        raise FileNotFoundError(f"local dailyliving dir not found: {local}")
    activity = local / "activity.parquet"
    if not activity.is_file():
        raise FileNotFoundError(
            f"missing {activity}; run scripts/release/build_dailyliving_activity.py first"
        )
    print(f"Repo: {HF_DATASET_REPO} (dataset)")
    print(f"Would upload {activity} -> fogathome_dailyliving/activity.parquet")
    if not args.execute:
        print("DRY RUN — pass --execute to upload.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=activity,
        path_in_repo="fogathome_dailyliving/activity.parquet",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )
    api.upload_file(path_or_fileobj=dataset_card(build_manifest()).encode(),
                    path_in_repo="README.md", repo_id=HF_DATASET_REPO, repo_type="dataset")
    listed = set(api.list_repo_files(HF_DATASET_REPO, repo_type="dataset"))
    if "fogathome_dailyliving/activity.parquet" not in listed:
        raise RuntimeError("upload verification FAILED: Activity sidecar missing on HF")
    print("Uploaded Activity sidecar + refreshed dataset card")

if __name__ == "__main__":
    main()

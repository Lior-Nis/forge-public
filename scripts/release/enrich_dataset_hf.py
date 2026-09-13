"""Add the missing fogathome_dailyliving split to Liornis/fog-dataset.

Dry-run by default. The repo already has fogathome/fogstar/kaggle_labeled/
kaggle_unlabeled/new; only fogathome_dailyliving is missing.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
from scripts.release.manifest import build_manifest, HF_DATASET_REPO
from scripts.release.hf_cards import dataset_card

LOCAL_DL = Path(os.path.expanduser("~/Datasets/fog-dataset/fogathome_dailyliving"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    p.add_argument("--local", default=str(LOCAL_DL))
    args = p.parse_args()
    local = Path(args.local)
    if not local.is_dir():
        raise FileNotFoundError(f"local dailyliving dir not found: {local}")
    n_files = sum(1 for _ in local.rglob("*") if _.is_file())
    print(f"Repo: {HF_DATASET_REPO} (dataset)")
    print(f"Would upload {n_files} files from {local} -> fogathome_dailyliving/")
    if not args.execute:
        print("DRY RUN — pass --execute to upload.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    # Resumable, per-file-retrying upload (plain upload_folder times out on this
    # 7.6 GB / 3.4k-file tree). Point at the parent and filter to the subtree so
    # the `fogathome_dailyliving/` prefix is preserved in the repo.
    api.upload_large_folder(repo_id=HF_DATASET_REPO, folder_path=str(local.parent),
                            repo_type="dataset",
                            allow_patterns=[f"{local.name}/**"])
    api.upload_file(path_or_fileobj=dataset_card(build_manifest()).encode(),
                    path_in_repo="README.md", repo_id=HF_DATASET_REPO, repo_type="dataset")
    listed = [f for f in api.list_repo_files(HF_DATASET_REPO, repo_type="dataset")
              if f.startswith("fogathome_dailyliving/")]
    if len(listed) < n_files:
        raise RuntimeError(
            f"upload verification FAILED: {len(listed)}/{n_files} "
            "fogathome_dailyliving files on HF")
    print(f"Uploaded {len(listed)} dailyliving files + refreshed dataset card")

if __name__ == "__main__":
    main()

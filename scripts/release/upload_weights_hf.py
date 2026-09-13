"""Publish the weights-only FORGE release to the HF model repo Liornis/forge-fog.

Dry-run by default. `--execute` uploads the staged files and deletes anything on
the repo that is no longer part of the release (e.g. superseded `.ckpt` files).

`--squash-history` additionally collapses the repo's history to a single commit,
so superseded files are not retrievable from earlier revisions. That is
irreversible; the files are gone from the Hub afterwards.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.release.manifest import build_manifest, REPO_ROOT, HF_WEIGHTS_REPO
from scripts.release.hf_cards import model_card

SIDECARS = ["manifest.yaml", "README.md", "checksums.json"]


def plan_upload(staging: Path, m: dict) -> list[str]:
    files = [e["hf_path"] for e in m["encoders"].values()]
    files += [c["hf_path"] for c in m["classification"]]
    files += SIDECARS
    missing = [f for f in files
               if f not in ("README.md", "manifest.yaml") and not (staging / f).exists()]
    if missing:
        raise FileNotFoundError(f"staging missing {missing}; run export_weights.py first")
    return files


def stale_files(existing: list[str], keep: list[str]) -> list[str]:
    """Files on the repo that the current release does not include."""
    return sorted(set(existing) - set(keep) - {".gitattributes"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--staging", default=str(REPO_ROOT / "release" / "staging"))
    p.add_argument("--execute", action="store_true", help="actually upload")
    p.add_argument("--squash-history", action="store_true",
                   help="collapse repo history to one commit (irreversible)")
    args = p.parse_args()
    staging = Path(args.staging)
    m = build_manifest()

    # Materialize README + manifest into staging for upload
    (staging / "README.md").write_text(model_card(m))
    if (REPO_ROOT / "release" / "manifest.yaml").exists():
        (staging / "manifest.yaml").write_text(
            (REPO_ROOT / "release" / "manifest.yaml").read_text())

    files = plan_upload(staging, m)

    from huggingface_hub import HfApi
    api = HfApi()
    try:
        existing = api.list_repo_files(HF_WEIGHTS_REPO, repo_type="model")
    except Exception:
        existing = []
    stale = stale_files(existing, files)

    print(f"Repo: {HF_WEIGHTS_REPO}\nWould upload {len(files)} files:")
    for f in files:
        print(f"  + {f}")
    print(f"Would delete {len(stale)} superseded files:")
    for f in stale:
        print(f"  - {f}")
    if args.squash_history:
        print("Would then SQUASH history to a single commit (irreversible).")
    if not args.execute:
        print("\nDRY RUN — pass --execute to upload.")
        return

    api.create_repo(HF_WEIGHTS_REPO, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(staging), repo_id=HF_WEIGHTS_REPO, repo_type="model",
                      commit_message="Weights-only release (safetensors)")
    for f in stale:
        api.delete_file(path_in_repo=f, repo_id=HF_WEIGHTS_REPO, repo_type="model",
                        commit_message=f"Remove superseded {f}")

    listed = set(api.list_repo_files(HF_WEIGHTS_REPO, repo_type="model"))
    missing = [f for f in files if f not in listed]
    leftover = [f for f in stale if f in listed]
    if missing or leftover:
        raise RuntimeError(f"verification FAILED: missing={missing} leftover={leftover}")
    print(f"\nUploaded + verified {len(files)} files to {HF_WEIGHTS_REPO}")

    if args.squash_history:
        api.super_squash_history(repo_id=HF_WEIGHTS_REPO, repo_type="model",
                                 commit_message="FORGE weights-only release")
        commits = api.list_repo_commits(HF_WEIGHTS_REPO, repo_type="model")
        print(f"History squashed: {len(commits)} commit(s) remain")


if __name__ == "__main__":
    main()

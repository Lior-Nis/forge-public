"""
Recompute patch_labels as fog_ratio (float32) for all labeled zarrs.

fog_ratio = fraction of timesteps in the window with label > 0 (any FOG class).
Replaces the old patch_labels which were either:
  - Multiclass int (0–3): old-style dominant-class label (longcontext zarrs)
  - Binary int (0/1):     pre-binarised any-fog flag (medcontext/shortcontext zarrs)

After this script:
  - All patch_labels are float32 in [0, 1]
  - binary_any_fog strategy: (patch_labels > 0) gives correct any-fog semantics
  - fog_ratio strategy: patch_labels IS the soft training target
  - Cross-context comparisons are now apples-to-apples
"""

import logging
from pathlib import Path

import numpy as np
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path("data/processed")

ZARRS = [
    # Training
    BASE / "len200_stride20_fogstride10_anyfog_kaggle_defog.zarr",
    BASE / "len500_stride200_fogstride100_anyfog_kaggle_defog.zarr",
    BASE / "len1000_stride200_kaggle.zarr",
    # FogAtHome evaluation
    BASE / "len200_stride20_fogstride10_anyfog_fogathome.zarr",
    BASE / "len500_stride200_fogstride100_anyfog_fogathome.zarr",
    BASE / "len1000_stride200_fogathome.zarr",
    # FogAtHome Daily Living evaluation
    BASE / "len200_stride20_fogstride10_anyfog_fogathome_dailyliving.zarr",
    BASE / "len500_stride200_fogstride100_anyfog_fogathome_dailyliving.zarr",
    BASE / "len1000_stride200_fogathome_dailyliving.zarr",
]

CHUNK = 8192  # patches per processing batch


def recompute_fog_ratio(zarr_path: Path) -> None:
    if not zarr_path.exists():
        logger.warning(f"Not found, skipping: {zarr_path}")
        return

    z = zarr.open_group(str(zarr_path), mode="a")

    if "labels" not in z:
        logger.info(f"No labels array, skipping: {zarr_path.name}")
        return

    labels_arr = z["labels"]
    n_patches, seq_len = labels_arr.shape
    old_max = z["patch_labels"][:].max() if "patch_labels" in z else "N/A"
    logger.info(
        f"{zarr_path.name}: {n_patches} patches × {seq_len} frames  "
        f"(old patch_labels max={old_max})"
    )

    # Compute fog_ratio in chunks to stay memory-efficient
    fog_ratio = np.empty(n_patches, dtype=np.float32)
    for start in range(0, n_patches, CHUNK):
        end = min(start + CHUNK, n_patches)
        chunk = labels_arr[start:end]          # int32, shape [chunk, seq_len]
        fog_ratio[start:end] = (chunk > 0).astype(np.float32).mean(axis=1)
        if start % (CHUNK * 10) == 0 and start > 0:
            logger.info(f"  {start}/{n_patches} processed …")

    # Replace patch_labels
    if "patch_labels" in z:
        del z["patch_labels"]

    z.create_array(
        "patch_labels",
        data=fog_ratio,  # dtype inferred from array (float32)
        chunks=(min(n_patches, CHUNK),),
        overwrite=True,
    )

    pos_rate = (fog_ratio > 0).mean()
    logger.info(
        f"  Done. fog_ratio mean={fog_ratio.mean():.4f}  "
        f"any-fog rate={(pos_rate):.4f}  "
        f"pure-fog rate={(fog_ratio == 1).mean():.4f}"
    )


def main():
    for path in ZARRS:
        recompute_fog_ratio(path)
    logger.info("All done.")


if __name__ == "__main__":
    main()

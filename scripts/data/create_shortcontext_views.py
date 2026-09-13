"""
Create view YAML files for the shortcontext (200-timestep) zarrs.
Run after generate_shortcontext_zarrs.sh completes.
"""

import numpy as np
import zarr
from pathlib import Path
import yaml

DATA_ROOT = Path("data/processed")
VIEWS_DIR = DATA_ROOT / "views"
VIEWS_DIR.mkdir(parents=True, exist_ok=True)


def compute_stats(zarr_path, query):
    z = zarr.open(str(zarr_path))
    meta = z["metadata"]
    purity = meta["purity"][:]
    validity = meta["validity"][:]

    if query == "pure_valid":
        mask = (purity == 1.0) & (validity == 1.0)
    elif query == "valid":
        mask = validity == 1.0
    else:
        raise ValueError(f"Unknown query: {query}")

    class_label = meta["class_label"][:]
    patient_id = meta["patient_id"][:]

    labels, counts = np.unique(class_label[mask], return_counts=True)
    class_dist = {int(l): int(c) for l, c in zip(labels, counts)}
    total = int(mask.sum())
    patients = len(set(patient_id[mask].tolist()))
    prevalence = round(float(counts[1] / counts.sum()), 3) if len(counts) > 1 else 0.0

    return total, patients, class_dist, prevalence


# ── View 1: defog pure+valid ───────────────────────────────────────────────────
zarr_path = DATA_ROOT / "len200_stride20_fogstride10_anyfog_kaggle_defog.zarr"
total, patients, class_dist, prevalence = compute_stats(zarr_path, "pure_valid")

view = {
    "view_metadata": {
        "name": "defog_pure_valid_shortcontext",
        "description": (
            "Kaggle defog patches (200-timestep, stride 20/10 adaptive, any-fog labeling). "
            "Filters to purity==1.0 AND validity==1.0. "
            "For purity==1.0 patches, any-fog and majority-vote labels are identical."
        ),
        "created_at": "2026-05-16T00:00:00Z",
        "version": "1.0",
        "format_version": "1.0",
    },
    "source": {
        "dataset_path": str(zarr_path.resolve()),
        "dataset_type": "physical",
    },
    "filters": {
        "custom_query": "(purity == 1.0) & (validity == 1.0)"
    },
    "computed": {
        "total_patches": total,
        "patient_count": patients,
        "class_distribution": class_dist,
        "prevalence": prevalence,
    },
}

out = VIEWS_DIR / "defog_pure_valid_shortcontext.yaml"
with open(out, "w") as f:
    yaml.dump(view, f, default_flow_style=False, sort_keys=False)
print(f"[view] {out} — {total} patches, {patients} patients, prevalence={prevalence}")


# ── View 2: fogathome dailyliving valid ────────────────────────────────────────
zarr_path = DATA_ROOT / "len200_stride20_fogstride10_anyfog_fogathome_dailyliving.zarr"
total, patients, class_dist, prevalence = compute_stats(zarr_path, "valid")

view = {
    "view_metadata": {
        "name": "fogathome_dailyliving_valid_shortcontext",
        "description": (
            "FogAtHome daily-living patches (200-timestep, stride 20/10 adaptive, any-fog). "
            "Filters to validity==1.0 only."
        ),
        "created_at": "2026-05-16T00:00:00Z",
        "version": "1.0",
        "format_version": "1.0",
    },
    "source": {
        "dataset_path": str(zarr_path.resolve()),
        "dataset_type": "physical",
    },
    "filters": {
        "custom_query": "validity == 1.0"
    },
    "computed": {
        "total_patches": total,
        "patient_count": patients,
        "class_distribution": class_dist,
        "prevalence": prevalence,
    },
}

out = VIEWS_DIR / "fogathome_dailyliving_valid_shortcontext.yaml"
with open(out, "w") as f:
    yaml.dump(view, f, default_flow_style=False, sort_keys=False)
print(f"[view] {out} — {total} patches, {patients} patients, prevalence={prevalence}")

print("Done.")

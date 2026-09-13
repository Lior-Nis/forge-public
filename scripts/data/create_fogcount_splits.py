"""
Generate 3-fold defog splits stratified by absolute FOG patch count (n_fog),
not FOG rate. This prevents large low-rate patients from dominating one fold.

Strategy:
  - Sort the 46 test-eligible FOG+ patients by n_fog
  - Assign to 3 groups using a serpentine pattern [0,1,2,2,1,0,...] so each
    group gets a balanced mix of low/medium/high FOG-count patients
  - Rotate groups across folds: fold k uses group k as test, k+1 as val,
    k+2 as train (plus always-train patients)

Output: configs/data/splits/kaggle_labeled/kfold_defog_fogcount_3fold{0,1,2}.yaml
"""

import zarr
import numpy as np
import pandas as pd
from pathlib import Path
ZARR_PATH = "data/processed/len1000_stride200_kaggle.zarr"
OUT_DIR = Path("configs/data/splits/kaggle_labeled")

# 11 always-train patients (zero/negligible FOG in the pure_valid_longcontext view)
ALWAYS_TRAIN = {
    "0967b2", "109122", "2ea8b1", "324cc0", "473568",
    "94fece", "a15b56", "c057de", "c25c22", "d79889", "eec0c9",
}

def get_patient_fog_stats():
    z = zarr.open(ZARR_PATH)
    patient_ids = z["metadata"]["patient_id"][:]
    protocols = z["metadata"]["protocol"][:]
    labels = z["labels"][:]
    fog_per_patch = (labels.max(axis=1) > 0).astype(int)

    df = pd.DataFrame({"patient_id": patient_ids, "protocol": protocols, "is_fog": fog_per_patch})
    defog = df[df["protocol"] == "defog"]
    stats = defog.groupby("patient_id").agg(
        n_patches=("is_fog", "count"),
        n_fog=("is_fog", "sum"),
    ).assign(fog_rate=lambda x: x.n_fog / x.n_patches)
    return stats


def serpentine_assign(patients_sorted: list, n_groups: int = 3) -> dict:
    """Assign patients to groups using serpentine order for balanced totals."""
    # Pattern: 0,1,2,2,1,0,0,1,2,... ensures each group gets low+high patients
    period = list(range(n_groups)) + list(range(n_groups - 1, -1, -1))  # [0,1,2,2,1,0]
    groups = {i: [] for i in range(n_groups)}
    for i, pid in enumerate(patients_sorted):
        g = period[i % len(period)]
        groups[g].append(pid)
    return groups


def write_yaml(fold: int, test: list, val: list, train: list, stats: pd.DataFrame, out_dir: Path):
    n_test = len(test)
    n_val = len(val)
    n_train = len(train)

    test_fog_patches = stats.loc[stats.index.isin(test), "n_fog"].sum()
    test_total_patches = stats.loc[stats.index.isin(test), "n_patches"].sum()
    test_fog_pct = test_fog_patches / test_total_patches * 100 if test_total_patches > 0 else 0

    header = (
        f"# 3-fold CV (fold {fold}) stratified by fog_count — FOG-positive patients only in test/val\n"
        f"# Stratification: serpentine assignment by n_fog (not fog_rate) to balance total FOG patches per fold\n"
        f"# Always-train patients (negligible FOG in pure_valid view) always in training\n"
        f"# Dataset: data/processed/len1000_stride200_kaggle.zarr\n"
        f"# Split: {n_train} train / {n_val} val / {n_test} test\n"
        f"# Test n_fog: {test_fog_patches} patches ({test_fog_pct:.1f}% of test)\n"
    )

    out_path = out_dir / f"kfold_defog_fogcount_3fold{fold}.yaml"
    with open(out_path, "w") as f:
        f.write(header)
        # Write quoted patient IDs to prevent YAML from misinterpreting hex-like
        # strings (e.g. "814e52") as scientific-notation floats.
        for split_name, pids in [("train", sorted(train)), ("val", sorted(val)), ("test", sorted(test))]:
            f.write(f'"{split_name}":\n')
            for pid in sorted(str(p) for p in pids):
                f.write(f'- "{pid}"\n')
    print(f"Wrote {out_path}  (train={n_train}, val={n_val}, test={n_test}, test_fog={test_fog_patches})")
    return test_fog_patches


def main():
    stats = get_patient_fog_stats()

    # 46 test-eligible patients: all defog patients with n_fog > 0, excluding always-train
    eligible = stats[(stats["n_fog"] > 0) & (~stats.index.isin(ALWAYS_TRAIN))].copy()
    eligible = eligible.sort_values("n_fog")
    print(f"\nTest-eligible patients: {len(eligible)}")
    print(f"n_fog range: {eligible.n_fog.min()}–{eligible.n_fog.max()}")
    print(f"Total FOG patches: {eligible.n_fog.sum()}")

    # Assign to 3 groups using serpentine
    groups = serpentine_assign(eligible.index.tolist(), n_groups=3)

    print("\nGroup assignments (fog_count balance):")
    for g, pids in groups.items():
        group_fog = stats.loc[stats.index.isin(pids), "n_fog"].sum()
        print(f"  Group {g}: {len(pids)} patients, {group_fog} total FOG patches")

    # Write 3 folds with rotation: fold k → test=group[k], val=group[(k+1)%3], train=group[(k+2)%3]+always_train
    always_train_list = sorted(ALWAYS_TRAIN)
    total_fog = []
    for fold in range(3):
        test = groups[fold]
        val = groups[(fold + 1) % 3]
        train = groups[(fold + 2) % 3] + always_train_list
        fog = write_yaml(fold, test, val, train, stats, OUT_DIR)
        total_fog.append(fog)

    print(f"\nTest FOG patches per fold: {total_fog}")
    print(f"Std / Mean = {np.std(total_fog)/np.mean(total_fog):.3f} (lower = more balanced)")


if __name__ == "__main__":
    main()

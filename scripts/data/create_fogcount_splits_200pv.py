"""
Generate 3-fold defog splits stratified by FOG patch count (n_fog),
using the 200-len pure+valid zarr (len200_stride20_fogstride10_anyfog_kaggle_defog.zarr,
filtered to purity==1.0 & validity==1.0).

200-len windows give many more pure FOG patches from short events (<5s) than
500-len or 1000-len windows, making the distribution more representative.

Output: configs/data/splits/kaggle_labeled/kfold_defog_fogcount_200pv_3fold{0,1,2}.yaml
"""

import zarr
import numpy as np
import pandas as pd
from pathlib import Path

ZARR_PATH = "data/processed/len200_stride20_fogstride10_anyfog_kaggle_defog.zarr"
OUT_DIR = Path("configs/data/splits/kaggle_labeled")
N_FOLDS = 3
MIN_FOG_PATCHES = 10  # slightly higher threshold — 200-len patches are ~5× more numerous


def get_patient_fog_stats():
    z = zarr.open(ZARR_PATH)
    meta = z["metadata"]
    patient_ids = meta["patient_id"][:]
    protocols  = meta["protocol"][:]
    purity     = meta["purity"][:]
    validity   = meta["validity"][:]
    class_label = meta["class_label"][:]   # binary any-fog

    mask = (purity == 1.0) & (validity == 1.0)
    df = pd.DataFrame({
        "patient_id": patient_ids[mask],
        "protocol":   protocols[mask],
        "is_fog":     class_label[mask],
    })
    defog = df[df["protocol"] == "defog"]
    stats = defog.groupby("patient_id").agg(
        n_patches=("is_fog", "count"),
        n_fog=("is_fog", "sum"),
    ).assign(fog_rate=lambda x: x.n_fog / x.n_patches)
    return stats


def serpentine_assign(patients_sorted: list, n_groups: int = 3) -> dict:
    period = list(range(n_groups)) + list(range(n_groups - 1, -1, -1))
    groups = {i: [] for i in range(n_groups)}
    for i, pid in enumerate(patients_sorted):
        groups[period[i % len(period)]].append(pid)
    return groups


def write_yaml(fold: int, test: list, val: list, train: list,
               stats: pd.DataFrame, out_dir: Path):
    n_test_fog    = int(stats.loc[stats.index.isin(test), "n_fog"].sum())
    n_test_patches = int(stats.loc[stats.index.isin(test), "n_patches"].sum())
    test_fog_pct  = n_test_fog / n_test_patches * 100 if n_test_patches > 0 else 0

    header = (
        f"# 3-fold CV (fold {fold}) stratified by fog_count — 200-len pure+valid dataset\n"
        f"# Zarr: len200_stride20_fogstride10_anyfog_kaggle_defog.zarr, filter: purity==1.0 & validity==1.0\n"
        f"# Labeling: any-fog binary. For purity==1.0 patches, any-fog == majority-vote.\n"
        f"# Stratification: serpentine assignment by n_fog to balance FOG patches per fold.\n"
        f"# Split: {len(train)} train / {len(val)} val / {len(test)} test\n"
        f"# Test n_fog: {n_test_fog} patches ({test_fog_pct:.1f}% of test)\n"
    )

    out_path = out_dir / f"kfold_defog_fogcount_200pv_3fold{fold}.yaml"
    with open(out_path, "w") as f:
        f.write(header)
        for split_name, pids in [("train", sorted(train)), ("val", sorted(val)), ("test", sorted(test))]:
            f.write(f'"{split_name}":\n')
            for pid in sorted(str(p) for p in pids):
                f.write(f'- "{pid}"\n')
    print(f"Wrote {out_path}  (train={len(train)}, val={len(val)}, test={len(test)}, "
          f"test_fog={n_test_fog}, {test_fog_pct:.1f}%)")
    return n_test_fog


def main():
    stats = get_patient_fog_stats()
    print(f"Total defog patients (pure+valid): {len(stats)}")
    print(f"n_fog range: {stats.n_fog.min()}–{stats.n_fog.max()}")

    always_train = set(stats[stats["n_fog"] < MIN_FOG_PATCHES].index.tolist())
    eligible = stats[stats["n_fog"] >= MIN_FOG_PATCHES].copy().sort_values("n_fog")

    print(f"Always-train (n_fog < {MIN_FOG_PATCHES}): {len(always_train)} patients")
    print(f"Test-eligible: {len(eligible)} patients")
    print(f"n_fog range (eligible): {eligible.n_fog.min()}–{eligible.n_fog.max()}")
    print(f"Total eligible FOG patches: {eligible.n_fog.sum()}")

    groups = serpentine_assign(eligible.index.tolist(), n_groups=N_FOLDS)

    print("\nGroup FOG balance:")
    for g, pids in groups.items():
        group_fog = int(stats.loc[stats.index.isin(pids), "n_fog"].sum())
        print(f"  Group {g}: {len(pids)} patients, {group_fog} FOG patches")

    always_train_list = sorted(always_train)
    total_fog = []
    for fold in range(N_FOLDS):
        test  = groups[fold]
        val   = groups[(fold + 1) % N_FOLDS]
        train = groups[(fold + 2) % N_FOLDS] + always_train_list
        total_fog.append(write_yaml(fold, test, val, train, stats, OUT_DIR))

    print(f"\nTest FOG patches per fold: {total_fog}")
    print(f"Std / Mean = {np.std(total_fog)/np.mean(total_fog):.3f} (lower = more balanced)")


if __name__ == "__main__":
    main()

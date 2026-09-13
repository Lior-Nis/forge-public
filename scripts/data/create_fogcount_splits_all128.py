"""
Regenerate 3-fold defog splits using ALL 128 DeFOG patients from the
fog-biased zarr (len500_stride200_fogstride100_anyfog_kaggle_defog.zarr).

Previous splits (create_fogcount_splits.py) used len1000_stride200_kaggle.zarr
which only captured 57 of 128 available patients.

Strategy (same as before):
  - Patients with n_fog == 0  → always-train (can't evaluate fog detection on them)
  - Patients with n_fog > 0   → test-eligible, serpentine-assign by n_fog across 3 groups
  - Fold k: test=group[k], val=group[(k+1)%3], train=group[(k+2)%3] + always-train
"""
import zarr, numpy as np, pandas as pd, json
from pathlib import Path

ZARR_PATH = "data/processed/len500_stride200_fogstride100_anyfog_kaggle_defog.zarr"
OUT_DIR   = Path("configs/data/splits/kaggle_labeled")


def get_patient_fog_stats():
    z = zarr.open_group(ZARR_PATH, mode="r")
    pats_raw = z["metadata"]["patient_id"][:]
    labels   = z["labels"][:]
    fog_per_patch = (labels.max(axis=1) > 0).astype(int)
    pats = np.array([str(p) for p in pats_raw])
    df = pd.DataFrame({"patient_id": pats, "is_fog": fog_per_patch})
    return df.groupby("patient_id").agg(
        n_patches=("is_fog", "count"),
        n_fog=("is_fog", "sum"),
    ).assign(fog_rate=lambda x: x.n_fog / x.n_patches)


def serpentine_assign(patients_sorted, n_groups=3):
    period = list(range(n_groups)) + list(range(n_groups - 1, -1, -1))
    groups = {i: [] for i in range(n_groups)}
    for i, pid in enumerate(patients_sorted):
        groups[i % len(period) if False else period[i % len(period)]].append(pid)
    return groups


def write_yaml(fold, test, val, train, stats, out_dir, tag="all128"):
    test_fog = int(stats.loc[stats.index.isin(test), "n_fog"].sum())
    test_tot = int(stats.loc[stats.index.isin(test), "n_patches"].sum())
    pct = test_fog / test_tot * 100 if test_tot > 0 else 0
    header = (
        f"# 3-fold CV (fold {fold}) — ALL 128 DeFOG patients\n"
        f"# Source zarr: {ZARR_PATH}\n"
        f"# Stratification: serpentine by n_fog; always-train = zero-fog patients\n"
        f"# Split: {len(train)} train / {len(val)} val / {len(test)} test\n"
        f"# Test n_fog: {test_fog} patches ({pct:.1f}% of test)\n"
    )
    out_path = out_dir / f"kfold_defog_{tag}_3fold{fold}.yaml"
    with open(out_path, "w") as f:
        f.write(header)
        for split_name, pids in [("train", train), ("val", val), ("test", test)]:
            f.write(f'"{ split_name}":\n')
            for pid in sorted(str(p) for p in pids):
                f.write(f'- "{pid}"\n')
    print(f"Wrote {out_path}  (train={len(train)}, val={len(val)}, test={len(test)}, test_fog={test_fog})")
    return test_fog


def main():
    stats = get_patient_fog_stats()
    print(f"Total patients: {len(stats)}")
    print(f"With fog: {(stats.n_fog > 0).sum()}, Without fog: {(stats.n_fog == 0).sum()}")
    print(f"Total fog patches: {stats.n_fog.sum()}")

    always_train = stats[stats.n_fog == 0].index.tolist()
    eligible     = stats[stats.n_fog > 0].sort_values("n_fog")
    print(f"\nAlways-train (zero fog): {len(always_train)}")
    print(f"Test-eligible (fog > 0): {len(eligible)}")
    print(f"n_fog range: {int(eligible.n_fog.min())}–{int(eligible.n_fog.max())}")

    groups = serpentine_assign(eligible.index.tolist(), n_groups=3)
    print("\nGroup fog balance:")
    for g, pids in groups.items():
        fog = int(stats.loc[stats.index.isin(pids), "n_fog"].sum())
        print(f"  Group {g}: {len(pids)} patients, {fog} fog patches")

    fog_counts = []
    for fold in range(3):
        test  = groups[fold]
        val   = groups[(fold + 1) % 3]
        train = groups[(fold + 2) % 3] + always_train
        fog_counts.append(write_yaml(fold, test, val, train, stats, OUT_DIR))

    print(f"\nTest fog patches per fold: {fog_counts}")
    print(f"Balance (std/mean): {np.std(fog_counts)/np.mean(fog_counts):.3f}")


if __name__ == "__main__":
    main()

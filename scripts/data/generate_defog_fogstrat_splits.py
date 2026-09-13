"""
Generate 3-fold fogstrat cross-validation splits for home-domain patients.

Strategy:
- Home-domain patients: anyone with defog OR notype recordings (same deployment domain)
- notype = home session with no FOG events (no sublabels — all negative)
- fog_ratio computed from defog patches only (notype is definitionally fog_ratio=0)
- FOG-positive patients (fog_ratio > 0) go into the 3-fold kfold pool for test/val
- Zero-FOG patients (including all notype-only patients) always in training
- For fold i: test=fold_i, val=fold_(i+1)%3, train=remaining FOG+ + all zero-FOG

Usage:
    python scripts/generate_defog_fogstrat_splits.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.zarr_helpers import read_zarr_metadata

ZARR_PATH = "data/processed/len1000_stride200_kaggle.zarr"
OUTPUT_DIR = Path("configs/data/splits/kaggle_labeled")
N_FOLDS = 3
SEED = 42

# ── Load patient-level metadata ──────────────────────────────────────────────
df = read_zarr_metadata(ZARR_PATH, '/metadata')

# ── Home-domain patients: defog OR notype ────────────────────────────────────
# notype is a home session protocol with no FOG sublabels (all negatives).
# fog_ratio is computed from defog patches only; notype patients get fog_ratio=0.
home_patients = df[df['protocol'].isin(['defog', 'notype'])]['patient_id'].unique()
defog_patches = df[df['protocol'] == 'defog']
defog_fog_ratio = defog_patches.groupby('patient_id', observed=True).agg(
    fog_ratio=('class_label', lambda x: (x > 0).mean()),
).reset_index()

# Patients with only notype (no defog) get fog_ratio=0
notype_only = pd.DataFrame({
    'patient_id': [p for p in home_patients if p not in defog_fog_ratio['patient_id'].values],
    'fog_ratio': 0.0,
})
defog_df = pd.concat([defog_fog_ratio, notype_only], ignore_index=True)
print(f"Home-domain patients: {len(defog_df)} (defog={len(defog_fog_ratio)}, notype-only={len(notype_only)})")

fog_pos = defog_df[defog_df['fog_ratio'] > 0].copy()
fog_zero = defog_df[defog_df['fog_ratio'] == 0].copy()
print(f"  FOG+ patients: {len(fog_pos)}")
print(f"  Zero-FOG patients: {len(fog_zero)}")

# ── Stratify FOG+ patients by fog_rate bins ───────────────────────────────────
bins = [-0.001, 0.02, 0.05, 0.15, 0.30, 1.0]
labels = ['0-2%', '2-5%', '5-15%', '15-30%', '>30%']
fog_pos['fog_bin'] = pd.cut(fog_pos['fog_ratio'], bins=bins, labels=labels)
print(f"\nFOG rate distribution:\n{fog_pos['fog_bin'].value_counts().sort_index()}")

# ── 3-fold stratified kfold on FOG+ ─────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
patient_ids = fog_pos['patient_id'].values
strat_labels = fog_pos['fog_bin'].astype(str).values

fold_indices = list(skf.split(patient_ids, strat_labels))

zero_fog_ids = sorted(fog_zero['patient_id'].tolist())

# ── Generate and write each fold ─────────────────────────────────────────────
for i in range(N_FOLDS):
    _, test_idx = fold_indices[i]
    _, val_idx = fold_indices[(i + 1) % N_FOLDS]

    test_ids = sorted(patient_ids[test_idx].tolist())
    val_ids = sorted(patient_ids[val_idx].tolist())

    test_set = set(test_ids)
    val_set = set(val_ids)
    train_fog_pos = sorted([p for p in patient_ids if p not in test_set and p not in val_set])
    train_ids = sorted(train_fog_pos + zero_fog_ids)

    # Compute test FOG%
    test_fog_pcts = fog_pos[fog_pos['patient_id'].isin(test_ids)]['fog_ratio']
    mean_test_fog = test_fog_pcts.mean() * 100

    split = {'train': train_ids, 'val': val_ids, 'test': test_ids}

    header = (
        f"# 3-fold CV (fold {i}) stratified by fog_rate — FOG-positive patients only in test/val\n"
        f"# Zero-FOG patients (incl. notype-only) always in training\n"
        f"# Dataset: {ZARR_PATH}\n"
        f"# Home-domain patients: {len(defog_df)} ({len(fog_pos)} FOG+, {len(fog_zero)} zero-FOG)\n"
        f"# Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test\n"
        f"# Test FOG%: {mean_test_fog:.1f}%"
    )

    output_path = OUTPUT_DIR / f"kfold_defog_fogstrat_3fold{i}.yaml"
    with open(output_path, 'w') as f:
        f.write(header + "\n\n")
        yaml.dump(split, f, default_flow_style=False, sort_keys=False, default_style='"')

    print(f"\nFold {i}: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test | test FOG%: {mean_test_fog:.1f}%")
    print(f"  Test patients:  {test_ids}")
    print(f"  Val patients:   {val_ids}")
    print(f"  Written: {output_path}")

print("\nDone.")

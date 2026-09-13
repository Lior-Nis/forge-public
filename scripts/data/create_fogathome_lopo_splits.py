"""Generate 12 leave-one-patient-out CV splits for FogAtHome (Fig 12 LOPO).

Each fold: test = 1 held-out patient, val = the next patient (rotating),
train = the remaining 10. Across folds every patient is test exactly once.
With the max_train_minutes budget, train data ranges up to ~35 min (10 patients)
— far more than the 3-fold version (~20 min), giving a fairer fine-tuning test.
"""
from pathlib import Path

PATIENTS = ["a00001", "a00002", "a00004", "a00006", "a00007", "a00008",
            "a00009", "a00010", "a00011", "a00012", "a00013", "a00014"]
OUT = Path("configs/data/splits")


def main():
    for i, test_pat in enumerate(PATIENTS):
        val_pat = PATIENTS[(i + 1) % len(PATIENTS)]
        train = [p for p in PATIENTS if p not in (test_pat, val_pat)]
        lines = [f"# FogAtHome LOPO fold {i}: test={test_pat}, val={val_pat}, train=10 patients.",
                 "train:"]
        lines += [f'  - "{p}"' for p in train]
        lines += ["val:", f'  - "{val_pat}"', "test:", f'  - "{test_pat}"', ""]
        (OUT / f"fogathome_lopo_fold{i}.yaml").write_text("\n".join(lines))
    print(f"Wrote {len(PATIENTS)} LOPO split files to {OUT}/fogathome_lopo_fold*.yaml")


if __name__ == "__main__":
    main()

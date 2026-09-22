"""Build the FORGE evaluation zarr from the public Stanford O'Day cohort."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.signal import resample_poly

G = 9.80665
FS_IN, FS_OUT = 128, 100
BLOCK, STRIDE = 500, 200
ACC_COLUMNS = ["imu_lumbar_ax", "imu_lumbar_ay", "imu_lumbar_az"]


def canonical_order(acceleration: np.ndarray) -> list[int]:
    """Place the gravity axis first and retain native order for the other axes."""
    vertical = int(np.argmax(np.abs(acceleration.mean(axis=0))))
    return [vertical, *[axis for axis in range(3) if axis != vertical]]


def build_zarr(raw_dir: Path, output: Path) -> None:
    files = sorted(raw_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No Stanford trials found under {raw_dir}")

    accelerations, labels, metadata, sessions = [], [], [], []
    for session_index, path in enumerate(files):
        frame = pd.read_excel(path, usecols=["subject_ID", *ACC_COLUMNS, "freeze_label"])
        raw = frame[ACC_COLUMNS].to_numpy(dtype=np.float64) / G
        order = canonical_order(raw)
        acceleration = resample_poly(raw[:, order], FS_OUT, FS_IN, axis=0).astype(np.float32)
        source_index = np.clip(
            np.round(np.arange(len(acceleration)) * FS_IN / FS_OUT).astype(int),
            0,
            len(frame) - 1,
        )
        native_labels = frame["freeze_label"].to_numpy(dtype=np.int32)[source_index]
        subject = int(frame["subject_ID"].iloc[0])

        for start in range(0, len(acceleration) - BLOCK + 1, STRIDE):
            accelerations.append(acceleration[start:start + BLOCK])
            labels.append(native_labels[start:start + BLOCK])
            metadata.append((subject, session_index, start))
        sessions.append(
            {
                "filename": path.name,
                "protocol": "stanford",
                "id": path.stem,
                "subject": subject,
                "axis_order": order,
            }
        )

    acc_array = np.stack(accelerations).astype(np.float32)
    label_array = np.stack(labels).astype(np.int32)
    subjects, session_indices, starts = map(np.asarray, zip(*metadata))
    fog_ratio = (label_array > 0).mean(axis=1).astype(np.float32)

    patient_names = [f"s{number:05d}" for number in sorted(set(subjects.tolist()))]
    patient_encoding = {name: index for index, name in enumerate(patient_names)}
    patient_codes = np.asarray(
        [patient_encoding[f"s{subject:05d}"] for subject in subjects], dtype=np.int16
    )
    session_encoding = {session["id"]: index for index, session in enumerate(sessions)}

    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(output, mode="w")
    root.create_array(
        "accs", shape=acc_array.shape, dtype="float32", chunks=(256, BLOCK, 3)
    )[:] = acc_array
    root.create_array(
        "labels", shape=label_array.shape, dtype="int32", chunks=(256, BLOCK)
    )[:] = label_array
    root.create_array(
        "valid_masks", shape=label_array.shape, dtype="bool", chunks=(256, BLOCK)
    )[:] = True
    root.create_array("patch_labels", shape=fog_ratio.shape, dtype="float32")[:] = fog_ratio

    count = len(acc_array)
    metadata_group = root.create_group("metadata")
    columns = {
        "patient_id": patient_codes,
        "session_id": session_indices.astype(np.int16),
        "session_idx": session_indices.astype(np.int64),
        "protocol": np.zeros(count, dtype=np.int16),
        "class_label": (fog_ratio >= 0.25).astype(np.int64),
        "purity": np.ones(count, dtype=np.float64),
        "validity": np.ones(count, dtype=np.float64),
        "start_frame": starts.astype(np.int32),
        "global_idx": np.arange(count, dtype=np.int64),
    }
    for name, values in columns.items():
        metadata_group.create_array(name, shape=values.shape, dtype=values.dtype)[:] = values
    metadata_group.attrs["columns"] = list(columns)
    metadata_group.attrs["encodings"] = {
        "patient_id": patient_encoding,
        "session_id": session_encoding,
        "protocol": {"stanford": 0},
    }

    root.attrs.update(
        {
            "block_len": BLOCK,
            "stride_len": STRIDE,
            "fog_stride_len": STRIDE,
            "is_unlabeled": False,
            "fog_ratio_labeling": True,
            "total_patches": count,
            "session_infos": json.dumps(sessions),
        }
    )
    print(
        f"wrote {output}: {count:,} windows, {len(files)} walks, "
        f"{len(patient_names)} participants"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/stanford_imu_fog/data/raw/imus6_subjects7"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/len500_stride200_stanford.zarr"),
    )
    args = parser.parse_args()
    build_zarr(args.raw_dir, args.output)


if __name__ == "__main__":
    main()

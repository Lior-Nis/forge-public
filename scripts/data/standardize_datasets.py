"""Standardize external FOG datasets to the fog-dataset format.

Converts two external datasets into the format expected by data/process/:
  - {protocol}/metadata.csv with columns: Id, Subject
  - {protocol}/sessions/{session_id}.csv with columns:
    Time, AccV, AccML, AccAP, StartHesitation, Turn, Walking, Valid

Usage:
    uv run python scripts/data/standardize_datasets.py --datasets-root /path/to/Datasets
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLS = ["Time", "AccV", "AccML", "AccAP", "StartHesitation", "Turn", "Walking", "Valid"]


def standardize_17838806(datasets_root: Path, fog_dataset_root: Path):
    """Convert the 17838806 (fogstar) dataset to fog-dataset format.

    Source: 31 session CSVs at 60 Hz with multi-sensor IMU data.
    Extracts back sensor accelerometer, resamples to 100 Hz, maps fog labels.
    """
    source_dir = datasets_root / "17838806" / "sessions"
    output_dir = fog_dataset_root / "fogstar" / "fogstar"
    sessions_dir = output_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(source_dir.glob("subject_*_session_*.csv"))
    logger.info(f"Found {len(source_files)} fogstar source files")

    metadata_rows = []

    for src_path in source_files:
        df = pd.read_csv(src_path)
        n_orig = len(df)

        # Extract back sensor accelerometer columns
        acc = df[["back_acc_x", "back_acc_y", "back_acc_z"]].values

        # Resample accelerometer from 60 Hz to 100 Hz
        n_target = int(round(n_orig * 100 / 60))
        acc_resampled = resample(acc, n_target, axis=0)

        # Resample binary fog label with nearest-neighbor (repeat)
        fog_orig = df["fog"].values
        indices_nn = np.round(np.linspace(0, n_orig - 1, n_target)).astype(int)
        fog_resampled = fog_orig[indices_nn]

        # Map fog label: fog=1 → all three labels=1
        labels = fog_resampled.astype(int)

        # Build session ID and subject ID
        subject_id = int(df["subjectID"].iloc[0])
        session_id_raw = int(df["sessionID"].iloc[0])
        subject_hex = f"b{subject_id:05d}"
        session_hex = f"b{subject_id:04d}{session_id_raw:05d}"

        # Build output DataFrame
        out_df = pd.DataFrame({
            "Time": np.arange(n_target),
            "AccV": acc_resampled[:, 0],
            "AccML": acc_resampled[:, 1],
            "AccAP": acc_resampled[:, 2],
            "StartHesitation": labels,
            "Turn": labels,
            "Walking": labels,
            "Valid": True,
        })
        out_df.to_csv(sessions_dir / f"{session_hex}.csv", index=False)
        metadata_rows.append({"Id": session_hex, "Subject": subject_hex})
        logger.info(f"  {src_path.name} → {session_hex}.csv ({n_orig}→{n_target} samples)")

    # Write metadata
    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(output_dir / "metadata.csv", index=False)
    logger.info(f"Wrote {len(metadata_rows)} sessions to {output_dir}")


def standardize_fogathome(datasets_root: Path, fog_dataset_root: Path):
    """Convert the fog@home dataset to fog-dataset format.

    Source: 97 acc CSVs (already at 100 Hz with correct column names) + labels.csv.
    Joins labels to accelerometer data by file hash and timestep index.
    """
    source_base = datasets_root / "fogathome_dataset_forlior" / "fogathome_dataset"
    acc_dir = source_base / "acc" / "fog@home_provoking250325"
    labels_path = source_base / "labels.csv"
    output_dir = fog_dataset_root / "fogathome" / "fogathome"
    sessions_dir = output_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Load labels and parse file hash + timestep index from Id column
    labels_df = pd.read_csv(labels_path)
    labels_df[["file_hash", "timestep_idx"]] = labels_df["Id"].str.rsplit("_", n=1, expand=True)
    labels_df["timestep_idx"] = labels_df["timestep_idx"].astype(int)

    # Build subject ID mapping: "Subject N" → "a{N:05d}"
    unique_subjects = sorted(labels_df["og_subject"].unique(), key=lambda s: int(s.split()[-1]))
    subject_map = {s: f"a{int(s.split()[-1]):05d}" for s in unique_subjects}
    logger.info(f"Found {len(subject_map)} unique subjects in labels")

    acc_files = sorted(acc_dir.glob("*.csv"))
    logger.info(f"Found {len(acc_files)} fogathome acc files")

    metadata_rows = []

    for acc_path in acc_files:
        file_hash = acc_path.stem
        acc_df = pd.read_csv(acc_path)

        # Get labels for this file
        file_labels = labels_df[labels_df["file_hash"] == file_hash].sort_values("timestep_idx")

        if file_labels.empty:
            logger.warning(f"  No labels for {file_hash}, skipping")
            continue

        # Verify alignment: label timestep indices should match acc row count
        max_label_idx = file_labels["timestep_idx"].max()
        if max_label_idx >= len(acc_df):
            logger.warning(
                f"  {file_hash}: label indices exceed acc length "
                f"({max_label_idx} >= {len(acc_df)}), truncating labels"
            )
            file_labels = file_labels[file_labels["timestep_idx"] < len(acc_df)]

        # Build label arrays aligned to acc rows (default 0 for unlabeled rows)
        n_samples = len(acc_df)
        start_hes = np.zeros(n_samples, dtype=int)
        turn = np.zeros(n_samples, dtype=int)
        walking = np.zeros(n_samples, dtype=int)

        idx = file_labels["timestep_idx"].values
        start_hes[idx] = file_labels["StartHesitation"].values.astype(int)
        turn[idx] = file_labels["Turn"].values.astype(int)
        walking[idx] = file_labels["Walking"].values.astype(int)

        # Determine subject from labels
        og_subject = file_labels["og_subject"].iloc[0]
        subject_hex = subject_map[og_subject]

        # Build output DataFrame
        out_df = pd.DataFrame({
            "Time": np.arange(n_samples),
            "AccV": acc_df["AccV"].values,
            "AccML": acc_df["AccML"].values,
            "AccAP": acc_df["AccAP"].values,
            "StartHesitation": start_hes,
            "Turn": turn,
            "Walking": walking,
            "Valid": True,
        })
        out_df.to_csv(sessions_dir / f"{file_hash}.csv", index=False)
        metadata_rows.append({"Id": file_hash, "Subject": subject_hex})

    # Write metadata
    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(output_dir / "metadata.csv", index=False)
    logger.info(f"Wrote {len(metadata_rows)} sessions to {output_dir}")


def standardize_fogathome_dailyliving(datasets_root: Path, fog_dataset_root: Path):
    """Convert the fogathome_dailyliving dataset to fog-dataset format.

    Source: 3428 parquet files at 100 Hz with columns:
      Time (datetime), AccV, AccML, AccAP, StartHesitation, Turn, Walking,
      Activity, Day, og_subject.
    Each file is one session (one subject, one recording segment).

    Outputs: metadata.csv (Id, Subject) + sessions/{id}.csv with standard columns.
    Subject IDs use the 'c' namespace (fogathome uses 'a', fogstar uses 'b').
    """
    source_dir = datasets_root / "fogathome_dailyliving" / "preprocesseddata"
    output_dir = fog_dataset_root / "fogathome_dailyliving" / "fogathome_dailyliving"
    sessions_dir = output_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(source_dir.glob("*.parquet"))
    logger.info(f"Found {len(source_files)} fogathome_dailyliving source files")

    # Build subject ID mapping: "Subject N" → "c{N:05d}" from first pass
    # Collect unique subjects by reading only the og_subject column
    subjects_seen = set()
    for f in source_files:
        subjects_seen.update(
            pd.read_parquet(f, columns=["og_subject"])["og_subject"].unique()
        )
    subject_map = {s: f"c{int(s.split()[-1]):05d}" for s in subjects_seen}
    logger.info(f"Subjects: {sorted(subject_map.items())}")

    metadata_rows = []

    for i, src_path in enumerate(source_files):
        df = pd.read_parquet(src_path)
        session_id = src_path.stem
        subject_hex = subject_map[df["og_subject"].iloc[0]]

        out_df = pd.DataFrame({
            "Time": np.arange(len(df)),
            "AccV": df["AccV"].values,
            "AccML": df["AccML"].values,
            "AccAP": df["AccAP"].values,
            "StartHesitation": df["StartHesitation"].values.astype(int),
            "Turn": df["Turn"].values.astype(int),
            "Walking": df["Walking"].values.astype(int),
            "Valid": True,
        })
        out_df.to_csv(sessions_dir / f"{session_id}.csv", index=False)
        metadata_rows.append({"Id": session_id, "Subject": subject_hex})
        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i + 1}/{len(source_files)} files...")

    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(output_dir / "metadata.csv", index=False)
    logger.info(f"Wrote {len(metadata_rows)} sessions to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        required=True,
        help="directory containing the source datasets",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="fog-dataset output root (default: <datasets-root>/fog-dataset)",
    )
    args = parser.parse_args()
    output_root = args.output_root or args.datasets_root / "fog-dataset"

    logger.info("=== Standardizing 17838806 (fogstar) ===")
    standardize_17838806(args.datasets_root, output_root)

    logger.info("\n=== Standardizing fog@home ===")
    standardize_fogathome(args.datasets_root, output_root)

    logger.info("\n=== Standardizing fogathome_dailyliving ===")
    standardize_fogathome_dailyliving(args.datasets_root, output_root)

    logger.info("\nDone! Run processing with:")
    logger.info("  python -m data.process paths=fogstar")
    logger.info("  python -m data.process paths=fogathome")
    logger.info("  python -m data.process paths=fogathome_dailyliving")

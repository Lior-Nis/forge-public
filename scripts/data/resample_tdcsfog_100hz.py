"""Resample the public tDCS-FOG cohort from 128 Hz to FORGE's 100 Hz rate."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

FS_IN, FS_OUT = 128, 100
UP, DOWN = 25, 32
ACC_COLUMNS = ["AccV", "AccML", "AccAP"]
LABEL_COLUMNS = ["StartHesitation", "Turn", "Walking"]


def resample_dataset(source: Path, output: Path) -> None:
    sessions = sorted((source / "sessions").glob("*.csv"))
    if not sessions:
        raise FileNotFoundError(f"No tDCS-FOG sessions found under {source / 'sessions'}")

    output_sessions = output / "sessions"
    output_sessions.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "metadata.csv", output / "metadata.csv")

    for index, path in enumerate(sessions, start=1):
        frame = pd.read_csv(path)
        acceleration = resample_poly(
            frame[ACC_COLUMNS].to_numpy(dtype=np.float64),
            UP,
            DOWN,
            axis=0,
            padtype="line",
        )
        source_index = np.clip(
            np.round(np.arange(len(acceleration)) * FS_IN / FS_OUT).astype(int),
            0,
            len(frame) - 1,
        )
        result = pd.DataFrame(acceleration, columns=ACC_COLUMNS)
        for column in LABEL_COLUMNS:
            result[column] = frame[column].to_numpy()[source_index]
        result.insert(0, "Time", np.arange(len(result)))
        result.to_csv(output_sessions / path.name, index=False)
        if index % 200 == 0:
            print(f"resampled {index}/{len(sessions)} sessions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/kaggle_labeled/tdcsfog"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/kaggle_tdcs100/tdcsfog"),
    )
    args = parser.parse_args()
    resample_dataset(args.source, args.output)


if __name__ == "__main__":
    main()

"""Build the compact DailyLiving Activity sidecar used by public reproduction."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        ("session_id", pa.string()),
        ("start_frame", pa.int32()),
        ("end_frame", pa.int32()),
        ("Activity", pa.uint8()),
    ]
)


def build_sidecar(source: Path, output: Path) -> None:
    files = sorted(source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No DailyLiving parquet files found under {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output, SCHEMA, compression="zstd")
    try:
        for index, path in enumerate(files, start=1):
            activity = pd.read_parquet(path, columns=["Activity"])["Activity"].to_numpy()
            starts = np.r_[0, np.flatnonzero(activity[1:] != activity[:-1]) + 1]
            ends = np.r_[starts[1:], len(activity)]
            table = pa.table(
                {
                    "session_id": pa.array([path.stem] * len(starts), type=pa.string()),
                    "start_frame": pa.array(starts, type=pa.int32()),
                    "end_frame": pa.array(ends, type=pa.int32()),
                    "Activity": pa.array(activity[starts], type=pa.uint8()),
                },
                schema=SCHEMA,
            )
            writer.write_table(table)
            if index % 500 == 0:
                print(f"processed {index}/{len(files)} sessions", flush=True)
    finally:
        writer.close()
    print(f"wrote {output} from {len(files)} sessions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_sidecar(args.source, args.output)


if __name__ == "__main__":
    main()

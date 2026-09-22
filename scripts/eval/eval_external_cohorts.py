"""Score the released detector on the four public external cohorts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_DATASETS = ("fogathome", "tdcsfog", "dailyliving", "stanford")
THRESHOLD = 0.35
DAILY_LIVING_ELIGIBLE_ACTIVITY = {1, 4}


def participant_icc(frame: pd.DataFrame, threshold: float) -> float:
    patients = frame["patient_id"]
    reference = frame["native_label"].groupby(patients, observed=True).mean() * 100
    predicted = (frame["pred_prob_fog"] >= threshold).astype(np.int8)
    prediction = predicted.groupby(patients, observed=True).mean() * 100
    endpoints = pd.DataFrame({"reference": reference, "prediction": prediction})
    if len(endpoints) < 3:
        return float("nan")
    long = pd.concat(
        [
            endpoints[["reference"]].rename(columns={"reference": "rating"}).assign(
                participant=endpoints.index, rater="reference"
            ),
            endpoints[["prediction"]].rename(columns={"prediction": "rating"}).assign(
                participant=endpoints.index, rater="model"
            ),
        ],
        ignore_index=True,
    )
    result = pg.intraclass_corr(
        data=long, targets="participant", raters="rater", ratings="rating"
    ).set_index("Type")
    return float(result.loc["ICC2", "ICC"])


def load_frames(cache_dir: Path, dataset: str) -> pd.DataFrame:
    path = cache_dir / f"{dataset}_mc_probe_safetensors_valid_frames.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing frame predictions for {dataset}: {path}")
    frame = pd.read_parquet(path)
    for column in ("patient_id", "session_id"):
        frame[column] = frame[column].astype("category")
    return frame


def restrict_daily_living(frame: pd.DataFrame, activity_path: Path) -> pd.DataFrame:
    if not activity_path.exists():
        raise FileNotFoundError(
            "Daily-living Activity annotations are required for the walking/standing "
            f"eligible domain: {activity_path}"
        )
    activity = pd.read_parquet(
        activity_path, columns=["session_id", "start_frame", "end_frame", "Activity"]
    )
    activity["session_id"] = activity["session_id"].astype(str)
    runs_by_session = {
        session_id: runs[runs["Activity"].isin(DAILY_LIVING_ELIGIBLE_ACTIVITY)]
        for session_id, runs in activity.groupby("session_id", sort=False)
    }
    selected = np.zeros(len(frame), dtype=bool)
    absolute_frames = frame["abs_frame"].to_numpy()
    for session_id, indices in frame.groupby("session_id", observed=True, sort=False).indices.items():
        runs = runs_by_session.get(str(session_id))
        if runs is None or runs.empty:
            continue
        indices = np.asarray(indices)
        absolute_frame = absolute_frames[indices]
        eligible = np.zeros(len(indices), dtype=bool)
        for run in runs.itertuples(index=False):
            eligible |= (absolute_frame >= run.start_frame) & (absolute_frame < run.end_frame)
        if eligible.any():
            selected[indices[eligible]] = True
    if not selected.any():
        raise RuntimeError("Daily-living Activity sidecar selected no walking/standing frames")
    return frame.loc[selected].reset_index(drop=True)


def score_dataset(frame: pd.DataFrame, dataset: str) -> dict:
    labels = frame["native_label"].to_numpy(dtype=np.int8)
    scores = frame["pred_prob_fog"].to_numpy(dtype=np.float64)
    return {
        "dataset": dataset,
        "AUROC": float(roc_auc_score(labels, scores)),
        "AP": float(average_precision_score(labels, scores)),
        "ICC_TF": participant_icc(frame, THRESHOLD),
        "threshold": THRESHOLD,
        "n_frames": len(frame),
        "n_participants": frame["patient_id"].nunique(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("logs/comprehensive_eval_cache"))
    parser.add_argument("--output", type=Path, default=Path("logs/RESULTS_external.csv"))
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument(
        "--dailyliving-activity",
        type=Path,
        default=Path("data/raw/fogathome_dailyliving/activity.parquet"),
    )
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        frame = load_frames(args.cache_dir, dataset)
        if dataset == "dailyliving":
            frame = restrict_daily_living(frame, args.dailyliving_activity)
        rows.append(score_dataset(frame, dataset))

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved {len(result)} cohorts -> {args.output}")


if __name__ == "__main__":
    main()

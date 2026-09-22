#!/usr/bin/env python
"""Reproduce FORGE evaluation results from released artifacts (cross-OS).

One self-contained, idempotent driver that:
  1. downloads the released weights  (HF: Liornis/forge-fog)
  2. downloads the evaluation datasets (HF: Liornis/fog-dataset)
  3. builds the evaluation zarrs       (data.process)
  4. runs the evaluation               (eval_comprehensive.py)
  5. scores the four external cohorts  (eval_external_cohorts.py)

It is pure Python on purpose: no `source activate`, no symlinks, no shell
globs -- so it runs identically on Linux, macOS, and Windows. The single
source of truth for paths/checkpoints is `release/manifest.yaml`; this script
reads it and mirrors the commands documented in the `onboard-repo` and
`reproduce-evaluations` skills.

Quick start from a bare clone (the wrappers run `uv sync` first):

    ./reproduce.sh                        # Linux/macOS: all four external cohorts
    reproduce.bat                         # Windows (cmd/PowerShell); WSL2 + ./reproduce.sh is more reliable

If the environment is already provisioned, run the module directly:

    uv run python reproduce.py [flags]    # same flags as the wrappers

Useful flags:
    --datasets fogathome stanford  # optionally restrict external cohorts
    --verify-assets-only           # check public release packaging, then exit
    --skip-download               # weights/data already present
    --skip-zarr                   # zarrs already built
    --no-score                    # run inference without the metric/check step
    --batch-scale 0.25            # smaller batches for low-VRAM GPUs

Evaluation is inference-only. A GPU is recommended but not required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MANIFEST = REPO_ROOT / "release" / "manifest.yaml"

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
LOGS_DIR = REPO_ROOT / "logs"
CACHE_DIR = LOGS_DIR / "comprehensive_eval_cache"
DEFAULT_DATASETS = ("fogathome", "tdcsfog", "dailyliving", "stanford")
STANFORD_REPOSITORY = "stanfordnmbl/imu-fog-detection"
STANFORD_REVISION = "e95687842801ca3565463486a50d562fff44d182"

# Windows: the console code page is often a legacy locale (e.g. Hebrew cp1255)
# that cannot encode the arrows/symbols (→, ×, …) our eval/build scripts print,
# which crashes a subprocess with UnicodeEncodeError -- discarding an otherwise
# successful eval. Force UTF-8 mode for THIS process and every child we spawn
# (build/eval/icc all inherit os.environ). No-op on POSIX, which is already UTF-8.
if os.name == "nt":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

# eval dataset key -> HF subdir under the dataset repo (also the data/raw subdir)
DATASET_SUBDIR = {
    "fogathome": "fogathome",
    "dailyliving": "fogathome_dailyliving",
    "tdcsfog": "kaggle_labeled/tdcsfog",
}

BUILD_RECIPE = {
    "process": "kaggle_medcontext",
    "paths": {
        "fogathome": "fogathome_medcontext",
        "dailyliving": "fogathome_dailyliving_medcontext",
        "tdcsfog": "kaggle_tdcs100",
    },
}

# Target eval zarr filenames, mirrored from scripts/eval/eval_comprehensive.py
# (ZARR_NAME). Used to verify a build produced what the eval expects.
ZARR_NAME = {
    ("mc", "fogathome"): "len500_stride200_fogstride100_fogathome.zarr",
    ("mc", "dailyliving"): "len500_stride200_fogstride100_anyfog_fogathome_dailyliving.zarr",
    ("mc", "tdcsfog"): "len500_stride200_fogstride100_anyfog_tdcs100.zarr",
    ("mc", "stanford"): "len500_stride200_stanford.zarr",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
# ANSI colour only on a real TTY and not on legacy Windows consoles (which would
# otherwise print raw escape codes).
_USE_COLOR = (os.name != "nt") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def log(msg: str) -> None:
    print(f"{_c('1;36', '[reproduce]')} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{_c('1;33', '[reproduce][warn]')} {msg}", flush=True)


def die(msg: str) -> None:
    print(f"{_c('1;31', '[reproduce][error]')} {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def load_manifest() -> dict:
    try:
        import yaml
    except ImportError:
        die("PyYAML not available. Run this via `uv run python reproduce.py`.")
    if not MANIFEST.exists():
        die(f"Manifest not found at {MANIFEST}")
    with open(MANIFEST) as f:
        return yaml.safe_load(f)


def run(cmd: list[str], env: dict | None = None) -> None:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if proc.returncode != 0:
        die(f"command failed (exit {proc.returncode}): {' '.join(cmd)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _released_artifacts(manifest: dict) -> list[dict]:
    """Return every model artifact referenced by the release manifest."""
    return [*manifest["encoders"].values(), *manifest["classification"]]


def verify_public_assets(manifest: dict, datasets: list[str]) -> None:
    """Verify the pinned public release contract without downloading large assets."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    weights_repo = manifest["meta"]["hf_weights_repo"]
    weights_revision = manifest["meta"]["hf_weights_revision"]
    dataset_repo = manifest["meta"]["hf_dataset_repo"]
    dataset_revision = manifest["meta"]["hf_dataset_revision"]

    log(f"Checking HF model repo: {weights_repo} @ {weights_revision}")
    remote_weights = set(
        api.list_repo_files(weights_repo, repo_type="model", revision=weights_revision)
    )
    expected_weights = {
        artifact["hf_path"] for artifact in _released_artifacts(manifest)
    }
    absent_weights = sorted(expected_weights - remote_weights)
    if absent_weights:
        die(f"HF model revision is missing released artifacts: {absent_weights}")

    checksum_file = hf_hub_download(
        repo_id=weights_repo,
        filename="checksums.json",
        repo_type="model",
        revision=weights_revision,
    )
    records = json.loads(Path(checksum_file).read_text())
    checksum_paths = {record["hf_path"] for record in records}
    missing_checksums = sorted(expected_weights - checksum_paths)
    if missing_checksums:
        die(f"checksums.json does not cover released artifacts: {missing_checksums}")
    malformed = sorted(
        record.get("hf_path", "<missing>")
        for record in records
        if len(record.get("sha256", "")) != 64
    )
    if malformed:
        die(f"checksums.json contains malformed SHA-256 records: {malformed}")
    log(
        "Verified presence and checksum coverage for all "
        f"{len(expected_weights)} model artifacts"
    )

    hf_datasets = [dataset for dataset in datasets if dataset in DATASET_SUBDIR]
    if hf_datasets:
        log(f"Checking HF dataset repo: {dataset_repo} @ {dataset_revision}")
        remote_data = set(
            api.list_repo_files(
                dataset_repo,
                repo_type="dataset",
                revision=dataset_revision,
            )
        )
        missing_prefixes = [
            DATASET_SUBDIR[dataset]
            for dataset in hf_datasets
            if not any(
                path.startswith(f"{DATASET_SUBDIR[dataset]}/")
                for path in remote_data
            )
        ]
        if missing_prefixes:
            die(f"HF dataset revision is missing required directories: {missing_prefixes}")
        if "dailyliving" in datasets:
            sidecar = manifest["datasets"]["dailyliving"]["activity_sidecar"]
            if sidecar not in remote_data:
                die(f"HF dataset revision is missing daily-living sidecar: {sidecar}")
        log(f"Verified public files for: {', '.join(hf_datasets)}")

    if "stanford" in datasets:
        stanford = manifest["datasets"]["stanford"]
        url = f"{stanford['source']}/archive/{stanford['revision']}.zip"
        request = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status >= 400:
                    die(f"Stanford release returned HTTP {response.status}: {url}")
        except OSError as exc:
            die(f"Stanford pinned revision is not reachable: {url} ({exc})")
        log(f"Verified Stanford source revision: {stanford['revision']}")


def _git_revision() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _environment_record() -> dict:
    import torch

    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device,
    }


def write_run_record(
    manifest: dict,
    datasets: list[str],
    started_at: datetime,
    elapsed_seconds: float,
    status: str,
) -> Path:
    """Write enough provenance to compare runs from independent machines."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "command": [sys.executable, *sys.argv],
        "git_revision": _git_revision(),
        "hf_weights_repo": manifest["meta"]["hf_weights_repo"],
        "hf_weights_revision": manifest["meta"]["hf_weights_revision"],
        "hf_dataset_repo": manifest["meta"]["hf_dataset_repo"],
        "hf_dataset_revision": manifest["meta"]["hf_dataset_revision"],
        "datasets": datasets,
        "environment": _environment_record(),
    }
    results = LOGS_DIR / "RESULTS_external.csv"
    if status == "completed" and results.exists():
        import pandas as pd

        rows = pd.read_csv(results)
        record["results"] = rows[rows["dataset"].isin(datasets)].to_dict(
            orient="records"
        )
    output = LOGS_DIR / "reproduction_run.json"
    output.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return output


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def preflight() -> None:
    log("Preflight checks")
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        die(
            "huggingface_hub not importable. Run inside the project env:\n"
            "    uv sync && uv run python reproduce.py"
        )
    free_gb = shutil.disk_usage(REPO_ROOT).free / 1e9
    log(f"Free disk at repo root: {free_gb:.0f} GB")
    if free_gb < 30:
        warn("Less than 30 GB free -- downloads + zarrs may not fit.")


def download_weights(manifest: dict, contexts: list[str], phases: set[str]) -> None:
    from huggingface_hub import hf_hub_download

    repo = manifest["meta"]["hf_weights_repo"]
    revision = manifest["meta"].get("hf_weights_revision")
    log(f"Downloading weights from HF model repo: {repo} @ {revision}")

    # encoders for the contexts in use (probe/finetune reload these at build time)
    wanted = []
    for ctx in contexts:
        enc = manifest["encoders"].get(ctx)
        if enc:
            wanted.append((enc["hf_path"], enc["local"]))
    # classifiers matching (context, phase)
    for entry in manifest["classification"]:
        if entry["context"] in contexts and entry["phase"] in phases:
            wanted.append((entry["hf_path"], entry["local"]))

    for hf_path, local in wanted:
        if Path(hf_path).suffix != ".safetensors" or Path(local).suffix != ".safetensors":
            die(f"release manifest contains non-safetensors weights: {hf_path} -> {local}")
        local_path = REPO_ROOT / local
        if local_path.exists():
            log(f"  exists, skip: {local}")
            continue
        cached = hf_hub_download(repo_id=repo, filename=hf_path, repo_type="model",
                                 revision=revision)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, local_path)
        log(f"  placed: {hf_path} -> {local}")

    checksum_file = hf_hub_download(
        repo_id=repo,
        filename="checksums.json",
        repo_type="model",
        revision=revision,
    )
    records = json.loads(Path(checksum_file).read_text())
    checksums = {record["hf_path"]: record["sha256"] for record in records}
    missing = [hf_path for hf_path, _ in wanted if hf_path not in checksums]
    if missing:
        die(f"checksums.json does not cover released files: {missing}")
    for hf_path, local in wanted:
        actual = sha256(REPO_ROOT / local)
        if actual != checksums[hf_path]:
            die(f"checksum mismatch for {hf_path}: expected {checksums[hf_path]}, got {actual}")
    log(f"Verified SHA-256 for all {len(wanted)} downloaded model artifacts")


def download_datasets(manifest: dict, datasets: list[str]) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import get_token

    repo = manifest["meta"]["hf_dataset_repo"]
    revision = manifest["meta"].get("hf_dataset_revision")
    subdirs = sorted({DATASET_SUBDIR[d] for d in datasets if d in DATASET_SUBDIR})
    # recursive globs: kaggle_labeled/* would silently skip nested session data
    patterns = [f"{s}/**" for s in subdirs]
    log(f"Downloading dataset splits {subdirs} from HF dataset repo: {repo} @ {revision}")
    if get_token() is None:
        warn("No HF token found. The dataset has many small per-session files; "
             "anonymous pulls can hit HTTP 429 rate limits (retried automatically, "
             "but slow). `hf auth login` raises the limit substantially.")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # max_workers kept modest: fewer concurrent requests => far fewer 429s on the
    # many small CSVs. snapshot_download retries 429s with backoff regardless.
    if patterns:
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
            local_dir=str(RAW_DIR),
            max_workers=4,
        )
    log(f"  datasets in {RAW_DIR}")
    if "stanford" in datasets:
        download_stanford()


def download_stanford() -> None:
    """Download the public O'Day dataset from its authoritative GitHub repository."""
    target = RAW_DIR / "stanford_imu_fog"
    expected = target / "data" / "raw" / "imus6_subjects7"
    if expected.exists() and any(expected.glob("*.xlsx")):
        log(f"  exists, skip: {target.relative_to(REPO_ROOT)}")
        return
    if target.exists():
        die(f"incomplete Stanford download at {target}; remove that directory and retry")

    url = (
        f"https://github.com/{STANFORD_REPOSITORY}/archive/"
        f"{STANFORD_REVISION}.zip"
    )
    log(f"Downloading Stanford O'Day dataset @ {STANFORD_REVISION}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="forge-stanford-") as temporary:
        archive = Path(temporary) / "stanford.zip"
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(temporary)
        extracted = Path(temporary) / f"imu-fog-detection-{STANFORD_REVISION}"
        if not extracted.exists():
            die("Stanford archive did not contain the expected repository directory")
        shutil.move(str(extracted), target)


def build_zarrs(contexts: list[str], datasets: list[str]) -> None:
    log("Building evaluation zarrs")
    for ctx in contexts:
        if ctx != "mc":
            die(f"External reproduction supports only the released MC detector, not {ctx!r}")
        for ds in datasets:
            target = ZARR_NAME.get((ctx, ds))
            if target and (PROCESSED_DIR / target).exists():
                log(f"  exists, skip: {ctx}/{ds} ({target})")
                continue
            if ds == "stanford":
                if ctx != "mc":
                    die("Stanford reproduction supports only the released MC detector")
                run([sys.executable, "scripts/data/build_stanford_zarr.py"])
                continue
            if ds == "tdcsfog":
                if ctx != "mc":
                    die("tDCS-FOG reproduction supports only the released MC detector")
                resampled = RAW_DIR / "kaggle_tdcs100" / "tdcsfog" / "sessions"
                if not resampled.exists() or not any(resampled.glob("*.csv")):
                    run([sys.executable, "scripts/data/resample_tdcsfog_100hz.py"])
            paths_cfg = BUILD_RECIPE["paths"].get(ds)
            if paths_cfg is None:
                die(f"No public build recipe for {ctx}/{ds}")
            run(
                [
                    sys.executable, "-m", "data.process",
                    f"paths={paths_cfg}",
                    f"process={BUILD_RECIPE['process']}",
                ]
            )


def verify_zarrs(contexts: list[str], datasets: list[str]) -> None:
    missing = []
    for ctx in contexts:
        for ds in datasets:
            name = ZARR_NAME.get((ctx, ds))
            if name is None:
                missing.append(f"{ctx}/{ds} -> unsupported combination")
            elif not (PROCESSED_DIR / name).exists():
                missing.append(f"{ctx}/{ds} -> data/processed/{name}")
    if missing:
        die(
            "Required eval zarrs are missing:\n  "
            + "\n  ".join(missing)
            + "\nBuild them (drop --skip-zarr) or create them manually; see release/manifest.yaml."
        )


def run_eval(datasets: list[str], contexts: list[str], models: list[str],
             batch_scale: float | None) -> tuple[Path, list[str]]:
    import pandas as pd

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOGS_DIR / "comprehensive_eval.csv"
    env = os.environ.copy()
    if batch_scale is not None:
        env["EVAL_BATCH_SCALE"] = str(batch_scale)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log(f"EVAL_BATCH_SCALE={batch_scale} (memory/speed only, results unchanged)")

    # Per-dataset isolation: one subprocess per dataset so a single dataset's
    # failure (e.g. dailyliving's large frame aggregation getting OOM-killed)
    # cannot abort the whole run. --no-context-ensembles drops the slow,
    # cross-context combos we don't report.
    per_ds_frames: list = []
    failed: list[str] = []
    for ds in datasets:
        ds_out = LOGS_DIR / f"eval_{ds}.csv"
        cmd = [
            sys.executable, "scripts/eval/eval_comprehensive.py",
            "--datasets", ds,
            "--contexts", *contexts,
            "--models", *models,
            "--threshold-source", "fixed_035",
            "--no-context-ensembles",
            "--output", str(ds_out),
            "--cache-dir", str(CACHE_DIR),
        ]
        log(f"Evaluating dataset: {ds}")
        log("$ " + " ".join(cmd))
        rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode
        ds_df = None
        if rc == 0 and ds_out.exists():
            try:
                ds_df = pd.read_csv(ds_out)
            except pd.errors.EmptyDataError:
                ds_df = None  # subprocess exited 0 but wrote a 0-row file
        if ds_df is not None and not ds_df.empty:
            per_ds_frames.append(ds_df)
        else:
            # Failed outright, or "succeeded" with 0 result rows (e.g. a corrupt zarr
            # or crashing DataLoader workers made every combo fail). Treat as failed so
            # an empty CSV can't crash the final concat (pandas EmptyDataError).
            reason = "OOM-killed (-9)" if rc in (-9, 137) else (
                "0 result rows" if rc == 0 else f"exit {rc}")
            warn(f"dataset '{ds}' eval failed ({reason}) -- continuing with the others")
            failed.append(ds)

    if not per_ds_frames:
        die("All dataset evaluations failed; nothing to report.")
    pd.concat(per_ds_frames, ignore_index=True).to_csv(out, index=False)
    return out, failed


def run_external_metrics(datasets: list[str]) -> Path:
    out = LOGS_DIR / "RESULTS_external.csv"
    run(
        [
            sys.executable, "scripts/eval/eval_external_cohorts.py",
            "--cache-dir", str(CACHE_DIR),
            "--output", str(out),
            "--datasets", *datasets,
        ]
    )
    return out


# How to read each manifest result id back out of the output files. Expected
# values come from the manifest (single source of truth); these only say where the
# produced number lives.
#
# This is a sanity check, not an exact-equality test. All rows use the public
# nine-head released detector and the fixed 0.35 operating point.
BALLPARK_TOL = 0.03

RESULT_CHECKS = {
    "fogathome_auroc": {"dataset": "fogathome", "column": "AUROC"},
    "fogathome_ap": {"dataset": "fogathome", "column": "AP"},
    "fogathome_icc_tf": {"dataset": "fogathome", "column": "ICC_TF"},
    "tdcs_auroc": {"dataset": "tdcsfog", "column": "AUROC"},
    "tdcs_ap": {"dataset": "tdcsfog", "column": "AP"},
    "tdcs_icc_tf": {"dataset": "tdcsfog", "column": "ICC_TF"},
    "stanford_auroc": {"dataset": "stanford", "column": "AUROC"},
    "stanford_ap": {"dataset": "stanford", "column": "AP"},
    "stanford_icc_tf": {"dataset": "stanford", "column": "ICC_TF"},
    "dailyliving_auroc": {"dataset": "dailyliving", "column": "AUROC"},
    "dailyliving_ap": {"dataset": "dailyliving", "column": "AP"},
    "dailyliving_icc_tf": {"dataset": "dailyliving", "column": "ICC_TF"},
}


def _read_external_metric(csv_path: Path, dataset: str, column: str):
    import pandas as pd
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    sub = df[df["dataset"] == dataset]
    return float(sub[column].iloc[0]) if not sub.empty else None


def verify_results(manifest: dict, datasets: list[str]) -> bool:
    """Compare produced numbers against expected tolerances; print PASS/FAIL."""
    csv_path = LOGS_DIR / "RESULTS_external.csv"
    specs = []
    for r in manifest.get("results", []):
        spec = RESULT_CHECKS.get(r["id"])
        if spec and spec["dataset"] in datasets:
            specs.append({**spec, "value": float(r["value"]), "ci": r.get("ci")})

    rows = []  # (label, expected_str, actual_str, absolute_difference, status)
    for s in specs:
        exp = s["value"]
        actual = _read_external_metric(csv_path, s["dataset"], s["column"])
        label = f"{s['dataset']} {s['column']}"
        if s["column"] in {"AUROC", "AP"}:
            exp_str = f"{exp:.3f} +/-{BALLPARK_TOL}"
            if actual is None:
                rows.append((label, exp_str, "n/a", "n/a", "FAIL"))
            else:
                ok = abs(actual - exp) <= BALLPARK_TOL
                rows.append((label, exp_str, f"{actual:.3f}", f"{abs(actual - exp):.3f}",
                             "PASS" if ok else "FAIL"))
        else:
            ci = s.get("ci") or [exp - 0.02, exp + 0.02]
            exp_str = f"{exp:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
            if actual is None:
                rows.append((label, exp_str, "n/a", "n/a", "FAIL"))
            else:
                ok = ci[0] <= actual <= ci[1]
                rows.append((label, exp_str, f"{actual:.3f}", f"{abs(actual - exp):.3f}",
                             "PASS" if ok else "FAIL"))

    width = max(len(r[0]) for r in rows)
    print()
    log(f"Sanity check vs the manuscript (ballpark: AP/AUC +/-{BALLPARK_TOL}; ICC within CI).")
    log("Small gaps are expected -- see RESULT_CHECKS for why. Large ones mean something is wrong:")
    for label, exp_str, act_str, difference, status in rows:
        tag = {"PASS": _c("1;32", "PASS"), "FAIL": _c("1;31", "FAIL"),
               "SKIP": _c("1;33", "SKIP")}[status]
        print(f"    [{tag}] {label:<{width}}  expected {exp_str:<22} "
              f"got {act_str:<6} |delta| {difference}")

    ran = rows
    all_ok = bool(ran) and all(r[4] == "PASS" for r in ran)
    print()
    if all_ok:
        log(_c("1;32", f"CHECKS PASSED — all {len(ran)} requested external metrics "
                       "are in the manuscript's ballpark."))
    elif not ran:
        die("NO CHECKS COULD RUN — expected outputs are missing, so nothing was "
            "verified. See messages above.")
    else:
        die("SOME CHECKS FAILED — produced numbers are far from the manuscript "
            "(see table above); check the downloaded checkpoints and data.")
    return all_ok


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description="Reproduce the released FORGE detector on four external cohorts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=DEFAULT_DATASETS,
        help="external cohorts to evaluate (default: all four)",
    )
    p.add_argument("--skip-download", action="store_true", help="skip HF weight + dataset download")
    p.add_argument("--skip-zarr", action="store_true", help="skip zarr building")
    p.add_argument("--no-score", action="store_true", help="skip external metric scoring")
    p.add_argument(
        "--verify-assets-only",
        action="store_true",
        help="verify pinned public model/data packaging without running inference",
    )
    p.add_argument("--batch-scale", type=float, default=None,
                   help="EVAL_BATCH_SCALE for low-VRAM GPUs, e.g. 0.25 (results unchanged)")
    args = p.parse_args()

    contexts = ["mc"]
    models = ["probe"]
    datasets = args.datasets
    phases = {"probe"}

    manifest = load_manifest()
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    status = "failed"
    try:
        log(f"Plan: contexts={contexts}  models={models}  datasets={datasets}")
        preflight()

        if args.verify_assets_only:
            verify_public_assets(manifest, datasets)
            status = "assets_verified"
            log("PUBLIC ASSET CHECKS PASSED")
            return

        if not args.skip_download:
            download_weights(manifest, contexts, phases)
            download_datasets(manifest, datasets)
        else:
            log("Skipping download (--skip-download)")

        if not args.skip_zarr:
            build_zarrs(contexts, datasets)
        else:
            log("Skipping zarr build (--skip-zarr)")

        verify_zarrs(contexts, datasets)

        csv, failed = run_eval(datasets, contexts, models, args.batch_scale)
        log(f"AP/AUC results -> {csv}")

        if failed:
            die(f"Datasets that did not complete: {failed}")
        if not args.no_score:
            metrics = run_external_metrics(datasets)
            log(f"External metrics -> {metrics}")
            verify_results(manifest, datasets)
        status = "completed"
        log("Done — see the 'logs' folder for full results.")
    finally:
        record = write_run_record(
            manifest,
            datasets,
            started_at,
            time.monotonic() - started,
            status,
        )
        log(f"Run provenance -> {record}")


if __name__ == "__main__":
    main()

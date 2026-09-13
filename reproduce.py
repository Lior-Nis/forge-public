#!/usr/bin/env python
"""Reproduce FORGE evaluation results from released artifacts (cross-OS).

One self-contained, idempotent driver that:
  1. downloads the released weights  (HF: Liornis/forge-fog)
  2. downloads the evaluation datasets (HF: Liornis/fog-dataset)
  3. builds the evaluation zarrs       (data.process)
  4. runs the evaluation               (eval_comprehensive.py)
  5. computes the clinical ICC table   (compute_icc_thresholds.py)

It is pure Python on purpose: no `source activate`, no symlinks, no shell
globs -- so it runs identically on Linux, macOS, and Windows. The single
source of truth for paths/checkpoints is `release/manifest.yaml`; this script
reads it and mirrors the commands documented in the `onboard-repo` and
`reproduce-evaluations` skills.

Quick start from a bare clone (the wrappers run `uv sync` first):

    ./reproduce.sh                        # Linux/macOS  -- headline MC: seg AUC 0.908, AP 0.823, ICC 0.909
    ./reproduce.sh --full                 # full paper table (all contexts x models, ~100+ GB)
    reproduce.bat                         # Windows (cmd/PowerShell); WSL2 + ./reproduce.sh is more reliable

If the environment is already provisioned, run the module directly:

    uv run python reproduce.py [flags]    # same flags as the wrappers

Useful flags:
    --contexts mc sc            # restrict contexts
    --datasets fogathome kaggle # restrict datasets
    --models probe              # restrict models
    --skip-download             # weights/data already present
    --skip-zarr                 # zarrs already built
    --no-icc                    # skip the clinical ICC step
    --batch-scale 0.25          # smaller inference batches for low-VRAM GPUs (results unchanged)

Evaluation is inference-only and reproduces bit-identically from the released
artifacts (verified 2026-06-22); a GPU is recommended but not required.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MANIFEST = REPO_ROOT / "release" / "manifest.yaml"

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
LOGS_DIR = REPO_ROOT / "logs"
CACHE_DIR = LOGS_DIR / "comprehensive_eval_cache"

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
    "kaggle": "kaggle_labeled",
}

# Model CLI names; these are also the manifest `phase` values and the
# eval_comprehensive.py CKPT model keys -- one namespace, no translation.
MODELS = ("probe", "finetune", "supervised")

# Authoritative zarr-build recipes, mirrored from the committed
# scripts/shell/generate_{med,short}context_zarrs.sh. Each maps an eval dataset
# key to its (paths, process) Hydra config pair. The kaggle_daily (unlabeled,
# 64 GB) pretraining build is intentionally excluded -- eval does not need it.
BUILD_RECIPES = {
    "mc": {
        "process": "kaggle_medcontext",
        "paths": {
            "kaggle": "kaggle_defog_medcontext",
            "fogathome": "fogathome_medcontext",
            "dailyliving": "fogathome_dailyliving_medcontext",
        },
    },
    "sc": {
        "process": "kaggle_shortcontext",
        "paths": {
            "kaggle": "kaggle_defog_shortcontext",
            "fogathome": "fogathome_shortcontext",
            "dailyliving": "fogathome_dailyliving_shortcontext",
        },
    },
    # No committed build script exists for long context, and the kaggle-LC
    # paths config is ambiguous; only the derivable fogathome/dailyliving
    # builds are wired. kaggle-LC must be built manually if needed for --full.
    "lc": {
        "process": "kaggle_longcontext",
        "paths": {
            "fogathome": "fogathome_longcontext",
            "dailyliving": "fogathome_dailyliving_longcontext",
        },
    },
}

# Target eval zarr filenames, mirrored from scripts/eval/eval_comprehensive.py
# (ZARR_NAME). Used to verify a build produced what the eval expects.
ZARR_NAME = {
    ("lc", "fogathome"): "len1000_stride200_fogstride100_fogathome.zarr",
    ("mc", "fogathome"): "len500_stride200_fogstride100_fogathome.zarr",
    ("sc", "fogathome"): "len200_stride20_fogstride10_fogathome.zarr",
    ("lc", "dailyliving"): "len1000_stride200_fogstride100_fogathome_dailyliving.zarr",
    ("mc", "dailyliving"): "len500_stride200_fogstride100_anyfog_fogathome_dailyliving.zarr",
    ("sc", "dailyliving"): "len200_stride20_fogstride10_anyfog_fogathome_dailyliving.zarr",
    ("lc", "kaggle"): "len1000_stride200_fogstride100_kaggle.zarr",
    ("mc", "kaggle"): "len500_stride200_fogstride100_anyfog_kaggle_defog.zarr",
    ("sc", "kaggle"): "len200_stride20_fogstride10_anyfog_kaggle_defog.zarr",
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
    log(f"Downloading weights from HF model repo: {repo}")

    # Classification weights only. Each released head already contains its
    # encoder, so encoders/*.safetensors are needed only to train new heads.
    wanted = []
    for entry in manifest["classification"]:
        if entry["context"] in contexts and entry["phase"] in phases:
            wanted.append((entry["hf_path"], entry["local"]))

    for hf_path, local in wanted:
        local_path = REPO_ROOT / local
        if local_path.exists():
            log(f"  exists, skip: {local}")
            continue
        cached = hf_hub_download(repo_id=repo, filename=hf_path, repo_type="model")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, local_path)
        log(f"  placed: {hf_path} -> {local}")


def download_datasets(manifest: dict, datasets: list[str]) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import get_token

    repo = manifest["meta"]["hf_dataset_repo"]
    subdirs = sorted({DATASET_SUBDIR[d] for d in datasets})
    # recursive globs: kaggle_labeled/* would silently skip nested session data
    patterns = [f"{s}/**" for s in subdirs]
    log(f"Downloading dataset splits {subdirs} from HF dataset repo: {repo}")
    if get_token() is None:
        warn("No HF token found. The dataset has many small per-session files; "
             "anonymous pulls can hit HTTP 429 rate limits (retried automatically, "
             "but slow). `hf auth login` raises the limit substantially.")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # max_workers kept modest: fewer concurrent requests => far fewer 429s on the
    # many small CSVs. snapshot_download retries 429s with backoff regardless.
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(RAW_DIR),
        max_workers=4,
    )
    log(f"  datasets in {RAW_DIR}")


def build_zarrs(contexts: list[str], datasets: list[str]) -> None:
    log("Building evaluation zarrs")
    for ctx in contexts:
        recipe = BUILD_RECIPES.get(ctx)
        if recipe is None:
            warn(f"no build recipe for context '{ctx}' -- skipping")
            continue
        for ds in datasets:
            target = ZARR_NAME.get((ctx, ds))
            if target and (PROCESSED_DIR / target).exists():
                log(f"  exists, skip: {ctx}/{ds} ({target})")
                continue
            paths_cfg = recipe["paths"].get(ds)
            if paths_cfg is None:
                warn(
                    f"no automated build for {ctx}/{ds}; build it manually, e.g.\n"
                    f"    uv run python -m data.process paths=<cfg> process={recipe['process']}\n"
                    f"  (expected output: data/processed/{target})"
                )
                continue
            run(
                [
                    sys.executable, "-m", "data.process",
                    f"paths={paths_cfg}",
                    f"process={recipe['process']}",
                ]
            )


def verify_zarrs(contexts: list[str], datasets: list[str]) -> None:
    missing = []
    for ctx in contexts:
        for ds in datasets:
            name = ZARR_NAME.get((ctx, ds))
            if name and not (PROCESSED_DIR / name).exists():
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


def run_icc() -> Path:
    out = LOGS_DIR / "RESULTS_icc.md"
    run(
        [
            sys.executable, "scripts/eval/compute_icc_thresholds.py",
            "--cache-dir", str(CACHE_DIR),
            "--out", str(out),
        ]
    )
    return out


# How to read each manifest result id back out of the output files. Expected
# values come from the manifest (single source of truth); these only say where the
# produced number lives.
#
# This is a sanity check, not an exact-equality test. The manifest reports the
# manuscript's released detector (nine heads = 3 folds x 3 seeds, scored on
# annotator-verified frames); this pipeline runs a seed-42 three-fold ensemble over
# the full window grid. Those bases differ by ~0.01-0.02, so the bar is a ballpark:
# wide enough to absorb the basis gap, tight enough that wrong checkpoints or
# missing data (off by ~0.1+) still show up as FAIL. ICC is checked against the
# manuscript's CI. Result ids with no entry here are simply not checked.
BALLPARK_TOL = 0.03

RESULT_CHECKS = {
    "fogathome_auroc":  {"where": "csv", "dataset": "fogathome", "level": "segmentation",
                         "thr": None, "col": "AUC", "tol": BALLPARK_TOL,
                         "label": "FogAtHome frame AUROC"},
    "fogathome_ap":     {"where": "csv", "dataset": "fogathome", "level": "segmentation",
                         "thr": None, "col": "AP", "tol": BALLPARK_TOL,
                         "label": "FogAtHome frame AP"},
    "fogathome_icc_tf": {"where": "icc", "model": "mc/probe", "label": "FogAtHome ICC(%TF)"},
    "defog_window_ap":  {"where": "csv", "dataset": "kaggle", "level": "classification",
                         "thr": "50", "col": "AP", "tol": BALLPARK_TOL,
                         "label": "DeFOG window AP (in-distribution)"},
}


def _read_csv_metric(csv_path: Path, dataset: str, level: str, thr, col: str):
    import pandas as pd
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    sub = df[(df["dataset"] == dataset) & (df["context"] == "mc") & (df["level"] == level)]
    if thr is not None:
        sub = sub[sub["threshold_pct"].astype(str) == thr]
    return float(sub[col].iloc[0]) if not sub.empty else None


def _read_icc_value(icc_path: Path, model: str):
    if not icc_path.exists():
        return None
    pat = re.compile(re.escape(model) + r"\b.*?(\d+\.\d+)\s*\[(\d+\.\d+),\s*(\d+\.\d+)\]")
    for line in icc_path.read_text().splitlines():
        m = pat.search(line)
        if m:
            return float(m.group(1)), (float(m.group(2)), float(m.group(3)))
    return None


def verify_results(manifest: dict) -> bool:
    """Compare produced numbers against expected tolerances; print PASS/FAIL."""
    csv_path = LOGS_DIR / "comprehensive_eval.csv"
    icc_path = LOGS_DIR / "RESULTS_icc.md"

    # Account for EVERY reported number. A result this pipeline cannot recompute
    # must be named as such, not dropped -- otherwise a run that verified 4 of 13
    # prints the same success line as one that verified all of them.
    specs, not_run = [], []
    for r in manifest.get("results", []):
        status = r.get("verification", "recorded")
        spec = RESULT_CHECKS.get(r["id"])
        if status == "one_click" and spec:
            specs.append({**spec, "value": float(r["value"]), "ci": r.get("ci")})
        else:
            not_run.append((r["id"], status,
                            r.get("verification_note") or r.get("script") or ""))

    rows = []  # (label, expected_str, actual_str, status)
    for s in specs:
        exp = s["value"]
        if s["where"] == "csv":
            actual = _read_csv_metric(csv_path, s["dataset"], s["level"], s["thr"], s["col"])
            exp_str = f"{exp:.3f} +/-{s['tol']}"
            if actual is None:
                rows.append((s["label"], exp_str, "n/a", "SKIP"))
            else:
                ok = abs(actual - exp) <= s["tol"]
                rows.append((s["label"], exp_str, f"{actual:.3f}", "PASS" if ok else "FAIL"))
        else:  # icc — within CI
            res = _read_icc_value(icc_path, s["model"])
            ci = s.get("ci") or [exp - 0.02, exp + 0.02]
            exp_str = f"{exp:.3f} [{ci[0]}, {ci[1]}]"
            if res is None:
                rows.append((s["label"], exp_str, "n/a", "SKIP"))
            else:
                actual, _ = res
                ok = ci[0] <= actual <= ci[1]
                rows.append((s["label"], exp_str, f"{actual:.3f}", "PASS" if ok else "FAIL"))

    width = max(len(r[0]) for r in rows)
    print()
    log(f"Sanity check vs the manuscript (ballpark: AP/AUC +/-{BALLPARK_TOL}; ICC within CI).")
    log("Small gaps are expected -- see RESULT_CHECKS for why. Large ones mean something is wrong:")
    for label, exp_str, act_str, status in rows:
        tag = {"PASS": _c("1;32", "PASS"), "FAIL": _c("1;31", "FAIL"),
               "SKIP": _c("1;33", "SKIP")}[status]
        print(f"    [{tag}] {label:<{width}}  expected {exp_str:<22} got {act_str}")

    if not_run:
        print()
        log(f"Not checked by this run ({len(not_run)} of "
            f"{len(rows) + len(not_run)} reported numbers):")
        for rid, status, why in not_run:
            tag = {"scripted": _c("1;36", "SCRIPTED"),
                   "recorded": _c("1;35", "RECORDED")}.get(status, status.upper())
            print(f"    [{tag}] {rid}")
            if why:
                print(f"             {why}")

    ran = [r for r in rows if r[3] != "SKIP"]
    all_ok = bool(ran) and all(r[3] == "PASS" for r in ran)
    print()
    if all_ok:
        log(_c("1;32", f"CHECKS PASSED — {len(ran)} of "
                       f"{len(rows) + len(not_run)} reported numbers verified here, "
                       "all in the manuscript's ballpark."))
        if not_run:
            log(f"The other {len(not_run)} were NOT verified by this run (listed above).")
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
        description="Reproduce FORGE evaluation results from released HF artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--full", action="store_true",
                   help="full paper table: all contexts (lc mc sc) x models (probe/finetune/supervised)")
    p.add_argument("--contexts", nargs="+", help="override contexts (default: mc, or lc mc sc with --full)")
    p.add_argument("--datasets", nargs="+", default=None, choices=list(DATASET_SUBDIR),
                   help="datasets to evaluate (default: fogathome kaggle; +dailyliving with --full). "
                        "The 4 headline numbers only need fogathome + kaggle; dailyliving is the "
                        "memory-heavy daily-living head-to-head.")
    p.add_argument("--models", nargs="+", choices=list(MODELS),
                   help="override models (default: probe, or all with --full)")
    p.add_argument("--skip-download", action="store_true", help="skip HF weight + dataset download")
    p.add_argument("--skip-zarr", action="store_true", help="skip zarr building")
    p.add_argument("--no-icc", action="store_true", help="skip the clinical ICC table step")
    p.add_argument("--batch-scale", type=float, default=None,
                   help="EVAL_BATCH_SCALE for low-VRAM GPUs, e.g. 0.25 (results unchanged)")
    args = p.parse_args()

    contexts = args.contexts or (["lc", "mc", "sc"] if args.full else ["mc"])
    models = args.models or (
        list(MODELS) if args.full else ["probe"]
    )
    datasets = args.datasets or (
        ["fogathome", "dailyliving", "kaggle"] if args.full else ["fogathome", "kaggle"]
    )
    phases = {m for m in models if m in MODELS}

    log(f"Plan: contexts={contexts}  models={models}  datasets={datasets}")

    manifest = load_manifest()
    preflight()

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

    if not args.no_icc:
        # ICC needs only the fogathome + kaggle frame parquets, which are cached
        # even if other datasets (e.g. dailyliving) failed above.
        icc = run_icc()
        log(f"Clinical ICC table -> {icc}")

    if failed:
        warn(f"Datasets that did not complete: {failed} (headline numbers are unaffected).")
    verify_results(manifest)  # prints PASS/FAIL table; exits non-zero on any FAIL
    log("Done — see the 'logs' folder for full results.")


if __name__ == "__main__":
    main()

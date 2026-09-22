"""
Comprehensive evaluation across all contexts, model types, datasets, and fog_ratio thresholds.

Produces a CSV with one row per (dataset, context, model_type, threshold_pct, level).
Metrics: AUC, AP, Precision, Recall, Specificity, Accuracy, F1, NormAP, Lift.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
import zarr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_MANIFEST = REPO_ROOT / "release" / "manifest.yaml"
CACHE_FORMAT = "safetensors_valid"

# ── Patient cohorts ────────────────────────────────────────────────────────────
FOGATHOME_PATIENTS = [
    "a00001", "a00002", "a00004", "a00006", "a00007", "a00008",
    "a00009", "a00010", "a00011", "a00012", "a00013", "a00014",
]
DAILYLIVING_PATIENTS = [
    "c00001", "c00002", "c00004", "c00005", "c00006", "c00007",
    "c00010", "c00011", "c00012", "c00013", "c00014",
]
STANFORD_PATIENTS = [f"s{n:05d}" for n in range(1, 8)]

# ── Physical zarr filenames (relative to processed_dir) per (context, dataset) ──
# Kaggle/DeFOG eval points at the physical .zarr directly and applies the
# validity==1.0 filter via KAGGLE_METADATA_QUERY at dataset-build time. This
# replaces the former gitignored, non-portable `views/defog_valid_*.yaml` files
# (whose only filter was `validity == 1.0`; protocol filtering is handled by the
# committed DeFOG patient splits). No view YAML is needed to reproduce these rows.
ZARR_NAME = {
    ("lc", "fogathome"):   "len1000_stride200_fogstride100_fogathome.zarr",
    ("mc", "fogathome"):   "len500_stride200_fogstride100_fogathome.zarr",
    ("sc", "fogathome"):   "len200_stride20_fogstride10_fogathome.zarr",
    ("lc", "dailyliving"): "len1000_stride200_fogstride100_fogathome_dailyliving.zarr",
    ("mc", "dailyliving"): "len500_stride200_fogstride100_anyfog_fogathome_dailyliving.zarr",
    ("sc", "dailyliving"): "len200_stride20_fogstride10_anyfog_fogathome_dailyliving.zarr",
    ("mc", "tdcsfog"):     "len500_stride200_fogstride100_anyfog_tdcs100.zarr",
    ("mc", "stanford"):    "len500_stride200_stanford.zarr",
    ("lc", "kaggle"):      "len1000_stride200_fogstride100_kaggle.zarr",
    ("mc", "kaggle"):      "len500_stride200_fogstride100_anyfog_kaggle_defog.zarr",
    ("sc", "kaggle"):      "len200_stride20_fogstride10_anyfog_kaggle_defog.zarr",
}
ZARR = {k: f"data/processed/{v}" for k, v in ZARR_NAME.items()}

# Metadata filter applied to Kaggle/DeFOG eval patches — the sole filter the
# former defog_valid_* views encoded.
KAGGLE_METADATA_QUERY = "validity == 1.0"

# Local processed-data dir. Do NOT trust the checkpoint config's baked processed_dir
# (it is the author's absolute path). Honors DATA_ROOT like the Hydra configs
# (${oc.env:DATA_ROOT, data}); defaults to <repo>/data/processed.
PROCESSED_DIR = str((Path(os.environ.get("DATA_ROOT", "data")) / "processed").resolve())


def _open_zarr(zarr_path: str):
    """Open a processed eval zarr, failing clearly if it has not been built."""
    if not Path(zarr_path).exists():
        raise FileNotFoundError(
            f"Eval zarr not found: {zarr_path}\n"
            "Build the evaluation zarrs first with `python -m data.process`; "
            "see scripts/README.md or the reproduction docs for the command shape. "
            "Only request contexts whose zarrs you have built."
        )
    return zarr.open_group(zarr_path, mode="r")

# seq_len (frames) and stride (frames) for frame-level aggregation
SEQ_LEN    = {"lc": 1000, "mc": 500, "sc": 200}
STRIDE_FR  = {"lc": 200,  "mc": 50,  "sc": 10}
# Inference batch size per context. Override with EVAL_BATCH_SCALE (e.g. 0.25) to fit a
# smaller GPU — batch size does not affect inference outputs, only memory/speed.
_BATCH_SCALE = float(os.environ.get("EVAL_BATCH_SCALE", "1.0"))
BATCH_SIZE = {c: max(1, int(b * _BATCH_SCALE))
              for c, b in {"lc": 32, "mc": 128, "sc": 512}.items()}
THRESHOLDS = [0.0, 0.25, 0.50, 0.75, 1.0]

# Decision-threshold protocol for F1/Recall/Precision/Specificity. Set in main().
#   test_youden     : ROC-Youden on the eval set (legacy default)
#   test_pr11       : PR point closest to (1,1) on the eval set (Salomon et al. 2026)
#   defog_val_pr11  : PR (1,1) on held-out DeFOG (kaggle) preds, applied to external sets
#                     (no test peeking — the rigorous cross-dataset operating point)
THRESHOLD_SOURCE = "test_youden"
_VAL_THR_CACHE: dict = {}

# Context-level ensembles: which context combinations to evaluate
CONTEXT_ENSEMBLES = [
    ("mc", "sc"),
    ("lc", "sc"),
    ("lc", "mc"),
    ("lc", "mc", "sc"),
]

# ── Zarr helpers ──────────────────────────────────────────────────────────────
def load_zarr_arrays(zarr_path: str):
    """Return (fog_ratios, session_idx_arr, patient_id_arr, session_id_arr, global_idx_arr)."""
    z = _open_zarr(zarr_path)
    fog_ratios   = z["patch_labels"][:]          # (N,) float32 — continuous fog fraction
    meta         = z["metadata"]
    global_idxs  = meta["global_idx"][:]         # (N,) int64
    session_idxs = meta["session_idx"][:]        # (N,) int64
    patient_ids  = meta["patient_id"][:]         # (N,) str/object
    session_ids  = meta["session_id"][:]         # (N,) str/object

    # Ensure fog_ratios are continuous (handle binary int32 zarrs)
    if fog_ratios.dtype != np.float32 or not (fog_ratios % 1 != 0).any():
        if "labels" in z:
            labs = z["labels"][:]
            fog_ratios = (labs > 0).mean(axis=1).astype(np.float32)

    # Map global_idx → array position (may not be identity if zarr was filtered)
    gidx_to_pos = {int(g): i for i, g in enumerate(global_idxs)}
    return fog_ratios, gidx_to_pos, session_idxs, patient_ids, session_ids, global_idxs


# ── Inference ─────────────────────────────────────────────────────────────────
def released_members(context, model_type, dataset):
    manifest = yaml.safe_load(RELEASE_MANIFEST.read_text())
    entries = [e for e in manifest["classification"]
               if e["context"] == context and e["phase"] == model_type]
    if dataset == "kaggle":
        entries = [e for e in entries if e["seed"] == 42]
    if context == "mc" and model_type == "probe" and dataset != "kaggle":
        expected = {(s, f) for s in (42, 43, 44) for f in range(3)}
        actual = {(e["seed"], e["fold"]) for e in entries}
        if actual != expected:
            raise RuntimeError(f"Incomplete released ensemble: {sorted(actual)}")
    invalid = [e["local"] for e in entries if Path(e["local"]).suffix != ".safetensors"]
    if invalid:
        raise ValueError(f"Released models must use .safetensors: {invalid}")
    return sorted(({**e, "path": REPO_ROOT / e["local"]} for e in entries),
                  key=lambda e: (e["seed"], e["fold"]))


def run_inference(context, model_type, dataset, member, cache_dir: Path) -> pd.DataFrame:
    """Run inference for one (context, model_type, dataset, fold). Cached."""
    fold = member["fold"]
    seed = member["seed"]
    key = f"{dataset}_{context}_{model_type}"
    weights_path = Path(member["path"])
    if weights_path.suffix != ".safetensors":
        raise ValueError(f"{weights_path}: released models must use .safetensors")
    pred_csv = cache_dir / f"{key}_{weights_path.stem}_preds.csv"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing released weights: {weights_path}")
    if pred_csv.exists():
        return pd.read_csv(pred_csv)

    zarr_name = ZARR_NAME[(context, dataset)]  # filename only (for config override)
    logger.info(f"Running inference: {key} seed={seed} fold={fold}")

    from data.datamodule.config import SplitsConfig
    from data.datamodule.datamodule import FOGDataModule
    from utils.paths import normalize_data_paths
    from utils.released_weights import load_released_model

    model, config = load_released_model(weights_path)
    normalize_data_paths(config.data.paths)

    # Determine patient filter
    if dataset == "fogathome":
        patients = FOGATHOME_PATIENTS
        splits_override = SplitsConfig(train=patients, val=[], test=patients)
    elif dataset == "dailyliving":
        patients = DAILYLIVING_PATIENTS
        splits_override = SplitsConfig(train=patients, val=[], test=patients)
    elif dataset == "stanford":
        splits_override = SplitsConfig(train=STANFORD_PATIENTS, val=[], test=STANFORD_PATIENTS)
    elif dataset == "tdcsfog":
        from utils.zarr_helpers import read_zarr_metadata

        metadata = read_zarr_metadata(ZARR[(context, dataset)], "/metadata")
        patients = sorted(metadata["patient_id"].astype(str).unique().tolist())
        splits_override = SplitsConfig(train=patients, val=[], test=patients)
    else:
        # Kaggle: use original fold split (test = held-out Kaggle patients)
        splits_override = None

    updates = {
        "dataloader": config.data.dataloader.model_copy(update={
            "batch_size": BATCH_SIZE[context],
            # num_workers=0 by default: the model's wavelet transform runs on the GPU,
            # so the DataLoader only does light zarr reads -- cheap to keep in-process.
            # On Windows, num_workers>0 spawns worker processes that crashed the long
            # sc daily-living eval ("Caught OSError in DataLoader worker", [Errno 22]).
            # Override with EVAL_NUM_WORKERS to benchmark; batch size via EVAL_BATCH_SCALE.
            "num_workers": int(os.environ.get("EVAL_NUM_WORKERS", "0")),
        }),
        # Override dataset_path AND processed_dir: the released checkpoint config bakes
        # the author's absolute processed_dir, which does not exist on another machine.
        # Rebase to the LOCAL processed dir so the eval reproduces from a fresh clone.
        "paths": config.data.paths.model_copy(update={
            "dataset_path": zarr_name,
            "processed_dir": PROCESSED_DIR,
        }),
    }
    if dataset == "kaggle":
        # Apply the validity==1.0 filter the former defog_valid_* view encoded,
        # directly on the physical zarr (no gitignored view YAML needed).
        updates["dataset"] = config.data.dataset.model_copy(
            update={"metadata_query": KAGGLE_METADATA_QUERY}
        )
    if splits_override is not None:
        updates["splits"] = splits_override

    data_cfg    = config.data.model_copy(update=updates)
    data_module = FOGDataModule(data_cfg=data_cfg, task_type=config.train.pipeline_type)
    data_module.setup("test")

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    # Inject normalization stats (best-effort; identity normalizer = no-op so safe to skip)
    try:
        data_module.setup("fit")
        if hasattr(model, "preprocessors") and model.preprocessors is not None:
            stats = data_module.train_dataset.get_normalization_stats()
            model.preprocessors.inject_stats(stats)
    except Exception as e:
        logger.warning(f"Stats injection skipped ({e})")

    rows = []
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            x    = batch["x"].to(device)
            meta = batch["metadata"]
            logits = model(x)
            probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            for i, m in enumerate(meta):
                rows.append({
                    "patient_id":   m.get("patient_id", ""),
                    "session_id":   m.get("session_id", ""),
                    "global_idx":   int(m.get("global_idx", -1)),
                    "session_idx":  int(m.get("session_idx", -1)),
                    "pred_prob_fog": float(probs[i]),
                    "fold":         fold,
                    "seed":         seed,
                })

    df = pd.DataFrame(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(pred_csv, index=False)
    logger.info(f"  → {len(df)} preds cached to {pred_csv}")

    # Explicit GPU cleanup to avoid OOM across sequential inference runs
    del model, data_module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df


# ── Ensemble: average or concatenate fold preds ───────────────────────────────
def build_ensemble(context, model_type, dataset, cache_dir: Path) -> pd.DataFrame:
    """Collect fold preds and return ensemble:
    - External cohorts: same patients in all folds → mean prob per global_idx
    - Kaggle: different test patients per fold → concatenate
    """
    members = released_members(context, model_type, dataset)
    if not members:
        raise RuntimeError(
            f"No released safetensors for context={context!r}, model={model_type!r}"
        )
    fold_dfs = []
    for member in members:
        df = run_inference(context, model_type, dataset, member, cache_dir)
        if df.empty or df["global_idx"].duplicated().any():
            raise RuntimeError(f"Invalid predictions for seed={member['seed']} fold={member['fold']}")
        df["fold"] = member["fold"]
        fold_dfs.append(df)

    if not fold_dfs:
        return pd.DataFrame()

    combined = pd.concat(fold_dfs, ignore_index=True)

    if dataset == "kaggle":
        # Each fold has distinct test patients → no averaging needed
        return combined.groupby("global_idx").agg(
            patient_id=("patient_id", "first"),
            session_id=("session_id", "first"),
            session_idx=("session_idx", "first"),
            pred_prob_fog=("pred_prob_fog", "first"),
        ).reset_index()
    else:
        # Average across folds for same-cohort evals
        counts = combined.groupby("global_idx").size()
        if not (counts == len(members)).all():
            raise RuntimeError("Released ensemble members produced different prediction rows")
        return combined.groupby("global_idx").agg(
            patient_id=("patient_id", "first"),
            session_id=("session_id", "first"),
            session_idx=("session_idx", "first"),
            pred_prob_fog=("pred_prob_fog", "mean"),
        ).reset_index()


# ── Metrics ───────────────────────────────────────────────────────────────────
def youden_threshold(y_true, y_score):
    fpr, tpr, thrs = roc_curve(y_true, y_score)
    return float(thrs[np.argmax(tpr - fpr)])


def pr11_threshold(y_true, y_score):
    """PR-curve point closest to (1,1) — Salomon et al. 2026 operating point."""
    prec, rec, thrs = precision_recall_curve(y_true, y_score)
    prec, rec = prec[:-1], rec[:-1]  # drop trailing point that has no threshold
    return float(thrs[np.argmin(np.sqrt((1.0 - prec) ** 2 + (1.0 - rec) ** 2))])


def resolve_threshold(y_true, y_score, context, model_type, cache_dir) -> float:
    """Pick the decision threshold per THRESHOLD_SOURCE.

    For defog_val_pr11 the threshold is computed once per (context, model_type) from
    the held-out DeFOG (kaggle) frame parquet and reused; falls back to test_pr11 if
    that parquet is absent (e.g. context ensembles not yet built).
    """
    if THRESHOLD_SOURCE == "fixed_035":
        return 0.35
    if THRESHOLD_SOURCE == "test_youden":
        return youden_threshold(y_true, y_score)
    if THRESHOLD_SOURCE == "test_pr11":
        return pr11_threshold(y_true, y_score)
    if THRESHOLD_SOURCE == "defog_val_pr11":
        key = (context, model_type)
        if key not in _VAL_THR_CACHE:
            p = Path(cache_dir) / f"kaggle_{context}_{model_type}_{CACHE_FORMAT}_frames.parquet"
            if p.exists():
                d = pd.read_parquet(p)
                _VAL_THR_CACHE[key] = pr11_threshold(
                    d["native_label"].values.astype(int), d["pred_prob_fog"].values
                )
            else:
                logger.error(
                    f"THRESHOLD WARNING: held-out DeFOG-validation parquet missing for {key} "
                    f"({p}). Falling back to test_pr11 — these F1/precision/recall numbers are "
                    f"TEST-SET-TUNED (optimistic) for this row. To get the held-out operating "
                    f"point, evaluate --datasets kaggle (same --cache-dir) before the external sets."
                )
                _VAL_THR_CACHE[key] = None
        t = _VAL_THR_CACHE[key]
        return t if t is not None else pr11_threshold(y_true, y_score)
    return youden_threshold(y_true, y_score)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, prevalence: float = None,
                    threshold: float = None) -> dict:
    nan = float("nan")
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return dict(AUC=nan, AP=nan, Precision=nan, Recall=nan,
                    Specificity=nan, Accuracy=nan, F1=nan, NormAP=nan, Lift=nan, Threshold=nan)
    auc = float(roc_auc_score(y_true, y_score))
    ap  = float(average_precision_score(y_true, y_score))
    thr = youden_threshold(y_true, y_score) if threshold is None else float(threshold)
    y_pred = (y_score >= thr).astype(int)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    acc  = float(accuracy_score(y_true, y_pred))
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    spec = tn / (tn + fp) if (tn + fp) > 0 else nan
    if prevalence is None:
        prevalence = float(y_true.mean())
    norm_ap = (ap - prevalence) / (1 - prevalence) if prevalence < 1.0 else nan
    lift    = prec / prevalence if prevalence > 0 else nan
    return dict(AUC=auc, AP=ap, Precision=prec, Recall=rec,
                Specificity=spec, Accuracy=acc, F1=f1, NormAP=norm_ap, Lift=lift,
                Threshold=round(thr, 4))


# ── Frame-level aggregation (vectorized per session) ─────────────────────────
def build_frame_df(ensemble: pd.DataFrame, zarr_path: str, context: str,
                   frame_cache: Path,
                   eligible_intervals: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Aggregate patch predictions to per-frame predictions.
    Stores fog_ratio (mean of overlapping patch fog_ratios) and native_label
    (majority vote of zarr binary annotations) per covered frame."""
    # Daily-living eligibility comes from a versioned public sidecar. Rebuild its
    # inexpensive frame aggregation so a stale, unfiltered cache cannot be reused.
    if eligible_intervals is not None and frame_cache.exists():
        frame_cache.unlink()

    # Invalidate cache if native_label column is missing (older cache format)
    if frame_cache.exists():
        try:
            cols = pq.ParquetFile(frame_cache).schema_arrow.names
            if "native_label" in cols:
                return _read_frame_cache(frame_cache)
            frame_cache.unlink()
            logger.info(f"  Rebuilding frame cache (adding native_label): {frame_cache.name}")
        except Exception:
            frame_cache.unlink()

    seq_len = SEQ_LEN[context]
    stride  = STRIDE_FR[context]

    z = _open_zarr(zarr_path)
    fog_ratios_all = z["patch_labels"][:]   # (N_patches,) float32
    has_labels     = "labels" in z
    labels_all     = z["labels"][:] if has_labels else None  # (N_patches, seq_len) int8/int32
    valid_masks_all = z["valid_masks"][:] if "valid_masks" in z else None
    meta           = z["metadata"]
    all_gidx       = meta["global_idx"][:]
    all_sidx       = meta["session_idx"][:]

    # Decode int-encoded patient/session IDs using the zarr encoding dict if present
    _enc = dict(meta.attrs.get("encodings", {}))
    _pat_dec = {str(v): k for k, v in _enc["patient_id"].items()} if "patient_id" in _enc else None
    _sid_dec = {str(v): k for k, v in _enc["session_id"].items()} if "session_id"  in _enc else None
    _raw_pats = meta["patient_id"][:]
    _raw_sids = meta["session_id"][:]
    all_pats = np.array([_pat_dec[str(v)] if _pat_dec else str(v) for v in _raw_pats], dtype=object)
    all_sids = np.array([_sid_dec[str(v)] if _sid_dec else str(v) for v in _raw_sids], dtype=object)
    # Use actual frame start if stored (new zarrs); fall back to session_idx * STRIDE_FR
    has_start_frame = "start_frame" in meta
    all_start_frame = meta["start_frame"][:] if has_start_frame else None

    # gidx → position in zarr arrays (identity for sequential global_idx)
    gidx_to_i = {int(g): i for i, g in enumerate(all_gidx)}

    ens = ensemble.copy()
    ens_gidxs = ens["global_idx"].values

    zarr_sidx = np.array([all_sidx[gidx_to_i[int(g)]] if int(g) in gidx_to_i else -1
                          for g in ens_gidxs], dtype=np.int64)
    zarr_fogr = np.array([float(fog_ratios_all[gidx_to_i[int(g)]]) if int(g) in gidx_to_i else 0.0
                          for g in ens_gidxs], dtype=np.float32)
    zarr_pat  = np.array([str(all_pats[gidx_to_i[int(g)]]) if int(g) in gidx_to_i else ""
                          for g in ens_gidxs], dtype=object)
    zarr_sid  = np.array([str(all_sids[gidx_to_i[int(g)]]) if int(g) in gidx_to_i else ""
                          for g in ens_gidxs], dtype=object)
    zarr_zpos  = np.array([gidx_to_i.get(int(g), -1) for g in ens_gidxs], dtype=np.int64)
    zarr_start = np.array(
        [int(all_start_frame[gidx_to_i[int(g)]]) if has_start_frame and int(g) in gidx_to_i else -1
         for g in ens_gidxs],
        dtype=np.int64,
    )

    ens["_sidx"]  = zarr_sidx
    ens["_fogr"]  = zarr_fogr
    ens["_pat"]   = zarr_pat
    ens["_sid"]   = zarr_sid
    ens["_zpos"]  = zarr_zpos
    ens["_start"] = zarr_start

    valid = (ens["_sidx"] >= 0) & (ens["_pat"] != "")
    ens   = ens[valid]

    offsets = np.arange(seq_len, dtype=np.int64)
    frame_cache.parent.mkdir(parents=True, exist_ok=True)
    temp_cache = frame_cache.with_suffix(frame_cache.suffix + ".tmp")
    temp_cache.unlink(missing_ok=True)
    writer = None
    total_frames = 0
    try:
        for (pat, sid), grp in ens.groupby(["_pat", "_sid"], sort=False):
            sidxs  = grp["_sidx"].values.astype(np.int64)
            probs  = grp["pred_prob_fog"].values.astype(np.float32)
            fogrs  = grp["_fogr"].values.astype(np.float32)
            raw_starts = grp["_start"].values.astype(np.int64)
            # Use actual frame starts if available; fall back to session_idx * stride for old zarrs
            starts = raw_starts if has_start_frame and (raw_starts >= 0).all() else sidxs * stride
            n_frames = int(starts.max()) + seq_len

            frame_mat = (starts[:, None] + offsets[None, :]).flatten()  # (K*seq_len,)
            prob_vals = np.repeat(probs, seq_len).astype(np.float64)
            fogr_vals = np.repeat(fogrs, seq_len).astype(np.float64)

            prob_sum = np.bincount(frame_mat, weights=prob_vals, minlength=n_frames)
            fogr_sum = np.bincount(frame_mat, weights=fogr_vals, minlength=n_frames)
            count    = np.bincount(frame_mat, minlength=n_frames)

            if valid_masks_all is not None:
                zpos = grp["_zpos"].values
                valid_sum = np.bincount(
                    frame_mat,
                    weights=valid_masks_all[zpos, :].astype(np.float64).flatten(),
                    minlength=n_frames,
                )

            if has_labels:
                zpos = grp["_zpos"].values  # (K,) zarr positions
                # Binarize: label > 0 → fog (matches fog_ratio computation in steps.py)
                patch_labs = (labels_all[zpos, :] > 0).astype(np.float64)  # (K, seq_len)
                native_vals = patch_labs.flatten()
                native_sum  = np.bincount(frame_mat, weights=native_vals, minlength=n_frames)

            covered = count > 0
            if valid_masks_all is not None:
                covered &= valid_sum > 0
            f_idxs = np.where(covered)[0]
            if eligible_intervals is not None:
                intervals = eligible_intervals.get(str(sid))
                if intervals is None:
                    continue
                eligible = np.zeros(len(f_idxs), dtype=bool)
                for start, end in intervals:
                    eligible |= (f_idxs >= start) & (f_idxs < end)
                f_idxs = f_idxs[eligible]
            n = len(f_idxs)
            if n == 0:
                continue

            part = {
                "patient_id": np.full(n, pat, dtype=object),
                "session_id": np.full(n, sid, dtype=object),
                "abs_frame": f_idxs.astype(np.int32),
                "pred_prob_fog": (prob_sum[f_idxs] / count[f_idxs]).astype(np.float32),
                "fog_ratio": (fogr_sum[f_idxs] / count[f_idxs]).astype(np.float32),
            }
            if has_labels:
                # Round majority vote: frame is FoG if majority of overlapping patch frames say so
                native_frac = native_sum[f_idxs] / count[f_idxs]
                part["native_label"] = np.round(native_frac).astype(np.int8)

            table = pa.Table.from_pydict(part)
            if writer is None:
                writer = pq.ParquetWriter(temp_cache, table.schema)
            writer.write_table(table)
            total_frames += n
    except BaseException:
        if writer is not None:
            writer.close()
        temp_cache.unlink(missing_ok=True)
        raise
    else:
        if writer is not None:
            writer.close()

    if writer is None:
        return pd.DataFrame(columns=["patient_id","session_id","abs_frame",
                                     "pred_prob_fog","fog_ratio","native_label"])

    temp_cache.replace(frame_cache)
    logger.info(f"Frame df: {total_frames} frames → {frame_cache}")
    return _read_frame_cache(frame_cache)


def _read_frame_cache(path: Path) -> pd.DataFrame:
    """Read a frame cache while keeping repeated cohort IDs compact in memory."""
    frame = pd.read_parquet(path)
    for column in ("patient_id", "session_id"):
        if column in frame:
            frame[column] = frame[column].astype("category")
    return frame


def _daily_living_eligible_intervals() -> dict[str, np.ndarray]:
    """Load public walking/standing intervals used by the manuscript evaluation."""
    path = REPO_ROOT / "data" / "raw" / "fogathome_dailyliving" / "activity.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing daily-living Activity sidecar: {path}")
    activity = pd.read_parquet(
        path, columns=["session_id", "start_frame", "end_frame", "Activity"]
    )
    activity = activity[activity["Activity"].isin({1, 4})]
    return {
        str(session_id): rows[["start_frame", "end_frame"]].to_numpy(dtype=np.int64)
        for session_id, rows in activity.groupby("session_id", sort=False)
    }


# ── Context-level ensemble helpers ───────────────────────────────────────────
def build_context_ensemble_frame_df(
    contexts: tuple, model_type: str, dataset: str, cache_dir: Path
) -> pd.DataFrame:
    """Inner-join per-context frame parquets and average pred_prob_fog.

    fog_ratio and native_label taken from the most granular available context
    (SC > MC > LC) for consistent classification threshold labelling.
    Returns empty DataFrame if any required frame parquet is missing.
    """
    ctx_str = "+".join(contexts)
    ens_cache = cache_dir / f"{dataset}_{ctx_str}_{model_type}_{CACHE_FORMAT}_frames.parquet"
    if ens_cache.exists():
        try:
            return pd.read_parquet(ens_cache)
        except Exception:
            ens_cache.unlink()

    ref_ctx = next((c for c in ("sc", "mc", "lc") if c in contexts), contexts[0])
    join_cols = ["patient_id", "session_id", "abs_frame"]

    merged = None
    for ctx in contexts:
        fp = cache_dir / f"{dataset}_{ctx}_{model_type}_{CACHE_FORMAT}_frames.parquet"
        if not fp.exists():
            logger.warning(f"Missing frame parquet for {dataset}_{ctx}_{model_type}; ensemble skipped")
            return pd.DataFrame()
        df = pd.read_parquet(fp)
        if "native_label" not in df.columns:
            logger.warning(f"No native_label in {fp}; ensemble skipped")
            return pd.DataFrame()

        df = df[join_cols + ["pred_prob_fog", "fog_ratio", "native_label"]].rename(
            columns={"pred_prob_fog": f"score_{ctx}", "fog_ratio": f"fogr_{ctx}"}
        )
        if ctx != ref_ctx:
            df = df.drop(columns=["native_label"])

        merged = df if merged is None else merged.merge(df, on=join_cols, how="inner")

    if merged is None or merged.empty:
        return pd.DataFrame()

    score_cols = [f"score_{ctx}" for ctx in contexts]
    merged["pred_prob_fog"] = merged[score_cols].mean(axis=1)
    merged["fog_ratio"] = merged[f"fogr_{ref_ctx}"]

    result = merged[join_cols + ["pred_prob_fog", "fog_ratio", "native_label"]].reset_index(drop=True)
    result.to_parquet(ens_cache, index=False)
    logger.info(f"Context ensemble frame df ({ctx_str}): {len(result)} frames → {ens_cache.name}")
    return result


def eval_context_ensemble(contexts: tuple, model_type: str, dataset: str, cache_dir: Path) -> list:
    """Evaluate a context-level ensemble on classification and segmentation tasks."""
    ctx_str = "+".join(contexts)
    logger.info(f"  Context ensemble {ctx_str}/{model_type} on {dataset} …")
    frame_df = build_context_ensemble_frame_df(contexts, model_type, dataset, cache_dir)
    if frame_df.empty:
        return []

    results = []

    # Classification: fog_ratio thresholds at frame level (ref context's fog_ratio as label)
    for thr in THRESHOLDS:
        y_true = (frame_df["fog_ratio"] > 0 if thr == 0.0 else frame_df["fog_ratio"] >= thr).astype(int).values
        y_score = frame_df["pred_prob_fog"].values
        prevalence = float(y_true.mean())
        dec_thr = resolve_threshold(y_true, y_score, ctx_str, model_type, cache_dir)
        m = compute_metrics(y_true, y_score, prevalence, threshold=dec_thr)
        results.append({
            "dataset": dataset, "context": ctx_str, "model_type": model_type,
            "threshold_pct": int(thr * 100), "level": "classification",
            "n_samples": len(y_true), "prevalence": round(prevalence, 4), **m,
        })

    # Segmentation: native binary label
    y_true = frame_df["native_label"].values
    y_score = frame_df["pred_prob_fog"].values
    prevalence = float(y_true.mean())
    dec_thr = resolve_threshold(y_true, y_score, ctx_str, model_type, cache_dir)
    m = compute_metrics(y_true, y_score, prevalence, threshold=dec_thr)
    results.append({
        "dataset": dataset, "context": ctx_str, "model_type": model_type,
        "threshold_pct": "native", "level": "segmentation",
        "n_samples": len(y_true), "prevalence": round(prevalence, 4), **m,
    })

    return results


# ── Main evaluation loop ──────────────────────────────────────────────────────
def eval_one(dataset, context, model_type, cache_dir: Path) -> list:
    logger.info("  ensemble …")
    ensemble = build_ensemble(context, model_type, dataset, cache_dir)
    if ensemble.empty:
        logger.warning("  No predictions — skipping.")
        return []

    zarr_path = ZARR[(context, dataset)]
    fog_ratios_zarr, gidx_to_pos, *_ = load_zarr_arrays(zarr_path)

    # Attach fog_ratio per global_idx
    ensemble["fog_ratio"] = ensemble["global_idx"].map(
        lambda g: float(fog_ratios_zarr[gidx_to_pos[int(g)]]) if int(g) in gidx_to_pos else np.nan
    )
    ensemble = ensemble.dropna(subset=["fog_ratio"])

    results = []

    # ── Segment level ──────────────────────────────────────────────────────
    for thr in THRESHOLDS:
        if thr == 0.0:
            y_true = (ensemble["fog_ratio"] > 0).astype(int).values
        else:
            y_true = (ensemble["fog_ratio"] >= thr).astype(int).values
        y_score    = ensemble["pred_prob_fog"].values
        prevalence = float(y_true.mean())
        dec_thr    = resolve_threshold(y_true, y_score, context, model_type, cache_dir)
        m          = compute_metrics(y_true, y_score, prevalence, threshold=dec_thr)
        results.append({
            "dataset": dataset, "context": context, "model_type": model_type,
            "threshold_pct": int(thr * 100), "level": "classification",
            "n_samples": len(y_true), "prevalence": round(prevalence, 4), **m,
        })

    # ── Frame level ────────────────────────────────────────────────────────
    model_key  = f"{dataset}_{context}_{model_type}"
    frame_cache = cache_dir / f"{model_key}_{CACHE_FORMAT}_frames.parquet"
    try:
        logger.info("  frame aggregation …")
        eligible_intervals = _daily_living_eligible_intervals() if dataset == "dailyliving" else None
        frame_df = build_frame_df(
            ensemble, zarr_path, context, frame_cache, eligible_intervals=eligible_intervals
        )
    except Exception as e:
        logger.error(f"  Frame aggregation failed: {e}")
        frame_df = pd.DataFrame()

    nan_row = {k: float("nan") for k in ["AUC","AP","Precision","Recall","Specificity","Accuracy","F1","NormAP","Lift","Threshold"]}

    # Native frame-level eval — zarr binary annotations directly (no threshold artifact)
    if frame_df.empty or "native_label" not in frame_df.columns:
        results.append({
            "dataset": dataset, "context": context, "model_type": model_type,
            "threshold_pct": "native", "level": "segmentation",
            "n_samples": 0, "prevalence": float("nan"), **nan_row,
        })
    else:
        y_true  = frame_df["native_label"].values
        y_score = frame_df["pred_prob_fog"].values
        prevalence = float(y_true.mean())
        dec_thr = resolve_threshold(y_true, y_score, context, model_type, cache_dir)
        m = compute_metrics(y_true, y_score, prevalence, threshold=dec_thr)
        results.append({
            "dataset": dataset, "context": context, "model_type": model_type,
            "threshold_pct": "native", "level": "segmentation",
            "n_samples": len(y_true), "prevalence": round(prevalence, 4), **m,
        })

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",    default="logs/comprehensive_eval.csv")
    parser.add_argument("--cache-dir", default="logs/comprehensive_eval_cache")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["fogathome", "tdcsfog", "dailyliving", "stanford"],
        choices=["kaggle", "fogathome", "tdcsfog", "dailyliving", "stanford"],
    )
    parser.add_argument("--contexts",  nargs="+", default=["lc", "mc", "sc"])
    parser.add_argument("--models",    nargs="+", default=["probe", "finetune", "supervised"])
    parser.add_argument("--threshold-source", default="test_youden",
                        choices=["fixed_035", "test_youden", "test_pr11", "defog_val_pr11"],
                        help="Decision-threshold protocol for F1/Recall/Precision/Specificity. "
                             "defog_val_pr11 = held-out DeFOG PR-(1,1), applied to external sets "
                             "(no test peeking). Requires kaggle frame parquets to exist first.")
    parser.add_argument("--no-context-ensembles", action="store_true",
                        help="Skip the cross-context ensemble pass (mc+sc, lc+mc, …). "
                             "Useful when only single-context results are needed; avoids the "
                             "slow per-combo frame aggregation on large datasets.")
    args = parser.parse_args()

    global THRESHOLD_SOURCE
    THRESHOLD_SOURCE = args.threshold_source
    logger.info(f"Threshold source: {THRESHOLD_SOURCE}")

    cache_dir   = Path(args.cache_dir)
    all_results = []
    combos = [(d, c, m) for d in args.datasets for c in args.contexts for m in args.models]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, (dataset, context, model_type) in enumerate(combos):
        logger.info(f"[{i+1}/{len(combos)}] {dataset}/{context}/{model_type}")
        try:
            rows = eval_one(dataset, context, model_type, cache_dir)
            all_results.extend(rows)
            logger.info(f"  → {len(rows)} result rows")
        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)

        # Progressive save after each combo
        if all_results:
            pd.DataFrame(all_results).to_csv(output_path, index=False)

    # ── Context-level ensembles (require individual frame parquets above) ──────
    ens_combos = [] if args.no_context_ensembles else [
        (d, ctxs, m)
        for d in args.datasets
        for ctxs in CONTEXT_ENSEMBLES
        for m in args.models
    ]
    for i, (dataset, contexts, model_type) in enumerate(ens_combos):
        ctx_str = "+".join(contexts)
        logger.info(f"[ens {i+1}/{len(ens_combos)}] {dataset}/{ctx_str}/{model_type}")
        try:
            rows = eval_context_ensemble(contexts, model_type, dataset, cache_dir)
            all_results.extend(rows)
            logger.info(f"  → {len(rows)} result rows")
        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)

        if all_results:
            pd.DataFrame(all_results).to_csv(output_path, index=False)

    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows → {args.output}")


if __name__ == "__main__":
    main()

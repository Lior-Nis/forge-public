"""Compute pooled Average Precision across all 5 defog fogstrat folds.

Pooled evaluation treats all 5 test sets as one combined set: since each
defog patient appears in exactly one test fold, pooling gives per-sample
predictions across all patients. Computing AP on this pooled set reduces
variance and gives a single more reliable estimate compared to mean ± std
over fold-level APs (which have high variance with only 5-6 test patients
per fold).

Two main modes:

  1. From WandB prediction tables (fast, no GPU needed):
       python scripts/pooled_ap_eval.py --wandb-project fog-classification --wandb-prefix kfold_defog_fogstrat

  2. From checkpoint files (re-runs inference, needs GPU):
       # Single checkpoint reused for all folds (e.g. one model, different split configs):
       python scripts/pooled_ap_eval.py \\
           experiment=classification/spectral_patch_simclr_finetune_defog \\
           --checkpoint checkpoints/classification/my-run/last.ckpt \\
           --splits-dir configs/data/splits/kaggle_labeled

       # Five separate fold checkpoints:
       python scripts/pooled_ap_eval.py \\
           experiment=classification/spectral_patch_simclr_finetune_defog \\
           --checkpoints \\
             checkpoints/fold0/last.ckpt \\
             checkpoints/fold1/last.ckpt \\
             checkpoints/fold2/last.ckpt \\
             checkpoints/fold3/last.ckpt \\
             checkpoints/fold4/last.ckpt

The inference modes use Hydra for config; pass the same experiment= and any
overrides you used during training so the model architecture matches the
checkpoint.

Note: inference mode does NOT launch a WandB run (logger is set to null).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WandB path (no Hydra / GPU required)
# ---------------------------------------------------------------------------

def fetch_wandb_predictions(project: str, prefix: str = None, n_folds: int = 5,
                             stage: str = "test",
                             run_names: list[str] | None = None) -> list[dict] | None:
    """Download patch-level prediction tables from WandB runs.

    Returns a list of dicts with keys: fold, label, prob_class_1, patient_id.
    Returns None if wandb is unavailable or no tables found.

    Can query runs by:
      - prefix + fold index (original mode): looks up config.global.experiment_name={prefix}{fold}
      - explicit run_names list: looks up runs by display name (e.g. "frosty-snowflake-362")
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        return None

    import json
    import pandas as pd

    api = wandb.Api()
    table_key = f"patch_analysis/{stage}_patch_details"
    fold_dfs = []

    def _fetch_table_from_run(run, fold: int) -> "pd.DataFrame | None":
        """Try to get prediction table from a WandB run (summary or artifact)."""
        df = None
        # Method 1: table stored in summary as artifact reference (table-file type)
        # The summary dict contains {'_type': 'table-file', 'path': 'media/table/...', ...}
        # Download the file directly using run.file(path) — much faster than iterating artifacts.
        try:
            raw_entry = run.summary._json_dict.get(table_key)
        except Exception:
            raw_entry = None

        if raw_entry and isinstance(raw_entry, dict) and raw_entry.get("_type") == "table-file":
            file_path = raw_entry.get("path")
            if file_path:
                try:
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        run.file(file_path).download(root=tmpdir, replace=True)
                        local_path = os.path.join(tmpdir, file_path)
                        if os.path.exists(local_path):
                            with open(local_path) as fp:
                                tbl_data = json.load(fp)
                            if "data" in tbl_data and "columns" in tbl_data:
                                df = pd.DataFrame(tbl_data["data"], columns=tbl_data["columns"])
                                logger.debug(f"    Downloaded table via run.file({file_path})")
                except Exception as e:
                    logger.warning(f"    run.file download failed: {e}")

        # Method 2: logged as artifact
        if df is None:
            for artifact in run.logged_artifacts():
                if "patch_details" in artifact.name and stage in artifact.name:
                    try:
                        artifact_dir = artifact.download()
                        for fname in os.listdir(artifact_dir):
                            if fname.endswith(".json"):
                                with open(os.path.join(artifact_dir, fname)) as f:
                                    data = json.load(f)
                                df = pd.DataFrame(data["data"], columns=data["columns"])
                                break
                    except Exception as e:
                        logger.warning(f"    Artifact download failed: {e}")
                    if df is not None:
                        break

        if df is None:
            logger.warning(f"  Fold {fold}: no prediction table found in run {run.id}")
            return None

        if "label" not in df.columns or "prob_class_1" not in df.columns:
            logger.warning(f"  Fold {fold}: table missing required columns "
                           f"(have: {list(df.columns)})")
            return None

        df = df.copy()
        df["fold"] = fold
        logger.info(f"    {len(df)} patches, "
                    f"{df['patient_id'].nunique() if 'patient_id' in df.columns else '?'} patients")
        return df

    if run_names is not None:
        # Query by explicit display names, one per fold
        for fold, run_name in enumerate(run_names):
            matching = api.runs(
                project,
                filters={"display_name": run_name},
                order="-created_at",
            )
            matching = list(matching)
            if not matching:
                logger.warning(f"No run found with display name: {run_name} (fold {fold})")
                continue
            run = matching[0]
            logger.info(f"  Fold {fold}: {run.name} ({run.id})")
            df = _fetch_table_from_run(run, fold)
            if df is not None:
                fold_dfs.append(df)
        if not fold_dfs:
            return None
        return pd.concat(fold_dfs, ignore_index=True)

    # Original mode: query by config.global.experiment_name prefix
    for fold in range(n_folds):
        name_pattern = f"{prefix}{fold}"
        matching = api.runs(
            project,
            filters={
                "config.global.experiment_name": name_pattern,
                "state": "finished",
            },
            order="-created_at",
        )
        matching = list(matching)
        if not matching:
            logger.warning(f"No finished run for fold {fold} (name: {name_pattern})")
            continue

        run = matching[0]
        logger.info(f"  Fold {fold}: {run.name} ({run.id})")

        df = _fetch_table_from_run(run, fold)
        if df is not None:
            fold_dfs.append(df)

    if not fold_dfs:
        return None

    all_preds = pd.concat(fold_dfs, ignore_index=True)
    return all_preds


# ---------------------------------------------------------------------------
# Inference path (Hydra + checkpoint)
# ---------------------------------------------------------------------------

def run_fold_inference(hydra_config, checkpoint_path: str, fold_idx: int,
                       splits_dir: str) -> "pd.DataFrame":
    """Load a checkpoint and run test inference for one fold.

    Overrides data/splits to the fold's kfold_defog_fogstrat<fold_idx> config.
    Returns a DataFrame with columns: prob_class_1, label, patient_id, session_id.
    """
    import torch
    import torch.nn.functional as F
    import pandas as pd
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    from data.datamodule.datamodule import FOGDataModule
    from data.datamodule import collate
    from pipeline.classification import ClassificationPipeline
    from utils.config_loaders import load_config
    from utils.paths import normalize_data_paths

    # Override splits to this fold's split config
    splits_key = f"kaggle_labeled/kfold_defog_fogstrat{fold_idx}"
    OmegaConf.update(hydra_config, "data.splits", OmegaConf.load(
        os.path.join(splits_dir, f"kfold_defog_fogstrat{fold_idx}.yaml")
    ), merge=True)

    normalize_data_paths(hydra_config.data.paths)
    config = load_config(hydra_config)

    logger.info(f"  Fold {fold_idx}: loading checkpoint {checkpoint_path}")
    model = ClassificationPipeline(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k not in model_state or v.shape == model_state[k].shape
    }
    skipped = set(state_dict.keys()) - set(filtered.keys())
    if skipped:
        logger.info(f"    Skipped {len(skipped)} shape-mismatched keys")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected:
        logger.warning(f"    Unexpected keys: {unexpected[:5]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Setup datamodule
    data_module = FOGDataModule(data_cfg=config.data, task_type="classification")
    data_module.setup(stage="test")

    # Inject normalization stats if available
    data_module.setup(stage="fit")
    stats = getattr(data_module, "train_normalization_stats", None)
    if stats is not None and model.preprocessors is not None:
        try:
            from model.preprocessors import PatientNormalizationPreprocessor
            for module in model.preprocessors.preprocessors:
                if isinstance(module, PatientNormalizationPreprocessor):
                    module.set_stats(stats)
                    logger.info("    Injected normalization stats")
                    break
        except Exception as e:
            logger.warning(f"    Could not inject normalization stats: {e}")

    data_module.setup(stage="test")
    dl_kwargs = dict(
        collate_fn=collate.classification,
        batch_size=config.data.dataloader.batch_size,
        num_workers=config.data.dataloader.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    test_loader = DataLoader(data_module.test_dataset, **dl_kwargs)
    logger.info(f"    Test dataset: {len(data_module.test_dataset)} patches")

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            labels = batch["patch_y"]      # [B] patch-level label
            valid_masks = batch["valid_mask"]  # [B] or [B, T]
            metadata = batch["metadata"]

            logits = model(x)

            # Handle both patch-level [B, C] and sequence [B, T, C]
            if logits.dim() == 3:
                # Sequence head — use mean over valid timesteps for patch-level score
                probs_seq = F.softmax(logits, dim=-1)[..., 1]  # [B, T]
                if valid_masks.dim() > 1:
                    mask = valid_masks.bool()
                    probs = (probs_seq * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    probs = probs_seq.mean(dim=1)
                # labels are [B, T] for sequence head
                if labels.dim() > 1:
                    if valid_masks.dim() > 1:
                        labels_patch = labels.float()
                        labels_patch = (labels_patch * valid_masks.float()).sum(dim=1) / valid_masks.float().sum(dim=1).clamp(min=1)
                        labels_patch = labels_patch.round().long()
                    else:
                        labels_patch = labels[:, 0]
                else:
                    labels_patch = labels
            else:
                probs = F.softmax(logits, dim=-1)[:, 1]  # [B]
                if labels.dim() > 1:
                    labels_patch = labels[:, 0]
                else:
                    labels_patch = labels

            probs = probs.cpu().numpy()
            labels_np = labels_patch.cpu().numpy()

            for i, meta in enumerate(metadata):
                rows.append({
                    "prob_class_1": float(probs[i]),
                    "label": int(labels_np[i]),
                    "patient_id": meta.get("patient_id", "unknown"),
                    "session_id": meta.get("session_id", "unknown"),
                    "fold": fold_idx,
                })

    df = pd.DataFrame(rows)
    logger.info(f"    Collected {len(df)} patches, "
                f"{df['patient_id'].nunique()} patients, "
                f"FOG rate: {df['label'].mean()*100:.1f}%")
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(all_preds: "pd.DataFrame") -> dict:
    """Compute pooled AP and related metrics."""
    from sklearn.metrics import average_precision_score, f1_score

    y_true = all_preds["label"].values.astype(int)
    y_score = all_preds["prob_class_1"].values
    y_pred = (y_score >= 0.5).astype(int)

    pooled_ap = average_precision_score(y_true, y_score)
    pooled_f1 = f1_score(y_true, y_pred, zero_division=0)
    prevalence = y_true.mean()

    n_patients = all_preds["patient_id"].nunique() if "patient_id" in all_preds.columns else "?"
    n_patches = len(all_preds)

    return dict(
        pooled_ap=pooled_ap,
        pooled_f1=pooled_f1,
        prevalence=prevalence,
        n_patients=n_patients,
        n_patches=n_patches,
    )


def compute_per_fold_ap(all_preds: "pd.DataFrame") -> dict:
    """Compute per-fold AP for comparison with pooled."""
    from sklearn.metrics import average_precision_score

    per_fold = {}
    for fold, grp in all_preds.groupby("fold"):
        y_true = grp["label"].values.astype(int)
        y_score = grp["prob_class_1"].values
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            per_fold[fold] = float("nan")
        else:
            per_fold[fold] = average_precision_score(y_true, y_score)
    return per_fold


def print_results(per_fold_ap: dict, pooled: dict) -> None:
    fold_vals = np.array([v for v in per_fold_ap.values() if not np.isnan(v)])

    print()
    print("=" * 60)
    print("  POOLED AVERAGE PRECISION — defog fogstrat 5-fold CV")
    print("=" * 60)

    print("\nPer-fold test AP:")
    for fold, ap in sorted(per_fold_ap.items()):
        marker = "  (NaN: constant labels)" if np.isnan(ap) else ""
        print(f"  Fold {fold}: {ap:.4f}{marker}")

    if len(fold_vals) > 0:
        print(f"\nMean ± std across folds: {fold_vals.mean():.4f} ± {fold_vals.std(ddof=1):.4f}")
        print(f"Range: [{fold_vals.min():.4f}, {fold_vals.max():.4f}]")

    print()
    print("Pooled evaluation (all test folds combined):")
    print(f"  Patients:       {pooled['n_patients']}")
    print(f"  Patches:        {pooled['n_patches']:,}")
    print(f"  FOG prevalence: {pooled['prevalence']*100:.1f}%")
    print()
    print(f"  Pooled AP:  {pooled['pooled_ap']:.4f}")
    print(f"  Pooled F1:  {pooled['pooled_f1']:.4f}  (threshold=0.5)")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Parse non-Hydra arguments before Hydra swallows sys.argv
    # We use a pre-parse step to separate our args from Hydra overrides.
    import argparse

    # Split sys.argv into our known args and Hydra overrides
    # Our flags are prefixed with '--'; Hydra overrides use key=value (no --)
    our_argv = []
    hydra_argv = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--"):
            our_argv.append(arg)
            # Consume next token if it looks like a value (not --flag or key=val)
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") and "=" not in sys.argv[i + 1]:
                # Could be a value for this flag — but nargs='+' flags are tricky.
                # We'll just put all --flag tokens in our_argv and rebuild later.
                pass
        else:
            hydra_argv.append(arg)
        i += 1

    # Use a proper pre-parser to find our flags
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--checkpoint", type=str, default=None)
    pre_parser.add_argument("--checkpoints", nargs="+", default=None)
    pre_parser.add_argument("--splits-dir", type=str, default="configs/data/splits/kaggle_labeled")
    pre_parser.add_argument("--fold-prefix", type=str, default="kfold_defog_fogstrat")
    pre_parser.add_argument("--n-folds", type=int, default=5)
    pre_parser.add_argument("--wandb-project", type=str, default=None)
    pre_parser.add_argument("--wandb-prefix", type=str, default="kfold_defog_fogstrat")
    pre_parser.add_argument("--wandb-runs", nargs="+", default=None,
                            help="Explicit WandB run display names (one per fold, e.g. "
                                 "frosty-snowflake-362 different-universe-363 ...). "
                                 "Overrides --wandb-prefix lookup.")
    pre_parser.add_argument("--wandb-metric-stage", type=str, default="test")
    pre_parser.add_argument("--output-csv", type=str, default=None,
                            help="Save pooled per-patch predictions to this CSV path")
    pre_parser.add_argument("--no-gpu", action="store_true",
                            help="Force CPU inference even if GPU is available")

    known, remaining = pre_parser.parse_known_args()

    # Determine mode
    use_wandb = known.wandb_project is not None
    use_inference = (known.checkpoint is not None or known.checkpoints is not None)

    if not use_wandb and not use_inference:
        print(__doc__)
        pre_parser.print_help()
        sys.exit(1)

    if use_wandb and use_inference:
        print("ERROR: Specify either --wandb-project OR --checkpoint/--checkpoints, not both.")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # -----------------------------------------------------------------------
    # WandB mode — no Hydra needed
    # -----------------------------------------------------------------------
    if use_wandb:
        import pandas as pd

        if known.wandb_runs:
            logger.info(f"Fetching predictions from WandB: project={known.wandb_project}, "
                        f"runs={known.wandb_runs}")
            all_preds = fetch_wandb_predictions(
                known.wandb_project, run_names=known.wandb_runs,
                stage=known.wandb_metric_stage,
            )
        else:
            logger.info(f"Fetching predictions from WandB: project={known.wandb_project}, "
                        f"prefix={known.wandb_prefix}")
            all_preds = fetch_wandb_predictions(
                known.wandb_project, known.wandb_prefix,
                n_folds=known.n_folds, stage=known.wandb_metric_stage,
            )

        if all_preds is None or len(all_preds) == 0:
            logger.error(
                "No prediction tables found in WandB runs. "
                "Ensure the runs logged patch_analysis tables (they are logged during test). "
                "Alternative: use --checkpoint to re-run inference."
            )
            sys.exit(1)

        # Sanity check: no patient in multiple folds
        if "patient_id" in all_preds.columns:
            patient_fold_counts = all_preds.groupby("patient_id")["fold"].nunique()
            leakers = patient_fold_counts[patient_fold_counts > 1]
            if not leakers.empty:
                logger.warning(
                    f"{len(leakers)} patients appear in multiple folds — "
                    f"split may be leaking: {leakers.index.tolist()}"
                )

        per_fold_ap = compute_per_fold_ap(all_preds)
        pooled = compute_metrics(all_preds)
        print_results(per_fold_ap, pooled)

        if known.output_csv:
            all_preds.to_csv(known.output_csv, index=False)
            logger.info(f"Predictions saved to {known.output_csv}")

        return

    # -----------------------------------------------------------------------
    # Inference mode — requires Hydra config matching the training setup
    # -----------------------------------------------------------------------
    if known.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # We need Hydra to build the config. Reconstruct sys.argv with only
    # Hydra-compatible overrides (key=value style) so Hydra doesn't choke on
    # our --checkpoint style flags.
    sys.argv = [sys.argv[0]] + remaining  # remaining = Hydra overrides

    import hydra
    from omegaconf import DictConfig, OmegaConf
    import pandas as pd

    # Collect per-fold predictions
    all_fold_dfs: list[pd.DataFrame] = []

    if known.checkpoints is not None:
        if len(known.checkpoints) != known.n_folds:
            logger.error(
                f"Expected {known.n_folds} checkpoints, got {len(known.checkpoints)}"
            )
            sys.exit(1)
        checkpoint_per_fold = {i: p for i, p in enumerate(known.checkpoints)}
    else:
        checkpoint_per_fold = {i: known.checkpoint for i in range(known.n_folds)}

    # We need to initialise Hydra once and then reuse the config.
    # Hydra is not designed for multi-call, so we compose manually.
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str(Path(__file__).parent.parent / "configs")

    GlobalHydra.instance().clear()

    # Build base hydra config from remaining overrides
    # Silence WandB for inference
    override_list = list(remaining) + ["train.logger=null"]

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        base_cfg = compose(config_name="config", overrides=override_list)

    for fold_idx in range(known.n_folds):
        ckpt = checkpoint_per_fold[fold_idx]

        # Re-compose with fold-specific split override
        GlobalHydra.instance().clear()
        split_name = f"kaggle_labeled/{known.fold_prefix}{fold_idx}"
        fold_overrides = list(remaining) + [
            f"data/splits={split_name}",
            "train.logger=null",
        ]
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            fold_cfg = compose(config_name="config", overrides=fold_overrides)

        logger.info(f"\nFold {fold_idx}: split={split_name}, checkpoint={ckpt}")
        try:
            fold_df = run_fold_inference(fold_cfg, ckpt, fold_idx, known.splits_dir)
            all_fold_dfs.append(fold_df)
        except Exception as e:
            logger.error(f"Fold {fold_idx} inference failed: {e}", exc_info=True)

    if not all_fold_dfs:
        logger.error("All folds failed. Check checkpoint paths and experiment config.")
        sys.exit(1)

    all_preds = pd.concat(all_fold_dfs, ignore_index=True)

    # Sanity check
    if "patient_id" in all_preds.columns:
        patient_fold_counts = all_preds.groupby("patient_id")["fold"].nunique()
        leakers = patient_fold_counts[patient_fold_counts > 1]
        if not leakers.empty:
            logger.warning(
                f"{len(leakers)} patients appear in multiple folds — "
                f"split may be leaking: {leakers.index.tolist()}"
            )

    per_fold_ap = compute_per_fold_ap(all_preds)
    pooled = compute_metrics(all_preds)
    print_results(per_fold_ap, pooled)

    if known.output_csv:
        all_preds.to_csv(known.output_csv, index=False)
        logger.info(f"Per-patch predictions saved to {known.output_csv}")


if __name__ == "__main__":
    main()

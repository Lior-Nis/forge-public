"""
Extract MAE backbone embeddings for embedding analysis.

Loads a probe checkpoint (fold 0 by default — backbone weights are identical across
folds), runs inference on all defog test patients (all 5 folds) and all FogAtHome
patients, and saves:
  - embeddings.npy  [N_patches, embed_dim]
  - metadata.csv    patient_id, true_label, dataset, fold

The backbone outputs [B, nH*nW, D] patch tokens. We mean-pool over all N tokens
to get one 512-d vector per 2-second patch — the sequence-level representation
used for all downstream analyses.

Usage:
    uv run python scripts/extract_embeddings.py \
        --probe-pattern "checkpoints/classification/probe_vit12_ep4_fold{fold}/last.ckpt" \
        --model-name vit12_ep4 \
        --n-folds 5 \
        --output-dir logs/embeddings

    # Skip FogAtHome:
    uv run python scripts/extract_embeddings.py ... --no-fogathome
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.set_float32_matmul_precision("high")

FOGATHOME_PATIENTS = [
    "a00001", "a00002", "a00004", "a00006", "a00007", "a00008",
    "a00009", "a00010", "a00011", "a00012", "a00013", "a00014",
]

logger = logging.getLogger(__name__)


def _load_pipeline(ckpt_path: str, batch_size: int):
    """Load ClassificationPipeline from a probe checkpoint."""
    from pipeline.classification import ClassificationPipeline
    from utils.paths import normalize_data_paths

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["hyper_parameters"]["config"]
    normalize_data_paths(config.data.paths)

    model = ClassificationPipeline(config)

    # Resize patient normalizer buffers if shape mismatch
    state_dict = ckpt["state_dict"]
    for key in [
        "preprocessors.preprocessors.3.normalizer.mean",
        "preprocessors.preprocessors.3.normalizer.stdev",
    ]:
        if key in state_dict:
            saved_shape = state_dict[key].shape
            parts = key.split(".")
            mod = model
            for part in parts[:-1]:
                mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
            current = getattr(mod, parts[-1])
            if current.shape != saved_shape:
                setattr(mod, parts[-1], torch.zeros(saved_shape))

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config


@torch.no_grad()
def _extract_from_loader(model, loader, device, dataset_label: str, fold: int) -> tuple[np.ndarray, list[dict]]:
    """Run forward pass up to backbone, mean-pool tokens → [N, D]."""
    all_embeddings = []
    all_meta = []

    for batch in loader:
        x = batch["x"].to(device)
        patch_y = batch["patch_y"]
        meta = batch["metadata"]

        # Replicate classification forward up to backbone (no head)
        if model.preprocessors is not None:
            x = model.preprocessors(x)
        x = model.transform(x)
        tokens = model.backbone(x)   # [B, N, D]

        # Global mean pool: [B, N, D] → [B, D]
        emb = tokens.mean(dim=1).cpu().numpy()
        labels = patch_y.numpy() if isinstance(patch_y, torch.Tensor) else np.array(patch_y)

        all_embeddings.append(emb)
        for i, m in enumerate(meta):
            all_meta.append({
                "patient_id": m.get("patient_id", "unknown"),
                "session_id": m.get("session_id", "unknown"),
                "true_label": int(labels[i]),
                "dataset": dataset_label,
                "fold": fold,
            })

    return np.concatenate(all_embeddings, axis=0), all_meta


def extract_defog(ckpt_pattern: str, n_folds: int, batch_size: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract embeddings for defog test splits across all folds."""
    from data.datamodule.datamodule import FOGDataModule

    all_embs, all_meta = [], []
    used_patients: set[str] = set()

    for fold in range(n_folds):
        ckpt_path = ckpt_pattern.format(fold=fold)
        if not Path(ckpt_path).exists():
            logger.warning(f"Missing fold {fold}: {ckpt_path}")
            continue

        logger.info(f"Defog fold {fold}: {ckpt_path}")
        model, config = _load_pipeline(ckpt_path, batch_size)

        data_cfg = config.data.model_copy(update={
            "dataloader": config.data.dataloader.model_copy(update={"batch_size": batch_size}),
        })
        dm = FOGDataModule(data_cfg=data_cfg, task_type=config.train.pipeline_type)
        dm.setup("test")

        # Inject norm stats from training fold
        dm.setup("fit")
        try:
            stats = dm.train_dataset.get_normalization_stats()
            model.preprocessors.inject_stats(stats)
        except Exception:
            pass

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        embs, meta = _extract_from_loader(model, dm.test_dataloader(), device, "defog", fold)

        # Guard against double-counting patients if test splits overlap
        fold_patients = {m["patient_id"] for m in meta}
        if fold_patients & used_patients:
            logger.warning(f"Fold {fold} patients overlap with previous folds — skipping duplicates")
            keep = [i for i, m in enumerate(meta) if m["patient_id"] not in used_patients]
            embs = embs[keep]
            meta = [meta[i] for i in keep]

        used_patients |= {m["patient_id"] for m in meta}
        all_embs.append(embs)
        all_meta.extend(meta)
        logger.info(f"  → {len(meta)} patches from {fold_patients - used_patients | fold_patients}")

    return np.concatenate(all_embs, axis=0), pd.DataFrame(all_meta)


def extract_fogathome(ckpt_path: str, batch_size: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract embeddings for all 12 FogAtHome patients using fold-0 backbone."""
    from data.datamodule.datamodule import FOGDataModule
    from data.datamodule.config import SplitsConfig

    logger.info(f"FogAtHome: {ckpt_path}")
    model, config = _load_pipeline(ckpt_path, batch_size)

    data_cfg = config.data.model_copy(update={
        "dataloader": config.data.dataloader.model_copy(update={"batch_size": batch_size}),
        "paths": config.data.paths.model_copy(update={
            "dataset_path": "len1000_stride200_fogathome.zarr",
        }),
        "splits": SplitsConfig(train=[], val=[], test=FOGATHOME_PATIENTS),
    })

    dm = FOGDataModule(data_cfg=data_cfg, task_type=config.train.pipeline_type)
    dm.setup("test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    embs, meta = _extract_from_loader(model, dm.test_dataloader(), device, "fogathome", fold=-1)
    logger.info(f"  → {len(meta)} patches from {len(FOGATHOME_PATIENTS)} FogAtHome patients")
    return embs, pd.DataFrame(meta)


def extract_single(ckpt_path: str, patient_ids: list[str], batch_size: int,
                   dataset_label: str, dataset_path_override: str | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Extract embeddings from a single checkpoint for a given list of patients.
    Used for supervised models that aren't structured as k-fold probes.
    """
    from data.datamodule.datamodule import FOGDataModule
    from data.datamodule.config import SplitsConfig

    logger.info(f"{dataset_label}: {ckpt_path} — {len(patient_ids)} patients")
    model, config = _load_pipeline(ckpt_path, batch_size)

    path_update = {}
    if dataset_path_override:
        path_update["dataset_path"] = dataset_path_override

    data_cfg = config.data.model_copy(update={
        "dataloader": config.data.dataloader.model_copy(update={"batch_size": batch_size}),
        "paths": config.data.paths.model_copy(update=path_update) if path_update else config.data.paths,
        "splits": SplitsConfig(train=[], val=[], test=patient_ids),
    })

    dm = FOGDataModule(data_cfg=data_cfg, task_type=config.train.pipeline_type)
    dm.setup("test")
    # No stats injection: patient normalizer weights are already in the checkpoint;
    # unseen patients (FogAtHome) rely on RevIN for session-level normalisation.

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    embs, meta = _extract_from_loader(model, dm.test_dataloader(), device, dataset_label, fold=0)
    logger.info(f"  → {len(meta)} patches from {len(set(m['patient_id'] for m in meta))} patients")
    return embs, pd.DataFrame(meta)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-pattern",
                        help="K-fold mode: e.g. checkpoints/classification/probe_vit12_ep4_fold{fold}/last.ckpt")
    parser.add_argument("--single-ckpt",
                        help="Single-checkpoint mode: path to one .ckpt file (supervised / non-kfold models)")
    parser.add_argument("--defog-patients",
                        help="Single-ckpt mode: comma-separated patient IDs for defog, "
                             "or path to a text file with one ID per line. "
                             "Defaults to the 28 defog patients from the reference metadata.")
    parser.add_argument("--ref-metadata", default="logs/embeddings/vit12_ep4/metadata.csv",
                        help="Reference metadata CSV to derive defog patient list from (default: vit12_ep4)")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="logs/embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-fogathome", action="store_true")
    args = parser.parse_args()

    if not args.probe_pattern and not args.single_ckpt:
        parser.error("Provide either --probe-pattern (k-fold) or --single-ckpt (single model)")

    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_embs, all_meta = [], []

    if args.single_ckpt:
        # ── Single-checkpoint mode ──────────────────────────────────────────
        # Resolve defog patient list
        if args.defog_patients:
            if Path(args.defog_patients).exists():
                defog_ids = Path(args.defog_patients).read_text().split()
            else:
                defog_ids = [p.strip() for p in args.defog_patients.split(",")]
        else:
            ref = pd.read_csv(args.ref_metadata)
            defog_ids = sorted(ref[ref["dataset"] == "defog"]["patient_id"].unique().tolist())
            logger.info(f"Using {len(defog_ids)} defog patients from {args.ref_metadata}")

        embs, meta = extract_single(args.single_ckpt, defog_ids, args.batch_size, "defog")
        all_embs.append(embs)
        all_meta.append(meta)
        logger.info(f"Defog: {len(meta)} patches from {meta['patient_id'].nunique()} patients")

        if not args.no_fogathome:
            embs_fah, meta_fah = extract_single(
                args.single_ckpt, FOGATHOME_PATIENTS, args.batch_size,
                "fogathome", dataset_path_override="len1000_stride200_fogathome.zarr",
            )
            all_embs.append(embs_fah)
            all_meta.append(meta_fah)
            logger.info(f"FogAtHome: {len(meta_fah)} patches from {meta_fah['patient_id'].nunique()} patients")

    else:
        # ── K-fold mode ─────────────────────────────────────────────────────
        embs, meta = extract_defog(args.probe_pattern, args.n_folds, args.batch_size)
        all_embs.append(embs)
        all_meta.append(meta)
        logger.info(f"Defog: {len(meta)} patches from {meta['patient_id'].nunique()} patients")

    # FogAtHome for k-fold mode
    if not args.single_ckpt and not args.no_fogathome:
        fog_ckpt = args.probe_pattern.format(fold=0)
        if Path(fog_ckpt).exists():
            embs_fah, meta_fah = extract_fogathome(fog_ckpt, args.batch_size)
            all_embs.append(embs_fah)
            all_meta.append(meta_fah)
            logger.info(f"FogAtHome: {len(meta_fah)} patches from {meta_fah['patient_id'].nunique()} patients")
        else:
            logger.warning(f"Fold-0 checkpoint not found, skipping FogAtHome: {fog_ckpt}")

    embeddings = np.concatenate(all_embs, axis=0)
    metadata = pd.concat(all_meta, ignore_index=True)

    emb_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "metadata.csv"
    np.save(emb_path, embeddings)
    metadata.to_csv(meta_path, index=False)

    logger.info(f"\nSaved {len(embeddings)} embeddings ({embeddings.shape[1]}-d) to {output_dir}")
    logger.info(f"  Defog patients:     {metadata[metadata['dataset']=='defog']['patient_id'].nunique()}")
    logger.info(f"  FogAtHome patients: {metadata[metadata['dataset']=='fogathome']['patient_id'].nunique()}")
    logger.info(f"  FOG-positive rate:  {metadata['true_label'].mean():.3f}")


if __name__ == "__main__":
    main()

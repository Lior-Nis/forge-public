"""
Precompute frozen backbone embeddings for a MAE checkpoint.

Key design: load ALL data into RAM first, then run GPU inference in large
batches with zero I/O during compute — maximises GPU utilisation.

Model is loaded ONCE per invocation; all folds share it.
Per-fold normalization stats are applied on-the-fly.

Usage:
    # Defog splits (all 5 folds at once):
    uv run python scripts/precompute_mae_embeddings.py \
        --mae-ckpt checkpoints/mae/worldly-moon-229/eepoch=00.ckpt \
        --ref-ckpt checkpoints/classification/probe_vit12_ep6_fold0/last.ckpt \
        --output-dir checkpoints/embeddings/vit12_ep6 \
        --n-folds 5 --gpu-batch-size 1024

    # FogAtHome (all 5 folds' norm stats applied to same 873 patches):
    uv run python scripts/precompute_mae_embeddings.py \
        --mae-ckpt checkpoints/mae/worldly-moon-229/eepoch=00.ckpt \
        --ref-ckpt checkpoints/classification/probe_vit12_ep6_fold0/last.ckpt \
        --output-dir checkpoints/embeddings/vit12_ep6 \
        --n-folds 5 --gpu-batch-size 1024 --fogathome
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)

FOGATHOME_PATIENTS = [
    "a00001", "a00002", "a00004", "a00006", "a00007", "a00008",
    "a00009", "a00010", "a00011", "a00012", "a00013", "a00014",
]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_backbone_and_transform(mae_ckpt_path: str):
    import hydra
    mae_ckpt = torch.load(mae_ckpt_path, map_location="cpu", weights_only=False)
    mae_cfg = mae_ckpt["hyper_parameters"]["config"]
    mae_sd = mae_ckpt["state_dict"]

    backbone = hydra.utils.instantiate(mae_cfg.model.backbone.model_dump())
    backbone.load_state_dict(
        {k[len("backbone."):]: v for k, v in mae_sd.items() if k.startswith("backbone.")},
        strict=True)
    backbone.eval()

    transform = hydra.utils.instantiate(mae_cfg.model.transform.model_dump())
    transform.load_state_dict(
        {k[len("transform."):]: v for k, v in mae_sd.items() if k.startswith("transform.")},
        strict=True)
    transform.eval()

    vit_depth = mae_cfg.model.backbone.vit_depth
    logger.info(f"Backbone: {type(backbone).__name__}, "
                f"{sum(p.numel() for p in backbone.parameters())/1e6:.1f}M params, "
                f"vit_depth={vit_depth}")
    return backbone, transform


def load_preprocessors(ref_ckpt_path: str):
    import hydra
    ref_ckpt = torch.load(ref_ckpt_path, map_location="cpu", weights_only=False)
    ref_cfg = ref_ckpt["hyper_parameters"]["config"]
    preprocessors = hydra.utils.instantiate(ref_cfg.model.preprocessors.model_dump())
    preprocessors.eval()
    return preprocessors, ref_cfg


# ── Data loading into RAM ─────────────────────────────────────────────────────

def load_split_to_ram(dataloader) -> Dict[str, Any]:
    """Drain dataloader into RAM tensors. Returns {x, labels, metadata}."""
    xs, labels, metadata = [], [], []
    for batch in dataloader:
        xs.append(batch["x"])
        py = batch["patch_y"]
        labels.append(py if isinstance(py, torch.Tensor) else torch.tensor(py))
        metadata.extend(batch["metadata"])
    return {
        "x": torch.cat(xs),            # [N, C, T]
        "labels": torch.cat(labels),   # [N]
        "metadata": metadata,
    }


def get_fold_datamodule(ref_cfg, fold: int, batch_size: int, fogathome: bool = False):
    from data.datamodule.datamodule import FOGDataModule
    from data.datamodule.config import SplitsConfig
    from utils.paths import normalize_data_paths
    import yaml

    normalize_data_paths(ref_cfg.data.paths)

    # Always load defog fold for normalization stats
    split_path = Path(f"configs/data/splits/kaggle_labeled/kfold_defog_fogstrat{fold}.yaml")
    with open(split_path) as f:
        raw = yaml.safe_load(f)
    defog_splits = SplitsConfig(train=raw["train"], val=raw.get("val", []), test=raw.get("test", []))

    if fogathome:
        # Norm stats from defog train fold, data from fogathome zarr
        defog_dm = FOGDataModule(
            data_cfg=ref_cfg.data.model_copy(update={"splits": defog_splits}),
            task_type=ref_cfg.train.pipeline_type)
        defog_dm.setup("fit")

        fah_cfg = ref_cfg.data.model_copy(update={
            "dataloader": ref_cfg.data.dataloader.model_copy(update={"batch_size": batch_size}),
            "paths": ref_cfg.data.paths.model_copy(update={"dataset_path": "len1000_stride200_fogathome.zarr"}),
            "splits": SplitsConfig(train=[], val=[], test=FOGATHOME_PATIENTS),
        })
        fah_dm = FOGDataModule(data_cfg=fah_cfg, task_type=ref_cfg.train.pipeline_type)
        fah_dm.setup("test")
        return defog_dm, fah_dm

    else:
        dm = FOGDataModule(
            data_cfg=ref_cfg.data.model_copy(update={
                "dataloader": ref_cfg.data.dataloader.model_copy(update={"batch_size": batch_size}),
                "splits": defog_splits,
            }),
            task_type=ref_cfg.train.pipeline_type)
        dm.setup("fit")
        dm.setup("test")
        return dm, None


# ── GPU inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(x_ram: torch.Tensor, preprocessors, transform, backbone,
                  norm_stats, device: str, gpu_batch_size: int) -> torch.Tensor:
    """
    x_ram: [N, C, T] on CPU
    Returns: [N, nW, D] on CPU
    """
    # Inject fold-specific norm stats
    if norm_stats is not None:
        for module in preprocessors.preprocessors:
            if hasattr(module, "set_stats"):
                module.set_stats(norm_stats)

    N = len(x_ram)
    all_emb = []

    for start in range(0, N, gpu_batch_size):
        x = x_ram[start:start + gpu_batch_size].to(device)
        x = preprocessors(x)
        x = transform(x)
        x = backbone(x)
        if hasattr(backbone, "_last_nH"):
            nH, nW = backbone._last_nH, backbone._last_nW
            B, Npatch, D = x.shape
            x = x.view(B, nH, nW, D).mean(dim=1)  # [B, nW, D]
        all_emb.append(x.cpu())

    return torch.cat(all_emb)  # [N, nW, D]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mae-ckpt", required=True)
    parser.add_argument("--ref-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gpu-batch-size", type=int, default=1024,
                        help="Batch size for GPU inference (data already in RAM)")
    parser.add_argument("--io-batch-size", type=int, default=256,
                        help="Batch size for zarr I/O into RAM")
    parser.add_argument("--fogathome", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check which folds need work
    suffix = "fogathome" if args.fogathome else None
    folds_needed = []
    for fold in range(args.n_folds):
        if args.fogathome:
            needed = not (out / f"fold{fold}_fogathome.pt").exists()
        else:
            needed = not all((out / f"fold{fold}_{s}.pt").exists()
                             for s in ["train", "val", "test"])
        if needed:
            folds_needed.append(fold)

    if not folds_needed:
        logger.info("All folds cached, nothing to do.")
        return

    # Load model ONCE
    backbone, transform = load_backbone_and_transform(args.mae_ckpt)
    preprocessors, ref_cfg = load_preprocessors(args.ref_ckpt)
    backbone.to(device); transform.to(device); preprocessors.to(device)

    for fold in folds_needed:
        logger.info(f"=== Fold {fold} {'fogathome' if args.fogathome else 'defog'} ===")

        defog_dm, fah_dm = get_fold_datamodule(ref_cfg, fold, args.io_batch_size, args.fogathome)
        norm_stats = getattr(defog_dm, "train_normalization_stats", None)

        if args.fogathome:
            out_path = out / f"fold{fold}_fogathome.pt"
            logger.info("  Loading FogAtHome into RAM...")
            ram = load_split_to_ram(fah_dm.test_dataloader())
            logger.info(f"  {len(ram['x'])} samples in RAM — running GPU inference...")
            emb = run_inference(ram["x"], preprocessors, transform, backbone,
                                norm_stats, device, args.gpu_batch_size)
            torch.save({"embeddings": emb, "labels": ram["labels"], "metadata": ram["metadata"]}, out_path)
            logger.info(f"  Saved {len(emb)} embeddings, pos={ram['labels'].float().mean():.3f} → {out_path}")

        else:
            loaders = {"train": defog_dm.train_dataloader(),
                       "val":   defog_dm.val_dataloader(),
                       "test":  defog_dm.test_dataloader()}
            for split, loader in loaders.items():
                out_path = out / f"fold{fold}_{split}.pt"
                if out_path.exists():
                    logger.info(f"  {split}: cached")
                    continue
                logger.info(f"  {split}: loading {len(loader.dataset)} samples into RAM...")
                ram = load_split_to_ram(loader)
                logger.info(f"  {split}: running GPU inference (batch_size={args.gpu_batch_size})...")
                emb = run_inference(ram["x"], preprocessors, transform, backbone,
                                    norm_stats, device, args.gpu_batch_size)
                torch.save({"embeddings": emb, "labels": ram["labels"], "metadata": ram["metadata"]}, out_path)
                logger.info(f"  {split}: {len(emb)} embeddings, pos={ram['labels'].float().mean():.3f} → {out_path}")

    logger.info("All done.")


if __name__ == "__main__":
    main()

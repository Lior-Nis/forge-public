"""
Train GRU probe head on precomputed backbone embeddings.

Loads {train,val,test}.pt files from --emb-dir, trains RNNHead, logs to WandB.
Much faster than full probe: no zarr I/O, no wavelet, no backbone forward.

Usage:
    uv run python scripts/train_embedding_probe.py \
        --emb-dir checkpoints/embeddings/vit12_ep6 \
        --fold 0 \
        --model-name vit12_ep6 \
        --output-dir checkpoints/classification/emb_probe_vit12_ep6_fold0
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)

# ── Focal loss (matches probe experiment) ────────────────────────────────────

class FocalBinaryCE(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # weight for positive class

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


# ── Training loop ─────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["patch_y"].to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())
    ap = average_precision_score(all_labels, all_probs)
    return ap


def train(args):
    from data.dataset.embedding import EmbeddingDataset
    import hydra
    from omegaconf import OmegaConf

    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if (out_dir / "last.ckpt").exists():
        logger.info("Already done, skipping.")
        return

    # Load embedding datasets
    train_ds = EmbeddingDataset(emb_dir / f"fold{args.fold}_train.pt")
    val_ds = EmbeddingDataset(emb_dir / f"fold{args.fold}_val.pt")
    test_ds = EmbeddingDataset(emb_dir / f"fold{args.fold}_test.pt")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Infer embedding shape from data
    sample = train_ds[0]
    nW, D = sample["x"].shape  # e.g. [20, 512]

    # Build RNNHead (matching probe experiment config)
    from model.heads import RNNHead
    head = RNNHead(
        input_dim=D,
        rnn_type="GRU",
        rnn_layers=1,
        rnn_hidden_mult=1,
        num_classes=2,
        dropout=0.3,
        sequence_output=False,
        bidirectional=True,
        input_seq_len=nW,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = head.to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = FocalBinaryCE(gamma=2.0, alpha=0.75)

    run = wandb.init(
        project="fog-classification",
        name=f"emb_probe_{args.model_name}_fold{args.fold}",
        config=vars(args),
    )

    best_val_ap, best_state = 0.0, None

    for epoch in range(args.epochs):
        head.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["patch_y"].to(device)
            optimizer.zero_grad()
            logits = head(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_ap = evaluate(head, val_loader, device)
        wandb.log({"epoch": epoch, "train_loss": total_loss / len(train_loader), "val_ap": val_ap})
        logger.info(f"Epoch {epoch:02d}  loss={total_loss/len(train_loader):.4f}  val_ap={val_ap:.4f}")

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    # Test with best val checkpoint
    head.load_state_dict(best_state)
    test_ap = evaluate(head, test_loader, device)
    wandb.log({"test_ap": test_ap, "best_val_ap": best_val_ap})
    logger.info(f"Test AP: {test_ap:.4f}  (best val AP: {best_val_ap:.4f})")

    # Save checkpoint
    torch.save({"state_dict": best_state, "test_ap": test_ap, "val_ap": best_val_ap,
                "args": vars(args)}, out_dir / "last.ckpt")
    run.finish()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

"""
Self-supervised pre-training for HSTGMatch (Section 4.2).

Usage:
    python scripts/pretrain.py --data_dir data/synthetic --config configs/default.yaml
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, Optional

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import Preprocessor
from src.data.dataset import TrajectoryDataset, pretrain_collate_fn
from src.models.hstgmatch import HSTGMatchPretrainer
from src.models.graph_embedding import build_ata_graph


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_nested(cfg: Dict, *keys, default=None):
    obj = cfg
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return default
        obj = obj[k]
    return obj


# ---------------------------------------------------------------------------
# Build ATA-Graph tensors
# ---------------------------------------------------------------------------

def build_graph(preprocessor: Preprocessor, device: torch.device):
    """Build edge_index and edge_weights for the ATA-Graph."""
    import numpy as np

    counts = torch.tensor(preprocessor.grid_point_counts, dtype=torch.long, device=device)
    positions = torch.tensor(
        preprocessor.grid_mapper.all_grid_centres(), dtype=torch.float32, device=device
    )
    edge_index, edge_weights = build_ata_graph(
        grid_points_count=counts,
        grid_positions=positions,
        distance_threshold=preprocessor.distance_threshold_m,
    )
    return edge_index, edge_weights


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: HSTGMatchPretrainer,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    device: torch.device,
    clip_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        grid_ids = batch["grid_ids"].to(device)
        coords = batch["coords"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        outputs = model(
            grid_ids=grid_ids,
            coords=coords,
            edge_index=edge_index,
            edge_weights=edge_weights,
            padding_mask=padding_mask,
            apply_mask=True,
        )

        loss = model.compute_loss(
            grid_logits=outputs["grid_logits"],
            coord_preds=outputs["coord_preds"],
            grid_ids=grid_ids,
            coords=coords,
            mask=outputs["mask"],
            padding_mask=padding_mask,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: HSTGMatchPretrainer,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        grid_ids = batch["grid_ids"].to(device)
        coords = batch["coords"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        outputs = model(
            grid_ids=grid_ids,
            coords=coords,
            edge_index=edge_index,
            edge_weights=edge_weights,
            padding_mask=padding_mask,
            apply_mask=True,
        )

        loss = model.compute_loss(
            grid_logits=outputs["grid_logits"],
            coord_preds=outputs["coord_preds"],
            grid_ids=grid_ids,
            coords=coords,
            mask=outputs["mask"],
            padding_mask=padding_mask,
        )

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HSTGMatch self-supervised pre-training")
    parser.add_argument("--data_dir", type=str, default="data/synthetic",
                        help="Directory containing trajectories.json and road_network.json")
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--config", type=str,
                        default=os.path.join(_project_root, "configs", "default.yaml"),
                        help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override config batch_size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override config learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Override checkpoint output directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of data used for validation")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Config
    cfg = load_config(args.config)

    epochs = args.epochs or get_nested(cfg, "pretrain", "epochs", default=50)
    batch_size = args.batch_size or get_nested(cfg, "pretrain", "batch_size", default=32)
    lr = args.lr or get_nested(cfg, "pretrain", "lr", default=1e-4)
    weight_decay = get_nested(cfg, "pretrain", "weight_decay", default=1e-5)
    checkpoint_dir = args.checkpoint_dir or get_nested(cfg, "pretrain", "checkpoint_dir",
                                                        default="checkpoints/pretrain")
    k = get_nested(cfg, "pretrain", "k", default=0.5)
    mask_ratio = get_nested(cfg, "pretrain", "mask_ratio", default=0.15)
    span_len = get_nested(cfg, "pretrain", "span_len", default=3)
    clip_grad = get_nested(cfg, "train", "clip_grad_norm", default=1.0)

    d_model = get_nested(cfg, "model", "d_model", default=128)
    n_enc_layers = get_nested(cfg, "model", "n_encoder_layers", default=4)
    n_heads = get_nested(cfg, "model", "n_heads", default=8)
    d_ff = get_nested(cfg, "model", "d_ff", default=512)
    dropout = get_nested(cfg, "model", "dropout", default=0.1)
    n_gat_layers = get_nested(cfg, "model", "n_gat_layers", default=2)
    n_gat_heads = get_nested(cfg, "model", "n_gat_heads", default=4)
    gat_dropout = get_nested(cfg, "model", "gat_dropout", default=0.1)
    max_seq_len = get_nested(cfg, "model", "max_seq_len", default=512)
    cell_size_m = get_nested(cfg, "data", "grid_size_meters", default=100)
    dist_thresh = get_nested(cfg, "data", "distance_threshold", default=200)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load & preprocess data
    traj_path = os.path.join(args.data_dir, "trajectories.json")
    road_path = os.path.join(args.data_dir, "road_network.json")
    print(f"Loading data from {args.data_dir}...")

    preprocessor = Preprocessor.from_files(
        traj_path, road_path,
        cell_size_m=cell_size_m,
        distance_threshold_m=dist_thresh,
    )
    preprocessor.fit()

    print(f"  Grid size: {preprocessor.grid_mapper.n_rows} × {preprocessor.grid_mapper.n_cols} = {preprocessor.grid_mapper.n_grids} cells")

    processed = preprocessor.process_trajectories()
    print(f"  Processed {len(processed)} trajectories.")

    # Save preprocessor state for later use
    preprocessor.save_state(os.path.join(checkpoint_dir, "preprocessor.pkl"))

    # Build ATA-Graph
    edge_index, edge_weights = build_graph(preprocessor, device)
    print(f"  ATA-Graph: {edge_index.shape[1]} edges over {preprocessor.grid_mapper.n_grids} nodes")

    # Datasets
    full_ds = TrajectoryDataset(processed, max_len=max_seq_len)
    n_val = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=pretrain_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=pretrain_collate_fn, num_workers=0)

    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Model
    model = HSTGMatchPretrainer(
        n_grids=preprocessor.grid_mapper.n_grids,
        d_model=d_model,
        n_encoder_layers=n_enc_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        n_gat_layers=n_gat_layers,
        n_gat_heads=n_gat_heads,
        gat_dropout=gat_dropout,
        max_len=max_seq_len,
        k=k,
        mask_ratio=mask_ratio,
        span_len=span_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_loss = float("inf")
    log_interval = get_nested(cfg, "logging", "log_interval", default=10)
    eval_interval = get_nested(cfg, "logging", "eval_interval", default=1)

    print(f"\nStarting pre-training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, edge_index, edge_weights, device, clip_grad
        )
        scheduler.step()

        if epoch % eval_interval == 0:
            val_loss = evaluate(model, val_loader, edge_index, edge_weights, device)

            if epoch % log_interval == 0 or epoch <= 5:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch:4d}/{epochs}  train_loss={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  lr={lr_now:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(checkpoint_dir, "best_encoder.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": cfg,
                        "n_grids": preprocessor.grid_mapper.n_grids,
                    },
                    ckpt_path,
                )

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "final_encoder.pt")
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "n_grids": preprocessor.grid_mapper.n_grids,
        },
        final_path,
    )

    print(f"\nPre-training complete.")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best checkpoint: {os.path.join(checkpoint_dir, 'best_encoder.pt')}")
    print(f"  Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()

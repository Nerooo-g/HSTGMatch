"""
Supervised fine-tuning for HSTGMatch (Equations 7-11).

Usage:
    python scripts/train.py \\
        --data_dir data/synthetic \\
        --config configs/default.yaml \\
        --pretrained_encoder checkpoints/pretrain/best_encoder.pt
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import Preprocessor
from src.data.dataset import (
    MapMatchingDataset,
    mapmatching_collate_fn,
    load_labels,
)
from src.models.hstgmatch import HSTGMatch, HSTGMatchPretrainer
from src.models.graph_embedding import build_ata_graph
from src.utils.metrics import RouteEvaluator


# ---------------------------------------------------------------------------
# Config helpers
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
# Build ATA-Graph
# ---------------------------------------------------------------------------

def build_graph(preprocessor: Preprocessor, device: torch.device):
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
# Training and evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: HSTGMatch,
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
        distances = batch["distances"].to(device)
        times = batch["times"].to(device)
        positions = batch["positions"].to(device)
        src_mask = batch["src_padding_mask"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        tgt_mask = batch["tgt_padding_mask"].to(device)

        logits = model(
            grid_ids=grid_ids,
            coords=coords,
            distances=distances,
            times=times,
            positions=positions,
            edge_index=edge_index,
            edge_weights=edge_weights,
            tgt_ids=tgt_input,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
        )

        # tgt_output has -100 at padded positions (ignored by cross-entropy)
        # logits shape: (B, L_dec, n_segments)
        # tgt_output shape: (B, L_dec) with raw 0-indexed seg_ids
        B, L_dec, V = logits.shape
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(B * L_dec, V),
            tgt_output.reshape(B * L_dec),
            ignore_index=-100,
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
    model: HSTGMatch,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    device: torch.device,
    n_segments: int,
) -> Tuple[float, float, float, float]:
    """
    Returns (mean_loss, precision, recall, f1).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    evaluator = RouteEvaluator()

    for batch in loader:
        grid_ids = batch["grid_ids"].to(device)
        coords = batch["coords"].to(device)
        distances = batch["distances"].to(device)
        times = batch["times"].to(device)
        positions = batch["positions"].to(device)
        src_mask = batch["src_padding_mask"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        tgt_mask = batch["tgt_padding_mask"].to(device)

        # Forward for loss
        logits = model(
            grid_ids=grid_ids,
            coords=coords,
            distances=distances,
            times=times,
            positions=positions,
            edge_index=edge_index,
            edge_weights=edge_weights,
            tgt_ids=tgt_input,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
        )

        B, L_dec, V = logits.shape
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(B * L_dec, V),
            tgt_output.reshape(B * L_dec),
            ignore_index=-100,
        )
        total_loss += loss.item()
        n_batches += 1

        # Greedy decode for metrics
        # Determine max_decode_len from non-padding output targets
        valid_lens = (tgt_output != -100).sum(dim=1)
        max_decode_len = int(valid_lens.max().item())

        pred_raw = model.predict(
            grid_ids=grid_ids,
            coords=coords,
            distances=distances,
            times=times,
            positions=positions,
            edge_index=edge_index,
            edge_weights=edge_weights,
            max_decode_len=max_decode_len,
            src_key_padding_mask=src_mask,
        )  # (B, max_decode_len)

        # Convert to Python lists, filter out-of-range predictions
        for b in range(B):
            valid_len = int(valid_lens[b].item())
            truth = tgt_output[b, :valid_len].cpu().tolist()
            pred_full = pred_raw[b].cpu().tolist()
            # Keep only valid seg IDs
            pred = [p for p in pred_full[:valid_len] if 0 <= p < n_segments]
            evaluator.update([pred], [truth])

    mean_loss = total_loss / max(n_batches, 1)
    p, r, f1 = evaluator.compute()
    return mean_loss, p, r, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HSTGMatch supervised training")
    parser.add_argument("--data_dir", type=str, default="data/synthetic",
                        help="Directory containing trajectories.json, road_network.json, labels.json")
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--config", type=str,
                        default=os.path.join(_project_root, "configs", "default.yaml"),
                        help="Path to YAML config file")
    parser.add_argument("--pretrained_encoder", type=str, default=None,
                        help="Path to pretrained encoder checkpoint (overrides config)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Config
    cfg = load_config(args.config)

    epochs = args.epochs or get_nested(cfg, "train", "epochs", default=100)
    batch_size = args.batch_size or get_nested(cfg, "train", "batch_size", default=32)
    lr = args.lr or get_nested(cfg, "train", "lr", default=5e-5)
    weight_decay = get_nested(cfg, "train", "weight_decay", default=1e-5)
    checkpoint_dir = args.checkpoint_dir or get_nested(cfg, "train", "checkpoint_dir",
                                                        default="checkpoints/train")
    clip_grad = get_nested(cfg, "train", "clip_grad_norm", default=1.0)

    pretrained_path = (
        args.pretrained_encoder
        or get_nested(cfg, "train", "pretrained_encoder", default=None)
    )

    train_ratio = args.train_ratio or get_nested(cfg, "data", "train_ratio", default=0.7)
    val_ratio = args.val_ratio or get_nested(cfg, "data", "val_ratio", default=0.1)

    d_model = get_nested(cfg, "model", "d_model", default=128)
    n_enc_layers = get_nested(cfg, "model", "n_encoder_layers", default=4)
    n_dec_layers = get_nested(cfg, "model", "n_decoder_layers", default=4)
    n_heads = get_nested(cfg, "model", "n_heads", default=8)
    d_ff = get_nested(cfg, "model", "d_ff", default=512)
    dropout = get_nested(cfg, "model", "dropout", default=0.1)
    n_gat_layers = get_nested(cfg, "model", "n_gat_layers", default=2)
    n_gat_heads = get_nested(cfg, "model", "n_gat_heads", default=4)
    gat_dropout = get_nested(cfg, "model", "gat_dropout", default=0.1)
    max_seq_len = get_nested(cfg, "model", "max_seq_len", default=512)
    n_dist_slots = get_nested(cfg, "model", "n_distance_slots", default=64)
    n_time_slots = get_nested(cfg, "model", "n_time_slots", default=64)
    max_distance = get_nested(cfg, "model", "max_distance", default=10000.0)
    max_time = get_nested(cfg, "model", "max_time", default=3600.0)
    cell_size_m = get_nested(cfg, "data", "grid_size_meters", default=100)
    dist_thresh = get_nested(cfg, "data", "distance_threshold", default=200)

    log_interval = get_nested(cfg, "logging", "log_interval", default=10)
    eval_interval = get_nested(cfg, "logging", "eval_interval", default=1)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load & preprocess data
    traj_path = os.path.join(args.data_dir, "trajectories.json")
    road_path = os.path.join(args.data_dir, "road_network.json")
    labels_path = os.path.join(args.data_dir, "labels.json")

    print(f"Loading data from {args.data_dir}...")

    # Try to reuse preprocessor state from pretrain if available
    pretrain_ckpt_dir = get_nested(cfg, "pretrain", "checkpoint_dir", default="checkpoints/pretrain")
    preprocessor_state_path = os.path.join(pretrain_ckpt_dir, "preprocessor.pkl")

    with open(traj_path) as f:
        trajectories = json.load(f)
    with open(road_path) as f:
        road_network = json.load(f)
    labels = load_labels(labels_path)

    if os.path.exists(preprocessor_state_path):
        print(f"  Loading preprocessor state from {preprocessor_state_path}")
        preprocessor = Preprocessor.load_state(preprocessor_state_path, trajectories, road_network)
    else:
        preprocessor = Preprocessor(
            trajectories, road_network,
            cell_size_m=cell_size_m,
            distance_threshold_m=dist_thresh,
        )
        preprocessor.fit()

    n_grids = preprocessor.grid_mapper.n_grids
    n_segments = preprocessor.get_n_segments()
    print(f"  Grids: {n_grids}  Segments: {n_segments}")

    processed = preprocessor.process_trajectories()

    # Build ATA-Graph
    edge_index, edge_weights = build_graph(preprocessor, device)
    print(f"  ATA-Graph edges: {edge_index.shape[1]}")

    # Datasets
    full_ds = MapMatchingDataset(
        processed, labels, max_traj_len=max_seq_len, max_route_len=max_seq_len
    )
    n_total = len(full_ds)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"  Split: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=mapmatching_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=mapmatching_collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=mapmatching_collate_fn, num_workers=0)

    # Build model
    model = HSTGMatch(
        n_grids=n_grids,
        n_segments=n_segments,
        d_model=d_model,
        n_encoder_layers=n_enc_layers,
        n_decoder_layers=n_dec_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        n_gat_layers=n_gat_layers,
        n_gat_heads=n_gat_heads,
        gat_dropout=gat_dropout,
        n_distance_slots=n_dist_slots,
        n_time_slots=n_time_slots,
        max_distance=max_distance,
        max_time=max_time,
        max_len=max_seq_len,
    ).to(device)

    # Load pretrained encoder if available
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained encoder from {pretrained_path}...")
        ckpt = torch.load(pretrained_path, map_location=device)
        # Reconstruct pretrainer and load state, then copy to model
        pretrainer = HSTGMatchPretrainer(
            n_grids=n_grids,
            d_model=d_model,
            n_encoder_layers=n_enc_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_gat_layers=n_gat_layers,
            n_gat_heads=n_gat_heads,
            gat_dropout=gat_dropout,
            max_len=max_seq_len,
        ).to(device)
        pretrainer.load_state_dict(ckpt["model_state_dict"])
        model.load_pretrained_encoder(pretrainer)
        print("  Pretrained encoder loaded successfully.")
    else:
        if pretrained_path:
            print(f"  Warning: pretrained encoder not found at {pretrained_path}. Training from scratch.")
        else:
            print("  No pretrained encoder specified. Training from scratch.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_f1 = -1.0
    best_val_loss = float("inf")

    print(f"\nStarting supervised training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, edge_index, edge_weights, device, clip_grad
        )
        scheduler.step()

        if epoch % eval_interval == 0:
            val_loss, val_p, val_r, val_f1 = evaluate(
                model, val_loader, edge_index, edge_weights, device, n_segments
            )

            if epoch % log_interval == 0 or epoch <= 5:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:4d}/{epochs}  "
                    f"train_loss={train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}  "
                    f"P={val_p:.4f}  R={val_r:.4f}  F1={val_f1:.4f}  "
                    f"lr={lr_now:.2e}"
                )

            if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_loss < best_val_loss):
                best_val_f1 = val_f1
                best_val_loss = val_loss
                ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "val_f1": val_f1,
                        "val_loss": val_loss,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": cfg,
                        "n_grids": n_grids,
                        "n_segments": n_segments,
                    },
                    ckpt_path,
                )

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "n_grids": n_grids,
            "n_segments": n_segments,
        },
        final_path,
    )

    # Final test evaluation
    print(f"\nEvaluating on test set...")
    best_ckpt = torch.load(os.path.join(checkpoint_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_p, test_r, test_f1 = evaluate(
        model, test_loader, edge_index, edge_weights, device, n_segments
    )

    print("\n=== Test Results ===")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Precision: {test_p:.4f}")
    print(f"  Recall:    {test_r:.4f}")
    print(f"  F1:        {test_f1:.4f}")
    print(f"\nBest model checkpoint: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print(f"Final model checkpoint: {final_path}")

    # Save test results
    results = {
        "test_loss": test_loss,
        "precision": test_p,
        "recall": test_r,
        "f1": test_f1,
        "best_val_f1": best_val_f1,
    }
    results_path = os.path.join(checkpoint_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Test results saved to {results_path}")


if __name__ == "__main__":
    main()

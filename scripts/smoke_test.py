"""
Smoke test: verifies the full pipeline runs end-to-end on synthetic data.

Run from the project root:
    python scripts/smoke_test.py
"""

import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch

print("=== HSTGMatch Smoke Test ===\n")

# -------------------------------------------------------------------
# 1. Imports
# -------------------------------------------------------------------
print("[1] Testing imports...")
from src.models.graph_embedding import build_ata_graph, OptGATLayer, GridGraphEncoder
from src.models.encoder import TransformerEncoder, SinusoidalPositionalEncoding
from src.models.decoder import TransformerDecoder
from src.models.spatial_temporal import IntervalEmbedding, SpatialTemporalEmbedding
from src.models.hstgmatch import HSTGMatchPretrainer, HSTGMatch
from src.data.preprocess import GridMapper, Preprocessor
from src.data.dataset import (
    TrajectoryDataset, MapMatchingDataset,
    pretrain_collate_fn, mapmatching_collate_fn, load_labels,
)
from src.utils.metrics import compute_precision_recall_f1, RouteEvaluator
print("    All imports OK\n")

device = torch.device("cpu")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "synthetic")

# -------------------------------------------------------------------
# 2. Preprocessing
# -------------------------------------------------------------------
print("[2] Preprocessing...")
preprocessor = Preprocessor.from_files(
    os.path.join(DATA_DIR, "trajectories.json"),
    os.path.join(DATA_DIR, "road_network.json"),
    cell_size_m=100.0,
    distance_threshold_m=200.0,
)
preprocessor.fit()
n_grids = preprocessor.grid_mapper.n_grids
n_segments = preprocessor.get_n_segments()
print(f"    Grids: {n_grids}  Segments: {n_segments}")

processed = preprocessor.process_trajectories()
print(f"    Processed {len(processed)} trajectories")

# -------------------------------------------------------------------
# 3. ATA-Graph
# -------------------------------------------------------------------
print("\n[3] Building ATA-Graph...")
counts = torch.tensor(preprocessor.grid_point_counts, dtype=torch.long)
positions = torch.tensor(preprocessor.grid_mapper.all_grid_centres(), dtype=torch.float32)
edge_index, edge_weights = build_ata_graph(counts, positions, 200.0)
print(f"    Edges: {edge_index.shape[1]}  Weights shape: {edge_weights.shape}")

# -------------------------------------------------------------------
# 4. GridGraphEncoder
# -------------------------------------------------------------------
print("\n[4] GridGraphEncoder forward pass...")
graph_enc = GridGraphEncoder(n_grids=n_grids, d_model=32, n_layers=2, n_heads=4)
grid_embs = graph_enc(edge_index, edge_weights)
print(f"    Output shape: {grid_embs.shape}  (expected [{n_grids}, 32])")
assert grid_embs.shape == (n_grids, 32)

# -------------------------------------------------------------------
# 5. TransformerEncoder
# -------------------------------------------------------------------
print("\n[5] TransformerEncoder forward pass...")
enc = TransformerEncoder(input_dim=64, d_model=32, n_heads=4, n_layers=2, d_ff=64)
x = torch.randn(2, 10, 64)
out = enc(x)
print(f"    Output shape: {out.shape}  (expected [2, 10, 32])")
assert out.shape == (2, 10, 32)

# -------------------------------------------------------------------
# 6. TransformerDecoder
# -------------------------------------------------------------------
print("\n[6] TransformerDecoder forward pass...")
dec = TransformerDecoder(n_segments=n_segments, d_model=32, n_heads=4, n_layers=2, d_ff=64)
memory = torch.randn(2, 10, 32)
tgt_ids = torch.randint(0, n_segments + 2, (2, 5))
logits = dec(memory, tgt_ids)
print(f"    Logits shape: {logits.shape}  (expected [2, 5, {n_segments}])")
assert logits.shape == (2, 5, n_segments)

# -------------------------------------------------------------------
# 7. IntervalEmbedding
# -------------------------------------------------------------------
print("\n[7] IntervalEmbedding forward pass...")
emb = IntervalEmbedding(n_slots=32, d_model=16, max_value=1000.0)
v_r = torch.tensor([[0.0, 100.0, 500.0, 999.0], [50.0, 200.0, 700.0, 900.0]])
pos = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
out_emb = emb(v_r, pos)
print(f"    Output shape: {out_emb.shape}  (expected [2, 4, 32])")
assert out_emb.shape == (2, 4, 32)

# -------------------------------------------------------------------
# 8. HSTGMatchPretrainer
# -------------------------------------------------------------------
print("\n[8] HSTGMatchPretrainer forward + loss...")
pretrainer = HSTGMatchPretrainer(
    n_grids=n_grids, d_model=32, n_encoder_layers=2, n_heads=4,
    d_ff=64, dropout=0.0, n_gat_layers=2, n_gat_heads=4, gat_dropout=0.0,
    max_len=50, k=0.5, mask_ratio=0.3, span_len=2,
)

# Build a small batch from processed trajectories
from torch.utils.data import DataLoader
ds = TrajectoryDataset(processed[:8], max_len=20)
loader = DataLoader(ds, batch_size=4, collate_fn=pretrain_collate_fn)
batch = next(iter(loader))

grid_ids = batch["grid_ids"]
coords = batch["coords"]
padding_mask = batch["padding_mask"]

out = pretrainer(
    grid_ids=grid_ids,
    coords=coords,
    edge_index=edge_index,
    edge_weights=edge_weights,
    padding_mask=padding_mask,
    apply_mask=True,
)
print(f"    grid_logits: {out['grid_logits'].shape}")
print(f"    coord_preds: {out['coord_preds'].shape}")
print(f"    mask: {out['mask'].shape}  (masked: {out['mask'].sum().item()})")

loss = pretrainer.compute_loss(
    out["grid_logits"], out["coord_preds"],
    grid_ids, coords, out["mask"], padding_mask,
)
print(f"    Loss: {loss.item():.4f}")
assert loss.item() > 0.0

# -------------------------------------------------------------------
# 9. HSTGMatch (full model)
# -------------------------------------------------------------------
print("\n[9] HSTGMatch full model forward pass...")
labels = load_labels(os.path.join(DATA_DIR, "labels.json"))
ds_mm = MapMatchingDataset(processed[:16], labels[:16], max_traj_len=20, max_route_len=15)
loader_mm = DataLoader(ds_mm, batch_size=4, collate_fn=mapmatching_collate_fn)
batch_mm = next(iter(loader_mm))

model = HSTGMatch(
    n_grids=n_grids, n_segments=n_segments,
    d_model=32, n_encoder_layers=2, n_decoder_layers=2,
    n_heads=4, d_ff=64, dropout=0.0,
    n_gat_layers=2, n_gat_heads=4, gat_dropout=0.0,
    n_distance_slots=32, n_time_slots=32,
    max_distance=5000.0, max_time=3600.0, max_len=50,
)

# Load pretrained encoder weights
model.load_pretrained_encoder(pretrainer)
print("    Pretrained encoder loaded OK")

grid_ids = batch_mm["grid_ids"]
coords = batch_mm["coords"]
distances = batch_mm["distances"]
times = batch_mm["times"]
positions = batch_mm["positions"]
src_mask = batch_mm["src_padding_mask"]
tgt_input = batch_mm["tgt_input"]
tgt_output = batch_mm["tgt_output"]
tgt_mask = batch_mm["tgt_padding_mask"]

logits = model(
    grid_ids=grid_ids, coords=coords,
    distances=distances, times=times, positions=positions,
    edge_index=edge_index, edge_weights=edge_weights,
    tgt_ids=tgt_input,
    src_key_padding_mask=src_mask,
    tgt_key_padding_mask=tgt_mask,
)
print(f"    Logits shape: {logits.shape}  (expected [4, *, {n_segments}])")
assert logits.shape[0] == 4
assert logits.shape[2] == n_segments

# Loss
loss_mm = model.compute_loss(logits, tgt_output, tgt_mask)
print(f"    CE Loss: {loss_mm.item():.4f}")
assert loss_mm.item() > 0.0

# Greedy decode
pred = model.predict(
    grid_ids=grid_ids, coords=coords,
    distances=distances, times=times, positions=positions,
    edge_index=edge_index, edge_weights=edge_weights,
    max_decode_len=10, src_key_padding_mask=src_mask,
)
print(f"    Prediction shape: {pred.shape}  (expected [4, 10])")
assert pred.shape == (4, 10)

# -------------------------------------------------------------------
# 10. Metrics
# -------------------------------------------------------------------
print("\n[10] Metrics...")
preds_list = [[0, 1, 2, 3], [5, 6, 7], [10, 11]]
truths_list = [[0, 1, 2, 4], [5, 6, 8], [10, 11]]
p, r, f1 = compute_precision_recall_f1(preds_list, truths_list)
print(f"    Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")
assert 0.0 <= p <= 1.0
assert 0.0 <= r <= 1.0
assert 0.0 <= f1 <= 1.0

evaluator = RouteEvaluator()
evaluator.update(preds_list, truths_list)
print(f"    Evaluator: {evaluator}")
evaluator.reset()

print("\n=== All smoke tests passed! ===")

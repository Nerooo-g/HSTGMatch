"""
PyTorch Dataset classes for HSTGMatch.

TrajectoryDataset   — for self-supervised pre-training (no labels needed).
MapMatchingDataset  — for supervised training (with route labels).

Both datasets return variable-length sequences; collate functions handle
padding to the maximum length in each batch.
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# TrajectoryDataset  (pre-training)
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """
    Dataset for self-supervised pre-training.

    Each sample contains the token-level features for one trajectory:
        grid_ids   : (L,)    int64 — grid cell IDs
        coords     : (L, 2)  float32 — normalised (lon, lat)
        distances  : (L,)    float32 — distance from p_0 in metres
        times      : (L,)    float32 — time delta from p_0 in seconds
        positions  : (L,)    int64   — 1-based position indices

    Args:
        processed_trajs: list of processed trajectory dicts returned by
                         Preprocessor.process_trajectories().
        max_len: if set, trajectories longer than max_len are truncated.
    """

    def __init__(
        self,
        processed_trajs: List[Dict],
        max_len: Optional[int] = None,
    ) -> None:
        self.trajs = processed_trajs
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.trajs)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        t = self.trajs[idx]

        grid_ids = t["grid_ids"]
        coords = t["coords"]
        distances = t["distances"]
        times = t["times"]
        positions = t["positions"]

        if self.max_len is not None:
            grid_ids = grid_ids[: self.max_len]
            coords = coords[: self.max_len]
            distances = distances[: self.max_len]
            times = times[: self.max_len]
            positions = positions[: self.max_len]

        return {
            "grid_ids": torch.tensor(grid_ids, dtype=torch.long),
            "coords": torch.tensor(coords, dtype=torch.float32),
            "distances": torch.tensor(distances, dtype=torch.float32),
            "times": torch.tensor(times, dtype=torch.float32),
            "positions": torch.tensor(positions, dtype=torch.long),
            "traj_id": t["traj_id"],
        }


# ---------------------------------------------------------------------------
# MapMatchingDataset  (supervised training)
# ---------------------------------------------------------------------------

class MapMatchingDataset(Dataset):
    """
    Dataset for supervised map-matching training.

    In addition to trajectory features, each sample includes a route label.

    Args:
        processed_trajs: list of processed trajectory dicts.
        labels:          list of {"traj_id": int, "route": List[int]} dicts.
        max_traj_len:    max encoder sequence length (truncates if exceeded).
        max_route_len:   max decoder sequence length (truncates if exceeded).
    """

    PAD_SEG_ID = -1  # placeholder; collate function uses this for masking

    def __init__(
        self,
        processed_trajs: List[Dict],
        labels: List[Dict],
        max_traj_len: Optional[int] = None,
        max_route_len: Optional[int] = None,
    ) -> None:
        # Build lookup: traj_id → processed_traj
        traj_lookup = {t["traj_id"]: t for t in processed_trajs}

        # Build aligned lists (only include trajectories that have labels)
        self.samples: List[Tuple[Dict, List[int]]] = []
        for label in labels:
            tid = label["traj_id"]
            if tid in traj_lookup:
                self.samples.append((traj_lookup[tid], label["route"]))

        self.max_traj_len = max_traj_len
        self.max_route_len = max_route_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        traj, route = self.samples[idx]

        grid_ids = traj["grid_ids"]
        coords = traj["coords"]
        distances = traj["distances"]
        times = traj["times"]
        positions = traj["positions"]

        if self.max_traj_len is not None:
            grid_ids = grid_ids[: self.max_traj_len]
            coords = coords[: self.max_traj_len]
            distances = distances[: self.max_traj_len]
            times = times[: self.max_traj_len]
            positions = positions[: self.max_traj_len]

        if self.max_route_len is not None:
            route = route[: self.max_route_len]

        return {
            "grid_ids": torch.tensor(grid_ids, dtype=torch.long),
            "coords": torch.tensor(coords, dtype=torch.float32),
            "distances": torch.tensor(distances, dtype=torch.float32),
            "times": torch.tensor(times, dtype=torch.float32),
            "positions": torch.tensor(positions, dtype=torch.long),
            # route: raw seg_ids (0-indexed), will be handled by decoder vocab mapping
            "route": torch.tensor(route, dtype=torch.long),
            "traj_id": traj["traj_id"],
        }


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def pretrain_collate_fn(batch: List[Dict]) -> Dict[str, Tensor]:
    """
    Collate a list of pre-training samples into padded batches.

    Padding:
        grid_ids   : pad with 0
        coords     : pad with 0.0
        distances  : pad with 0.0
        times      : pad with 0.0
        positions  : pad with 0
        padding_mask: True at padded positions
    """
    max_len = max(s["grid_ids"].shape[0] for s in batch)
    B = len(batch)

    grid_ids = torch.zeros(B, max_len, dtype=torch.long)
    coords = torch.zeros(B, max_len, 2, dtype=torch.float32)
    distances = torch.zeros(B, max_len, dtype=torch.float32)
    times = torch.zeros(B, max_len, dtype=torch.float32)
    positions = torch.zeros(B, max_len, dtype=torch.long)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool)  # True = padded

    for i, s in enumerate(batch):
        L = s["grid_ids"].shape[0]
        grid_ids[i, :L] = s["grid_ids"]
        coords[i, :L] = s["coords"]
        distances[i, :L] = s["distances"]
        times[i, :L] = s["times"]
        positions[i, :L] = s["positions"]
        padding_mask[i, :L] = False  # not padded

    return {
        "grid_ids": grid_ids,
        "coords": coords,
        "distances": distances,
        "times": times,
        "positions": positions,
        "padding_mask": padding_mask,
    }


def mapmatching_collate_fn(batch: List[Dict]) -> Dict[str, Tensor]:
    """
    Collate map-matching samples into padded batches for teacher-forcing.

    The decoder targets are prepared as:
        tgt_input  : [BOS, seg0, seg1, ..., seg_{L-1}]  (vocab IDs)
        tgt_output : [seg0, seg1, ..., seg_{L-1}]       (raw 0-indexed IDs)

    BOS vocab ID = 1  (TransformerDecoder.BOS_ID)
    Vocab ID for raw seg_id k = k + 2
    """
    BOS_ID = 1   # vocab space
    PAD_VOCAB = 0

    max_traj_len = max(s["grid_ids"].shape[0] for s in batch)
    max_route_len = max(s["route"].shape[0] for s in batch)
    B = len(batch)

    # Encoder inputs
    grid_ids = torch.zeros(B, max_traj_len, dtype=torch.long)
    coords = torch.zeros(B, max_traj_len, 2, dtype=torch.float32)
    distances = torch.zeros(B, max_traj_len, dtype=torch.float32)
    times = torch.zeros(B, max_traj_len, dtype=torch.float32)
    positions = torch.zeros(B, max_traj_len, dtype=torch.long)
    src_padding_mask = torch.ones(B, max_traj_len, dtype=torch.bool)

    # Decoder inputs (length = max_route_len + 1 for BOS)
    dec_len = max_route_len + 1
    tgt_input = torch.full((B, dec_len), PAD_VOCAB, dtype=torch.long)
    tgt_output = torch.full((B, dec_len), -100, dtype=torch.long)  # -100 = ignore in CE
    tgt_padding_mask = torch.ones(B, dec_len, dtype=torch.bool)

    for i, s in enumerate(batch):
        L = s["grid_ids"].shape[0]
        R = s["route"].shape[0]

        grid_ids[i, :L] = s["grid_ids"]
        coords[i, :L] = s["coords"]
        distances[i, :L] = s["distances"]
        times[i, :L] = s["times"]
        positions[i, :L] = s["positions"]
        src_padding_mask[i, :L] = False

        # tgt_input: [BOS, seg0+2, seg1+2, ..., segR+2, PAD, ...]
        tgt_input[i, 0] = BOS_ID
        tgt_input[i, 1 : R + 1] = s["route"] + 2  # vocab offset
        tgt_padding_mask[i, : R + 1] = False

        # tgt_output: shifted target — decoder predicts token at position t+1
        # We want: at dec position 0 (after BOS) predict seg0, ..., at R-1 predict segR-1
        # tgt_output[i, :R] = raw seg_ids (0-indexed)
        tgt_output[i, :R] = s["route"]
        # positions R onward remain -100 (ignored)

    return {
        "grid_ids": grid_ids,
        "coords": coords,
        "distances": distances,
        "times": times,
        "positions": positions,
        "src_padding_mask": src_padding_mask,
        "tgt_input": tgt_input,
        "tgt_output": tgt_output,
        "tgt_padding_mask": tgt_padding_mask,
    }


# ---------------------------------------------------------------------------
# Label loader helper
# ---------------------------------------------------------------------------

def load_labels(labels_path: str) -> List[Dict]:
    with open(labels_path, "r") as f:
        return json.load(f)

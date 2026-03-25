"""
ATA-Graph construction and opt-GATs (Section 4.1 of HSTGMatch paper).

The ATA-Graph G_T = (V_T, E_T, W_T) connects grid nodes whose geographic
distance is below a threshold.  Edge weights reflect how often each
destination grid is visited.

opt-GATs implement Equations 1-2:
  alpha_ij = softmax_j ( (1/sqrt(d)) * (W h_i) · (W h_j) )
  h'_i = (1/K) * sum_k sum_{j in N_i} alpha^k_ij * W^k * h_j
       + sum_{j in N_i} W_l * h_j * gamma_j
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def haversine_distance_matrix(positions: Tensor) -> Tensor:
    """
    Compute pairwise Haversine distances (metres) for a set of grid centres.

    Args:
        positions: (N, 2) tensor of (lon, lat) in decimal degrees.

    Returns:
        dist: (N, N) tensor of distances in metres.
    """
    R = 6_371_000.0  # Earth radius in metres
    lon = positions[:, 0]
    lat = positions[:, 1]

    lon_r = torch.deg2rad(lon)
    lat_r = torch.deg2rad(lat)

    # Broadcast to (N, N)
    dlat = lat_r.unsqueeze(1) - lat_r.unsqueeze(0)  # (N, N)
    dlon = lon_r.unsqueeze(1) - lon_r.unsqueeze(0)  # (N, N)

    a = torch.sin(dlat / 2) ** 2 + (
        torch.cos(lat_r.unsqueeze(1))
        * torch.cos(lat_r.unsqueeze(0))
        * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))
    return R * c


def build_ata_graph(
    grid_points_count: Tensor,
    grid_positions: Tensor,
    distance_threshold: float,
) -> Tuple[Tensor, Tensor]:
    """
    Build the ATA-Graph edge_index and edge_weights.

    Args:
        grid_points_count: (N,) tensor — number of trajectory points falling
                           in each grid cell.
        grid_positions:    (N, 2) tensor — (lon, lat) of each grid cell centre.
        distance_threshold: Maximum distance (metres) for connecting two grids.

    Returns:
        edge_index: (2, E) long tensor — [source_nodes; destination_nodes].
        edge_weights: (E,) float tensor — gamma_j for each destination node j.
    """
    n = grid_positions.shape[0]
    total_points = grid_points_count.sum().float().clamp(min=1.0)

    # Edge weights gamma_j = count_j / total
    gamma = grid_points_count.float() / total_points  # (N,)

    # Pairwise distances
    dist = haversine_distance_matrix(grid_positions.float())  # (N, N)

    # Connect grids within threshold (exclude self-loops)
    mask = (dist < distance_threshold) & (~torch.eye(n, dtype=torch.bool, device=dist.device))

    src, dst = mask.nonzero(as_tuple=True)  # both (E,)
    edge_weights = gamma[dst]

    edge_index = torch.stack([src, dst], dim=0)  # (2, E)
    return edge_index, edge_weights


# ---------------------------------------------------------------------------
# opt-GAT Layer  (Equations 1-2)
# ---------------------------------------------------------------------------

class OptGATLayer(nn.Module):
    """
    A single opt-GAT layer implementing Equations 1-2.

    The output of node i is the sum of two terms:
      Term 1: (1/K) * sum_k  sum_{j in N_i}  alpha^k_ij * W^k * h_j
              (multi-head dot-product attention aggregation)
      Term 2: sum_{j in N_i}  W_l * h_j * gamma_j
              (linearly projected, visit-frequency weighted sum)

    No non-linear activation is applied (following LightGCN spirit).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Per-head projection matrices W^k  (shared key/value projection)
        self.W_heads = nn.Linear(in_dim, out_dim, bias=False)

        # Frequency-weighted branch W_l
        self.W_l = nn.Linear(in_dim, out_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weights: Tensor,
    ) -> Tensor:
        """
        Args:
            x:            (N, in_dim) node features.
            edge_index:   (2, E) — [src; dst].
            edge_weights: (E,) — gamma_j for each edge (src -> dst).

        Returns:
            h_out: (N, out_dim) updated node features.
        """
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]  # (E,)

        # ---------- Term 1: multi-head attention ----------
        # Project all nodes: (N, out_dim) -> (N, K, head_dim)
        h_proj = self.W_heads(x).view(N, self.n_heads, self.head_dim)  # (N, K, D_h)

        # Gather src and dst projected features for each edge
        h_src = h_proj[src]  # (E, K, D_h)
        h_dst = h_proj[dst]  # (E, K, D_h)

        # Attention score: (1/sqrt(d)) * h_i · h_j  per head
        attn_scores = (h_src * h_dst).sum(dim=-1) * self.scale  # (E, K)

        # Softmax over neighbours of each node (destination == "query" node)
        # We aggregate messages arriving at node dst, so we softmax over
        # all edges that share the same dst.
        attn_scores_exp = attn_scores.exp()  # (E, K)

        # Sum exp scores per dst node (for normalisation)
        denom = torch.zeros(N, self.n_heads, device=x.device)
        denom.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.n_heads), attn_scores_exp)
        denom = denom.clamp(min=1e-9)

        alpha = attn_scores_exp / denom[dst]  # (E, K)
        alpha = self.dropout(alpha)

        # Weighted sum of source projected features
        # weighted messages: (E, K, D_h)
        weighted = alpha.unsqueeze(-1) * h_proj[src]  # (E, K, D_h)

        # Aggregate at destination
        agg1 = torch.zeros(N, self.n_heads, self.head_dim, device=x.device)
        idx = dst.view(-1, 1, 1).expand(-1, self.n_heads, self.head_dim)
        agg1.scatter_add_(0, idx, weighted)
        # Average over heads: (N, out_dim)
        term1 = (agg1 / self.n_heads).view(N, self.out_dim)

        # ---------- Term 2: frequency-weighted branch ----------
        h_l = self.W_l(x)  # (N, out_dim)

        # gamma_j weighting: edge_weights is gamma for dst node j
        msg2 = h_l[src] * edge_weights.view(-1, 1)  # (E, out_dim)

        agg2 = torch.zeros(N, self.out_dim, device=x.device)
        idx2 = dst.view(-1, 1).expand_as(msg2)
        agg2.scatter_add_(0, idx2, msg2)

        return term1 + agg2  # (N, out_dim)


# ---------------------------------------------------------------------------
# Multi-layer GridGraphEncoder
# ---------------------------------------------------------------------------

class GridGraphEncoder(nn.Module):
    """
    Stack of opt-GAT layers that produces a final embedding for each grid cell.

    An initial linear projection lifts raw node features (typically a simple
    one-hot or learnable embedding) to d_model before the GAT layers.
    """

    def __init__(
        self,
        n_grids: int,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_grids = n_grids
        self.d_model = d_model

        # Learnable initial node embedding (analogous to an ID embedding)
        self.node_embedding = nn.Embedding(n_grids, d_model)

        self.layers = nn.ModuleList(
            [
                OptGATLayer(d_model, d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        edge_index: Tensor,
        edge_weights: Tensor,
        node_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            edge_index:   (2, E) long tensor.
            edge_weights: (E,) float tensor.
            node_ids:     (N,) long tensor of grid IDs. If None, uses
                          arange(n_grids).

        Returns:
            h: (N, d_model) grid embeddings.
        """
        if node_ids is None:
            device = edge_index.device
            node_ids = torch.arange(self.n_grids, device=device)

        h = self.node_embedding(node_ids)  # (N, d_model)

        for layer in self.layers:
            h = layer(h, edge_index, edge_weights)

        return self.norm(h)

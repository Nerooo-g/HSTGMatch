"""
Spatial-Temporal Factor module (Section 4.3 of HSTGMatch paper).

Distance interval: absolute Haversine distance from p_0 to p_i (metres).
Time interval:     absolute timestamp difference from p_0 to p_i (seconds).

Slot-based transformation (Equation 4):
    r = ( W_{u(v_r)} * [u(v_r) - v_r]
        + W_{l(v_r)} * [v_r - l(v_r)] ) / (u(v_r) - l(v_r))

Decay coefficient (Equations 5-6):
    r' = (log(v_r) / s) * W_i * r      # s = 1-based position index
    r'' = r ⊕ r'                        # concatenate

The final spatial/temporal embedding has dimension 2 * d_model so that
when distance and time embeddings are concatenated downstream the combined
size is 4 * d_model (before any projection).
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class IntervalEmbedding(nn.Module):
    """
    Slot-based interval embedding with logarithmic decay.

    Attributes:
        n_slots:   Number of uniform slots partitioning [0, max_value].
        d_model:   Output embedding dimension (before concatenation with decay).
        max_value: Upper bound of the interval range.

    Output dimension per interval: 2 * d_model  (original r ⊕ decayed r').
    """

    def __init__(
        self,
        n_slots: int,
        d_model: int,
        max_value: float,
    ) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.d_model = d_model
        self.max_value = max_value

        # Slot embedding matrix W ∈ R^{N_r × D}  (Eq. 4)
        self.slot_embeddings = nn.Embedding(n_slots, d_model)

        # W_i for the decay branch  (Eq. 5)
        self.W_decay = nn.Linear(d_model, d_model, bias=False)

        nn.init.xavier_uniform_(self.slot_embeddings.weight)
        nn.init.xavier_uniform_(self.W_decay.weight)

        # Register slot boundary positions as buffers (not trainable)
        boundaries = torch.linspace(0.0, max_value, n_slots + 1)
        self.register_buffer("boundaries", boundaries)  # (n_slots+1,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _slot_interpolate(self, v_r: Tensor) -> Tensor:
        """
        Eq. 4: slot-based linear interpolation between adjacent slot embeddings.

        Args:
            v_r: (B, L) or (B,) absolute interval values.

        Returns:
            r: same leading shape + (d_model,) — interpolated embedding.
        """
        shape = v_r.shape
        v_flat = v_r.reshape(-1)  # (N,)

        # Clamp to valid range
        v_clamped = v_flat.clamp(0.0, self.max_value - 1e-6)

        # Identify lower slot index
        slot_size = self.max_value / self.n_slots
        lo_idx = (v_clamped / slot_size).long().clamp(0, self.n_slots - 1)  # (N,)
        hi_idx = (lo_idx + 1).clamp(0, self.n_slots - 1)

        lo_val = lo_idx.float() * slot_size   # l(v_r)
        hi_val = hi_idx.float() * slot_size   # u(v_r)

        W_lo = self.slot_embeddings(lo_idx)   # (N, D)
        W_hi = self.slot_embeddings(hi_idx)   # (N, D)

        # Eq. 4:  r = (W_hi * (hi - v) + W_lo * (v - lo)) / (hi - lo)
        # Note: paper names upper slot W_{u(v_r)} and lower slot W_{l(v_r)}
        slot_width = (hi_val - lo_val).clamp(min=1e-6).unsqueeze(-1)  # (N, 1)
        dist_to_hi = (hi_val - v_clamped).unsqueeze(-1)               # (N, 1)
        dist_to_lo = (v_clamped - lo_val).unsqueeze(-1)               # (N, 1)

        r = (W_hi * dist_to_hi + W_lo * dist_to_lo) / slot_width  # (N, D)
        return r.reshape(*shape, self.d_model)

    def _decay(self, r: Tensor, v_r: Tensor, positions: Tensor) -> Tensor:
        """
        Eq. 5: r' = (log(v_r + 1) / s) * W_i * r
               (log(v_r + 1) to avoid log(0))

        Args:
            r:         (..., d_model) slot-interpolated embeddings.
            v_r:       (...) absolute interval values.
            positions: (...) 1-based position indices (s in Eq. 5).

        Returns:
            r_prime: (..., d_model)
        """
        log_vr = torch.log1p(v_r.float())                         # (...)
        s = positions.float().clamp(min=1.0)                       # (...)
        decay_scale = (log_vr / s).unsqueeze(-1)                   # (..., 1)

        r_transformed = self.W_decay(r)                            # (..., d_model)
        return decay_scale * r_transformed

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, v_r: Tensor, positions: Tensor) -> Tensor:
        """
        Compute interval embedding r'' = r ⊕ r'  (Eq. 6).

        Args:
            v_r:       (B, L) absolute interval values (distances or times).
            positions: (B, L) 1-based position indices within the trajectory.

        Returns:
            r_out: (B, L, 2 * d_model) concatenated [r; r'].
        """
        r = self._slot_interpolate(v_r)            # (B, L, d_model)
        r_prime = self._decay(r, v_r, positions)   # (B, L, d_model)
        return torch.cat([r, r_prime], dim=-1)      # (B, L, 2*d_model)


class SpatialTemporalEmbedding(nn.Module):
    """
    Combines distance and time interval embeddings into a single tensor.

    Output dimension: 4 * d_model  (2*d_model per interval type, concatenated).
    """

    def __init__(
        self,
        n_distance_slots: int,
        n_time_slots: int,
        d_model: int,
        max_distance: float,
        max_time: float,
    ) -> None:
        super().__init__()
        self.distance_emb = IntervalEmbedding(n_distance_slots, d_model, max_distance)
        self.time_emb = IntervalEmbedding(n_time_slots, d_model, max_time)

    def forward(
        self,
        distances: Tensor,
        times: Tensor,
        positions: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            distances: (B, L) distance intervals in metres.
            times:     (B, L) time intervals in seconds.
            positions: (B, L) 1-based position indices.

        Returns:
            s_emb: (B, L, 2*d_model) spatial interval embedding.
            t_emb: (B, L, 2*d_model) temporal interval embedding.
        """
        s_emb = self.distance_emb(distances, positions)
        t_emb = self.time_emb(times, positions)
        return s_emb, t_emb

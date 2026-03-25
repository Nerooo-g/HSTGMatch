"""
Full HSTGMatch model integrating all modules (Section 4).

Two classes:
  HSTGMatchPretrainer — self-supervised phase (encoder + two FC heads).
  HSTGMatch           — supervised map-matching (full Seq2Seq).
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graph_embedding import GridGraphEncoder
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .spatial_temporal import SpatialTemporalEmbedding


# ---------------------------------------------------------------------------
# Pre-training model
# ---------------------------------------------------------------------------

class HSTGMatchPretrainer(nn.Module):
    """
    Self-supervised pre-training module.

    Input representation (Section 4.2):
        Each token i = concat( graph_grid_embedding[grid_id_i], lon_i, lat_i )

    Masking: span masking with a single shared sentinel token substituted
             for masked positions.

    Loss (Equation 3):
        L = k  * CE(grid_prediction)
          + (1-k) * RMSE(coord_prediction)

    Only the encoder is trained; no seq2seq decoder is used here.
    """

    def __init__(
        self,
        n_grids: int,
        d_model: int = 128,
        n_encoder_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        n_gat_layers: int = 2,
        n_gat_heads: int = 4,
        gat_dropout: float = 0.1,
        max_len: int = 512,
        k: float = 0.5,
        mask_ratio: float = 0.15,
        span_len: int = 3,
    ) -> None:
        super().__init__()
        self.n_grids = n_grids
        self.d_model = d_model
        self.k = k
        self.mask_ratio = mask_ratio
        self.span_len = span_len

        # Graph encoder producing grid embeddings
        self.graph_encoder = GridGraphEncoder(
            n_grids=n_grids,
            d_model=d_model,
            n_layers=n_gat_layers,
            n_heads=n_gat_heads,
            dropout=gat_dropout,
        )

        # Coordinate projection: (lon, lat) → d_model
        self.coord_proj = nn.Linear(2, d_model)

        # Sentinel embedding — lives in raw input space (2 * d_model)
        # so it can be substituted before the input projection.
        self.sentinel = nn.Parameter(torch.randn(2 * d_model))

        # Transformer encoder
        # Input dim: d_model (grid emb) + d_model (coord) = 2 * d_model
        self.encoder = TransformerEncoder(
            input_dim=2 * d_model,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )

        # Prediction heads
        self.grid_head = nn.Linear(d_model, n_grids)      # classification
        self.coord_head = nn.Linear(d_model, 2)           # regression (lon, lat)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.coord_proj.weight)
        nn.init.zeros_(self.coord_proj.bias)
        nn.init.xavier_uniform_(self.grid_head.weight)
        nn.init.zeros_(self.grid_head.bias)
        nn.init.xavier_uniform_(self.coord_head.weight)
        nn.init.zeros_(self.coord_head.bias)

    # ------------------------------------------------------------------
    # Span masking
    # ------------------------------------------------------------------

    def _create_span_mask(self, seq_len: int, batch_size: int, device: torch.device) -> Tensor:
        """
        Create a span mask for a batch.  For each sequence, independently
        sample spans of length `span_len` until approximately `mask_ratio`
        of positions are masked.

        Returns:
            mask: (B, L) bool tensor — True where position is masked.
        """
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        target_masked = max(1, int(seq_len * self.mask_ratio))

        for b in range(batch_size):
            n_masked = 0
            attempts = 0
            while n_masked < target_masked and attempts < seq_len * 10:
                start = torch.randint(0, seq_len, (1,)).item()
                end = min(start + self.span_len, seq_len)
                mask[b, start:end] = True
                n_masked = mask[b].sum().item()
                attempts += 1
        return mask

    # ------------------------------------------------------------------
    # Build token features
    # ------------------------------------------------------------------

    def build_token_features(
        self,
        grid_ids: Tensor,
        coords: Tensor,
        edge_index: Tensor,
        edge_weights: Tensor,
    ) -> Tensor:
        """
        Construct per-token feature vectors.

        Args:
            grid_ids:     (B, L) long tensor of grid cell IDs.
            coords:       (B, L, 2) normalised (lon, lat) pairs.
            edge_index:   (2, E) graph edges.
            edge_weights: (E,) graph edge weights.

        Returns:
            features: (B, L, 2*d_model)
        """
        B, L = grid_ids.shape

        # Get all grid embeddings (N_grids, d_model)
        all_grid_embs = self.graph_encoder(edge_index, edge_weights)  # (N, d_model)

        # Index-select for each token  (B, L, d_model)
        flat_ids = grid_ids.reshape(-1)                         # (B*L,)
        flat_embs = all_grid_embs[flat_ids]                     # (B*L, d_model)
        grid_embs = flat_embs.view(B, L, self.d_model)          # (B, L, d_model)

        # Project coordinates
        coord_embs = self.coord_proj(coords)                    # (B, L, d_model)

        return torch.cat([grid_embs, coord_embs], dim=-1)       # (B, L, 2*d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        grid_ids: Tensor,
        coords: Tensor,
        edge_index: Tensor,
        edge_weights: Tensor,
        padding_mask: Optional[Tensor] = None,
        apply_mask: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Args:
            grid_ids:     (B, L) long — grid cell IDs.
            coords:       (B, L, 2) float — normalised (lon, lat).
            edge_index:   (2, E) long.
            edge_weights: (E,) float.
            padding_mask: (B, L) bool — True at padded positions.
            apply_mask:   If True, span-mask the input for self-supervised training.

        Returns:
            dict with keys:
              'grid_logits'  : (B, L, n_grids) — grid classification logits.
              'coord_preds'  : (B, L, 2)       — predicted (lon, lat).
              'mask'         : (B, L) bool      — positions that were masked.
              'enc_output'   : (B, L, d_model)  — encoder output.
        """
        B, L = grid_ids.shape
        device = grid_ids.device

        features = self.build_token_features(grid_ids, coords, edge_index, edge_weights)
        # (B, L, 2*d_model)

        mask = None
        # sentinel lives in input space (2 * d_model)
        sentinel = self.sentinel  # (2 * d_model,)

        if apply_mask:
            # Create span mask
            mask = self._create_span_mask(L, B, device)
            # sentinel already matches input_dim (2 * d_model)
            sentinel_input = sentinel.view(1, 1, 2 * self.d_model)

            enc_out = self.encoder(
                features,
                src_key_padding_mask=padding_mask,
                mask_positions=mask,
                sentinel_embedding=sentinel_input,
            )
        else:
            enc_out = self.encoder(features, src_key_padding_mask=padding_mask)

        grid_logits = self.grid_head(enc_out)    # (B, L, n_grids)
        coord_preds = self.coord_head(enc_out)   # (B, L, 2)

        return {
            "grid_logits": grid_logits,
            "coord_preds": coord_preds,
            "mask": mask,
            "enc_output": enc_out,
        }

    def compute_loss(
        self,
        grid_logits: Tensor,
        coord_preds: Tensor,
        grid_ids: Tensor,
        coords: Tensor,
        mask: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Equation 3:
            L = k * CE(grid) + (1-k) * RMSE(coord)

        Only computes loss on masked positions.
        """
        # Combine masks: compute only on positions that are masked AND not padded
        active = mask.clone()
        if padding_mask is not None:
            active = active & ~padding_mask

        if active.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=grid_logits.device)

        # --- Grid classification loss (cross-entropy) ---
        # (N_active, n_grids) vs (N_active,)
        logits_flat = grid_logits[active]          # (N, n_grids)
        targets_flat = grid_ids[active]            # (N,)
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")

        # --- Coordinate regression loss (RMSE) ---
        pred_coords = coord_preds[active]          # (N, 2)
        true_coords = coords[active]               # (N, 2)
        mse = ((pred_coords - true_coords) ** 2).mean()
        rmse_loss = torch.sqrt(mse + 1e-8)

        return self.k * ce_loss + (1.0 - self.k) * rmse_loss

    def get_encoder(self) -> TransformerEncoder:
        """Return the encoder module (used to initialise supervised model)."""
        return self.encoder


# ---------------------------------------------------------------------------
# Full supervised model
# ---------------------------------------------------------------------------

class HSTGMatch(nn.Module):
    """
    Full HSTGMatch model for supervised map-matching (Equations 7-11).

    Architecture (Eq. 7):
        E = Encoder( W' * [ W_s * h_s  ⊕  s  ⊕  t ] )

    where:
        h_s = pretrained graph-encoder output for each grid token
        s   = spatial interval embedding  (2 * d_model)
        t   = temporal interval embedding (2 * d_model)
        Total input to Encoder: d_model + 2*d_model + 2*d_model = 5*d_model

    E' = Decoder(E, R)          — teacher forcing with road segment IDs
    Z  = E' * W_q^T
    P  = softmax(Z)
    """

    def __init__(
        self,
        n_grids: int,
        n_segments: int,
        d_model: int = 128,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        n_gat_layers: int = 2,
        n_gat_heads: int = 4,
        gat_dropout: float = 0.1,
        n_distance_slots: int = 64,
        n_time_slots: int = 64,
        max_distance: float = 10000.0,
        max_time: float = 3600.0,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_grids = n_grids
        self.n_segments = n_segments

        # ------ Graph encoder (shared with pretrainer if weights loaded) ------
        self.graph_encoder = GridGraphEncoder(
            n_grids=n_grids,
            d_model=d_model,
            n_layers=n_gat_layers,
            n_heads=n_gat_heads,
            dropout=gat_dropout,
        )

        # ------ Coordinate projection ------
        self.coord_proj = nn.Linear(2, d_model)

        # ------ Spatial-temporal embeddings ------
        self.st_embedding = SpatialTemporalEmbedding(
            n_distance_slots=n_distance_slots,
            n_time_slots=n_time_slots,
            d_model=d_model,
            max_distance=max_distance,
            max_time=max_time,
        )

        # ------ Input projection W' (Eq. 7) ------
        # h_s: d_model; s: 2*d_model; t: 2*d_model  → total: 5*d_model
        encoder_input_dim = 5 * d_model
        self.input_proj = nn.Linear(encoder_input_dim, d_model)

        # ------ Transformer encoder ------
        # input_dim = d_model because we project above before passing to encoder
        self.encoder = TransformerEncoder(
            input_dim=d_model,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )

        # ------ Transformer decoder ------
        self.decoder = TransformerDecoder(
            n_segments=n_segments,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.coord_proj.weight)
        nn.init.zeros_(self.coord_proj.bias)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def load_pretrained_encoder(self, pretrainer: HSTGMatchPretrainer) -> None:
        """
        Copy graph encoder and transformer backbone weights from a pretrained
        HSTGMatchPretrainer instance.

        The pretrainer's encoder has input_dim=2*d_model, while this model's
        encoder has input_dim=d_model (because we apply input_proj externally
        before passing to the encoder).  Therefore only the Transformer
        backbone layers (pos_enc + transformer) are transferred; the
        input_proj is intentionally kept different and re-trained.

        The graph encoder and coord_proj weights are transferred in full.
        """
        self.graph_encoder.load_state_dict(pretrainer.graph_encoder.state_dict())
        self.coord_proj.load_state_dict(pretrainer.coord_proj.state_dict())

        # Copy only the Transformer backbone, skip input_proj (different shape)
        pretrain_enc_state = pretrainer.encoder.state_dict()
        own_enc_state = self.encoder.state_dict()
        transferable = {
            k: v for k, v in pretrain_enc_state.items()
            if k in own_enc_state and own_enc_state[k].shape == v.shape
        }
        own_enc_state.update(transferable)
        self.encoder.load_state_dict(own_enc_state)

    def _build_encoder_input(
        self,
        grid_ids: Tensor,
        coords: Tensor,
        distances: Tensor,
        times: Tensor,
        positions: Tensor,
        edge_index: Tensor,
        edge_weights: Tensor,
    ) -> Tensor:
        """
        Build the combined input token features for the encoder.

        Returns:
            features: (B, L, d_model) — projected encoder input.
        """
        B, L = grid_ids.shape

        # Grid embeddings via graph encoder
        all_grid_embs = self.graph_encoder(edge_index, edge_weights)  # (N_grids, d_model)
        flat_ids = grid_ids.reshape(-1)
        h_s = all_grid_embs[flat_ids].view(B, L, self.d_model)         # (B, L, d_model)

        # Spatial-temporal interval embeddings
        s_emb, t_emb = self.st_embedding(distances, times, positions)  # each (B, L, 2*d_model)

        # Concatenate: [h_s; s; t] → (B, L, 5*d_model)
        combined = torch.cat([h_s, s_emb, t_emb], dim=-1)

        # Project W' → (B, L, d_model)
        return self.input_proj(combined)

    def encode(
        self,
        grid_ids: Tensor,
        coords: Tensor,
        distances: Tensor,
        times: Tensor,
        positions: Tensor,
        edge_index: Tensor,
        edge_weights: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Run the encoder part (Eq. 7).

        Returns:
            memory: (B, L, d_model)
        """
        features = self._build_encoder_input(
            grid_ids, coords, distances, times, positions, edge_index, edge_weights
        )
        return self.encoder(features, src_key_padding_mask=src_key_padding_mask)

    def forward(
        self,
        grid_ids: Tensor,
        coords: Tensor,
        distances: Tensor,
        times: Tensor,
        positions: Tensor,
        edge_index: Tensor,
        edge_weights: Tensor,
        tgt_ids: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Full forward pass (teacher forcing).

        Args:
            grid_ids:     (B, L_enc) trajectory grid IDs.
            coords:       (B, L_enc, 2) normalised coordinates.
            distances:    (B, L_enc) distance intervals.
            times:        (B, L_enc) time intervals.
            positions:    (B, L_enc) 1-based position indices.
            edge_index:   (2, E)
            edge_weights: (E,)
            tgt_ids:      (B, L_dec) decoder target sequence (vocab IDs,
                          starting with BOS).
            src_key_padding_mask: (B, L_enc) bool.
            tgt_key_padding_mask: (B, L_dec) bool.

        Returns:
            logits: (B, L_dec, n_segments)
        """
        memory = self.encode(
            grid_ids, coords, distances, times, positions,
            edge_index, edge_weights,
            src_key_padding_mask=src_key_padding_mask,
        )
        logits = self.decoder(
            memory,
            tgt_ids,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return logits

    def compute_loss(
        self,
        logits: Tensor,
        target_seg_ids: Tensor,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Cross-entropy loss over non-padded decoder positions (Eq. 11).

        Args:
            logits:           (B, L_dec, n_segments).
            target_seg_ids:   (B, L_dec) raw seg_ids (0-indexed).
            tgt_key_padding_mask: (B, L_dec) bool — True at padded positions.

        Returns:
            loss: scalar.
        """
        B, L, V = logits.shape
        logits_flat = logits.reshape(B * L, V)
        targets_flat = target_seg_ids.reshape(B * L)

        if tgt_key_padding_mask is not None:
            # Ignore padded positions: replace target with -100
            pad_flat = tgt_key_padding_mask.reshape(B * L)
            targets_flat = targets_flat.masked_fill(pad_flat, -100)

        return F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)

    def predict(
        self,
        grid_ids: Tensor,
        coords: Tensor,
        distances: Tensor,
        times: Tensor,
        positions: Tensor,
        edge_index: Tensor,
        edge_weights: Tensor,
        max_decode_len: int,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Greedy decoding for inference.

        Returns:
            pred_seg_ids: (B, max_decode_len) — raw 0-indexed seg IDs.
        """
        memory = self.encode(
            grid_ids, coords, distances, times, positions,
            edge_index, edge_weights,
            src_key_padding_mask=src_key_padding_mask,
        )
        return self.decoder.greedy_decode(
            memory, max_decode_len,
            memory_key_padding_mask=src_key_padding_mask,
        )

"""
Transformer Encoder for HSTGMatch (Section 4).

During pre-training the encoder receives:
    [graph_grid_embedding + (lon, lat) projection]  with optional span masking.

During supervised training the encoder receives the full feature vector
described in Equation 7:
    E = Encoder( W' * [ W_s * h_s  ⊕  s  ⊕  t ] )

where h_s is the pretrained grid embedding, s is the spatial interval
embedding, and t is the temporal interval embedding.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            x + pe[:, :L, :] with dropout applied.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Encoder
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    Standard Transformer encoder with sinusoidal positional encoding.

    Supports:
    - A projection layer (input_dim -> d_model) so the caller can pass
      concatenated feature vectors of arbitrary width.
    - Padding mask for variable-length sequences.
    - A boolean mask for self-supervised span masking.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Project input features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
        sentinel_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (B, L, input_dim) input feature sequences.
            src_key_padding_mask: (B, L) bool tensor — True where padded.
            mask_positions: (B, L) bool tensor — True at masked positions
                            (used during self-supervised pre-training).
            sentinel_embedding: (input_dim,) or (1, 1, input_dim) — the single
                                shared sentinel token substituted at masked
                                positions in the *raw* input space (before
                                the input projection).

        Returns:
            enc_out: (B, L, d_model) contextualised representations.
        """
        # Apply span masking on raw input features (before projection)
        if mask_positions is not None and sentinel_embedding is not None:
            input_dim = x.shape[-1]
            sent = sentinel_embedding.view(1, 1, input_dim)
            mask_3d = mask_positions.unsqueeze(-1).float()  # (B, L, 1)
            x = x * (1.0 - mask_3d) + sent * mask_3d

        # Project to d_model
        h = self.input_proj(x)  # (B, L, d_model)

        h = self.pos_enc(h)

        enc_out = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        return enc_out

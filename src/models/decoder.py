"""
Transformer Decoder for HSTGMatch (Equations 8-10).

During supervised training (teacher forcing):
    E' = Decoder(E, R)

where E is the encoder output and R is the road segment embedding sequence
(right-shifted by one position with a start token).

The decoder output is projected to road-segment logits via W_q (Eq. 10):
    Z = E' * W_q^T
    P(y=c | ...) = softmax(Z)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .encoder import SinusoidalPositionalEncoding


class TransformerDecoder(nn.Module):
    """
    Standard Transformer decoder with:
    - Road segment embedding table (n_segments + 2 special tokens)
    - Sinusoidal positional encoding
    - Cross-attention over encoder memory
    - Output projection to segment logits

    Special tokens:
        0 : PAD
        1 : BOS  (beginning of sequence — start token for teacher forcing)
    Segment IDs start from 2 (raw seg_id + 2 internally).
    """

    PAD_ID = 0
    BOS_ID = 1

    def __init__(
        self,
        n_segments: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_vocab = n_segments + 2  # PAD + BOS + segments

        # Road segment embeddings (includes PAD and BOS)
        self.seg_embedding = nn.Embedding(self.n_vocab, d_model, padding_idx=self.PAD_ID)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection W_q  (Eq. 10):  Z = E' * W_q^T
        self.output_proj = nn.Linear(d_model, n_segments, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.seg_embedding.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
        """Upper-triangular causal mask (True = ignore)."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
        return mask

    def forward(
        self,
        memory: Tensor,
        tgt_ids: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            memory:  (B, L_enc, d_model) encoder output.
            tgt_ids: (B, L_dec) long tensor of target road segment IDs
                     (already shifted: starts with BOS_ID, then seg_id+2).
            memory_key_padding_mask: (B, L_enc) bool — True at padded encoder positions.
            tgt_key_padding_mask:    (B, L_dec) bool — True at padded decoder positions.

        Returns:
            logits: (B, L_dec, n_segments) unnormalised scores.
        """
        tgt_emb = self.seg_embedding(tgt_ids)  # (B, L_dec, d_model)
        tgt_emb = self.pos_enc(tgt_emb)

        L_dec = tgt_ids.shape[1]
        causal_mask = self._make_causal_mask(L_dec, device=tgt_ids.device)

        dec_out = self.transformer(
            tgt_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, L_dec, d_model)

        logits = self.output_proj(dec_out)  # (B, L_dec, n_segments)
        return logits

    def decode_segment_ids(self, raw_seg_ids: Tensor) -> Tensor:
        """
        Convert raw 0-indexed seg_ids to internal vocab IDs (adds 2 offset).
        """
        return raw_seg_ids + 2

    def encode_segment_ids(self, vocab_ids: Tensor) -> Tensor:
        """
        Convert internal vocab IDs back to 0-indexed seg_ids (subtracts 2).
        """
        return vocab_ids - 2

    def greedy_decode(
        self,
        memory: Tensor,
        max_len: int,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Greedy auto-regressive decoding (inference only).

        Args:
            memory: (B, L_enc, d_model)
            max_len: maximum number of decoding steps.
            memory_key_padding_mask: (B, L_enc) bool mask.

        Returns:
            pred_ids: (B, max_len) predicted segment IDs (0-indexed, raw).
        """
        B = memory.shape[0]
        device = memory.device

        # Start with BOS token
        decoded = torch.full((B, 1), self.BOS_ID, dtype=torch.long, device=device)

        for _ in range(max_len):
            logits = self.forward(
                memory,
                decoded,
                memory_key_padding_mask=memory_key_padding_mask,
            )  # (B, t, n_segments)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1) — vocab offset
            # next_id here is the raw seg_id index in n_segments space (no +2 offset in logits)
            # We need to store it as vocab_id for next embedding lookup
            next_vocab_id = next_id + 2  # convert to vocab ID
            decoded = torch.cat([decoded, next_vocab_id], dim=1)

        # Return raw seg_ids (remove BOS, subtract 2)
        return decoded[:, 1:] - 2  # (B, max_len)

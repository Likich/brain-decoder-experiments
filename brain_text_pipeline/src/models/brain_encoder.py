"""Temporal brain encoder."""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def sinusoidal_positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(length, dim, device=device)
    position = torch.arange(0, length, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class BrainEncoder(nn.Module):
    def __init__(
        self,
        brain_dim: int,
        d_model: int,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(brain_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, brain_seq: torch.Tensor, brain_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # brain_seq: [B, T, D]
        assert brain_seq.ndim == 3, f"brain_seq must be [B,T,D], got {brain_seq.shape}"
        b, t, _ = brain_seq.shape
        h = self.proj(brain_seq)
        pe = sinusoidal_positional_encoding(t, h.size(-1), h.device)
        h = h + pe.unsqueeze(0)
        h = self.dropout(h)
        key_padding_mask = None
        if brain_mask is not None:
            key_padding_mask = brain_mask.eq(0)
        return self.encoder(h, src_key_padding_mask=key_padding_mask)

"""Training losses."""
from __future__ import annotations

import torch
from torch import nn


def mse_next_step(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask: [B, T]
    diff = (pred - target) ** 2
    diff = diff.mean(dim=-1)
    diff = diff * mask
    return diff.sum() / mask.sum().clamp_min(1.0)

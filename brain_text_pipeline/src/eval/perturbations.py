"""Brain signal perturbations."""
from __future__ import annotations

import torch


def temporal_shuffle(x: torch.Tensor) -> torch.Tensor:
    idx = torch.randperm(x.size(1), device=x.device)
    return x[:, idx, :]


def add_noise(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    return x + sigma * torch.randn_like(x)


def channel_dropout(x: torch.Tensor, drop_prob: float = 0.2) -> torch.Tensor:
    mask = torch.rand(x.size(-1), device=x.device) > drop_prob
    return x * mask


def time_shift(x: torch.Tensor, shift: int = 5) -> torch.Tensor:
    if shift == 0:
        return x
    return torch.roll(x, shifts=shift, dims=1)


def smooth_time(x: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    if kernel <= 1:
        return x
    pad = kernel // 2
    x_pad = torch.nn.functional.pad(x, (0, 0, pad, pad), mode="replicate")
    w = torch.ones(kernel, device=x.device) / kernel
    out = torch.einsum("btd,k->btd", x_pad.unfold(1, kernel, 1).mean(dim=-1), w)
    return out

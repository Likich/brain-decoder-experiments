"""Evaluation metrics."""
from __future__ import annotations

import torch


def js_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.clamp_min(eps).log() - b.clamp_min(eps).log())).sum(dim=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def top1_agreement(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p.argmax(dim=-1) == q.argmax(dim=-1)).float().mean()

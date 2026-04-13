"""Minimal training utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import torch


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0


def run_epoch(
    model,
    dataloader: Iterable,
    optimizer,
    device: torch.device,
    loss_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    train: bool = True,
) -> float:
    if train:
        model.train()
    else:
        model.eval()
    total = 0.0
    count = 0
    for batch in dataloader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(batch)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = loss_fn(batch)
        total += loss.item()
        count += 1
    return total / max(1, count)

"""Utilities for TVB synthetic data."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def load_tvb_sequence(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "brain_seq" in data:
            return data["brain_seq"]
        # fallback: first array
        return data[list(data.files)[0]]
    raise ValueError(f"Unsupported TVB file: {path}")


def list_tvb_files(root: Path, pattern: str = "*.npy") -> List[Path]:
    return sorted(root.rglob(pattern))

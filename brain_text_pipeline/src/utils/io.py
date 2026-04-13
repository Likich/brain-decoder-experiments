"""Shard IO and manifest helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def read_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


class ShardWriter:
    """Accumulate examples and write to sharded NPZ files."""

    def __init__(
        self,
        out_dir: Path,
        prefix: str,
        shard_size: int = 5000,
    ):
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.buffers: dict[str, list] = {}
        self.shard_idx = 0
        self.count = 0
        self.shards: list[dict] = []

    def add(self, item: dict[str, Any]) -> None:
        for k, v in item.items():
            self.buffers.setdefault(k, []).append(v)
        self.count += 1
        if self.count % self.shard_size == 0:
            self.flush()

    def flush(self) -> None:
        if not self.buffers:
            return
        shard_name = f"{self.prefix}_shard_{self.shard_idx:04d}.npz"
        shard_path = self.out_dir / shard_name
        arrays = {}
        for k, vals in self.buffers.items():
            # Use object arrays for variable-length sequences
            arrays[k] = np.array(vals, dtype=object)
        save_npz(shard_path, arrays)
        self.shards.append({"path": str(shard_path), "size": len(next(iter(self.buffers.values())))})
        self.buffers = {}
        self.shard_idx += 1

    def finalize(self) -> dict:
        if self.buffers:
            self.flush()
        return {
            "prefix": self.prefix,
            "shard_size": self.shard_size,
            "num_shards": len(self.shards),
            "total_examples": sum(s["size"] for s in self.shards),
            "shards": self.shards,
        }

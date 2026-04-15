"""Dataset utilities for sharded brain-text datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from brain_text_pipeline.src.utils.io import read_manifest


class ShardedExampleDataset(Dataset):
    def __init__(self, manifest_path: Path):
        self.manifest = read_manifest(manifest_path)
        self.shards = self.manifest["shards"]
        self._index = []
        for shard_idx, shard in enumerate(self.shards):
            for i in range(shard["size"]):
                self._index.append((shard_idx, i))
        self._cache = {}

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, shard_idx: int) -> dict[str, Any]:
        if shard_idx in self._cache:
            return self._cache[shard_idx]
        shard_path = Path(self.shards[shard_idx]["path"])
        data = np.load(shard_path, allow_pickle=True)
        self._cache[shard_idx] = data
        return data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_idx, item_idx = self._index[idx]
        data = self._load_shard(shard_idx)
        out = {}
        for key in data.files:
            val = data[key][item_idx]
            if key == "meta" and isinstance(val, (str, bytes)):
                out[key] = json.loads(val)
            else:
                out[key] = val
        return out


class TVBSequenceDataset(Dataset):
    def __init__(self, manifest_path: Path):
        self.manifest = read_manifest(manifest_path)
        self.shards = self.manifest["shards"]
        self._index = []
        for shard_idx, shard in enumerate(self.shards):
            for i in range(shard["size"]):
                self._index.append((shard_idx, i))
        self._cache = {}

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, shard_idx: int) -> dict[str, Any]:
        if shard_idx in self._cache:
            return self._cache[shard_idx]
        shard_path = Path(self.shards[shard_idx]["path"])
        data = np.load(shard_path, allow_pickle=True)
        self._cache[shard_idx] = data
        return data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_idx, item_idx = self._index[idx]
        data = self._load_shard(shard_idx)
        return {k: data[k][item_idx] for k in data.files}

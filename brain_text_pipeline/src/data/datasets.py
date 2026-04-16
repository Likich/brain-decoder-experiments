"""Dataset utilities for sharded brain-text datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from brain_text_pipeline.src.utils.io import read_manifest


def _load_npz_as_dict(path: Path) -> dict[str, Any]:
    """Load an NPZ once.

    NumPy's ``NpzFile`` lazily reads arrays from the zip container. If we keep
    the ``NpzFile`` object and index ``data[key]`` inside every ``__getitem__``,
    shuffled training repeatedly re-reads whole arrays from disk. Materializing
    the arrays once per shard makes random access cheap.
    """
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _coerce_numeric_item(key: str, value: Any) -> Any:
    """Best-effort conversion for object-dtype shards."""
    if key == "meta":
        return value
    if not isinstance(value, np.ndarray) or value.dtype != object:
        return value

    if key in {"brain_seq", "brain_target"}:
        return value.astype(np.float32, copy=False)
    if key in {"input_ids_context", "decoder_target_ids", "attention_mask_text", "brain_mask"}:
        return value.astype(np.int64, copy=False)
    return value


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
        data = _load_npz_as_dict(shard_path)
        self._cache[shard_idx] = data
        return data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_idx, item_idx = self._index[idx]
        data = self._load_shard(shard_idx)
        out = {}
        for key in data.keys():
            val = data[key][item_idx]
            if key == "meta" and isinstance(val, (str, bytes)):
                out[key] = json.loads(val)
            else:
                out[key] = _coerce_numeric_item(key, val)
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
        data = _load_npz_as_dict(shard_path)
        self._cache[shard_idx] = data
        return data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_idx, item_idx = self._index[idx]
        data = self._load_shard(shard_idx)
        return {k: _coerce_numeric_item(k, data[k][item_idx]) for k in data.keys()}

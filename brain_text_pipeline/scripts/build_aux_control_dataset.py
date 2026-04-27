#!/usr/bin/env python3
"""Build a matched-shape auxiliary-control dataset from a MEG manifest.

This keeps the text targets, metadata, sequence lengths, and temporal shapes
unchanged while replacing ``brain_seq`` with a synthetic auxiliary stream.
It lets us test whether the same T5+brain-encoder path improves prediction for
arbitrary continuous side channels, without changing the training or eval code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_text_pipeline.src.utils.io import ShardWriter, read_manifest, write_manifest
from brain_text_pipeline.src.utils.logging import log


def shard_path(source_manifest: Path, shard: dict[str, Any]) -> Path:
    path = Path(shard["path"])
    if path.is_absolute() or path.exists():
        return path
    candidate = source_manifest.parent / path
    return candidate if candidate.exists() else path


def iter_shard_items(manifest_path: Path, manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for shard in manifest["shards"]:
        path = shard_path(manifest_path, shard)
        with np.load(path, allow_pickle=True) as data:
            keys = list(data.files)
            size = int(shard.get("size", len(data[keys[0]])))
            arrays = {key: data[key] for key in keys}
            for item_idx in range(size):
                yield {key: arrays[key][item_idx] for key in keys}


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    meta = out.get("meta", {})
    if isinstance(meta, dict):
        out["meta"] = json.dumps(meta)
    return out


def coerce_float_matrix(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = arr.astype(np.float32, copy=False)
    else:
        arr = arr.astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"brain_seq must be 2D [T,D], got shape {arr.shape}")
    return arr


def fit_global_channel_stats(manifest_path: Path, manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    sum_x: np.ndarray | None = None
    sum_x2: np.ndarray | None = None
    count = 0
    for item in iter_shard_items(manifest_path, manifest):
        brain_seq = coerce_float_matrix(item["brain_seq"])
        if sum_x is None:
            sum_x = brain_seq.sum(axis=0, dtype=np.float64)
            sum_x2 = np.square(brain_seq, dtype=np.float64).sum(axis=0, dtype=np.float64)
        else:
            sum_x += brain_seq.sum(axis=0, dtype=np.float64)
            sum_x2 += np.square(brain_seq, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += int(brain_seq.shape[0])
    if sum_x is None or sum_x2 is None or count <= 0:
        raise ValueError("source dataset is empty; cannot estimate channel statistics")
    mean = sum_x / count
    var = np.maximum(sum_x2 / count - np.square(mean), 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def synth_aux(
    brain_seq: np.ndarray,
    *,
    rng: np.random.Generator,
    mode: str,
    global_mean: np.ndarray | None,
    global_std: np.ndarray | None,
) -> np.ndarray:
    shape = brain_seq.shape
    if mode == "gaussian_iid":
        return rng.standard_normal(shape, dtype=np.float32)
    if mode == "gaussian_global":
        assert global_mean is not None and global_std is not None
        sample = rng.standard_normal(shape, dtype=np.float32)
        return sample * global_std[None, :] + global_mean[None, :]
    raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--source_manifest", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument(
        "--mode",
        choices=["gaussian_iid", "gaussian_global"],
        default="gaussian_iid",
        help=(
            "gaussian_iid samples N(0,1) per timepoint/channel; "
            "gaussian_global matches the source dataset's per-channel mean/std"
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard_size", type=int, default=None, help="defaults to source shard size")
    ap.add_argument("--report_every", type=int, default=10000)
    args = ap.parse_args()

    manifest = read_manifest(args.source_manifest)
    rng = np.random.default_rng(args.seed)
    shard_size = int(args.shard_size or manifest.get("shard_size", 5000))

    global_mean = None
    global_std = None
    if args.mode == "gaussian_global":
        log("estimating global per-channel mean/std from source dataset")
        global_mean, global_std = fit_global_channel_stats(args.source_manifest, manifest)

    writer = ShardWriter(args.out_dir, prefix="aux", shard_size=shard_size)
    total = 0
    for item in iter_shard_items(args.source_manifest, manifest):
        out = normalize_item(item)
        brain_seq = coerce_float_matrix(out["brain_seq"])
        out["brain_seq"] = synth_aux(
            brain_seq,
            rng=rng,
            mode=args.mode,
            global_mean=global_mean,
            global_std=global_std,
        )
        writer.add(out)
        total += 1
        if args.report_every and total % args.report_every == 0:
            log(f"processed {total}/{manifest.get('total_examples', '?')} examples")

    out_manifest = writer.finalize()
    source_metadata = {
        key: value
        for key, value in manifest.items()
        if key not in {"shards", "num_shards", "total_examples", "prefix", "shard_size"}
    }
    out_manifest.update(source_metadata)
    out_manifest.update(
        {
            "source_manifest": str(args.source_manifest),
            "aux_control": {
                "mode": args.mode,
                "seed": args.seed,
            },
        }
    )
    write_manifest(args.out_dir / "manifest.json", out_manifest)
    log(f"saved {total} auxiliary-control examples to {args.out_dir}")


if __name__ == "__main__":
    main()

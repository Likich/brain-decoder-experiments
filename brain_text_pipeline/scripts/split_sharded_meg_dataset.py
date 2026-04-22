#!/usr/bin/env python3
"""Split a sharded MEG dataset into train/test shards.

This materializes new shard directories so training and evaluation can use
separate manifests without changing the dataset class. It streams source shards
instead of indexing through ``ShardedExampleDataset`` so large MEG arrays do not
all stay cached in memory during split construction.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_text_pipeline.src.utils.io import ShardWriter, read_manifest, write_manifest
from brain_text_pipeline.src.utils.logging import log


def meta_value(meta: Any, key: str) -> str:
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            return ""
    if isinstance(meta, dict):
        value = meta.get(key, "")
        return "" if value is None else str(value)
    return ""


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    meta = out.get("meta", {})
    if isinstance(meta, dict):
        out["meta"] = json.dumps(meta)
    return out


def shard_path(source_manifest: Path, shard: dict[str, Any]) -> Path:
    path = Path(shard["path"])
    if path.is_absolute() or path.exists():
        return path
    candidate = source_manifest.parent / path
    return candidate if candidate.exists() else path


def iter_shard_items(manifest_path: Path, manifest: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any]]]:
    global_idx = 0
    for shard in manifest["shards"]:
        path = shard_path(manifest_path, shard)
        with np.load(path, allow_pickle=True) as data:
            keys = list(data.files)
            size = int(shard.get("size", len(data[keys[0]])))
            arrays = {key: data[key] for key in keys}
            for item_idx in range(size):
                yield global_idx, {key: arrays[key][item_idx] for key in keys}
                global_idx += 1


def load_group_values(manifest_path: Path, manifest: dict[str, Any], key: str) -> list[str]:
    values: list[str] = []
    for _, item in iter_shard_items(manifest_path, manifest):
        values.append(meta_value(item.get("meta", {}), key))
    return values


def choose_group_values(values: list[str], fraction: float, seed: int) -> set[str]:
    unique = sorted({v for v in values if v})
    if not unique:
        return set()
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * fraction)))
    return set(unique[:n_test])


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--train_out", type=Path, required=True)
    ap.add_argument("--test_out", type=Path, required=True)
    ap.add_argument(
        "--split",
        choices=["random", "subject", "story", "session", "task", "sound", "sequence_id"],
        default="random",
        help="Held-out grouping. Use story/sound for leakage-resistant language splits.",
    )
    ap.add_argument("--test_fraction", type=float, default=0.1)
    ap.add_argument("--test_subjects", nargs="*", default=None)
    ap.add_argument("--test_stories", nargs="*", default=None)
    ap.add_argument("--test_values", nargs="*", default=None, help="Explicit held-out values for the chosen split key")
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_every", type=int, default=10000)
    args = ap.parse_args()

    manifest = read_manifest(args.manifest)
    total_examples = int(manifest.get("total_examples", sum(int(s["size"]) for s in manifest["shards"])))
    rng = random.Random(args.seed)

    test_indices: set[int]
    if args.split == "random":
        idxs = list(range(total_examples))
        rng.shuffle(idxs)
        n_test = max(1, int(round(len(idxs) * args.test_fraction)))
        test_indices = set(idxs[:n_test])
        split_info = {"type": "random", "test_fraction": args.test_fraction}
    else:
        group_key = args.split
        log(f"scanning metadata for split={group_key}")
        values = load_group_values(args.manifest, manifest, group_key)
        if args.test_values:
            test_values = {str(v) for v in args.test_values}
        elif args.split == "subject" and args.test_subjects:
            test_values = {str(v) for v in args.test_subjects}
        elif args.split == "story" and args.test_stories:
            test_values = {str(v) for v in args.test_stories}
        else:
            test_values = choose_group_values(values, args.test_fraction, args.seed)
        test_indices = {i for i, value in enumerate(values) if value in test_values}
        split_info = {
            "type": args.split,
            "test_fraction": args.test_fraction,
            "test_values": sorted(test_values),
        }

    train_writer = ShardWriter(args.train_out, prefix="meg_train", shard_size=args.shard_size)
    test_writer = ShardWriter(args.test_out, prefix="meg_test", shard_size=args.shard_size)

    train_n = 0
    test_n = 0
    for i, item in iter_shard_items(args.manifest, manifest):
        item = normalize_item(item)
        if i in test_indices:
            test_writer.add(item)
            test_n += 1
        else:
            train_writer.add(item)
            train_n += 1
        if args.report_every and (i + 1) % args.report_every == 0:
            log(f"processed {i + 1}/{total_examples} examples")

    train_manifest = train_writer.finalize()
    test_manifest = test_writer.finalize()
    source_metadata = {
        key: value
        for key, value in manifest.items()
        if key not in {"shards", "num_shards", "total_examples", "prefix", "shard_size"}
    }
    for out_manifest, name, count in ((train_manifest, "train", train_n), (test_manifest, "test", test_n)):
        out_manifest.update(source_metadata)
        out_manifest.update(
            {
                "source_manifest": str(args.manifest),
                "split": split_info,
                "split_name": name,
                "total_examples": count,
            }
        )

    write_manifest(args.train_out / "manifest.json", train_manifest)
    write_manifest(args.test_out / "manifest.json", test_manifest)
    log(f"wrote train={train_n} to {args.train_out}")
    log(f"wrote test={test_n} to {args.test_out}")


if __name__ == "__main__":
    main()

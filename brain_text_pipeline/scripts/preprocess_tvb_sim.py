#!/usr/bin/env python3
"""Preprocess TVB simulation outputs into sharded sequences."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_text_pipeline.src.utils.io import ShardWriter, write_manifest
from brain_text_pipeline.src.utils.logging import log


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input_dir", type=Path, required=True, help="Directory with TVB .npy sequences")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--shard_size", type=int, default=1000)
    ap.add_argument("--zscore", action="store_true")
    args = ap.parse_args()

    writer = ShardWriter(args.out_dir, prefix="tvb", shard_size=args.shard_size)
    files = sorted(args.input_dir.glob("*.npy"))
    if not files:
        raise SystemExit("No .npy files found")

    total = 0
    for f in files:
        seq = np.load(f)
        if args.zscore:
            seq = (seq - seq.mean(axis=0, keepdims=True)) / (seq.std(axis=0, keepdims=True) + 1e-6)
        item = {
            "brain_seq": np.array(seq, dtype=np.float32),
            "brain_mask": np.ones(seq.shape[0], dtype=np.int32),
            "meta": str(f),
        }
        writer.add(item)
        total += 1

    manifest = writer.finalize()
    manifest.update({"zscore": args.zscore})
    write_manifest(args.out_dir / "manifest.json", manifest)
    log(f"Saved {total} sequences to {args.out_dir}")


if __name__ == "__main__":
    main()

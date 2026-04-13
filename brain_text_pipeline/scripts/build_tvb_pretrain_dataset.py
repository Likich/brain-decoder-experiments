#!/usr/bin/env python3
"""Build TVB pretraining dataset for brain encoder.

Creates pairs (brain_seq, target_next) for next-step prediction.
"""
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
    ap.add_argument("--input_manifest", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--max_seq_len", type=int, default=200)
    args = ap.parse_args()

    manifest = np.load(args.input_manifest, allow_pickle=True)
    # If manifest is JSON, load differently
    if args.input_manifest.suffix == ".json":
        import json
        with args.input_manifest.open("r") as f:
            m = json.load(f)
        shards = m["shards"]
    else:
        raise SystemExit("input_manifest must be JSON from preprocess_tvb_sim")

    writer = ShardWriter(args.out_dir, prefix="tvb_pretrain", shard_size=args.shard_size)

    total = 0
    for shard in shards:
        data = np.load(shard["path"], allow_pickle=True)
        for seq in data["brain_seq"]:
            seq = np.array(seq, dtype=np.float32)
            if len(seq) < 2:
                continue
            if len(seq) > args.max_seq_len:
                seq = seq[: args.max_seq_len]
            # next-step prediction targets
            brain_in = seq[:-1]
            brain_tgt = seq[1:]
            writer.add(
                {
                    "brain_seq": brain_in.astype(np.float32),
                    "brain_target": brain_tgt.astype(np.float32),
                    "brain_mask": np.ones(brain_in.shape[0], dtype=np.int32),
                }
            )
            total += 1

    manifest_out = writer.finalize()
    write_manifest(args.out_dir / "manifest.json", manifest_out)
    log(f"Saved {total} pretrain examples to {args.out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Slice a brain-conditioned dataset to a smaller brain_dim.

Example:
  python3 scripts/slice_brain_dataset.py \
    --in_npz data/brain_ctx_pairs_100k_qwen_544.npz \
    --out_npz data/brain_ctx_pairs_100k_qwen_136.npz \
    --brain_dim 136
"""

import argparse
import numpy as np
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in_npz", type=Path, required=True)
    ap.add_argument("--out_npz", type=Path, required=True)
    ap.add_argument("--brain_dim", type=int, required=True)
    args = ap.parse_args()

    data = np.load(args.in_npz, allow_pickle=True)
    if "brain" not in data:
        raise SystemExit("Input NPZ missing 'brain' array")

    brain = data["brain"]
    cur_dim = brain.shape[1]
    target_dim = args.brain_dim

    if target_dim > cur_dim:
        raise SystemExit(f"brain_dim {target_dim} > current dim {cur_dim}")

    brain_sliced = brain[:, :target_dim].astype(np.float32)

    out = {
        "contexts": data["contexts"],
        "brain": brain_sliced,
        "targets": data["targets"].astype(np.int64),
        "brain_dim": int(target_dim),
    }
    # Preserve optional metadata if present
    for key in ("tokenizer", "schedule"):
        if key in data.files:
            out[key] = data[key]

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **out)
    print(f"Wrote {args.out_npz} with brain_dim={target_dim}")


if __name__ == "__main__":
    main()

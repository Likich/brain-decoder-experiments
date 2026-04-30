#!/usr/bin/env python3
"""Combine matched MEG and text-aux datasets into one feature stream.

This is for the incremental ``BERT + MEG`` test. The output dataset keeps the
original text targets and temporal length, but replaces ``brain_seq`` with a
feature-wise concatenation ``[MEG ; text_aux]`` at each time bin. Evaluation
can then zero/shuffle only the MEG slice while keeping the text-aux slice fixed.
"""
from __future__ import annotations

import argparse
import itertools
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


def shard_path(manifest_path: Path, shard: dict[str, Any]) -> Path:
    path = Path(shard["path"])
    if path.is_absolute() or path.exists():
        return path
    candidate = manifest_path.parent / path
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


def arrays_match(left: Any, right: Any) -> bool:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    return left_arr.shape == right_arr.shape and np.array_equal(left_arr, right_arr)


def validate_pair(idx: int, meg_item: dict[str, Any], aux_item: dict[str, Any]) -> None:
    for key in ("input_ids_context", "decoder_target_ids"):
        if key in meg_item and key in aux_item and not arrays_match(meg_item[key], aux_item[key]):
            raise ValueError(f"dataset mismatch at example {idx}: field {key!r} differs")

    meg_seq = coerce_float_matrix(meg_item["brain_seq"])
    aux_seq = coerce_float_matrix(aux_item["brain_seq"])
    if meg_seq.shape[0] != aux_seq.shape[0]:
        raise ValueError(
            f"dataset mismatch at example {idx}: temporal lengths differ "
            f"{meg_seq.shape[0]} vs {aux_seq.shape[0]}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--meg_manifest", type=Path, required=True)
    ap.add_argument("--text_aux_manifest", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--shard_size", type=int, default=None, help="defaults to MEG shard size")
    ap.add_argument("--report_every", type=int, default=10000)
    args = ap.parse_args()

    meg_manifest = read_manifest(args.meg_manifest)
    aux_manifest = read_manifest(args.text_aux_manifest)
    if int(meg_manifest.get("total_examples", -1)) != int(aux_manifest.get("total_examples", -1)):
        raise ValueError(
            "input manifests disagree on total_examples: "
            f"{meg_manifest.get('total_examples')} vs {aux_manifest.get('total_examples')}"
        )

    shard_size = int(args.shard_size or meg_manifest.get("shard_size", 5000))
    writer = ShardWriter(args.out_dir, prefix="meg_text_combo", shard_size=shard_size)

    total = 0
    meg_dim: int | None = None
    aux_dim: int | None = None
    meg_iter = iter_shard_items(args.meg_manifest, meg_manifest)
    aux_iter = iter_shard_items(args.text_aux_manifest, aux_manifest)

    for pair in itertools.zip_longest(meg_iter, aux_iter, fillvalue=None):
        meg_item, aux_item = pair
        if meg_item is None or aux_item is None:
            raise ValueError("input datasets ended at different lengths")
        validate_pair(total, meg_item, aux_item)

        meg_seq = coerce_float_matrix(meg_item["brain_seq"])
        aux_seq = coerce_float_matrix(aux_item["brain_seq"])
        cur_meg_dim = int(meg_seq.shape[1])
        cur_aux_dim = int(aux_seq.shape[1])
        if meg_dim is None:
            meg_dim = cur_meg_dim
            aux_dim = cur_aux_dim
        elif cur_meg_dim != meg_dim or cur_aux_dim != aux_dim:
            raise ValueError(
                f"feature dimensions changed at example {total}: "
                f"MEG {cur_meg_dim} vs {meg_dim}, AUX {cur_aux_dim} vs {aux_dim}"
            )

        out = normalize_item(meg_item)
        out["brain_seq"] = np.concatenate([meg_seq, aux_seq], axis=1).astype(np.float32, copy=False)
        writer.add(out)
        total += 1
        if args.report_every and total % args.report_every == 0:
            log(f"processed {total}/{meg_manifest.get('total_examples', '?')} examples")

    out_manifest = writer.finalize()
    source_metadata = {
        key: value
        for key, value in meg_manifest.items()
        if key not in {"shards", "num_shards", "total_examples", "prefix", "shard_size"}
    }
    out_manifest.update(source_metadata)
    out_manifest.update(
        {
            "source_manifest": str(args.meg_manifest),
            "combined_aux": {
                "mode": "meg_plus_text_aux",
                "meg_manifest": str(args.meg_manifest),
                "text_aux_manifest": str(args.text_aux_manifest),
                "feature_order": "meg_then_aux",
                "meg_dim": int(meg_dim or 0),
                "aux_dim": int(aux_dim or 0),
                "total_dim": int((meg_dim or 0) + (aux_dim or 0)),
            },
        }
    )
    write_manifest(args.out_dir / "manifest.json", out_manifest)
    log(
        "saved "
        f"{total} combined examples to {args.out_dir} "
        f"(MEG dim={meg_dim}, text-aux dim={aux_dim})"
    )


if __name__ == "__main__":
    main()

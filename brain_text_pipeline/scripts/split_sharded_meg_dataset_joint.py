#!/usr/bin/env python3
"""Build a simultaneous held-out subject + held-out story/sound split.

Train:
  all examples whose subject is NOT in heldout_subjects
  and whose story_key value is NOT in heldout_story_values

Test:
  only examples whose subject IS in heldout_subjects
  and whose story_key value IS in heldout_story_values

Examples from the held-out subject on other stories, and examples from other
subjects on the held-out story, are dropped. This gives a clean
unseen-subject + unseen-story evaluation rather than a mixed train/test reuse
of one factor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "brain_text_pipeline").is_dir():
            return candidate
    raise RuntimeError(f"could not locate repo root from {start}")


ROOT = find_repo_root(Path(__file__).resolve().parent)
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


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--train_out", type=Path, required=True)
    ap.add_argument("--test_out", type=Path, required=True)
    ap.add_argument("--heldout_subjects", nargs="+", required=True)
    ap.add_argument(
        "--heldout_story_values",
        nargs="+",
        required=True,
        help="Held-out values for the story-like grouping key (e.g. story or sound).",
    )
    ap.add_argument("--subject_key", type=str, default="subject")
    ap.add_argument(
        "--story_key",
        type=str,
        default="story",
        help="Metadata key used for the story-like grouping. Use 'sound' if that is your stable story identifier.",
    )
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--report_every", type=int, default=10000)
    args = ap.parse_args()

    manifest = read_manifest(args.manifest)
    total_examples = int(manifest.get("total_examples", sum(int(s["size"]) for s in manifest["shards"])))

    heldout_subjects = {str(v) for v in args.heldout_subjects}
    heldout_story_values = {str(v) for v in args.heldout_story_values}

    train_writer = ShardWriter(args.train_out, prefix="meg_train", shard_size=args.shard_size)
    test_writer = ShardWriter(args.test_out, prefix="meg_test", shard_size=args.shard_size)

    train_n = 0
    test_n = 0
    dropped_subject_only = 0
    dropped_story_only = 0
    dropped_missing = 0

    for i, item in iter_shard_items(args.manifest, manifest):
        item = normalize_item(item)
        meta = item.get("meta", {})
        subject_value = meta_value(meta, args.subject_key)
        story_value = meta_value(meta, args.story_key)
        subject_hit = subject_value in heldout_subjects
        story_hit = story_value in heldout_story_values

        if subject_value == "" or story_value == "":
            dropped_missing += 1
        elif subject_hit and story_hit:
            test_writer.add(item)
            test_n += 1
        elif subject_hit:
            dropped_subject_only += 1
        elif story_hit:
            dropped_story_only += 1
        else:
            train_writer.add(item)
            train_n += 1

        if args.report_every and (i + 1) % args.report_every == 0:
            log(
                f"processed {i + 1}/{total_examples} "
                f"(train={train_n}, test={test_n}, drop_subj={dropped_subject_only}, "
                f"drop_story={dropped_story_only}, drop_missing={dropped_missing})"
            )

    train_manifest = train_writer.finalize()
    test_manifest = test_writer.finalize()
    source_metadata = {
        key: value
        for key, value in manifest.items()
        if key not in {"shards", "num_shards", "total_examples", "prefix", "shard_size"}
    }
    split_info = {
        "type": "subject_story_intersection",
        "subject_key": args.subject_key,
        "story_key": args.story_key,
        "heldout_subjects": sorted(heldout_subjects),
        "heldout_story_values": sorted(heldout_story_values),
        "dropped_subject_only": dropped_subject_only,
        "dropped_story_only": dropped_story_only,
        "dropped_missing": dropped_missing,
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
    log(
        "dropped examples: "
        f"subject-only={dropped_subject_only}, story-only={dropped_story_only}, missing={dropped_missing}"
    )


if __name__ == "__main__":
    main()

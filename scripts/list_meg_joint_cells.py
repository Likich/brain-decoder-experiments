#!/usr/bin/env python3
"""List the largest joint subject x story/sound cells in a sharded MEG dataset."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "brain_text_pipeline").is_dir():
            return candidate
    raise RuntimeError(f"could not locate repo root from {start}")


ROOT = find_repo_root(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def meta_value(meta: Any, key: str) -> str:
    if isinstance(meta, bytes):
        meta = meta.decode("utf-8")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            return ""
    if isinstance(meta, dict):
        value = meta.get(key, "")
        return "" if value is None else str(value)
    return ""


def shard_path(source_manifest: Path, shard: dict[str, Any]) -> Path:
    path = Path(shard["path"])
    if path.is_absolute() or path.exists():
        return path
    candidate = source_manifest.parent / path
    return candidate if candidate.exists() else path


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--subject_key", type=str, default="subject")
    ap.add_argument("--story_key", type=str, default="sound", help="Use 'sound' or 'story'.")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_count", type=int, default=1)
    args = ap.parse_args()

    with args.manifest.open() as f:
        manifest = json.load(f)

    pair_counts: Counter[tuple[str, str]] = Counter()
    subject_counts: Counter[str] = Counter()
    story_counts: Counter[str] = Counter()

    for shard in manifest["shards"]:
        path = shard_path(args.manifest, shard)
        with np.load(path, allow_pickle=True) as data:
            metas = data["meta"]
            for raw in metas:
                subject = meta_value(raw, args.subject_key)
                story = meta_value(raw, args.story_key)
                if not subject or not story:
                    continue
                pair_counts[(subject, story)] += 1
                subject_counts[subject] += 1
                story_counts[story] += 1

    print(f"# top {args.top_k} ({args.subject_key}, {args.story_key}) cells")
    print("count\tsubject\tstory_like")
    shown = 0
    for (subject, story), count in pair_counts.most_common():
        if count < args.min_count:
            continue
        print(f"{count}\t{subject}\t{story}")
        shown += 1
        if shown >= args.top_k:
            break

    print(f"\n# subject totals ({len(subject_counts)})")
    for subject, count in subject_counts.most_common():
        print(f"{count}\t{subject}")

    print(f"\n# {args.story_key} totals ({len(story_counts)})")
    for story, count in story_counts.most_common(args.top_k):
        print(f"{count}\t{story}")


if __name__ == "__main__":
    main()

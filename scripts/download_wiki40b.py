"""
Helper script to pull a small shard of Wiki40B locally so we can build
token-level training data without manual downloads.

Usage:
    python scripts/download_wiki40b.py \
        --lang en \
        --split train \
        --max_articles 200 \
        --out data/wiki40b_en.jsonl

Requires the `datasets` package (declared in pyproject). The resulting
JSONL file contains one JSON object per article:
    {"id": 0, "title": "...", "text": "..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset, load_dataset_builder


def _parse_split_spec(split: str, builder) -> tuple[str, int | None]:
    """
    Convert strings like 'train', 'train[:1000]', 'train[:0.1%]' into
    (base_split, numeric_limit). Returns limit=None if no slicing.
    """
    split = split.strip()
    if "[" not in split:
        return split, None
    if not split.endswith("]"):
        raise ValueError(f"Malformed split spec: {split}")
    base, slice_part = split.split("[", 1)
    slice_part = slice_part[:-1]

    if ":" not in slice_part:
        raise ValueError(f"Slice spec must be like [:N], got {split}")
    start, end = slice_part.split(":", 1)
    start = start.strip()
    end = end.strip()
    if start not in ("", "0"):
        raise ValueError(f"Only slices starting at 0 supported (got {split})")
    if not end:
        raise ValueError(f"Slice end missing in {split}")

    total = None
    if builder is not None and base in builder.info.splits:
        total = builder.info.splits[base].num_examples

    if end.endswith("%"):
        if total is None:
            raise ValueError(f"Cannot resolve percentage slice {split} without split metadata")
        frac = float(end[:-1]) / 100.0
        limit = max(1, int(total * frac))
    else:
        limit = int(end)

    return base, limit



def normalize_text(raw: str, min_len: int) -> str | None:
    """
    Clean up Wiki40B article text and discard extremely short entries.
    """
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    # Collapse internal blank lines and trim whitespace.
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return None
    text = "\n".join(lines)
    if len(text) < min_len:
        return None
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", help="Wiki40B language code")
    ap.add_argument(
        "--split",
        default="train",
        help="Split spec (train, train[:1000], train[:0.5%]); quote it in shells!",
    )
    ap.add_argument(
        "--max_articles",
        type=int,
        default=500,
        help="Stop after this many accepted articles (post-filter)",
    )
    ap.add_argument(
        "--min_chars",
        type=int,
        default=200,
        help="Drop articles shorter than this many characters",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/wiki40b_en.jsonl"),
        help="Output JSONL file",
    )
    args = ap.parse_args()

    print(f"Preparing to stream Wiki40B lang={args.lang!r} split={args.split!r}")
    builder = load_dataset_builder("wiki40b", args.lang)
    base_split, slice_limit = _parse_split_spec(args.split, builder)
    dataset = load_dataset("wiki40b", args.lang, split=base_split, streaming=True)
    print(f"Streaming base split {base_split!r}")
    total_seen = 0
    total_written = 0
    effective_limit = args.max_articles if args.max_articles and args.max_articles > 0 else None
    if slice_limit is not None:
        effective_limit = min(slice_limit, effective_limit) if effective_limit else slice_limit
        print(f"Applying slice limit: <= {effective_limit} articles")
    elif effective_limit:
        print(f"Stopping after {effective_limit} accepted articles")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in dataset:
            total_seen += 1
            text = normalize_text(row.get("text", ""), args.min_chars)
            if text is None:
                continue

            payload = {
                "id": total_written,
                "title": row.get("title", "") or "",
                "text": text,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            total_written += 1

            if effective_limit and total_written >= effective_limit:
                break

    print(
        f"Wrote {total_written} articles (kept {100.0 * total_written / max(1, total_seen):.2f}% of seen rows) "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()

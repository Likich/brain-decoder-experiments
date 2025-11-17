"""
Encode the Wiki40B shard into token-id sequences using a trained tokenizer.

The output is a JSONL file where each line contains:
    {"id": 0, "title": "...", "tokens": [int, ...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/wiki40b_en.jsonl"))
    ap.add_argument("--tokenizer", type=Path, default=Path("models/wiki_tokenizer.json"))
    ap.add_argument("--out", type=Path, default=Path("data/wiki40b_tokens.jsonl"))
    ap.add_argument("--max_articles", type=int, default=1000)
    ap.add_argument("--max_tokens_per_article", type=int, default=2048)
    ap.add_argument("--min_chars", type=int, default=200)
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input corpus not found: {args.input}")
    if not args.tokenizer.exists():
        raise SystemExit(f"Tokenizer file not found: {args.tokenizer}")

    tok = Tokenizer.from_file(str(args.tokenizer))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with args.out.open("w", encoding="utf-8") as fout, args.input.open(
        "r", encoding="utf-8"
    ) as fin:
        for line in fin:
            if args.max_articles and kept >= args.max_articles:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            if len(text) < args.min_chars:
                continue

            enc = tok.encode(text)
            ids = enc.ids
            if args.max_tokens_per_article:
                ids = ids[: args.max_tokens_per_article]

            payload = {
                "id": kept,
                "title": row.get("title", "") or "",
                "tokens": ids,
            }
            fout.write(json.dumps(payload) + "\n")
            kept += 1

    print(f"Wrote {kept} tokenized articles to {args.out}")


if __name__ == "__main__":
    main()


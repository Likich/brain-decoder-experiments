"""Convert the Databricks Dolly 15k dataset into a token schedule JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/dolly_tokens.jsonl"))
    ap.add_argument("--tokenizer", type=Path, default=Path("models/wiki_tokenizer.json"))
    ap.add_argument("--max_examples", type=int, default=5000,
                    help="Limit number of Dolly rows (0 = all)")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if args.max_examples and i >= args.max_examples:
                break
            parts = [row.get("instruction") or ""]
            context = row.get("context")
            if context:
                parts.append(context)
            parts.append(row.get("response") or "")
            text = "\n".join(part.strip() for part in parts if part and part.strip())
            if not text:
                continue
            ids = tok.encode(text).ids
            if not ids:
                continue
            f.write(json.dumps({"tokens": ids}) + "\n")
            written += 1

    print(f"Wrote {written} rows to {args.out}")


if __name__ == "__main__":
    main()

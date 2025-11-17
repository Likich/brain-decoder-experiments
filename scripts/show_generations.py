"""
Decode and display generated token sequences from outputs/experiment.jsonl.

Usage:
    python scripts/show_generations.py \
        --input outputs/experiment.jsonl \
        --tokenizer models/wiki_tokenizer.json \
        --min_conf 0.8 \
        --max_rows 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("outputs/experiment.jsonl"))
    ap.add_argument("--tokenizer", type=Path, default=Path("models/wiki_tokenizer.json"))
    ap.add_argument("--min_conf", type=float, default=0.0, help="Min decision confidence to display")
    ap.add_argument("--max_rows", type=int, default=20, help="Limit number of rows printed")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    if not args.tokenizer.exists():
        raise SystemExit(f"Tokenizer file not found: {args.tokenizer}")

    tok = Tokenizer.from_file(str(args.tokenizer))
    shown = 0
    for row in load_rows(args.input):
        conf = row.get("confidence")
        if conf is None or conf < args.min_conf:
            continue
        gen_ids = row.get("generated_token_ids") or []
        decoded = tok.decode(gen_ids) if gen_ids else ""
        stimulus = row.get("stimulus_token") or row.get("stimulus")
        report = row.get("report")
        print(f"stimulus={stimulus!r} | choice={row.get('choice')} | conf={conf:.2f}")
        print(f"  generated: {decoded!r}")
        print(f"  report: {report}")
        print()
        shown += 1
        if args.max_rows and shown >= args.max_rows:
            break

    if shown == 0:
        print("No rows matched the filters.")


if __name__ == "__main__":
    main()

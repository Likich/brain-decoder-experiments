"""
Train a byte-level BPE tokenizer on the downloaded Wiki40B shard.

Usage:
    python scripts/train_tokenizer.py \
        --input data/wiki40b_en.jsonl \
        --out models/wiki_tokenizer.json \
        --vocab_size 2048
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


def iter_texts(path: Path, max_articles: int | None, min_chars: int) -> Iterable[str]:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_articles and count >= max_articles:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = (data.get("text") or "").strip()
            if len(text) < min_chars:
                continue
            count += 1
            yield text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/wiki40b_en.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("models/wiki_tokenizer.json"))
    ap.add_argument("--vocab_size", type=int, default=2048)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--max_articles", type=int, default=1000)
    ap.add_argument("--min_chars", type=int, default=200)
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input corpus not found: {args.input}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
    )

    print(
        f"Training tokenizer on {args.input} "
        f"(max_articles={args.max_articles}, vocab_size={args.vocab_size})"
    )
    iterator = iter_texts(args.input, args.max_articles, args.min_chars)
    tokenizer.train_from_iterator(iterator, trainer=trainer, length=args.max_articles)
    tokenizer.save(str(args.out))
    print(f"Saved tokenizer to {args.out}")


if __name__ == "__main__":
    main()


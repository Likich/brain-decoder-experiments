from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


class TokenSchedule:
    """
    Simple helper that streams token IDs from a JSONL file produced by
    scripts/encode_corpus_tokens.py. Each line must have a `tokens` list.
    The schedule loops when it reaches the end.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Token schedule not found: {self.path}")
        self._tokens = self._load_tokens(self.path)
        if not self._tokens:
            raise ValueError(f"No tokens found in {self.path}")
        self._cursor = 0

    @staticmethod
    def _load_tokens(path: Path) -> list[int]:
        buf: list[int] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                toks = row.get("tokens", [])
                if not toks:
                    continue
                buf.extend(int(t) for t in toks)
        return buf

    def next_token(self) -> int:
        tok = self._tokens[self._cursor]
        self._cursor = (self._cursor + 1) % len(self._tokens)
        return tok

    def peek(self, offset: int = 0) -> int:
        idx = (self._cursor + offset) % len(self._tokens)
        return self._tokens[idx]

    def reset(self) -> None:
        self._cursor = 0

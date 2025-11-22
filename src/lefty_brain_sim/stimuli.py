from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator


class TokenSchedule:
    """
    Simple helper that streams token IDs from a JSONL file produced by
    scripts/encode_corpus_tokens.py. Each line must have a `tokens` list.
    The schedule loops when it reaches the end.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        state_path: str | Path | None = None,
        random_start: bool = False,
        seed: int | None = None,
        persist_every: int = 128,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Token schedule not found: {self.path}")
        self._tokens = self._load_tokens(self.path)
        if not self._tokens:
            raise ValueError(f"No tokens found in {self.path}")
        self.persist_every = max(1, int(persist_every))
        self._since_persist = 0

        if state_path is None:
            self.state_path = self.path.with_suffix(self.path.suffix + ".state.json")
        else:
            self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        self._cursor = self._load_state()
        if self._cursor is None:
            if random_start:
                rng = random.Random(seed)
                self._cursor = rng.randrange(len(self._tokens))
            else:
                self._cursor = 0

        self._cursor %= len(self._tokens)
        self._persist_state(force=True)

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

    def _load_state(self) -> int | None:
        if not self.state_path.exists():
            return None
        try:
            data = json.loads(self.state_path.read_text())
            cursor = int(data.get("cursor", 0))
        except Exception:
            return None
        if 0 <= cursor < len(self._tokens):
            return cursor
        return None

    def _persist_state(self, force: bool = False) -> None:
        if not force and self._since_persist < self.persist_every:
            return
        payload = {"cursor": self._cursor}
        self.state_path.write_text(json.dumps(payload))
        self._since_persist = 0

    def next_token(self) -> int:
        tok = self._tokens[self._cursor]
        self._cursor = (self._cursor + 1) % len(self._tokens)
        self._since_persist += 1
        self._persist_state()
        return tok

    def peek(self, offset: int = 0) -> int:
        idx = (self._cursor + offset) % len(self._tokens)
        return self._tokens[idx]

    def reset(self) -> None:
        self._cursor = 0
        self._since_persist = 0
        self._persist_state(force=True)

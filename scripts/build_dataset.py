import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


CANONICAL_CLASSES = ["APPLE", "BANANA", "GRAPE", "ORANGE", "PEAR"]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def vocab_from_tokenizer(tok_path: Path) -> list[str]:
    tok = Tokenizer.from_file(str(tok_path))
    vocab = tok.get_vocab()
    if not vocab:
        raise ValueError(f"Tokenizer {tok_path} has empty vocab")
    max_id = max(vocab.values())
    inv = ["<unk>"] * (max_id + 1)
    for token, idx in vocab.items():
        inv[idx] = token
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("outputs/experiment.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/brain_multiclass.npz"))
    ap.add_argument("--tokenizer", type=Path, default=None, help="Optional tokenizer.json to derive vocab order")
    ap.add_argument("--use-target", action="store_true", help="Use target_token_id labels (next-token training)")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"File not found: {args.input}")

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit("No JSON rows parsed. Check results file.")

    if args.tokenizer:
        class_names = vocab_from_tokenizer(args.tokenizer)
    else:
        class_names = CANONICAL_CLASSES

    label_map = {lab: i for i, lab in enumerate(class_names)}

    X, y = [], []
    expected_len = None

    for r in rows:
        snap = r.get("activity_snapshot")
        if snap is None:
            continue

        if args.use_target and r.get("target_token_id") is not None:
            idx = int(r["target_token_id"])
        else:
            stim_id = r.get("stimulus_id")
            if stim_id is not None:
                idx = int(stim_id)
            else:
                lab = r.get("stimulus_token") or r.get("stimulus")
                if lab not in label_map:
                    continue
                idx = label_map[lab]

        if idx >= len(class_names):
            continue

        if expected_len is None:
            expected_len = len(snap)
        if len(snap) != expected_len:
            continue

        X.append(snap)
        y.append(idx)

    if not X:
        raise SystemExit("No usable snapshots found (maybe inconsistent lengths?).")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        X=X,
        y=y,
        label_names=np.array(class_names),
        num_classes=len(class_names),
    )
    print(f"Saved dataset {X.shape} → {args.out}")


if __name__ == "__main__":
    main()

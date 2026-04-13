#!/usr/bin/env python3
"""Build MEG-MASC word-aligned dataset shards.

Requires preprocessed brain.npy from preprocess_meg_masc.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_text_pipeline.src.data.meg_masc import events_path
from brain_text_pipeline.src.utils.io import ShardWriter, write_manifest
from brain_text_pipeline.src.utils.logging import log


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--meg_root", type=Path, required=True, help="MEG-MASC root")
    ap.add_argument("--preprocessed_root", type=Path, required=True, help="Output of preprocess_meg_masc")
    ap.add_argument("--tokenizer", type=str, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--max_context_words", type=int, default=100)
    ap.add_argument(
        "--max_context_tokens",
        type=int,
        default=None,
        help="Max encoder tokens; defaults to tokenizer.model_max_length if reasonable",
    )
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--tmin", type=float, default=-0.5, help="seconds before word onset")
    ap.add_argument("--tmax", type=float, default=0.0, help="seconds before word onset")
    ap.add_argument("--word_column", type=str, default="word", help="events.tsv column for word")
    ap.add_argument("--subject", type=str, default=None)
    ap.add_argument("--session", type=str, default=None)
    ap.add_argument("--task", type=str, default=None)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    writer = ShardWriter(args.out_dir, prefix="meg", shard_size=args.shard_size)
    # decide max token length for encoder
    max_ctx_tokens = args.max_context_tokens
    if max_ctx_tokens is None:
        tok_max = getattr(tokenizer, "model_max_length", None)
        if tok_max and tok_max < 100000:
            max_ctx_tokens = int(tok_max)
        else:
            max_ctx_tokens = 512
    # avoid tokenizer warnings by enforcing truncation
    tokenizer.model_max_length = max_ctx_tokens

    subjects = [args.subject] if args.subject else sorted([p.name for p in args.preprocessed_root.glob("sub-*")])

    total = 0
    for sub in subjects:
        sessions = [args.session] if args.session else sorted([p.name for p in (args.preprocessed_root / sub).glob("ses-*")])
        for ses in sessions:
            task_dirs = list((args.preprocessed_root / sub / ses).glob("task-*"))
            if args.task:
                task_dirs = [args.preprocessed_root / sub / ses / f"task-{args.task}"]
            for task_dir in task_dirs:
                task = task_dir.name.replace("task-", "")
                brain_path = task_dir / "brain.npy"
                meta_path = task_dir / "meta.json"
                if not brain_path.exists() or not meta_path.exists():
                    continue
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                sfreq = float(meta["sfreq"])
                brain = np.load(brain_path, mmap_mode="r")

                events_file = events_path(args.meg_root, sub, ses, task)
                if not events_file.exists():
                    log(f"Missing events {events_file}")
                    continue
                events = np.loadtxt(events_file, delimiter="\t", dtype=str, skiprows=1)
                # fallback: use pandas if needed
                try:
                    import pandas as pd
                    df = pd.read_csv(events_file, sep="\t")
                except Exception:
                    df = None

                words = []
                onsets = []
                if df is not None and args.word_column in df.columns:
                    for _, row in df.iterrows():
                        word = str(row[args.word_column])
                        onset = float(row.get("onset", row.get("onset_sec", 0.0)))
                        if word.strip() == "" or word == "nan":
                            continue
                        words.append(word)
                        onsets.append(onset)
                else:
                    # fallback to trial_type column
                    if df is not None and "trial_type" in df.columns:
                        for _, row in df.iterrows():
                            word = str(row["trial_type"])
                            onset = float(row.get("onset", row.get("onset_sec", 0.0)))
                            words.append(word)
                            onsets.append(onset)
                    else:
                        log(f"No word column in {events_file}, skipping")
                        continue

                for i, (word, onset) in enumerate(zip(words, onsets)):
                    ctx_words = words[max(0, i - args.max_context_words) : i]
                    ctx_text = " ".join(ctx_words)
                    ctx_ids = tokenizer(
                        ctx_text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_ctx_tokens,
                    ).input_ids
                    tgt_ids = tokenizer.encode(word, add_special_tokens=False)
                    if len(tgt_ids) == 0:
                        continue

                    start = int((onset + args.tmin) * sfreq)
                    stop = int((onset + args.tmax) * sfreq)
                    start = max(start, 0)
                    stop = max(stop, start + 1)
                    brain_seq = brain[start:stop]

                    item = {
                        "input_ids_context": np.array(ctx_ids, dtype=np.int32),
                        "decoder_target_ids": np.array(tgt_ids, dtype=np.int32),
                        "brain_seq": np.array(brain_seq, dtype=np.float32),
                        "attention_mask_text": np.ones(len(ctx_ids), dtype=np.int32),
                        "brain_mask": np.ones(brain_seq.shape[0], dtype=np.int32),
                        "meta": json.dumps(
                            {
                                "subject": sub,
                                "session": ses,
                                "task": task,
                                "onset_sec": float(onset),
                                "target_text": word,
                            }
                        ),
                    }
                    writer.add(item)
                    total += 1
                    if args.max_examples and total >= args.max_examples:
                        break
                if args.max_examples and total >= args.max_examples:
                    break
            if args.max_examples and total >= args.max_examples:
                break
        if args.max_examples and total >= args.max_examples:
            break

    manifest = writer.finalize()
    manifest.update({
        "tokenizer": args.tokenizer,
        "tmin": args.tmin,
        "tmax": args.tmax,
        "max_context_words": args.max_context_words,
        "max_context_tokens": max_ctx_tokens,
    })
    write_manifest(args.out_dir / "manifest.json", manifest)
    log(f"Saved {total} examples to {args.out_dir}")


if __name__ == "__main__":
    main()

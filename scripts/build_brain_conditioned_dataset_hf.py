#!/usr/bin/env python3
"""
Build brain-conditioned dataset using a Hugging Face tokenizer.

This script:
  1) Loads an HF tokenizer and saves it (tokenizer.json).
  2) Tokenizes a JSONL text file (e.g., wiki40b_en.jsonl) into token IDs.
  3) Runs the TVB Experiment to collect brain snapshots per token.
  4) Saves NPZ with contexts/brain/targets compatible with training scripts.

Notes:
- The tokenizer used here MUST match the model you train (Qwen/T5/etc).
- We only tokenize as many tokens as needed for max_samples + start_offset.

Example:
  python3 scripts/build_brain_conditioned_dataset_hf.py \
    --config configs/default.yaml \
    --hf_model Qwen/Qwen2.5-0.5B \
    --text_jsonl data/wiki40b_en.jsonl \
    --tokenizer_out models/qwen_tokenizer \
    --token_out data/wiki40b_tokens_qwen.jsonl \
    --out data/brain_ctx_pairs_100k_qwen.npz \
    --brain_dim 136 --max_samples 100000 --snr high --report_every 500
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import List

import numpy as np

from transformers import AutoTokenizer

from lefty_brain_sim.experiment import Experiment
from lefty_brain_sim.utils import set_seed


def iter_texts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") or row.get("content") or row.get("body") or ""
            title = row.get("title") or ""
            yield row.get("id", None), title, text


def tokenize_jsonl(
    text_path: Path,
    tokenizer: AutoTokenizer,
    token_out: Path,
    max_tokens: int | None,
    report_every: int,
) -> List[int]:
    token_out.parent.mkdir(parents=True, exist_ok=True)
    all_tokens: List[int] = []
    count = 0
    with token_out.open("w", encoding="utf-8") as f:
        for idx, (rid, title, text) in enumerate(iter_texts(text_path)):
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue

            # Clip if we only need a prefix of tokens
            if max_tokens is not None and len(all_tokens) + len(ids) > max_tokens:
                need = max_tokens - len(all_tokens)
                if need <= 0:
                    break
                ids = ids[:need]

            row = {"id": rid if rid is not None else idx, "title": title, "tokens": ids}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            all_tokens.extend(ids)
            count += 1
            if report_every and count % report_every == 0:
                print(f"Tokenized {count} docs, total tokens={len(all_tokens)}")

            if max_tokens is not None and len(all_tokens) >= max_tokens:
                break

    if not all_tokens:
        raise SystemExit(f"No tokens written to {token_out}")
    return all_tokens


def snapshot_from_trial(tr):
    return tr.decoder_snapshot or tr.activity_snapshot


_WORK = {}


def _init_worker(
    config_path: str,
    tokenizer_json: str,
    token_out: str,
    tokens: list[int],
    snr_levels: list[str],
    snr_override: str | None,
    context_window: int | None,
    base_seed: int,
):
    global _WORK
    proc = mp.current_process()
    worker_id = proc._identity[0] if proc._identity else 0
    set_seed(base_seed + worker_id)

    exp = Experiment.from_yaml(config_path)
    exp.stimuli_cfg["tokenizer"] = tokenizer_json
    exp.stimuli_cfg["schedule"] = token_out
    exp.stimulus_mode = "tokens"
    exp.predict_next = True
    exp.class_names, exp.stim_patterns = exp._init_stimuli()

    _WORK = {
        "exp": exp,
        "tokens": tokens,
        "snr_levels": snr_levels,
        "snr_override": snr_override,
        "context_window": context_window,
    }


def _process_sample(task):
    i, pos = task
    exp = _WORK["exp"]
    tokens = _WORK["tokens"]
    snr_levels = _WORK["snr_levels"]
    snr_override = _WORK["snr_override"]
    context_window = _WORK["context_window"]

    stim_id = int(tokens[pos])
    target_id = int(tokens[pos + 1])
    snr = snr_override or snr_levels[i % len(snr_levels)]

    trial = exp.run_trial(
        snr=snr,
        stim_idx=stim_id,
        allow_generation=False,
        log_debug=False,
    )
    snap = snapshot_from_trial(trial)
    if snap is None:
        return None

    if context_window is not None:
        start = max(0, pos + 1 - context_window)
        ctx = tokens[start : pos + 1]
    else:
        ctx = tokens[: pos + 1]
    return i, ctx, snap, target_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Build brain dataset with HF tokenizer")
    ap.add_argument("--config", type=Path, required=True, help="Experiment YAML config")
    ap.add_argument("--hf_model", type=str, required=True, help="HF model/tokenizer name")
    ap.add_argument("--text_jsonl", type=Path, required=True, help="JSONL with a 'text' field")
    ap.add_argument("--tokenizer_out", type=Path, default=Path("models/hf_tokenizer"))
    ap.add_argument("--token_out", type=Path, default=Path("data/wiki40b_tokens_hf.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/brain_ctx_pairs_hf.npz"))
    ap.add_argument("--brain_dim", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=100000)
    ap.add_argument("--start_offset", type=int, default=0)
    ap.add_argument("--snr", type=str, default=None)
    ap.add_argument("--report_every", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=1, help="Parallel workers for TVB trials")
    ap.add_argument("--chunksize", type=int, default=16, help="Chunk size for multiprocessing pool")
    ap.add_argument(
        "--context_window",
        type=int,
        default=None,
        help="If set, store only the last N tokens per context to reduce file size",
    )
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    args.tokenizer_out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.tokenizer_out)

    tokenizer_json = args.tokenizer_out / "tokenizer.json"
    if not tokenizer_json.exists():
        raise SystemExit(
            f"tokenizer.json not found in {args.tokenizer_out}. "
            "Use a fast tokenizer or ensure tokenizer.save_pretrained wrote tokenizer.json."
        )

    # We need start_offset + max_samples + 1 tokens
    need_tokens = None
    if args.max_samples is not None:
        need_tokens = args.start_offset + args.max_samples + 1

    tokens = tokenize_jsonl(
        args.text_jsonl,
        tokenizer,
        args.token_out,
        max_tokens=need_tokens,
        report_every=args.report_every,
    )

    # Build experiment (for config access / snr levels)
    exp = Experiment.from_yaml(str(args.config))

    total_possible = len(tokens) - 1 - args.start_offset
    if total_possible <= 0:
        raise SystemExit("start_offset too large for token sequence")
    limit = total_possible if args.max_samples is None else min(total_possible, args.max_samples)

    snr_levels = exp.cfg.snr_levels or ["high"]
    snr_override = args.snr
    contexts: List[List[int]] = []
    brain_vecs: List[list[float]] = []
    targets: List[int] = []
    skipped = 0

    tasks = [(i, args.start_offset + i) for i in range(limit)]

    if args.num_workers and args.num_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(
                str(args.config),
                str(tokenizer_json),
                str(args.token_out),
                tokens,
                snr_levels,
                snr_override,
                args.context_window,
                exp.cfg.seed,
            ),
        ) as pool:
            results = []
            for out in pool.imap_unordered(_process_sample, tasks, chunksize=args.chunksize):
                if out is None:
                    skipped += 1
                    continue
                results.append(out)
                if args.report_every and len(results) % args.report_every == 0:
                    print(f"Saved {len(results)} samples (skipped {skipped})")
        # sort by original index
        results.sort(key=lambda x: x[0])
        for _, ctx_tokens, snap, target_id in results:
            contexts.append(ctx_tokens)
            brain_vecs.append(snap)
            targets.append(target_id)
    else:
        # Single-process fallback
        exp.stimuli_cfg["tokenizer"] = str(tokenizer_json)
        exp.stimuli_cfg["schedule"] = str(args.token_out)
        exp.stimulus_mode = "tokens"
        exp.predict_next = True
        exp.class_names, exp.stim_patterns = exp._init_stimuli()

        exp.token_schedule.reset() if exp.token_schedule else None

        for i in range(limit):
            pos = args.start_offset + i
            stim_id = int(tokens[pos])
            target_id = int(tokens[pos + 1])
            snr = snr_override or snr_levels[i % len(snr_levels)]

            trial = exp.run_trial(
                snr=snr,
                stim_idx=stim_id,
                allow_generation=False,
                log_debug=False,
            )
            snap = snapshot_from_trial(trial)
            if snap is None:
                skipped += 1
                continue

            if args.context_window is not None:
                start = max(0, pos + 1 - args.context_window)
                contexts.append(tokens[start : pos + 1])
            else:
                contexts.append(tokens[: pos + 1])
            brain_vecs.append(snap)
            targets.append(target_id)

            if args.report_every and len(contexts) % args.report_every == 0:
                print(f"Saved {len(contexts)} samples (skipped {skipped})")

    if not contexts:
        raise SystemExit("No samples recorded. Try adjusting start_offset/snr/config.")

    brain_arr = np.asarray(brain_vecs, dtype=np.float32)
    if args.brain_dim is not None:
        target_dim = args.brain_dim
        cur_dim = brain_arr.shape[1]
        if target_dim < cur_dim:
            brain_arr = brain_arr[:, :target_dim]
        elif target_dim > cur_dim:
            padded = np.zeros((brain_arr.shape[0], target_dim), dtype=np.float32)
            padded[:, :cur_dim] = brain_arr
            brain_arr = padded

    target_arr = np.asarray(targets, dtype=np.int64)
    contexts_arr = np.array(contexts, dtype=object)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        contexts=contexts_arr,
        brain=brain_arr,
        targets=target_arr,
        brain_dim=int(brain_arr.shape[1]),
        tokenizer=str(tokenizer_json),
        schedule=str(args.token_out),
    )
    print(
        f"Saved {len(contexts_arr)} samples → {args.out}\n"
        f" brain shape: {brain_arr.shape}, targets shape: {target_arr.shape}"
    )


if __name__ == "__main__":
    main()

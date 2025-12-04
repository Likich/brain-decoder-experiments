import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from lefty_brain_sim.experiment import Experiment


def load_tokens(path: Path) -> List[int]:
    if not path.exists():
        raise FileNotFoundError(f"Token file not found: {path}")
    buf: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            toks = row.get("tokens") or []
            if not toks:
                continue
            buf.extend(int(t) for t in toks)
    if not buf:
        raise ValueError(f"No tokens parsed from {path}")
    return buf


def snapshot_from_trial(tr) -> list[float] | None:
    return tr.decoder_snapshot or tr.activity_snapshot


def main():
    ap = argparse.ArgumentParser(description="Build paired (context, brain, next-token) dataset")
    ap.add_argument("--config", type=Path, required=True, help="Experiment YAML config (must use token stimuli)")
    ap.add_argument("--token_file", type=Path, default=None, help="JSONL with encoded tokens (defaults to config stimuli schedule)")
    ap.add_argument("--out", type=Path, default=Path("data/brain_ctx_pairs.npz"))
    ap.add_argument(
        "--brain_dim",
        type=int,
        default=None,
        help="Optional brain vector dimension; trims or zero-pads snapshots to this size",
    )
    ap.add_argument("--max_samples", type=int, default=None, help="Maximum number of sequential samples to record")
    ap.add_argument("--start_offset", type=int, default=0, help="Skip this many tokens before logging samples")
    ap.add_argument("--snr", type=str, default=None, help="Override SNR level (otherwise cycle config levels)")
    ap.add_argument("--report_every", type=int, default=100, help="Print progress every N saved samples")
    args = ap.parse_args()

    exp = Experiment.from_yaml(str(args.config))

    token_path = args.token_file or Path(exp.cfg.stimuli.get("schedule"))
    if token_path is None:
        raise SystemExit("Token file not provided and config has no stimuli.schedule")
    tokens = load_tokens(token_path)
    if len(tokens) < 2:
        raise SystemExit("Need at least two tokens to form context/target pairs")

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
        tokenizer=str(exp.stimuli_cfg.get("tokenizer")),
        schedule=str(token_path),
    )
    print(
        f"Saved {len(contexts_arr)} samples → {args.out}\n"
        f" brain shape: {brain_arr.shape}, targets shape: {target_arr.shape}"
    )


if __name__ == "__main__":
    main()

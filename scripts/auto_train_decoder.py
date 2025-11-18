"""
Automated pipeline:
  1) Run multiple experiment batches (mock decoder) until the log file
     has at least `target_trials` JSON lines.
  2) Build a next-token dataset from that log.
  3) Train the brain decoder with the requested (attention) hyperparameters.

Example:
  python scripts/auto_train_decoder.py \
      --config configs/default.yaml \
      --log outputs/experiment_attn_train.jsonl \
      --target_trials 30000 \
      --chunk_runs 5 \
      --dataset data/brain_next_token_136_attn.npz \
      --epochs 40 \
      --hidden_dim 1024 \
      --num_layers 2 \
      --use_attention \
      --attn_heads 8 \
      --attn_layers 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], append_to: Path | None = None) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if append_to is not None:
        with append_to.open("a", encoding="utf-8") as f:
            f.write(res.stdout)
    else:
        sys.stdout.write(res.stdout)
    if res.stderr:
        sys.stderr.write(res.stderr)


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f if _.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--log", default="outputs/experiment_attn_train.jsonl")
    ap.add_argument("--target_trials", type=int, default=20000)
    ap.add_argument("--chunk_runs", type=int, default=5, help="How many experiment runs per progress report")
    ap.add_argument("--dataset", default="data/brain_next_token_136_attn.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--use_attention", action="store_true")
    ap.add_argument("--attn_heads", type=int, default=8)
    ap.add_argument("--attn_layers", type=int, default=1)
    args = ap.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting auto pipeline. Target trials: {args.target_trials}")
    trials = line_count(log_path)
    while trials < args.target_trials:
        for _ in range(args.chunk_runs):
            print(f"Running experiment batch (current trials={trials})")
            run_cmd(
                [
                    sys.executable,
                    "scripts/run_experiment.py",
                    "--config",
                    args.config,
                ],
                append_to=log_path,
            )
            trials = line_count(log_path)
            if trials >= args.target_trials:
                break
        print(f"Trials collected: {trials}/{args.target_trials}")

    print("Building dataset...")
    run_cmd(
        [
            sys.executable,
            "scripts/build_dataset.py",
            "--input",
            str(log_path),
            "--tokenizer",
            "models/wiki_tokenizer.json",
            "--use-target",
            "--out",
            args.dataset,
        ]
    )

    train_cmd = [
        sys.executable,
        "scripts/train_brain_decoder.py",
        "--data",
        args.dataset,
        "--tokenizer",
        "models/wiki_tokenizer.json",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--hidden_dim",
        str(args.hidden_dim),
        "--num_layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
    ]
    if args.use_attention:
        train_cmd += [
            "--use_attention",
            "--attn_heads",
            str(args.attn_heads),
            "--attn_layers",
            str(args.attn_layers),
        ]

    print("Training decoder...")
    run_cmd(train_cmd)
    print("Pipeline completed.")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Alpha-scaling analysis for brain-conditioned LMs.

For each alpha in a list, scales the brain vector z -> alpha * z and compares
to the zero-brain baseline. Reports JS divergence and NLL deltas for:
  - REAL: paired brain vector
  - SHUF: shuffled brain vector

Usage example:
  python3 scripts/analysis_alpha.py \
    --ckpt models/lm_brain136/language_model.pt \
    --tokenizer models/wiki_tokenizer.json \
    --brain_dataset data/brain_ctx_pairs_100k.npz \
    --alphas 0 0.25 0.5 1 2 \
    --hidden_dim 384 --num_layers 2 --attn_heads 8 --dropout 0.11049 \
    --block_size 96 --samples 5000 --device cuda --out_csv alpha_sweep.csv
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from tokenizers import Tokenizer

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_language_model import LanguageModel, BrainCrossAttentionLM  # noqa: E402


def resolve_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def js_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.clamp_min(eps).log() - b.clamp_min(eps).log())).sum(dim=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def crop_pad(tokens: List[int], block_size: int, pad_id: int = 0) -> List[int]:
    tokens = tokens[-block_size:]
    if len(tokens) < block_size:
        tokens = [pad_id] * (block_size - len(tokens)) + tokens
    return tokens


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Trained language_model.pt")
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    ap.add_argument("--brain_dataset", required=True, help="NPZ with contexts/brain/targets")
    ap.add_argument("--alphas", nargs="+", type=float, required=True, help="Alpha scaling values")
    ap.add_argument("--hidden_dim", type=int, default=384)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--attn_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.11)
    ap.add_argument("--block_size", type=int, default=96)
    ap.add_argument("--pad_token_id", type=int, default=0)
    ap.add_argument("--samples", type=int, default=5000, help="Number of samples to evaluate")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--brain_fusion",
        type=str,
        default="add",
        choices=["add", "cross_attn"],
        help="Brain conditioning mechanism",
    )
    ap.add_argument("--brain_tokens", type=int, default=4, help="Number of brain memory tokens (cross_attn)")
    ap.add_argument("--out_csv", type=str, default=None, help="Optional CSV path to save results")
    args = ap.parse_args()

    device = resolve_device(args.device)

    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    data = np.load(args.brain_dataset, allow_pickle=True)
    contexts = data["contexts"].tolist()
    brain = torch.tensor(data["brain"], dtype=torch.float32)
    targets = torch.tensor(data["targets"], dtype=torch.long)
    brain_dim = brain.shape[1]

    if args.brain_fusion == "cross_attn":
        model = BrainCrossAttentionLM(
            vocab_size=vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            attn_heads=args.attn_heads,
            dropout=args.dropout,
            brain_dim=brain_dim,
            brain_tokens=args.brain_tokens,
            pad_token_id=args.pad_token_id,
        ).to(device)
    else:
        model = LanguageModel(
            vocab_size=vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            attn_heads=args.attn_heads,
            dropout=args.dropout,
            brain_dim=brain_dim,
            pad_token_id=args.pad_token_id,
        ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    subset = min(args.samples, len(contexts))
    idxs = torch.randperm(len(contexts))[:subset]

    shuffle_map = torch.randperm(len(contexts))

    iterator = idxs.tolist()
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Evaluating", leave=False)

    results = []

    with torch.no_grad():
        for i in iterator:
            ctx_tokens = crop_pad(contexts[i], args.block_size, args.pad_token_id)
            x = torch.tensor(ctx_tokens, dtype=torch.long, device=device).unsqueeze(0)
            y = targets[i].view(1, 1).to(device)

            z_real_base = brain[i].unsqueeze(0).to(device)
            z_shuf_base = brain[shuffle_map[i]].unsqueeze(0).to(device)
            z_zero = torch.zeros_like(z_real_base)

            def dist_and_nll(z: torch.Tensor):
                logits = model(x, z)[:, -1, :]
                logp = torch.log_softmax(logits, dim=-1)
                nll = -logp.gather(-1, y).squeeze().item()
                probs = logp.exp()
                return probs, nll

            # baseline zero once
            p_zero, nll_zero = dist_and_nll(z_zero)

            for alpha in args.alphas:
                z_real = alpha * z_real_base
                z_shuf = alpha * z_shuf_base

                p_real, nll_real = dist_and_nll(z_real)
                p_shuf, nll_shuf = dist_and_nll(z_shuf)

                js_real = js_div(p_real, p_zero).item()
                js_shuf = js_div(p_shuf, p_zero).item()

                results.append(
                    {
                        "alpha": alpha,
                        "condition": "real",
                        "js": js_real,
                        "nll": nll_real,
                        "nll_zero": nll_zero,
                    }
                )
                results.append(
                    {
                        "alpha": alpha,
                        "condition": "shuf",
                        "js": js_shuf,
                        "nll": nll_shuf,
                        "nll_zero": nll_zero,
                    }
                )

    # Aggregate
    def summarize(cond: str):
        js_vals = [r["js"] for r in results if r["condition"] == cond]
        nll_vals = [r["nll"] for r in results if r["condition"] == cond]
        zero_vals = [r["nll_zero"] for r in results if r["condition"] == cond]
        js_t = torch.tensor(js_vals)
        nll_t = torch.tensor(nll_vals)
        zero_t = torch.tensor(zero_vals)
        delta = (nll_t - zero_t).mean().item()
        print(
            f"{cond.upper()}: JS mean={js_t.mean():.4f} median={js_t.median():.4f} std={js_t.std():.4f} | "
            f"NLL mean={nll_t.mean():.4f} | ΔNLL (vs zero) mean={delta:.4f}"
        )

    print(f"Samples evaluated: {subset}")
    for cond in ["real", "shuf"]:
        summarize(cond)

    if args.out_csv:
        import csv

        fieldnames = ["alpha", "condition", "js", "nll", "nll_zero"]
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote alpha sweep to {args.out_csv}")


if __name__ == "__main__":
    main()

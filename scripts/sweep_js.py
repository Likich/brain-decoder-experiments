#!/usr/bin/env python3
"""
Sweep brain embedding dimensionality and measure its influence on next-token
distributions via JS divergence and top-1 agreement.

Example:
  python3 scripts/sweep_js.py \
    --ckpt models/language_model.pt \
    --tokenizer models/wiki_tokenizer.json \
    --brain_dataset data/brain_ctx_pairs_100k.npz \
    --dims 68 136 272 544 \
    --hidden_dim 384 --num_layers 2 --attn_heads 8 --dropout 0.11049089681925957 \
    --block_size 96 --samples 5000 --device cuda --out_csv js_sweep.csv
"""

import argparse
import csv
from typing import List

import numpy as np
import torch
from tokenizers import Tokenizer

from scripts.train_language_model import LanguageModel


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


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Trained language_model.pt")
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    ap.add_argument("--brain_dataset", required=True, help="NPZ with contexts/brain/targets")
    ap.add_argument("--dims", nargs="+", type=int, required=True, help="Brain dims to test (trim/pad)")
    ap.add_argument("--hidden_dim", type=int, default=384)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--attn_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.11)
    ap.add_argument("--block_size", type=int, default=96)
    ap.add_argument("--pad_token_id", type=int, default=0)
    ap.add_argument("--samples", type=int, default=5000, help="Number of samples to evaluate per dim")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=None, help="Optional CSV to persist results")
    args = ap.parse_args()

    device = resolve_device(args.device)

    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    data = np.load(args.brain_dataset, allow_pickle=True)
    contexts = data["contexts"].tolist()
    brain_full = torch.tensor(data["brain"], dtype=torch.float32)
    base_dim = brain_full.shape[1]

    model = LanguageModel(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        brain_dim=base_dim,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    subset = min(args.samples, len(contexts))
    idxs = torch.randperm(len(contexts))[:subset]

    results = []
    for dim in args.dims:
        dim = int(dim)
        dim_used = min(dim, base_dim)
        brain_slice = brain_full[:, :dim_used]
        if dim_used < base_dim:
            pad = torch.zeros(brain_full.shape[0], base_dim - dim_used)
            brain_use = torch.cat([brain_slice, pad], dim=1)
        else:
            brain_use = brain_slice[:, :base_dim]  # ensure shape (N, base_dim)

        js_vals = []
        agree = 0
        with torch.no_grad():
            for i in idxs.tolist():
                ctx_tokens = crop_pad(contexts[i], args.block_size, args.pad_token_id)
                x = torch.tensor(ctx_tokens, dtype=torch.long, device=device).unsqueeze(0)
                z = brain_use[i].unsqueeze(0).to(device)

                logits_with = model(x, z)
                logits_zero = model(x, torch.zeros_like(z))
                p = torch.softmax(logits_with[:, -1, :], dim=-1)
                q = torch.softmax(logits_zero[:, -1, :], dim=-1)
                js = js_div(p, q).item()
                js_vals.append(js)
                agree += (p.argmax(dim=-1) == q.argmax(dim=-1)).sum().item()

        js_t = torch.tensor(js_vals)
        mean_js = js_t.mean().item()
        median_js = js_t.median().item()
        std_js = js_t.std().item()
        agree_rate = agree / len(js_vals)
        results.append(
            {
                "brain_dim": dim,
                "js_mean": mean_js,
                "js_median": median_js,
                "js_std": std_js,
                "top1_agree": agree_rate,
                "samples": len(js_vals),
            }
        )
        print(
            f"dim={dim:4d} | JS mean={mean_js:.4f} median={median_js:.4f} std={std_js:.4f} | "
            f"top1_agree={agree_rate:.3f} (n={len(js_vals)})"
        )

    if args.out_csv:
        fieldnames = list(results[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote sweep results to {args.out_csv}")


if __name__ == "__main__":
    main()

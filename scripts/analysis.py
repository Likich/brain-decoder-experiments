#!/usr/bin/env python3
"""
Analysis utilities for brain-conditioned vs. unconditioned LM behavior.

Currently implements:
  - JS divergence and top-1 agreement between "with brain" and "zeroed brain"
    next-token distributions on a paired dataset.

Usage:
  python3 scripts/analysis.py \
    --ckpt models/language_model.pt \
    --tokenizer models/wiki_tokenizer.json \
    --brain_dataset data/brain_ctx_pairs_100k.npz \
    --hidden_dim 384 --num_layers 2 --attn_heads 8 --dropout 0.11049089681925957 \
    --block_size 96 --samples 5000 --device cuda
"""

import argparse
import math
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


def crop_pad(tokens: list[int], block_size: int, pad_id: int = 0) -> list[int]:
    tokens = tokens[-block_size:]
    if len(tokens) < block_size:
        tokens = [pad_id] * (block_size - len(tokens)) + tokens
    return tokens


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Trained language_model.pt")
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    ap.add_argument("--brain_dataset", required=True, help="NPZ with contexts/brain/targets")
    ap.add_argument("--hidden_dim", type=int, default=384)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--attn_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.11)
    ap.add_argument("--block_size", type=int, default=96)
    ap.add_argument("--pad_token_id", type=int, default=0)
    ap.add_argument("--samples", type=int, default=5000, help="Number of samples to evaluate")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)

    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    data = np.load(args.brain_dataset, allow_pickle=True)
    contexts = data["contexts"].tolist()
    brain = torch.tensor(data["brain"], dtype=torch.float32)

    brain_dim = brain.shape[1]
    model = LanguageModel(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        brain_dim=brain_dim,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    subset = min(args.samples, len(contexts))
    idxs = torch.randperm(len(contexts))[:subset]

    js_vals = []
    agree = 0

    with torch.no_grad():
        for i in idxs.tolist():
            ctx_tokens = crop_pad(contexts[i], args.block_size, args.pad_token_id)
            x = torch.tensor(ctx_tokens, dtype=torch.long, device=device).unsqueeze(0)
            z = brain[i].unsqueeze(0).to(device)

            logits_with = model(x, z)
            logits_zero = model(x, torch.zeros_like(z))
            p = torch.softmax(logits_with[:, -1, :], dim=-1)
            q = torch.softmax(logits_zero[:, -1, :], dim=-1)
            js = js_div(p, q).item()
            js_vals.append(js)
            agree += (p.argmax(dim=-1) == q.argmax(dim=-1)).sum().item()

    js_t = torch.tensor(js_vals)
    print(f"Samples evaluated: {len(js_vals)}")
    print(f"JS mean={js_t.mean():.4f}, median={js_t.median():.4f}, std={js_t.std():.4f}")
    print(f"Top-1 agreement: {agree/len(js_vals):.3f}")


if __name__ == "__main__":
    main()

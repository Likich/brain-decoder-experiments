#!/usr/bin/env python3
"""
True-token alignment analysis: how much does brain conditioning push the gold token?

For each sample:
  - Compute logit deltas vs zero for REAL and SHUF:
        delta = logits(z) - logits(0)
  - Record gold_delta = delta[y] for REAL and SHUF (y = true next token)
  - Also record log-prob deltas: logp(z)[y] - logp(0)[y]

Outputs summary stats and optional per-sample CSV for plotting histograms
to show REAL shifts gold token up, SHUF does not.

Example:
  python3 scripts/true_token_alignment.py \
    --ckpt models/lm_brain272/language_model.pt \
    --tokenizer models/wiki_tokenizer.json \
    --brain_dataset data/brain_ctx_pairs_272.npz \
    --hidden_dim 384 --num_layers 2 --attn_heads 8 --dropout 0.11049 \
    --block_size 96 --samples 5000 --device cuda \
    --out_csv true_token_alignment_272.csv
"""

import argparse
import sys
from pathlib import Path

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


def crop_pad(tokens, block_size: int, pad_id: int = 0):
    tokens = tokens[-block_size:]
    if len(tokens) < block_size:
        tokens = [pad_id] * (block_size - len(tokens)) + tokens
    return tokens


def main() -> None:
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
    ap.add_argument(
        "--brain_fusion",
        type=str,
        default="add",
        choices=["add", "cross_attn"],
        help="Brain conditioning mechanism",
    )
    ap.add_argument("--brain_tokens", type=int, default=4, help="Number of brain memory tokens (cross_attn)")
    ap.add_argument("--out_csv", type=str, default=None, help="Optional per-sample CSV output")
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
        iterator = tqdm(iterator, desc="True-token alignment", leave=False)

    rows = []

    with torch.no_grad():
        for i in iterator:
            ctx_tokens = crop_pad(contexts[i], args.block_size, args.pad_token_id)
            x = torch.tensor(ctx_tokens, dtype=torch.long, device=device).unsqueeze(0)
            y = targets[i].item()
            y_t = torch.tensor([[y]], device=device)

            z_zero = torch.zeros((1, brain_dim), device=device)
            z_real = brain[i].unsqueeze(0).to(device)
            z_shuf = brain[shuffle_map[i]].unsqueeze(0).to(device)

            def logits_and_gold(z: torch.Tensor):
                logits = model(x, z)[:, -1, :]  # [1, V]
                logp = torch.log_softmax(logits, dim=-1)
                gold_logit = logits[0, y].item()
                gold_logp = logp[0, y].item()
                return gold_logit, gold_logp, logits

            gold_zero, logp_zero, logits_zero = logits_and_gold(z_zero)
            gold_real, logp_real, logits_real = logits_and_gold(z_real)
            gold_shuf, logp_shuf, logits_shuf = logits_and_gold(z_shuf)

            rows.append(
                {
                    "idx": int(i),
                    "gold_token": int(y),
                    "delta_logit_real": gold_real - gold_zero,
                    "delta_logit_shuf": gold_shuf - gold_zero,
                    "delta_logp_real": logp_real - logp_zero,
                    "delta_logp_shuf": logp_shuf - logp_zero,
                }
            )

    # Summaries
    def summarize(field: str):
        vals = torch.tensor([r[field] for r in rows])
        return vals.mean().item(), vals.median().item(), vals.std().item()

    m_rl, md_rl, s_rl = summarize("delta_logit_real")
    m_sh, md_sh, s_sh = summarize("delta_logit_shuf")
    print(f"Δlogit@gold REAL: mean={m_rl:.4f}, median={md_rl:.4f}, std={s_rl:.4f}")
    print(f"Δlogit@gold SHUF: mean={m_sh:.4f}, median={md_sh:.4f}, std={s_sh:.4f}")

    if args.out_csv:
        import csv

        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["idx", "gold_token", "delta_logit_real", "delta_logit_shuf", "delta_logp_real", "delta_logp_shuf"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote per-sample alignment metrics to {args.out_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Token-level sensitivity analysis for brain-conditioned LMs.

For each sample, compares logits with brain conditioning vs zero-brain baseline:
    delta = logits(cond) - logits(zero)
and measures mean |delta| over token buckets:
    - punctuation
    - stopwords
    - content (everything else, excluding specials/empty)
Also reports delta on the gold target logit.

Supports conditioning modes:
    real  : paired brain vector
    shuf  : shuffled brain vector
    gauss : Gaussian brain with dataset mean/std per dimension

Outputs per-mode summaries and optional per-sample CSV.

Example:
  python3 scripts/token_sensitivity.py \
    --ckpt models/lm_brain136/language_model.pt \
    --tokenizer models/wiki_tokenizer.json \
    --brain_dataset data/brain_ctx_pairs_100k.npz \
    --modes real shuf gauss \
    --hidden_dim 384 --num_layers 2 --attn_heads 8 --dropout 0.11049 \
    --block_size 96 --samples 5000 --device cuda \
    --out_csv token_sensitivity_136.csv
"""

import argparse
import sys
import string
import unicodedata
from pathlib import Path
from typing import List, Dict, Set

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


STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "at", "by", "from",
    "with", "as", "that", "this", "it", "is", "are", "was", "were", "be", "been",
    "being", "which", "who", "whom", "what", "when", "where", "why", "how", "not",
    "but", "if", "about", "into", "through", "during", "before", "after", "above",
    "below", "up", "down", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "should", "now", "i", "you", "he", "she", "they",
    "we", "me", "him", "her", "them", "us", "my", "your", "his", "their", "our",
    "mine", "yours", "hers", "theirs", "ours", "do", "does", "did", "doing",
}


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


def is_punct_token(t: str) -> bool:
    if not t:
        return False
    return all(unicodedata.category(ch).startswith("P") or ch in string.punctuation for ch in t)


def normalize_token(t: str) -> str:
    # Strip common whitespace markers from BPE/SentencePiece tokens
    for marker in ("Ġ", "▁", "▂", "▃", "Ċ", "ĉ", " "):
        t = t.replace(marker, " ")
    return t.strip().lower()


def build_buckets(tok: Tokenizer) -> Dict[str, List[int]]:
    vocab_size = tok.get_vocab_size()
    punct_ids, stop_ids, content_ids = [], [], []
    # Attempt to gather special ids from tokenizer; fallback to none
    special_tokens = set()
    try:
        special_tokens.update(tok.get_special_tokens().keys())  # type: ignore
    except Exception:
        pass

    for tid in range(vocab_size):
        t_str = tok.id_to_token(tid)
        if t_str is None:
            continue
        t_norm = normalize_token(t_str)
        if not t_norm:
            continue
        if t_norm in special_tokens:
            continue
        if is_punct_token(t_norm):
            punct_ids.append(tid)
        elif t_norm in STOPWORDS:
            stop_ids.append(tid)
        else:
            content_ids.append(tid)
    return {"punct": punct_ids, "stop": stop_ids, "content": content_ids}


def tensor_mean_on_indices(v: torch.Tensor, idx: List[int]) -> float:
    if not idx:
        return float("nan")
    return v[idx].mean().item()


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Trained language_model.pt")
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    ap.add_argument("--brain_dataset", required=True, help="NPZ with contexts/brain/targets")
    ap.add_argument("--modes", nargs="+", default=["real", "shuf", "gauss"], help="Which conditioning modes to run")
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

    modes = set(m.lower() for m in args.modes)

    device = resolve_device(args.device)

    tok = Tokenizer.from_file(args.tokenizer)
    buckets = build_buckets(tok)

    data = np.load(args.brain_dataset, allow_pickle=True)
    contexts = data["contexts"].tolist()
    brain = torch.tensor(data["brain"], dtype=torch.float32)
    targets = torch.tensor(data["targets"], dtype=torch.long)
    brain_dim = brain.shape[1]

    if args.brain_fusion == "cross_attn":
        model = BrainCrossAttentionLM(
            vocab_size=tok.get_vocab_size(),
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
            vocab_size=tok.get_vocab_size(),
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

    mu = brain.mean(dim=0, keepdim=True).to(device)
    sigma = brain.std(dim=0, keepdim=True).to(device)
    shuffle_map = torch.randperm(len(contexts))

    subset = min(args.samples, len(contexts))
    idxs = torch.randperm(len(contexts))[:subset]

    iterator = idxs.tolist()
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Token sensitivity", leave=False)

    per_sample = []

    with torch.no_grad():
        for i in iterator:
            ctx_tokens = crop_pad(contexts[i], args.block_size, args.pad_token_id)
            x = torch.tensor(ctx_tokens, dtype=torch.long, device=device).unsqueeze(0)
            y = targets[i].view(1, 1).to(device)

            z_zero = torch.zeros((1, brain_dim), device=device)

            def logits_and_delta(z: torch.Tensor):
                logits = model(x, z)[:, -1, :]  # [1, V]
                logp = torch.log_softmax(logits, dim=-1)
                nll = -logp.gather(-1, y).squeeze().item()
                return logits.squeeze(0), nll

            logits_zero, nll_zero = logits_and_delta(z_zero)

            # Prepare conditioned vectors
            z_real = brain[i].unsqueeze(0).to(device) if "real" in modes else None
            z_shuf = brain[shuffle_map[i]].unsqueeze(0).to(device) if "shuf" in modes else None
            z_gauss = mu + sigma * torch.randn_like(z_zero) if "gauss" in modes else None

            conds = {
                "real": z_real,
                "shuf": z_shuf,
                "gauss": z_gauss,
            }

            for mode, z in conds.items():
                if mode not in modes or z is None:
                    continue
                logits_cond, nll_cond = logits_and_delta(z)
                delta = logits_cond - logits_zero  # [V]
                sens = delta.abs()

                punct = tensor_mean_on_indices(sens, buckets["punct"])
                stop = tensor_mean_on_indices(sens, buckets["stop"])
                content = tensor_mean_on_indices(sens, buckets["content"])
                gold_delta = delta[targets[i].item()].item()

                per_sample.append(
                    {
                        "mode": mode,
                        "idx": int(i),
                        "nll": nll_cond,
                        "nll_zero": nll_zero,
                        "delta_gold": gold_delta,
                        "sens_punct": punct,
                        "sens_stop": stop,
                        "sens_content": content,
                        "content_over_stop": content / stop if stop == stop else float("nan"),
                        "content_over_punct": content / punct if punct == punct else float("nan"),
                    }
                )

    def summarize(field: str, mode: str) -> float:
        vals = [r[field] for r in per_sample if r["mode"] == mode]
        t = torch.tensor(vals)
        return t.mean().item()

    modes_present = sorted(set(r["mode"] for r in per_sample))
    print(f"Samples evaluated: {subset}")
    for m in modes_present:
        mean_punct = summarize("sens_punct", m)
        mean_stop = summarize("sens_stop", m)
        mean_content = summarize("sens_content", m)
        mean_delta_gold = summarize("delta_gold", m)
        mean_nll = summarize("nll", m)
        mean_nll_zero = summarize("nll_zero", m)
        print(
            f"{m.upper()}: "
            f"sens punct={mean_punct:.4f}, stop={mean_stop:.4f}, content={mean_content:.4f} | "
            f"content/stop={mean_content/mean_stop if mean_stop else float('nan'):.3f} "
            f"content/punct={mean_content/mean_punct if mean_punct else float('nan'):.3f} | "
            f"Δlogit_gold={mean_delta_gold:.4f} | NLL={mean_nll:.4f} vs zero={mean_nll_zero:.4f}"
        )

    if args.out_csv:
        import csv

        fieldnames = list(per_sample[0].keys()) if per_sample else []
        if fieldnames:
            with open(args.out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_sample)
            print(f"Wrote per-sample metrics to {args.out_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Control analysis for brain-conditioned LMs.

Computes JS divergence and top-1 agreement between the zeroed-brain condition
and three controls on the same samples:
  - REAL:   true paired brain vector
  - SHUF:   brain vector shuffled across samples
  - GAUSS:  Gaussian sample with dataset mean/std per dimension

Usage example:
  python3 scripts/analysis_controls.py \
    --ckpt models/lm_brain68/language_model.pt \
    --tokenizer models/wiki_tokenizer.json \
    --brain_dataset data/brain_ctx_pairs_68.npz \
    --hidden_dim 384 --num_layers 2 --attn_heads 8 --dropout 0.11049 \
    --block_size 96 --samples 5000 --device cuda
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


def js_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.clamp_min(eps).log() - b.clamp_min(eps).log())).sum(dim=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def crop_pad(tokens: list[int], block_size: int, pad_id: int = 0) -> list[int]:
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
    ap.add_argument("--freeze_brain_proj", action="store_true", help="Zero-out brain projection weights before eval")
    ap.add_argument(
        "--brain_fusion",
        type=str,
        default="add",
        choices=["add", "cross_attn"],
        help="Brain conditioning mechanism",
    )
    ap.add_argument("--brain_tokens", type=int, default=4, help="Number of brain memory tokens (cross_attn)")
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
    if args.freeze_brain_proj and getattr(model, "brain_proj", None) is not None:
        with torch.no_grad():
            model.brain_proj.weight.zero_()
            if model.brain_proj.bias is not None:
                model.brain_proj.bias.zero_()
        for p in model.brain_proj.parameters():
            p.requires_grad = False
        print("Brain projection frozen to zero.")

    # Precompute stats and permutation
    mu = brain.mean(dim=0, keepdim=True)
    sigma = brain.std(dim=0, keepdim=True)
    shuf_idx = torch.randperm(brain.size(0))

    subset = min(args.samples, len(contexts))
    idxs = torch.randperm(len(contexts))[:subset]

    js_real, js_shuf, js_gauss = [], [], []
    nll_real, nll_shuf, nll_gauss, nll_zero = [], [], [], []
    agree_real = agree_shuf = agree_gauss = 0

    iterator = idxs.tolist()
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Evaluating", leave=False)

    mu_d = mu.to(device)
    sigma_d = sigma.to(device)

    with torch.no_grad():
        for i in iterator:
            ctx_tokens = crop_pad(contexts[i], args.block_size, args.pad_token_id)
            x = torch.tensor(ctx_tokens, dtype=torch.long, device=device).unsqueeze(0)
            y = targets[i].view(1, 1).to(device)

            z_real = brain[i].unsqueeze(0).to(device)
            z_zero = torch.zeros_like(z_real)
            z_shuf = brain[shuf_idx[i]].unsqueeze(0).to(device)
            z_gauss = mu_d + sigma_d * torch.randn_like(z_real)

            def dist_and_nll(z: torch.Tensor):
                logits = model(x, z)[:, -1, :]           # [1, vocab]
                logp = torch.log_softmax(logits, dim=-1)  # [1, vocab]
                nll = -logp.gather(-1, y).squeeze().item()
                probs = logp.exp()
                return probs, nll

            p_real, nll_r = dist_and_nll(z_real)
            p_zero, nll_z = dist_and_nll(z_zero)
            p_shuf, nll_s = dist_and_nll(z_shuf)
            p_gauss, nll_g = dist_and_nll(z_gauss)

            js_real.append(js_div(p_real, p_zero).item())
            js_shuf.append(js_div(p_shuf, p_zero).item())
            js_gauss.append(js_div(p_gauss, p_zero).item())

            nll_real.append(nll_r)
            nll_zero.append(nll_z)
            nll_shuf.append(nll_s)
            nll_gauss.append(nll_g)

            agree_real += (p_real.argmax(dim=-1) == p_zero.argmax(dim=-1)).sum().item()
            agree_shuf += (p_shuf.argmax(dim=-1) == p_zero.argmax(dim=-1)).sum().item()
            agree_gauss += (p_gauss.argmax(dim=-1) == p_zero.argmax(dim=-1)).sum().item()

    def summarize_js(name: str, vals: list[float], agree: int) -> None:
        t = torch.tensor(vals)
        print(
            f"{name}: JS mean={t.mean():.4f} median={t.median():.4f} std={t.std():.4f} | "
            f"top1_agree={agree/len(vals):.3f} (n={len(vals)})"
        )

    def summarize_nll(name: str, vals: list[float]) -> None:
        t = torch.tensor(vals)
        print(f"{name}: NLL mean={t.mean():.4f} median={t.median():.4f} std={t.std():.4f}")

    print(f"Samples evaluated: {subset}")
    summarize_js("REAL", js_real, agree_real)
    summarize_js("SHUF", js_shuf, agree_shuf)
    summarize_js("GAUSS", js_gauss, agree_gauss)
    summarize_nll("NLL REAL", nll_real)
    summarize_nll("NLL ZERO", nll_zero)
    summarize_nll("NLL SHUF", nll_shuf)
    summarize_nll("NLL GAUSS", nll_gauss)


if __name__ == "__main__":
    main()

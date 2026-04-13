#!/usr/bin/env python3
"""
Evaluate Qwen brain-prefix model: NLL + JS for REAL vs SHUF vs ZERO.

Assumes model directory contains:
  - base model weights (HF)
  - brain_prefix.pt (brain_proj weights + prefix_tokens + brain_dim)

Example:
  python3 scripts/eval_qwen_prefix_controls.py \
    --model_dir models/qwen_brain_prefix_136_frozen \
    --brain_dataset data/brain_ctx_pairs_100k_qwen_136.npz \
    --block_size 96 --samples 5000 --device cuda \
    --trust_remote_code
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def resolve_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def js_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.clamp_min(eps).log() - b.clamp_min(eps).log())).sum(dim=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def build_input(context, target, block_size: int, pad_id: int):
    max_ctx = block_size - 1
    tokens = list(context)[-max_ctx:]
    if len(tokens) < max_ctx:
        tokens = [pad_id] * (max_ctx - len(tokens)) + tokens
    input_ids = tokens + [int(target)]
    attention_mask = [0 if t == pad_id else 1 for t in tokens] + [1]
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


class BrainPrefixWrapper(nn.Module):
    def __init__(self, base, brain_dim: int, prefix_tokens: int):
        super().__init__()
        self.base = base
        hidden = base.config.hidden_size
        self.prefix_tokens = prefix_tokens
        self.brain_proj = nn.Linear(brain_dim, hidden * prefix_tokens)

    def forward(self, input_ids, attention_mask, brain_vec):
        embed_dtype = self.base.get_input_embeddings().weight.dtype
        inputs_embeds = self.base.get_input_embeddings()(input_ids).to(embed_dtype)
        batch = input_ids.size(0)
        prefix = self.brain_proj(brain_vec).view(batch, self.prefix_tokens, -1).to(embed_dtype)
        inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)

        prefix_mask = torch.ones(batch, self.prefix_tokens, device=attention_mask.device)
        attn = torch.cat([prefix_mask, attention_mask], dim=1)

        return self.base(inputs_embeds=inputs_embeds, attention_mask=attn)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model_dir", type=Path, required=True)
    ap.add_argument("--brain_dataset", type=Path, required=True)
    ap.add_argument("--block_size", type=int, default=96)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--no_tqdm", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=args.trust_remote_code, torch_dtype="auto", device_map=None
    ).to(device)
    base.eval()

    state = torch.load(args.model_dir / "brain_prefix.pt", map_location="cpu")
    brain_dim = int(state["brain_dim"])
    prefix_tokens = int(state["prefix_tokens"])

    model = BrainPrefixWrapper(base, brain_dim=brain_dim, prefix_tokens=prefix_tokens).to(device)
    model.brain_proj.load_state_dict(state["brain_proj"])
    model.eval()

    data = np.load(args.brain_dataset, allow_pickle=True)
    contexts = data["contexts"].tolist()
    brain = torch.tensor(data["brain"], dtype=torch.float32)
    targets = torch.tensor(data["targets"], dtype=torch.long)

    subset = min(args.samples, len(contexts))
    idxs = torch.randperm(len(contexts))[:subset]
    shuffle_map = torch.randperm(len(contexts))

    js_real, js_shuf = [], []
    nll_real, nll_zero, nll_shuf = [], [], []
    agree_real = agree_shuf = 0

    iterator = idxs.tolist()
    use_tqdm = tqdm is not None and (not args.no_tqdm) and sys.stderr.isatty()
    if use_tqdm:
        iterator = tqdm(iterator, desc="Evaluating", leave=False)

    with torch.no_grad():
        for i in iterator:
            input_ids, attn = build_input(contexts[i], targets[i].item(), args.block_size, tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device)
            attn = attn.unsqueeze(0).to(device)

            z_real = brain[i].unsqueeze(0).to(device)
            z_zero = torch.zeros_like(z_real)
            z_shuf = brain[shuffle_map[i]].unsqueeze(0).to(device)

            def logits_and_nll(z):
                out = model(input_ids, attn, z)
                logits = out.logits[:, -1, :]
                logp = torch.log_softmax(logits, dim=-1)
                nll = -logp.gather(-1, targets[i].view(1, 1).to(device)).squeeze().item()
                probs = logp.exp()
                return logits, probs, nll

            _, p_real, nll_r = logits_and_nll(z_real)
            _, p_zero, nll_z = logits_and_nll(z_zero)
            _, p_shuf, nll_s = logits_and_nll(z_shuf)

            js_real.append(js_div(p_real, p_zero).item())
            js_shuf.append(js_div(p_shuf, p_zero).item())

            nll_real.append(nll_r)
            nll_zero.append(nll_z)
            nll_shuf.append(nll_s)

            agree_real += (p_real.argmax(dim=-1) == p_zero.argmax(dim=-1)).sum().item()
            agree_shuf += (p_shuf.argmax(dim=-1) == p_zero.argmax(dim=-1)).sum().item()

    def summarize_js(name, vals, agree):
        t = torch.tensor(vals)
        print(
            f"{name}: JS mean={t.mean():.4f} median={t.median():.4f} std={t.std():.4f} | "
            f"top1_agree={agree/len(vals):.3f} (n={len(vals)})"
        )

    def summarize_nll(name, vals):
        t = torch.tensor(vals)
        print(f"{name}: NLL mean={t.mean():.4f} median={t.median():.4f} std={t.std():.4f}")

    print(f"Samples evaluated: {subset}")
    summarize_js("REAL", js_real, agree_real)
    summarize_js("SHUF", js_shuf, agree_shuf)
    summarize_nll("NLL REAL", nll_real)
    summarize_nll("NLL ZERO", nll_zero)
    summarize_nll("NLL SHUF", nll_shuf)

    t_real = torch.tensor(nll_real)
    t_zero = torch.tensor(nll_zero)
    t_shuf = torch.tensor(nll_shuf)
    print(f"ΔNLL real-zero: {(t_real - t_zero).mean().item():.4f}")
    print(f"ΔNLL shuf-zero: {(t_shuf - t_zero).mean().item():.4f}")


if __name__ == "__main__":
    main()

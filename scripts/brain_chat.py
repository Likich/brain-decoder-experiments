#!/usr/bin/env python3
"""
Interactive brain-conditioned chat loop.

Usage example:
  python3 scripts/brain_chat.py \
    --tokenizer models/wiki_tokenizer.json \
    --ckpt models/language_model.pt \
    --brain_dataset data/brain_ctx_pairs_100k.npz \
    --brain_index 0 \
    --block_size 96 --max_new_tokens 40 --device cuda

Type 'exit' or 'quit' to stop.
"""

import argparse
import numpy as np
import torch
from tokenizers import Tokenizer

# Reuse the language model definition
from scripts.train_language_model import LanguageModel, BrainCrossAttentionLM  # noqa: E402


def resolve_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_brain_vector(path: str, index: int) -> torch.Tensor:
    data = np.load(path, allow_pickle=True)
    brain = torch.tensor(data["brain"], dtype=torch.float32)
    if index < 0 or index >= brain.shape[0]:
        raise IndexError(f"brain_index {index} out of range 0..{brain.shape[0]-1}")
    return brain[index]


def crop_context(tokens: list[int], block_size: int, pad_id: int) -> torch.Tensor:
    """Pad/trim token list to fixed block_size."""
    tokens = tokens[-block_size:]
    if len(tokens) < block_size:
        tokens = [pad_id] * (block_size - len(tokens)) + tokens
    return torch.tensor(tokens, dtype=torch.long)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    Sample a token from logits with temperature and optional top-k.
    logits: (batch, vocab)
    returns: (batch, 1) token ids
    """
    if temperature <= 0.0:
        # Degenerates to argmax if someone sets temperature<=0
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        return next_id

    logits = logits / temperature

    if top_k > 0:
        # Keep only top_k logits, set others to -inf
        top_vals, top_idx = torch.topk(logits, k=top_k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(dim=-1, index=top_idx, src=top_vals)
        logits = mask

    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)  # (batch, 1)
    return next_id


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    ap.add_argument("--ckpt", required=True, help="Path to trained language_model.pt")
    ap.add_argument("--brain_dataset", required=True, help="NPZ with 'brain' vectors")
    ap.add_argument("--brain_index", type=int, default=0, help="Row to use from brain matrix")
    ap.add_argument("--block_size", type=int, default=96, help="Context length used during training")
    ap.add_argument("--hidden_dim", type=int, default=384, help="Model hidden dim (must match checkpoint)")
    ap.add_argument("--num_layers", type=int, default=2, help="Transformer layers (match checkpoint)")
    ap.add_argument("--attn_heads", type=int, default=8, help="Attention heads (match checkpoint)")
    ap.add_argument("--dropout", type=float, default=0.11, help="Dropout used at training (match checkpoint)")
    ap.add_argument("--max_new_tokens", type=int, default=40, help="Tokens to generate per turn")
    ap.add_argument("--pad_token_id", type=int, default=0, help="Pad token id for context")
    ap.add_argument(
        "--brain_fusion",
        type=str,
        default="add",
        choices=["add", "cross_attn"],
        help="Brain conditioning mechanism",
    )
    ap.add_argument("--brain_tokens", type=int, default=4, help="Number of brain memory tokens (cross_attn)")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    ap.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    ap.add_argument("--device", type=str, default=None, help="cpu/cuda")
    args = ap.parse_args()

    device = resolve_device(args.device)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    # Load a single brain vector and keep it fixed for the session
    brain_vec = load_brain_vector(args.brain_dataset, args.brain_index).unsqueeze(0).to(device)

    # Load model
    if args.brain_fusion == "cross_attn":
        model = BrainCrossAttentionLM(
            vocab_size=vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            attn_heads=args.attn_heads,
            dropout=args.dropout,
            brain_dim=brain_vec.shape[-1],
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
            brain_dim=brain_vec.shape[-1],
            pad_token_id=args.pad_token_id,
        ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("Interactive brain chat (type 'exit' or 'quit' to stop)")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        # --- Turn-level context: only current user message ---
        user_tokens = tokenizer.encode(user).ids
        if not user_tokens:
            print("Brain: ")
            continue

        # Prepare model input for this turn
        ctx = crop_context(user_tokens, args.block_size, args.pad_token_id).unsqueeze(0).to(device)

        with torch.no_grad():
            cur_ids = ctx
            start_len = cur_ids.shape[1]
            for _ in range(args.max_new_tokens):
                logits = model(cur_ids, brain_vec)  # (1, seq_len, vocab)
                next_logits = logits[:, -1, :]      # (1, vocab)
                next_id = sample_next_token(
                    next_logits,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                cur_ids = torch.cat([cur_ids, next_id], dim=1)

        generated = cur_ids[0].tolist()
        new_tokens = generated[start_len:]
        # Strip pads from the printed response
        response_tokens = [t for t in new_tokens if t != args.pad_token_id]
        text = tokenizer.decode(response_tokens) if response_tokens else ""
        print(f"Brain: {text}")


if __name__ == "__main__":
    main()

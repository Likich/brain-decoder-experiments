#!/usr/bin/env python3
"""
Train an encoder-decoder (seq2seq) model with brain prefix tokens.

Brain vector is projected into K prefix tokens and prepended to encoder inputs.
Decoder predicts the true next token (single-token target).

IMPORTANT: The brain dataset must be built with the SAME tokenizer as the
model you train here.

Example:
  python3 scripts/train_seq2seq_brain_prefix.py \
    --model_name t5-small \
    --brain_dataset data/brain_ctx_pairs_100k_t5.npz \
    --prefix_tokens 8 --block_size 96 \
    --epochs 3 --batch_size 16 --lr 2e-4 --device cuda \
    --out_dir models/t5_brain_prefix
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def resolve_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BrainPrefixSeq2SeqDataset(Dataset):
    def __init__(
        self,
        contexts: List[List[int]],
        brains: np.ndarray,
        targets: np.ndarray,
        block_size: int,
        pad_token_id: int,
    ):
        self.contexts = [list(map(int, ctx)) for ctx in contexts]
        self.brains = torch.tensor(brains, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.block_size = block_size
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.targets)

    def _build_text(self, tokens: List[int]):
        tokens = tokens[-self.block_size :]
        if len(tokens) < self.block_size:
            pad_len = self.block_size - len(tokens)
            tokens = [self.pad_token_id] * pad_len + tokens
        attention_mask = [0 if t == self.pad_token_id else 1 for t in tokens]
        return tokens, attention_mask

    def __getitem__(self, idx: int):
        input_ids, attention_mask = self._build_text(self.contexts[idx])
        label = [int(self.targets[idx])]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            self.brains[idx],
            torch.tensor(label, dtype=torch.long),
        )


class BrainPrefixSeq2Seq(nn.Module):
    def __init__(self, base_model: AutoModelForSeq2SeqLM, brain_dim: int, prefix_tokens: int):
        super().__init__()
        self.base = base_model
        hidden = getattr(base_model.config, "d_model", base_model.config.hidden_size)
        self.prefix_tokens = prefix_tokens
        self.brain_proj = nn.Linear(brain_dim, hidden * prefix_tokens)

    def forward(self, input_ids, attention_mask, brain_vec, labels=None):
        embed_dtype = self.base.get_input_embeddings().weight.dtype
        inputs_embeds = self.base.get_input_embeddings()(input_ids).to(embed_dtype)
        batch = input_ids.size(0)
        prefix = self.brain_proj(brain_vec).view(batch, self.prefix_tokens, -1).to(embed_dtype)
        encoder_embeds = torch.cat([prefix, inputs_embeds], dim=1)

        prefix_mask = torch.ones(batch, self.prefix_tokens, device=attention_mask.device)
        encoder_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        return self.base(inputs_embeds=encoder_embeds, attention_mask=encoder_mask, labels=labels)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model_name", required=True, help="HF seq2seq model name or local path")
    ap.add_argument("--brain_dataset", required=True, help="NPZ with contexts/brain/targets")
    ap.add_argument("--out_dir", type=Path, default=Path("models/seq2seq_brain_prefix"))
    ap.add_argument("--prefix_tokens", type=int, default=8)
    ap.add_argument("--block_size", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--freeze_base", action="store_true", help="Train only brain_proj")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--no_tqdm", action="store_true", help="Disable progress bar")
    ap.add_argument("--log_interval", type=int, default=200, help="Steps between loss logs when tqdm is off")
    args = ap.parse_args()

    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    data = np.load(args.brain_dataset, allow_pickle=True)
    contexts = data["contexts"].tolist()
    brain = data["brain"]
    targets = data["targets"].astype(np.int64)
    brain_dim = brain.shape[1]

    max_id = max(max(ctx) for ctx in contexts if len(ctx))
    if max_id >= tokenizer.vocab_size:
        print(
            f"WARNING: dataset token ids max={max_id} >= tokenizer vocab={tokenizer.vocab_size}.\n"
            "This dataset likely uses a different tokenizer. Rebuild dataset with the same tokenizer."
        )

    dataset = BrainPrefixSeq2SeqDataset(contexts, brain, targets, args.block_size, tokenizer.pad_token_id)
    n_total = len(dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    base.config.use_cache = False
    model = BrainPrefixSeq2Seq(base, brain_dim=brain_dim, prefix_tokens=args.prefix_tokens).to(device)

    if args.freeze_base:
        for p in model.base.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = (len(train_loader) * args.epochs) // max(1, args.grad_accum_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        iterator = train_loader
        use_tqdm = tqdm is not None and (not args.no_tqdm) and sys.stderr.isatty()
        if use_tqdm:
            iterator = tqdm(iterator, desc=f"epoch {epoch}", leave=False, mininterval=10)
        optimizer.zero_grad(set_to_none=True)
        for step, (input_ids, attn, brain_vec, labels) in enumerate(iterator, start=1):
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            brain_vec = brain_vec.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attn, brain_vec, labels=labels)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            running += loss.item()

            if step % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not use_tqdm and args.log_interval and step % args.log_interval == 0:
                avg = running / step
                print(f"epoch {epoch} step {step}/{len(train_loader)} train_loss={avg:.4f}")

        avg_train = running / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, attn, brain_vec, labels in val_loader:
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                brain_vec = brain_vec.to(device)
                labels = labels.to(device)
                outputs = model(input_ids, attn, brain_vec, labels=labels)
                val_loss += outputs.loss.item()
        avg_val = val_loss / max(1, len(val_loader))

        print(f"epoch {epoch}: train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

    model.base.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    torch.save(
        {
            "brain_proj": model.brain_proj.state_dict(),
            "prefix_tokens": args.prefix_tokens,
            "brain_dim": brain_dim,
        },
        args.out_dir / "brain_prefix.pt",
    )
    print(f"Saved base model + brain prefix to {args.out_dir}")


if __name__ == "__main__":
    main()

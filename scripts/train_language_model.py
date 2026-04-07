import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from tokenizers import Tokenizer


def resolve_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        attn_heads: int = 4,
        dropout: float = 0.1,
        brain_dim: int | None = None,
        pad_token_id: int | None = None,
        max_pos: int = 2048,
    ):
        super().__init__()
        if hidden_dim % attn_heads != 0:
            raise ValueError("hidden_dim must be divisible by attn_heads")

        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.positional_embedding = nn.Embedding(max_pos, hidden_dim)  # Max sequence length
        self.brain_proj = nn.Linear(brain_dim, hidden_dim) if brain_dim is not None else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, brain_z: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = x.shape
        pad_mask = None
        if self.pad_token_id is not None:
            pad_mask = x.eq(self.pad_token_id)
            # Positions only count non-pad tokens (left padding safe)
            positions = (x.ne(self.pad_token_id).cumsum(dim=1) - 1).clamp(min=0)
        else:
            positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        tok_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(positions)
        h = tok_emb + pos_emb

        if self.brain_proj is not None:
            if brain_z is None:
                raise ValueError("brain_z must be provided when brain_dim is set")
            brain_emb = self.brain_proj(brain_z).unsqueeze(1).expand(batch_size, seq_len, -1)
            h = h + brain_emb

        h = self.dropout(h)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=pad_mask)
        return self.head(h)


class BrainCrossAttentionLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        attn_heads: int = 4,
        dropout: float = 0.1,
        brain_dim: int | None = None,
        brain_tokens: int = 4,
        pad_token_id: int | None = None,
        max_pos: int = 2048,
    ):
        super().__init__()
        if hidden_dim % attn_heads != 0:
            raise ValueError("hidden_dim must be divisible by attn_heads")
        if brain_dim is None:
            raise ValueError("brain_dim must be provided for cross-attention model")

        self.pad_token_id = pad_token_id
        self.brain_tokens = brain_tokens
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.positional_embedding = nn.Embedding(max_pos, hidden_dim)
        self.brain_proj = nn.Linear(brain_dim, hidden_dim * brain_tokens)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, brain_z: torch.Tensor | None = None) -> torch.Tensor:
        if brain_z is None:
            raise ValueError("brain_z must be provided for cross-attention model")
        batch_size, seq_len = x.shape
        pad_mask = None
        if self.pad_token_id is not None:
            pad_mask = x.eq(self.pad_token_id)
            positions = (x.ne(self.pad_token_id).cumsum(dim=1) - 1).clamp(min=0)
        else:
            positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        tok_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(positions)
        h = self.dropout(tok_emb + pos_emb)

        # Brain memory tokens
        mem = self.brain_proj(brain_z).view(batch_size, self.brain_tokens, -1)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        h = self.decoder(h, mem, tgt_mask=causal_mask, tgt_key_padding_mask=pad_mask)
        return self.head(h)


class TextDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx: int):
        chunk = self.token_ids[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class BrainConditionedDataset(Dataset):
    def __init__(
        self,
        contexts: Sequence[Sequence[int]],
        brains: np.ndarray,
        targets: Sequence[int],
        block_size: int,
        pad_token_id: int,
    ):
        self.contexts: List[List[int]] = [list(map(int, ctx)) for ctx in contexts]
        self.brains = torch.tensor(brains, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.block_size = block_size
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.targets)

    def _process_tokens(self, tokens: Sequence[int]) -> torch.Tensor:
        tokens = list(tokens)[-self.block_size :]
        if len(tokens) < self.block_size:
            tokens = [self.pad_token_id] * (self.block_size - len(tokens)) + tokens
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx: int):
        x = self._process_tokens(self.contexts[idx])
        z = self.brains[idx]
        y = self.targets[idx]
        return x, z, y


def load_brain_dataset(path: Path, block_size: int, pad_token_id: int):
    data = np.load(path, allow_pickle=True)
    required = {"contexts", "brain", "targets"}
    if not required.issubset(data.files):
        raise SystemExit(f"Brain dataset must contain keys {required}, found {set(data.files)}")

    contexts = data["contexts"].tolist()
    brain = data["brain"]
    targets = data["targets"].astype(np.int64)
    dataset = BrainConditionedDataset(contexts, brain, targets, block_size, pad_token_id)
    brain_dim = brain.shape[1]

    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    return train_ds, val_ds, brain_dim


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--data_file",
        type=Path,
        required=True,
        help="Path to raw text file (e.g., wikitext-2/wiki.train.tokens)",
    )
    ap.add_argument("--tokenizer_file", type=Path, required=True, help="Path to tokenizer.json")
    ap.add_argument(
        "--brain_dataset",
        type=Path,
        default=None,
        help="Optional NPZ file with contexts/brain/targets for brain-conditioned training",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("models"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=128, help="Sequence length")
    ap.add_argument("--hidden_dim", type=int, default=384)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--attn_heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--pad_token_id", type=int, default=0, help="Padding id for contexts")
    ap.add_argument(
        "--brain_fusion",
        type=str,
        default="add",
        choices=["add", "cross_attn"],
        help="Brain conditioning mechanism",
    )
    ap.add_argument("--brain_tokens", type=int, default=4, help="Number of brain memory tokens (cross_attn)")
    ap.add_argument("--max_pos", type=int, default=2048, help="Max positional embedding length")
    args = ap.parse_args()

    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(str(args.tokenizer_file))
    brain_mode = args.brain_dataset is not None

    if brain_mode:
        print(f"Loading brain-conditioned dataset from {args.brain_dataset}...")
        train_ds, val_ds, brain_dim = load_brain_dataset(
            args.brain_dataset, args.block_size, args.pad_token_id
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        print(
            f"Loaded {len(train_ds) + len(val_ds)} paired samples -> train {len(train_ds)}, val {len(val_ds)}"
        )
    else:
        print(f"Loading data from {args.data_file}...")
        all_tokens: list[int] = []
        if "tokens" in args.data_file.name:
            print("Detected pre-tokenized data file.")
            with open(args.data_file, "r", encoding="utf-8") as f:
                for line in f:
                    article = json.loads(line)
                    all_tokens.extend(article["tokens"])
        else:
            print("Detected raw text data file. Tokenizing now...")
            if args.data_file.name.endswith(".jsonl"):
                texts = []
                with open(args.data_file, "r", encoding="utf-8") as f:
                    for line in f:
                        texts.append(json.loads(line)["text"])
                text = "\n\n".join(texts)
            else:
                with open(args.data_file, "r", encoding="utf-8") as f:
                    text = f.read()
            all_tokens = tokenizer.encode(text).ids

        if not all_tokens:
            raise SystemExit("No tokens found in the data file. Please check the inputs.")

        n = len(all_tokens)
        n_train = int(0.9 * n)
        train_tokens = all_tokens[:n_train]
        val_tokens = all_tokens[n_train:]
        train_ds = TextDataset(train_tokens, args.block_size)
        val_ds = TextDataset(val_tokens, args.block_size)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        brain_dim = None

        print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
        print(f"Total tokens: {n}")
        print(f"Training with {len(train_ds)} sequences, validating with {len(val_ds)}")

    if brain_mode and args.brain_fusion == "cross_attn":
        model = BrainCrossAttentionLM(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            attn_heads=args.attn_heads,
            dropout=args.dropout,
            brain_dim=brain_dim,
            brain_tokens=args.brain_tokens,
            pad_token_id=args.pad_token_id,
            max_pos=args.max_pos,
        ).to(device)
    else:
        model = LanguageModel(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            attn_heads=args.attn_heads,
            dropout=args.dropout,
            brain_dim=brain_dim,
            pad_token_id=args.pad_token_id if brain_mode else None,
            max_pos=args.max_pos,
        ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    def evaluate(loader: DataLoader) -> float:
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                if brain_mode:
                    xb, zb, yb = batch
                    xb, zb, yb = xb.to(device), zb.to(device), yb.to(device)
                    logits = model(xb, zb)
                    loss = crit(logits[:, -1, :], yb)
                else:
                    xb, yb = batch
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = crit(logits.view(-1, logits.size(-1)), yb.view(-1))
                total_loss += loss.item()
        return total_loss / len(loader)

    for ep in range(1, args.epochs + 1):
        model.train()
        for i, batch in enumerate(train_loader):
            if brain_mode:
                xb, zb, yb = batch
                xb, zb, yb = xb.to(device), zb.to(device), yb.to(device)
                logits = model(xb, zb)
                loss = crit(logits[:, -1, :], yb)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits.view(-1, logits.size(-1)), yb.view(-1))

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 200 == 0:
                print(f"Epoch {ep:02d} | Batch {i:04d}/{len(train_loader)} | Loss: {loss.item():.4f}")

        val_loss = evaluate(val_loader)
        print(f"Epoch {ep:02d} | Validation Loss: {val_loss:.4f}")

    ckpt_path = args.out_dir / "language_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model to {ckpt_path}")

    print("\n--- Generating Text ---")
    model.eval()
    if brain_mode:
        seed_x, seed_z, _ = val_ds[0]
        input_ids = seed_x.unsqueeze(0).to(device)
        brain_vec = seed_z.unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(30):
                logits = model(input_ids, brain_vec)
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
        generated_tokens = input_ids[0].tolist()
        generated_text = tokenizer.decode(generated_tokens)
        print("Brain-conditioned sample (seeded by validation example 0):")
        print(generated_text)
    else:
        prompt = "Hello I am a language model and"
        prompt_tokens = tokenizer.encode(prompt).ids
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(50):
                logits = model(input_ids)
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
        generated_tokens = input_ids[0].tolist()
        generated_text = tokenizer.decode(generated_tokens)
        print(generated_text)


if __name__ == "__main__":
    main()

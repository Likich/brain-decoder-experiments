import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def resolve_device(arg):
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BrainDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_attention: bool = False,
        attn_heads: int = 4,
        attn_layers: int = 1,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.dropout = dropout
        if use_attention:
            if hidden_dim % attn_heads != 0:
                raise ValueError("hidden_dim must be divisible by attn_heads")
            self.input_proj = nn.Linear(1, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            layers = []
            dim = in_dim
            for _ in range(max(0, num_layers)):
                layers.append(nn.Linear(dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(hidden_dim))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                dim = hidden_dim
            layers.append(nn.Linear(dim, num_classes))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        if not self.use_attention:
            return self.net(x)
        h = x.unsqueeze(-1)
        h = self.input_proj(h)
        h = self.attn(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/brain_multiclass.npz"))
    ap.add_argument("--out_dir", type=Path, default=Path("models"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_attention", action="store_true")
    ap.add_argument("--attn_heads", type=int, default=4)
    ap.add_argument("--attn_layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--clip_grad_norm", type=float, default=0.0)
    ap.add_argument("--use_scheduler", action="store_true")
    ap.add_argument("--scheduler_factor", type=float, default=0.5)
    ap.add_argument("--scheduler_patience", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--tokenizer", type=str, default=None)
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    label_names = data["label_names"].tolist()
    num_classes = int(data["num_classes"])

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1e-6
    X = (X - mean) / std

    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y))
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = BrainDecoder(
        in_dim=X.shape[1],
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_attention=args.use_attention,
        attn_heads=args.attn_heads,
        attn_layers=args.attn_layers,
    )
    device = resolve_device(args.device)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            verbose=True,
        )

    def eval_loader(loader):
        model.eval()
        correct = total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                loss_sum += loss.item() * yb.size(0)
                pred = logits.argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        acc = correct / total if total else 0.0
        avg_loss = loss_sum / total if total else 0.0
        return acc, avg_loss

    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            optim.zero_grad()
            loss.backward()
            if args.clip_grad_norm and args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optim.step()

        acc_train, train_loss = eval_loader(train_loader)
        acc_val, val_loss = eval_loader(val_loader)
        if scheduler is not None:
            scheduler.step(val_loss)
        print(
            f"Epoch {ep:02d} | "
            f"train acc={acc_train:.3f} loss={train_loss:.4f} | "
            f"val acc={acc_val:.3f} loss={val_loss:.4f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "brain_decoder.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": X.shape[1],
            "num_classes": num_classes,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "use_attention": args.use_attention,
            "attn_heads": args.attn_heads,
            "attn_layers": args.attn_layers,
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        },
        ckpt_path,
    )
    print("Saved model to", ckpt_path)

    meta = {"class_names": label_names}
    if args.tokenizer:
        meta["tokenizer"] = args.tokenizer
    meta_path = args.out_dir / "brain_decoder_meta.json"
    meta_path.write_text(json.dumps(meta))
    print("Saved metadata to", meta_path)


if __name__ == "__main__":
    main()

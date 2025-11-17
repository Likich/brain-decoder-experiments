import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class BrainDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
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
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/brain_multiclass.npz"))
    ap.add_argument("--out_dir", type=Path, default=Path("models"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer path for metadata")
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Data file not found: {args.data}")

    data = np.load(args.data, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    label_names = data["label_names"].tolist()
    num_classes = int(data["num_classes"])

    print("Loaded data:", X.shape, y.shape, "num_classes:", num_classes)

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1e-6
    X = (X - mean) / std

    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y))
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = BrainDecoder(
        in_dim=X.shape[1],
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    def eval_loader(loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        return correct / total if total else 0.0

    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            optim.zero_grad()
            loss.backward()
            optim.step()

        acc_train = eval_loader(train_loader)
        acc_val = eval_loader(val_loader)
        print(f"Epoch {ep:02d} | train acc={acc_train:.3f} | val acc={acc_val:.3f}")

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
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        },
        ckpt_path,
    )
    print("Saved model to", ckpt_path)

    meta = {
        "class_names": label_names,
    }
    if args.tokenizer:
        meta["tokenizer"] = args.tokenizer
    meta_path = args.out_dir / "brain_decoder_meta.json"
    meta_path.write_text(json.dumps(meta))
    print("Saved metadata to", meta_path)


if __name__ == "__main__":
    main()

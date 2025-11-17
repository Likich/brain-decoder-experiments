# scripts/train_brain_decoder.py
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

DATA_PATH = Path("data/brain_multiclass.npz")
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)


class BrainDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Data file not found: {DATA_PATH}")

    data = np.load(DATA_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    label_names = data["label_names"].tolist()
    num_classes = int(data["num_classes"])

    print("Loaded data:", X.shape, y.shape, "num_classes:", num_classes)
    print("Classes:", label_names)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    dataset = TensorDataset(X_t, y_t)

    # simple 80/20 split
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    model = BrainDecoder(in_dim=X.shape[1], num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
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

    epochs = 20
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()

        acc_train = eval_loader(train_loader)
        acc_val = eval_loader(val_loader)
        print(f"Epoch {ep:02d} | train acc={acc_train:.3f} | val acc={acc_val:.3f}")

    ckpt_path = OUT_DIR / "brain_decoder.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": X.shape[1],
            "num_classes": num_classes,
        },
        ckpt_path,
    )
    print("Saved model to", ckpt_path)

    meta = {
        "class_names": label_names,
    }
    (OUT_DIR / "brain_decoder_meta.json").write_text(json.dumps(meta))
    print("Saved metadata to", OUT_DIR / "brain_decoder_meta.json")


if __name__ == "__main__":
    main()

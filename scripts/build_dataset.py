# scripts/build_dataset.py
import json
import sys
from pathlib import Path

import numpy as np


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/results.json")
    if not path.exists():
        print(f"File not found: {path}")
        return

    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # ignore stray debug / log lines
                continue

    if not rows:
        print("No JSON rows parsed. Check results.json.")
        return

    # --- Discover labels that appear with snapshots ---
    labels = sorted(
        {r["stimulus"] for r in rows if r.get("activity_snapshot") is not None}
    )
    if not labels:
        print("No labels with activity_snapshot found.")
        return

    print("Discovered labels:", labels)
    label_map = {lab: i for i, lab in enumerate(labels)}

    # --- Build X, y ---
    X = []
    y = []

    # Infer expected length from the *first* valid snapshot
    expected_len = None

    for r in rows:
        snap = r.get("activity_snapshot")
        stim = r.get("stimulus")

        if snap is None or stim not in label_map:
            continue

        if expected_len is None:
            expected_len = len(snap)

        # Skip any weird rows with different length (old runs)
        if len(snap) != expected_len:
            continue

        X.append(snap)
        y.append(label_map[stim])

    if not X:
        print("No usable snapshots found (maybe all had inconsistent lengths?).")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print("Dataset shape:", X.shape, y.shape)

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    np.savez(
        out_dir / "brain_multiclass.npz",
        X=X,
        y=y,
        label_names=np.array(labels),
        num_classes=len(labels),
    )
    print("Saved to data/brain_multiclass.npz")


if __name__ == "__main__":
    main()

import argparse
import json
import re
from collections import defaultdict

import numpy as np

CANONICAL_CLASSES = ["APPLE", "BANANA", "GRAPE", "ORANGE", "PEAR"]


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            snippet = m.group(0)
            try:
                rows.append(json.loads(snippet))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="runs/results.json")
    args = ap.parse_args()

    rows = load_rows(args.path)
    if not rows:
        print("No valid JSON rows found.")
        return

    token_mode = any(isinstance(r.get("stimulus_id"), int) for r in rows)

    by_snr = defaultdict(list)
    for r in rows:
        by_snr[r["snr"]].append(r)

    for snr in sorted(by_snr.keys()):
        trials = by_snr[snr]
        n = len(trials)
        ign = sum(bool(t["ignited"]) for t in trials) / max(1, n)
        labeled = [t for t in trials if t.get("choice") is not None]

        correct = 0
        for t in labeled:
            choice = t.get("choice")
            if token_mode and isinstance(t.get("target_token_id"), int):
                target = int(t["target_token_id"])
            elif token_mode and isinstance(t.get("stimulus_id"), int):
                target = int(t["stimulus_id"])
            else:
                stim = t.get("stimulus")
                if stim not in CANONICAL_CLASSES:
                    continue
                target = CANONICAL_CLASSES.index(stim)
            if choice == target:
                correct += 1

        acc = correct / len(labeled) if labeled else 0.0
        print(
            f"{snr:>4} | n={n:4d} | ignite={ign:0.2f} | "
            f"decision-acc={acc:0.2f} | decisions={len(labeled)}"
        )

    sample_by_stim = {}
    n_with_snap = 0
    for r in rows:
        snap = r.get("activity_snapshot")
        if snap is None:
            continue
        n_with_snap += 1
        stim = r.get("stimulus")
        if stim not in sample_by_stim:
            sample_by_stim[stim] = snap

    print(f"\nFound {n_with_snap} trials with non-null activity_snapshot.")

    if sample_by_stim:
        for stim, snap in sample_by_stim.items():
            arr = np.array(snap, dtype=float)
            top5 = np.argsort(-arr)[:5]
            print(f"\nExample top-5 regions for {stim}:")
            for idx in top5:
                print(f"  region {idx:2d}: {arr[idx]: .3f}")
    else:
        print("No activity_snapshot data found in rows.")


if __name__ == "__main__":
    main()

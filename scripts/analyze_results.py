import json
import re
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "runs/results.json"

# -------- load lines as JSON --------
rows = []
with open(path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Extract the first {...} JSON block from the line
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        snippet = m.group(0)
        try:
            rows.append(json.loads(snippet))
        except json.JSONDecodeError:
            # Skip garbage lines silently
            continue

if not rows:
    print("No valid JSON rows found.")
    raise SystemExit(0)

# -------- per-SNR stats --------
by_snr = defaultdict(list)
for r in rows:
    by_snr[r["snr"]].append(r)

for snr in sorted(by_snr.keys()):
    trials = by_snr[snr]
    n = len(trials)
    ign = sum(t["ignited"] for t in trials) / n

    labeled = [t for t in trials if t.get("report")]
    correct = 0
    for t in labeled:
        rep = t["report"]
        stim = t["stimulus"]
        if "APPLE" in rep and stim == "APPLE":
            correct += 1
        elif "PEAR" in rep and stim == "PEAR":
            correct += 1
    acc = correct / len(labeled) if labeled else 0.0

    print(
        f"{snr:>4} | n={n:2d} | ignite={ign:0.2f} | "
        f"report-acc={acc:0.2f} | labeled={len(labeled)}"
    )

# -------- example brain snapshots per stimulus --------
sample_by_stim = {}
n_with_snap = 0
for r in rows:
    # BE VERY EXPLICIT HERE
    if "activity_snapshot" in r and r["activity_snapshot"] is not None:
        n_with_snap += 1
        stim = r["stimulus"]
        if stim not in sample_by_stim:
            sample_by_stim[stim] = r["activity_snapshot"]

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

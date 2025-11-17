import json
from collections import Counter, defaultdict

path = "outputs/experiment.jsonl"  # whatever file you're using

trials = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        trials.append(json.loads(line))

print("N trials:", len(trials))

# Map label → index, if your model uses fixed order
classes = ["APPLE", "BANANA", "GRAPE", "ORANGE", "PEAR"]
label2idx = {c: i for i, c in enumerate(classes)}


# Basic stats
by_snr = defaultdict(list)
for tr in trials:
    by_snr[tr["snr"]].append(tr)

for snr, ts in by_snr.items():
    n = len(ts)
    n_ignited = sum(t["ignited"] for t in ts)
    n_choices = sum(t["choice"] is not None for t in ts)

    # accuracy where a choice was made and stimulus is known
    correct = 0
    total_decisions = 0
    for t in ts:
        if t["choice"] is None:
            continue
        if t["stimulus"] not in label2idx:
            continue
        total_decisions += 1
        if t["choice"] == label2idx[t["stimulus"]]:
            correct += 1

    acc = correct / total_decisions if total_decisions else 0.0
    print(
        f"SNR={snr:4} | trials={n:3} | ignited={n_ignited:3} | "
        f"decisions={total_decisions:3} | acc={acc:0.3f}"
    )

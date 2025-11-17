import argparse
import json
from collections import defaultdict

CANONICAL_CLASSES = ["APPLE", "BANANA", "GRAPE", "ORANGE", "PEAR"]
LABEL2IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}


def load_trials(path: str) -> list[dict]:
    trials = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trials.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return trials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/experiment.jsonl", help="Path to experiment JSONL")
    args = ap.parse_args()

    trials = load_trials(args.input)
    print("N trials:", len(trials))

    # Token mode if we have integer stimulus_id values
    token_mode = any(isinstance(t.get("stimulus_id"), int) for t in trials)

    by_snr = defaultdict(list)
    for tr in trials:
        by_snr[tr["snr"]].append(tr)

    for snr in sorted(by_snr.keys()):
        ts = by_snr[snr]
        n = len(ts)
        n_ignited = sum(bool(t["ignited"]) for t in ts)
        correct = 0
        total_decisions = 0

        for t in ts:
            choice = t.get("choice")
            if choice is None:
                continue

            if token_mode and isinstance(t.get("target_token_id"), int):
                target = int(t["target_token_id"])
            elif token_mode and isinstance(t.get("stimulus_id"), int):
                target = int(t["stimulus_id"])
            else:
                stim = t.get("stimulus")
                if stim not in LABEL2IDX:
                    continue
                target = LABEL2IDX[stim]

            total_decisions += 1
            if choice == target:
                correct += 1

        acc = correct / total_decisions if total_decisions else 0.0
        print(
            f"SNR={snr:4} | trials={n:4} | ignited={n_ignited:4} | "
            f"decisions={total_decisions:4} | acc={acc:0.3f}"
        )

    if token_mode and not any(isinstance(t.get("stimulus_id"), int) for t in trials if t.get("choice") is not None):
        print("Note: token-mode detected but no matching stimulus_id values for decisions.")


if __name__ == "__main__":
    main()

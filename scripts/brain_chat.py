"""
Simple REPL to feed user text through the brain simulator and read the generated response.

Usage:
    python scripts/brain_chat.py --config configs/default.yaml --snr med \
        --max_response 20 --min_conf 0.0
"""

from __future__ import annotations

import argparse

from tokenizers import Tokenizer

from lefty_brain_sim.experiment import Experiment


def run_prompt(exp: Experiment, snr: str, token_ids: list[int], max_response: int) -> list[int]:
    """
    Feed prompt tokens one by one (generation disabled), then enable generation
    on the final token to collect a response sequence.
    """
    last_result = None
    for idx, tok in enumerate(token_ids):
        allow_gen = idx == len(token_ids) - 1
        last_result = exp.run_trial(
            snr=snr,
            stim_idx=tok,
            allow_generation=allow_gen,
            log_debug=False,
        )
    if not token_ids:
        last_result = exp.run_trial(snr=snr, allow_generation=True, log_debug=False)

    response_ids = []
    if last_result and last_result.generated_token_ids:
        response_ids = last_result.generated_token_ids[:max_response]
    return response_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--snr", default="med", choices=["low", "med", "high"])
    ap.add_argument("--max_response", type=int, default=20)
    ap.add_argument("--min_conf", type=float, default=0.0)
    args = ap.parse_args()

    exp = Experiment.from_yaml(args.config)
    if not getattr(exp, "tokenizer", None):
        raise SystemExit("Current config is not in token stimulus mode.")
    tok = Tokenizer.from_file(exp.stimuli_cfg.get("tokenizer"))

    print("Interactive brain chat (type 'exit' to quit)")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break

        encoded = tok.encode(text)
        response_ids = run_prompt(exp, args.snr, encoded.ids, args.max_response)
        if not response_ids:
            print("Brain: (no response)")
            continue
        decoded = tok.decode(response_ids)
        print(f"Brain: {decoded}")


if __name__ == "__main__":
    main()

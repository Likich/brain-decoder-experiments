import argparse
import sys
import json
from pathlib import Path
from lefty_brain_sim.experiment import Experiment

OUT_PATH = Path("outputs/experiment.jsonl")
OUT_PATH.parent.mkdir(exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    exp = Experiment.from_yaml(args.config)

    print(
        f"DEBUG cfg: trials={exp.cfg.trials}, "
        f"snr_levels={exp.cfg.snr_levels}",
        file=sys.stderr,
    )

    if not exp.cfg.snr_levels or exp.cfg.trials <= 0:
        print("ERROR: snr_levels empty or trials <= 0", file=sys.stderr)
        sys.exit(1)

    for snr in exp.cfg.snr_levels:
        for _ in range(exp.cfg.trials):
            tr = exp.run_trial(snr)

            # JSON string for the trial
            js = tr.to_json()

            # print to stdout
            print(js)
            sys.stdout.flush()

            # append to file
            with OUT_PATH.open("a") as f:
                f.write(js + "\n")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METRICS = [
    "delta_real_zero",
    "delta_real_shuf",
    "nll_real",
    "nll_zero",
    "nll_shuf",
    "top1_real_zero",
    "top1_shuf_zero",
    "js_real",
    "js_shuf",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_config(eval_path: Path) -> dict:
    cfg_path = eval_path.parent / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return load_json(cfg_path)
    except Exception:
        return {}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def summarize_metric(rows: list[dict], key: str) -> dict:
    values = [float(row[key]) for row in rows]
    return {
        "mean": mean(values),
        "sd": sample_std(values),
        "min": min(values),
        "max": max(values),
        "n_runs": len(values),
    }


def build_run_record(eval_path: Path) -> dict:
    metrics = load_json(eval_path)
    config = maybe_load_config(eval_path)
    row = {
        "eval_json": str(eval_path),
        "run_dir": str(eval_path.parent),
        "seed": config.get("seed"),
        "brain_norm": config.get("brain_norm"),
    }
    for key in METRICS:
        if key in metrics:
            row[key] = metrics[key]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize multi-seed subject-blocked MEG eval JSONs.")
    ap.add_argument("eval_jsons", nargs="+", type=Path)
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    rows = [build_run_record(path) for path in args.eval_jsons]
    summary = {
        "runs": rows,
        "metrics": {},
    }
    for key in METRICS:
        if all(key in row for row in rows):
            summary["metrics"][key] = summarize_metric(rows, key)

    for row in rows:
        print(
            f"{Path(row['run_dir']).name}: seed={row.get('seed')} "
            f"norm={row.get('brain_norm')} "
            f"dRZ={row.get('delta_real_zero'):.4f} "
            f"dRS={row.get('delta_real_shuf'):.4f}"
        )

    for key, stats in summary["metrics"].items():
        print(
            f"{key}: mean={stats['mean']:.4f} sd={stats['sd']:.4f} "
            f"range=[{stats['min']:.4f}, {stats['max']:.4f}] n={stats['n_runs']}"
        )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

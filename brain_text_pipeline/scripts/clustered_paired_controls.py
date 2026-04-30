#!/usr/bin/env python3
"""Clustered bootstrap summaries for paired-control MEG evaluations."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_text_pipeline.src.utils.logging import log, save_json


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def parse_cluster_spec(spec: str) -> list[str]:
    keys = [part.strip() for part in spec.split(",") if part.strip()]
    if not keys:
        raise ValueError(f"invalid empty cluster spec: {spec!r}")
    return keys


def cluster_id(row: dict, keys: list[str]) -> tuple[str, ...]:
    values = []
    for key in keys:
        value = row.get(key)
        if value is None or value == "":
            value = "__missing__"
        values.append(str(value))
    return tuple(values)


def cluster_counts(rows: list[dict], keys: list[str]) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        counts[cluster_id(row, keys)] += 1
    return counts


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


def paired_cluster_bootstrap(
    rows: list[dict],
    *,
    keys: list[str],
    bootstrap_samples: int,
    seed: int,
) -> dict:
    groups: dict[tuple[str, ...], list[dict]] = {}
    for row in rows:
        groups.setdefault(cluster_id(row, keys), []).append(row)

    if len(groups) < 2:
        return {
            "cluster_keys": keys,
            "n_clusters": int(len(groups)),
            "n_examples": int(sum(len(group_rows) for group_rows in groups.values())),
            "skipped": True,
            "reason": f"cluster spec {','.join(keys)!r} yields only {len(groups)} cluster(s); need at least 2",
        }

    cluster_sizes = np.asarray([len(group_rows) for group_rows in groups.values()], dtype=np.int64)
    cluster_sum_rz = np.asarray(
        [sum(float(r["delta_real_zero"]) for r in group_rows) for group_rows in groups.values()],
        dtype=np.float64,
    )
    cluster_sum_rs = np.asarray(
        [sum(float(r["delta_real_shuf"]) for r in group_rows) for group_rows in groups.values()],
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(groups), size=(bootstrap_samples, len(groups)))
    sampled_sizes = cluster_sizes[sample_idx].sum(axis=1).astype(np.float64)
    sampled_rz = cluster_sum_rz[sample_idx].sum(axis=1) / sampled_sizes
    sampled_rs = cluster_sum_rs[sample_idx].sum(axis=1) / sampled_sizes

    observed_rz = float(cluster_sum_rz.sum() / cluster_sizes.sum())
    observed_rs = float(cluster_sum_rs.sum() / cluster_sizes.sum())
    ci_rz = percentile_ci(sampled_rz)
    ci_rs = percentile_ci(sampled_rs)

    return {
        "cluster_keys": keys,
        "n_clusters": int(len(groups)),
        "n_examples": int(cluster_sizes.sum()),
        "skipped": False,
        "delta_real_zero": {
            "mean": observed_rz,
            "bootstrap_mean": float(sampled_rz.mean()),
            "bootstrap_se": float(sampled_rz.std(ddof=1)),
            "ci95_low": ci_rz[0],
            "ci95_high": ci_rz[1],
            "bootstrap_p_nonnegative": float((sampled_rz >= 0).mean()),
        },
        "delta_real_shuf": {
            "mean": observed_rs,
            "bootstrap_mean": float(sampled_rs.mean()),
            "bootstrap_se": float(sampled_rs.std(ddof=1)),
            "ci95_low": ci_rs[0],
            "ci95_high": ci_rs[1],
            "bootstrap_p_nonnegative": float((sampled_rs >= 0).mean()),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--examples_jsonl", type=Path, required=True)
    ap.add_argument(
        "--cluster_spec",
        action="append",
        required=True,
        help=(
            "Cluster key(s) to resample, e.g. 'subject', 'story', or 'subject,story'. "
            "Pass multiple times to compute multiple clustered summaries."
        ),
    )
    ap.add_argument("--bootstrap_samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument(
        "--print_cluster_counts",
        action="store_true",
        help="Log the number of unique clusters and the largest few cluster sizes for each cluster spec.",
    )
    args = ap.parse_args()

    rows = load_jsonl(args.examples_jsonl)
    result = {
        "examples_jsonl": str(args.examples_jsonl),
        "n_examples": int(len(rows)),
        "bootstrap_samples": int(args.bootstrap_samples),
        "seed": int(args.seed),
        "clustered_bootstrap": {},
    }
    for spec in args.cluster_spec:
        keys = parse_cluster_spec(spec)
        if args.print_cluster_counts:
            counts = cluster_counts(rows, keys)
            largest = counts.most_common(5)
            log(f"{','.join(keys)}: unique_clusters={len(counts)} top_counts={largest}")
        summary = paired_cluster_bootstrap(
            rows,
            keys=keys,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        result["clustered_bootstrap"][",".join(keys)] = summary
        if summary.get("skipped"):
            log(f"{','.join(keys)}: skipped ({summary['reason']})")
            continue
        log(
            f"{','.join(keys)}: "
            f"dRZ={summary['delta_real_zero']['mean']:.4f} "
            f"[{summary['delta_real_zero']['ci95_low']:.4f}, {summary['delta_real_zero']['ci95_high']:.4f}] ; "
            f"dRS={summary['delta_real_shuf']['mean']:.4f} "
            f"[{summary['delta_real_shuf']['ci95_low']:.4f}, {summary['delta_real_shuf']['ci95_high']:.4f}]"
        )

    save_json(args.out_json, result)
    log(f"wrote clustered bootstrap summary to {args.out_json}")


if __name__ == "__main__":
    main()

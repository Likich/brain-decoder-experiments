#!/usr/bin/env python3
"""Characterize what the MEG effect captures beyond aggregate delta NLL."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_text_pipeline.src.data.datasets import ShardedExampleDataset
from brain_text_pipeline.src.utils.logging import log, save_json

STOPWORD_TARGETS = {
    "a", "an", "and", "as", "at", "be", "but", "by", "for", "from", "had",
    "he", "her", "his", "i", "if", "in", "is", "it", "its", "me", "my",
    "of", "on", "or", "our", "she", "that", "the", "their", "there", "they",
    "this", "to", "we", "with", "you", "your",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def normalize_target(text: str | None) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def alpha_chars(text: str | None) -> int:
    return sum(ch.isalpha() for ch in str(text or ""))


def word_type(row: dict[str, Any]) -> str:
    text = normalize_target(row.get("target_text"))
    if alpha_chars(text) == 0:
        return "punct_or_symbol"
    if text in STOPWORD_TARGETS:
        return "function"
    return "content"


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def quantile_edges(values: list[float], q: int) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    return [float(np.quantile(arr, i / q)) for i in range(q + 1)]


def quantile_bin(value: float | None, edges: list[float], prefix: str) -> str:
    if value is None:
        return f"{prefix}_missing"
    n_bins = len(edges) - 1
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        is_last = i == n_bins - 1
        if (value >= lo and value < hi) or (is_last and value <= hi):
            return f"{prefix}_{i+1}"
    return f"{prefix}_{n_bins}"


def summarize_quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
    }


def mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(mean(values))


def topk_hit(rank: int | None, k: int) -> float | None:
    if rank is None:
        return None
    return 1.0 if rank <= k else 0.0


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    d_rz = [float(r["delta_real_zero"]) for r in rows]
    d_rs = [float(r["delta_real_shuf"]) for r in rows]

    prob_rz = []
    prob_rs = []
    rank_imp_rz = []
    rank_imp_rs = []
    mrr_real = []
    mrr_zero = []
    mrr_shuf = []
    top1_real = []
    top1_zero = []
    top1_shuf = []
    top5_real = []
    top5_zero = []
    top5_shuf = []
    top10_real = []
    top10_zero = []
    top10_shuf = []
    unchanged_but_prob_up = 0
    unchanged_but_rank_up = 0

    for row in rows:
        p_r = safe_float(row.get("first_token_prob_real"))
        p_z = safe_float(row.get("first_token_prob_zero"))
        p_s = safe_float(row.get("first_token_prob_shuf"))
        r_r = safe_int(row.get("first_token_rank_real"))
        r_z = safe_int(row.get("first_token_rank_zero"))
        r_s = safe_int(row.get("first_token_rank_shuf"))

        if p_r is not None and p_z is not None:
            prob_rz.append(p_r - p_z)
        if p_r is not None and p_s is not None:
            prob_rs.append(p_r - p_s)
        if r_r is not None and r_z is not None:
            rank_imp_rz.append(float(r_z - r_r))
        if r_r is not None and r_s is not None:
            rank_imp_rs.append(float(r_s - r_r))
        if r_r is not None:
            mrr_real.append(1.0 / r_r)
            top1_real.append(topk_hit(r_r, 1))
            top5_real.append(topk_hit(r_r, 5))
            top10_real.append(topk_hit(r_r, 10))
        if r_z is not None:
            mrr_zero.append(1.0 / r_z)
            top1_zero.append(topk_hit(r_z, 1))
            top5_zero.append(topk_hit(r_z, 5))
            top10_zero.append(topk_hit(r_z, 10))
        if r_s is not None:
            mrr_shuf.append(1.0 / r_s)
            top1_shuf.append(topk_hit(r_s, 1))
            top5_shuf.append(topk_hit(r_s, 5))
            top10_shuf.append(topk_hit(r_s, 10))

        same_top1 = float(row.get("top1_real_zero", 0.0)) >= 0.5
        if same_top1 and p_r is not None and p_z is not None and p_r > p_z:
            unchanged_but_prob_up += 1
        if same_top1 and r_r is not None and r_z is not None and r_r < r_z:
            unchanged_but_rank_up += 1

    n = len(rows)
    return {
        "n": n,
        "delta_real_zero_mean": float(mean(d_rz)),
        "delta_real_shuf_mean": float(mean(d_rs)),
        "delta_real_zero_quantiles": summarize_quantiles(d_rz),
        "delta_real_shuf_quantiles": summarize_quantiles(d_rs),
        "first_token_prob_gain_real_zero_mean": mean_or_none(prob_rz),
        "first_token_prob_gain_real_shuf_mean": mean_or_none(prob_rs),
        "first_token_prob_gain_real_zero_quantiles": None if not prob_rz else summarize_quantiles(prob_rz),
        "first_token_prob_gain_real_shuf_quantiles": None if not prob_rs else summarize_quantiles(prob_rs),
        "first_token_rank_improvement_zero_to_real_mean": mean_or_none(rank_imp_rz),
        "first_token_rank_improvement_shuf_to_real_mean": mean_or_none(rank_imp_rs),
        "fraction_prob_up_real_vs_zero": None if not prob_rz else float(np.mean(np.asarray(prob_rz) > 0)),
        "fraction_prob_up_real_vs_shuf": None if not prob_rs else float(np.mean(np.asarray(prob_rs) > 0)),
        "fraction_rank_improved_real_vs_zero": None if not rank_imp_rz else float(np.mean(np.asarray(rank_imp_rz) > 0)),
        "fraction_rank_improved_real_vs_shuf": None if not rank_imp_rs else float(np.mean(np.asarray(rank_imp_rs) > 0)),
        "mrr_real": mean_or_none(mrr_real),
        "mrr_zero": mean_or_none(mrr_zero),
        "mrr_shuf": mean_or_none(mrr_shuf),
        "delta_mrr_real_zero": None if not mrr_real or not mrr_zero else float(mean(mrr_real) - mean(mrr_zero)),
        "delta_mrr_real_shuf": None if not mrr_real or not mrr_shuf else float(mean(mrr_real) - mean(mrr_shuf)),
        "top1_real": mean_or_none([x for x in top1_real if x is not None]),
        "top1_zero": mean_or_none([x for x in top1_zero if x is not None]),
        "top1_shuf": mean_or_none([x for x in top1_shuf if x is not None]),
        "top5_real": mean_or_none([x for x in top5_real if x is not None]),
        "top5_zero": mean_or_none([x for x in top5_zero if x is not None]),
        "top5_shuf": mean_or_none([x for x in top5_shuf if x is not None]),
        "top10_real": mean_or_none([x for x in top10_real if x is not None]),
        "top10_zero": mean_or_none([x for x in top10_zero if x is not None]),
        "top10_shuf": mean_or_none([x for x in top10_shuf if x is not None]),
        "delta_top1_real_zero": None if not top1_real or not top1_zero else float(mean(top1_real) - mean(top1_zero)),
        "delta_top1_real_shuf": None if not top1_real or not top1_shuf else float(mean(top1_real) - mean(top1_shuf)),
        "delta_top5_real_zero": None if not top5_real or not top5_zero else float(mean(top5_real) - mean(top5_zero)),
        "delta_top5_real_shuf": None if not top5_real or not top5_shuf else float(mean(top5_real) - mean(top5_shuf)),
        "delta_top10_real_zero": None if not top10_real or not top10_zero else float(mean(top10_real) - mean(top10_zero)),
        "delta_top10_real_shuf": None if not top10_real or not top10_shuf else float(mean(top10_real) - mean(top10_shuf)),
        "fraction_argmax_unchanged_but_prob_up_vs_zero": float(unchanged_but_prob_up / n),
        "fraction_argmax_unchanged_but_rank_up_vs_zero": float(unchanged_but_rank_up / n),
    }


def count_target_frequency(manifest_path: Path) -> Counter[str]:
    ds = ShardedExampleDataset(manifest_path)
    counts: Counter[str] = Counter()
    for i in range(len(ds)):
        meta = ds[i].get("meta") or {}
        tgt = normalize_target(meta.get("target_text"))
        if tgt:
            counts[tgt] += 1
        if (i + 1) % 10000 == 0:
            log(f"counted {i+1}/{len(ds)} training examples")
    return counts


def assign_story_position_bins(rows: list[dict[str, Any]]) -> None:
    seq_max: dict[str, int] = {}
    for row in rows:
        seq = row.get("sequence_id")
        idx = safe_int(row.get("word_index"))
        if seq is None or idx is None:
            continue
        seq_max[str(seq)] = max(seq_max.get(str(seq), -1), idx)
    for row in rows:
        seq = row.get("sequence_id")
        idx = safe_int(row.get("word_index"))
        if seq is None or idx is None or str(seq) not in seq_max or seq_max[str(seq)] <= 0:
            row["story_position_bin"] = "position_missing"
            continue
        rel = float(idx) / float(seq_max[str(seq)])
        row["story_position_rel"] = rel
        if rel < 1 / 3:
            row["story_position_bin"] = "early"
        elif rel < 2 / 3:
            row["story_position_bin"] = "middle"
        else:
            row["story_position_bin"] = "late"


def build_stratified_summaries(rows: list[dict[str, Any]], key: str, ordered_labels: list[str] | None = None) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "missing"))].append(row)
    labels = ordered_labels or sorted(groups.keys())
    out: dict[str, Any] = {}
    for label in labels:
        if label not in groups:
            continue
        out[label] = group_summary(groups[label])
    for label in groups.keys():
        if label not in out:
            out[label] = group_summary(groups[label])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--examples_jsonl", type=Path, required=True)
    ap.add_argument("--train_manifest", type=Path, default=None, help="Optional train manifest for target frequency bins.")
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--quantile_bins", type=int, default=3)
    args = ap.parse_args()

    rows = load_jsonl(args.examples_jsonl)

    # Basic per-row derived fields.
    for row in rows:
        tgt = str(row.get("target_text") or "")
        row["target_norm"] = normalize_target(tgt)
        row["word_type"] = word_type(row)
        row["alpha_chars"] = alpha_chars(tgt)
        row["char_len"] = len(tgt.strip())
        row["word_length_bin"] = (
            "len_0_3" if row["char_len"] <= 3 else
            "len_4_6" if row["char_len"] <= 6 else
            "len_7plus"
        )

    assign_story_position_bins(rows)

    surprisal_values = [float(r["nll_zero"]) for r in rows]
    surprisal_edges = quantile_edges(surprisal_values, args.quantile_bins)
    for row in rows:
        row["surprisal_bin"] = quantile_bin(float(row["nll_zero"]), surprisal_edges, "surprisal")

    if args.train_manifest is not None:
        freq_counts = count_target_frequency(args.train_manifest)
        freq_values = [float(freq_counts.get(r["target_norm"], 0)) for r in rows]
        freq_edges = quantile_edges(freq_values, args.quantile_bins)
        for row, freq in zip(rows, freq_values):
            row["train_target_freq"] = int(freq)
            row["frequency_bin"] = quantile_bin(float(freq), freq_edges, "freq")
    else:
        freq_counts = None

    result = {
        "examples_jsonl": str(args.examples_jsonl),
        "train_manifest": None if args.train_manifest is None else str(args.train_manifest),
        "n_examples": len(rows),
        "global": group_summary(rows),
        "stratified": {
            "word_type": build_stratified_summaries(rows, "word_type", ["content", "function", "punct_or_symbol"]),
            "word_length_bin": build_stratified_summaries(rows, "word_length_bin", ["len_0_3", "len_4_6", "len_7plus"]),
            "surprisal_bin": build_stratified_summaries(
                rows, "surprisal_bin", [f"surprisal_{i+1}" for i in range(args.quantile_bins)]
            ),
            "story_position_bin": build_stratified_summaries(rows, "story_position_bin", ["early", "middle", "late"]),
        },
    }
    if args.train_manifest is not None and freq_counts is not None:
        result["stratified"]["frequency_bin"] = build_stratified_summaries(
            rows, "frequency_bin", [f"freq_{i+1}" for i in range(args.quantile_bins)]
        )

    save_json(args.out_json, result)

    g = result["global"]
    log(
        "global: "
        f"dRZ={g['delta_real_zero_mean']:.4f} dRS={g['delta_real_shuf_mean']:.4f} ; "
        f"prob_up(R>Z)={g['fraction_prob_up_real_vs_zero']:.3f} "
        f"rank_up(R>Z)={g['fraction_rank_improved_real_vs_zero']:.3f} "
        f"delta_mrr(R-Z)={g['delta_mrr_real_zero']:.4f}"
    )
    log(
        "argmax-stable but still improving: "
        f"prob_up={g['fraction_argmax_unchanged_but_prob_up_vs_zero']:.3f} "
        f"rank_up={g['fraction_argmax_unchanged_but_rank_up_vs_zero']:.3f}"
    )
    for label, summary in result["stratified"]["word_type"].items():
        log(f"word_type={label}: n={summary['n']} dRZ={summary['delta_real_zero_mean']:.4f} dRS={summary['delta_real_shuf_mean']:.4f}")
    for label, summary in result["stratified"]["surprisal_bin"].items():
        log(f"surprisal={label}: n={summary['n']} dRZ={summary['delta_real_zero_mean']:.4f} dRS={summary['delta_real_shuf_mean']:.4f}")


if __name__ == "__main__":
    main()

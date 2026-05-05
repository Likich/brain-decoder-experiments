#!/usr/bin/env python3
"""Post-hoc covariate-matched checks for paired-control MEG results.

This script does not retrain the model. Instead, it asks whether the
paired-control gain survives after matching or stratifying examples by
available nuisance covariates such as subject, sound, story position,
word type, target length, training frequency, and (optionally) event-level
duration / phoneme metadata recovered from MEG-MASC event files.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "brain_text_pipeline").is_dir():
            return candidate
    raise RuntimeError(f"could not locate repo root from {start}")


ROOT = find_repo_root(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_text_pipeline.src.data.datasets import ShardedExampleDataset
from brain_text_pipeline.src.data.meg_masc import events_path
from brain_text_pipeline.src.utils.logging import log, save_json

STOPWORD_TARGETS = {
    "a", "an", "and", "as", "at", "be", "but", "by", "for", "from", "had",
    "he", "her", "his", "i", "if", "in", "is", "it", "its", "me", "my",
    "of", "on", "or", "our", "she", "that", "the", "their", "there", "they",
    "this", "to", "we", "with", "you", "your",
}


def parse_trial_type(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    text = str(value)
    if text == "" or text == "nan":
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def word_type(text: str) -> str:
    norm = normalize_target(text)
    if alpha_chars(norm) == 0:
        return "punct_or_symbol"
    if norm in STOPWORD_TARGETS:
        return "function"
    return "content"


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
        if rel < 1 / 3:
            row["story_position_bin"] = "early"
        elif rel < 2 / 3:
            row["story_position_bin"] = "middle"
        else:
            row["story_position_bin"] = "late"


def infer_duration(event_meta: dict[str, Any], tsv_row: dict[str, str]) -> float | None:
    for key in ("duration", "word_duration", "dur", "duration_sec"):
        val = safe_float(event_meta.get(key))
        if val is not None:
            return val
    return safe_float(tsv_row.get("duration"))


def infer_phoneme_count(event_meta: dict[str, Any]) -> int | None:
    for key in ("phonemes", "phoneme_sequence", "phones"):
        val = event_meta.get(key)
        if isinstance(val, (list, tuple)):
            return len(val)
        if isinstance(val, str) and val.strip():
            parts = [p for p in val.replace("|", " ").replace(",", " ").split() if p]
            if parts:
                return len(parts)
    for key in ("phoneme_count", "n_phonemes", "num_phonemes"):
        val = safe_int(event_meta.get(key))
        if val is not None:
            return val
    return None


def event_lookup_candidates(
    *,
    onset_sec: float | None,
    target_norm: str,
    sound: str | None,
    story: str | None,
    sequence_id: str | None,
    word_index: int | None,
) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    if sequence_id is not None and word_index is not None:
        out.append(("seq_idx_word", str(sequence_id), str(word_index), target_norm))
    if sound is not None and word_index is not None:
        out.append(("sound_idx_word", str(sound), str(word_index), target_norm))
    if story is not None and word_index is not None:
        out.append(("story_idx_word", str(story), str(word_index), target_norm))
    if onset_sec is not None:
        out.append(("onset_word", f"{onset_sec:.6f}", target_norm))
        out.append(("onset_word", f"{onset_sec:.4f}", target_norm))
    return out


def build_event_index(
    meg_root: Path,
    subject: str,
    session: str,
    task: str,
) -> dict[tuple[str, ...], dict[str, Any]]:
    path = events_path(meg_root, subject, session, task)
    if not path.exists():
        raise FileNotFoundError(f"missing events file: {path}")

    index: dict[tuple[str, ...], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            event_meta = parse_trial_type(row.get("trial_type"))
            if event_meta.get("kind") not in {None, "word"}:
                continue
            word = str(event_meta.get("word") or row.get("word") or "").strip()
            if not word or word == "nan":
                continue
            target_norm = normalize_target(word)
            onset = safe_float(row.get("onset") or row.get("onset_sec"))
            sound = event_meta.get("sound")
            story = event_meta.get("story")
            sequence_id = event_meta.get("sequence_id")
            word_index = safe_int(event_meta.get("word_index"))
            payload = {
                "event_meta": event_meta,
                "tsv_row": row,
            }
            for key in event_lookup_candidates(
                onset_sec=onset,
                target_norm=target_norm,
                sound=None if sound is None else str(sound),
                story=None if story is None else str(story),
                sequence_id=None if sequence_id is None else str(sequence_id),
                word_index=word_index,
            ):
                index.setdefault(key, payload)
    return index


def enrich_with_event_covariates(rows: list[dict[str, Any]], meg_root: Path) -> dict[str, Any]:
    cache: dict[tuple[str, str, str], dict[tuple[str, ...], dict[str, Any]]] = {}
    available_event_keys: Counter[str] = Counter()
    matched = 0
    duration_count = 0
    phoneme_count = 0

    for row in rows:
        subject = row.get("subject")
        session = row.get("session")
        task = row.get("task")
        if not subject or not session or not task:
            continue
        triplet = (str(subject), str(session), str(task))
        if triplet not in cache:
            try:
                cache[triplet] = build_event_index(meg_root, *triplet)
            except FileNotFoundError:
                cache[triplet] = {}
        idx = cache[triplet]
        target_norm = row.get("target_norm", "")
        onset_sec = safe_float(row.get("onset_sec"))
        sound = None if row.get("sound") is None else str(row.get("sound"))
        story = None if row.get("story") is None else str(row.get("story"))
        sequence_id = None if row.get("sequence_id") is None else str(row.get("sequence_id"))
        word_index = safe_int(row.get("word_index"))
        payload = None
        for key in event_lookup_candidates(
            onset_sec=onset_sec,
            target_norm=target_norm,
            sound=sound,
            story=story,
            sequence_id=sequence_id,
            word_index=word_index,
        ):
            payload = idx.get(key)
            if payload is not None:
                break
        if payload is None:
            continue

        matched += 1
        event_meta = payload["event_meta"]
        tsv_row = payload["tsv_row"]
        for key in event_meta.keys():
            available_event_keys[key] += 1

        duration = infer_duration(event_meta, tsv_row)
        phonemes = infer_phoneme_count(event_meta)
        if duration is not None:
            row["duration_sec"] = float(duration)
            duration_count += 1
        if phonemes is not None:
            row["phoneme_count"] = int(phonemes)
            phoneme_count += 1

    return {
        "matched_rows": matched,
        "available_event_keys_top20": available_event_keys.most_common(20),
        "duration_rows": duration_count,
        "phoneme_rows": phoneme_count,
        "n_triplets_loaded": len(cache),
    }


def derive_covariates(
    rows: list[dict[str, Any]],
    *,
    quantile_bins: int,
    train_manifest: Path | None,
) -> dict[str, Any]:
    for row in rows:
        tgt = str(row.get("target_text") or "")
        row["target_norm"] = normalize_target(tgt)
        row["word_type"] = word_type(tgt)
        row["char_len"] = len(tgt.strip())
        row["word_length_bin"] = (
            "len_0_3" if row["char_len"] <= 3 else
            "len_4_6" if row["char_len"] <= 6 else
            "len_7plus"
        )

    assign_story_position_bins(rows)

    surprisal_values = [float(r["nll_zero"]) for r in rows]
    surprisal_edges = quantile_edges(surprisal_values, quantile_bins)
    for row in rows:
        row["surprisal_bin"] = quantile_bin(float(row["nll_zero"]), surprisal_edges, "surprisal")

    freq_counts = None
    if train_manifest is not None:
        freq_counts = count_target_frequency(train_manifest)
        freq_values = [float(freq_counts.get(r["target_norm"], 0)) for r in rows]
        freq_edges = quantile_edges(freq_values, quantile_bins)
        for row, freq in zip(rows, freq_values):
            row["train_target_freq"] = int(freq)
            row["frequency_bin"] = quantile_bin(float(freq), freq_edges, "freq")

    duration_values = [float(r["duration_sec"]) for r in rows if safe_float(r.get("duration_sec")) is not None]
    if duration_values:
        duration_edges = quantile_edges(duration_values, quantile_bins)
        for row in rows:
            row["duration_bin"] = quantile_bin(safe_float(row.get("duration_sec")), duration_edges, "dur")

    phoneme_values = [float(r["phoneme_count"]) for r in rows if safe_int(r.get("phoneme_count")) is not None]
    if phoneme_values:
        phoneme_edges = quantile_edges(phoneme_values, quantile_bins)
        for row in rows:
            row["phoneme_count_bin"] = quantile_bin(float(row.get("phoneme_count")), phoneme_edges, "phon")

    return {
        "has_frequency_bins": freq_counts is not None,
        "has_duration_bins": bool(duration_values),
        "has_phoneme_bins": bool(phoneme_values),
    }


def build_auto_cell_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys = ["subject", "sound", "word_type", "word_length_bin", "story_position_bin"]
    if any("frequency_bin" in r for r in rows):
        keys.append("frequency_bin")
    if any("duration_bin" in r for r in rows):
        keys.append("duration_bin")
    if any("phoneme_count_bin" in r for r in rows):
        keys.append("phoneme_count_bin")
    return keys


def cell_value(row: dict[str, Any], key: str) -> str:
    val = row.get(key)
    if val is None or val == "":
        return "__missing__"
    return str(val)


def summarize_matched_cells(
    rows: list[dict[str, Any]],
    *,
    cell_keys: list[str],
    min_cell_size: int,
) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(cell_value(row, k) for k in cell_keys)].append(row)

    kept_cells: list[list[dict[str, Any]]] = [g for g in groups.values() if len(g) >= min_cell_size]
    if not kept_cells:
        return {
            "cell_keys": cell_keys,
            "min_cell_size": min_cell_size,
            "n_cells_total": len(groups),
            "n_cells_kept": 0,
            "n_rows_total": len(rows),
            "n_rows_covered": 0,
            "coverage": 0.0,
        }

    cell_mean_rz = np.asarray([mean(float(r["delta_real_zero"]) for r in cell) for cell in kept_cells], dtype=np.float64)
    cell_mean_rs = np.asarray([mean(float(r["delta_real_shuf"]) for r in cell) for cell in kept_cells], dtype=np.float64)
    kept_rows = [r for cell in kept_cells for r in cell]
    row_mean_rz = np.asarray([float(r["delta_real_zero"]) for r in kept_rows], dtype=np.float64)
    row_mean_rs = np.asarray([float(r["delta_real_shuf"]) for r in kept_rows], dtype=np.float64)
    cell_sizes = np.asarray([len(cell) for cell in kept_cells], dtype=np.int64)

    def mean_ci(arr: np.ndarray) -> dict[str, float]:
        m = float(arr.mean())
        se = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
        return {
            "mean": m,
            "se": se,
            "ci95_low": m - 1.96 * se,
            "ci95_high": m + 1.96 * se,
        }

    return {
        "cell_keys": cell_keys,
        "min_cell_size": min_cell_size,
        "n_cells_total": len(groups),
        "n_cells_kept": len(kept_cells),
        "n_rows_total": len(rows),
        "n_rows_covered": len(kept_rows),
        "coverage": float(len(kept_rows) / max(1, len(rows))),
        "median_cell_size": float(np.median(cell_sizes)),
        "row_weighted_delta_real_zero": mean_ci(row_mean_rz),
        "row_weighted_delta_real_shuf": mean_ci(row_mean_rs),
        "cell_balanced_delta_real_zero": mean_ci(cell_mean_rz),
        "cell_balanced_delta_real_shuf": mean_ci(cell_mean_rs),
        "fraction_cells_neg_dRZ": float((cell_mean_rz < 0).mean()),
        "fraction_cells_neg_dRS": float((cell_mean_rs < 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--examples_jsonl", type=Path, required=True)
    ap.add_argument("--train_manifest", type=Path, default=None)
    ap.add_argument("--meg_root", type=Path, default=None, help="Optional MEG-MASC root to recover duration/phoneme covariates from events.tsv.")
    ap.add_argument("--cell_keys", type=str, default=None, help="Comma-separated matching keys. Default is an auto-selected nuisance set.")
    ap.add_argument("--min_cell_size", type=int, default=10)
    ap.add_argument("--quantile_bins", type=int, default=3)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    rows = load_jsonl(args.examples_jsonl)
    event_info = None
    if args.meg_root is not None:
        event_info = enrich_with_event_covariates(rows, args.meg_root)

    derive_info = derive_covariates(rows, quantile_bins=args.quantile_bins, train_manifest=args.train_manifest)

    if args.cell_keys:
        cell_keys = [k.strip() for k in args.cell_keys.split(",") if k.strip()]
    else:
        cell_keys = build_auto_cell_keys(rows)

    matched_all = summarize_matched_cells(rows, cell_keys=cell_keys, min_cell_size=args.min_cell_size)
    matched_content = summarize_matched_cells(
        [r for r in rows if r.get("word_type") == "content"],
        cell_keys=[k for k in cell_keys if k != "word_type"],
        min_cell_size=args.min_cell_size,
    )

    result = {
        "examples_jsonl": str(args.examples_jsonl),
        "train_manifest": None if args.train_manifest is None else str(args.train_manifest),
        "meg_root": None if args.meg_root is None else str(args.meg_root),
        "n_examples": len(rows),
        "cell_keys": cell_keys,
        "derive_info": derive_info,
        "event_info": event_info,
        "matched_all": matched_all,
        "matched_content_only": matched_content,
    }
    save_json(args.out_json, result)

    for label, summary in (("all", matched_all), ("content_only", matched_content)):
        if summary.get("n_cells_kept", 0) == 0:
            log(f"{label}: no matched cells kept for keys={summary['cell_keys']}")
            continue
        log(
            f"{label}: keys={','.join(summary['cell_keys'])} "
            f"cells={summary['n_cells_kept']}/{summary['n_cells_total']} "
            f"coverage={summary['coverage']:.3f} "
            f"cell-balanced dRZ={summary['cell_balanced_delta_real_zero']['mean']:.4f} "
            f"[{summary['cell_balanced_delta_real_zero']['ci95_low']:.4f}, {summary['cell_balanced_delta_real_zero']['ci95_high']:.4f}] "
            f"dRS={summary['cell_balanced_delta_real_shuf']['mean']:.4f} "
            f"[{summary['cell_balanced_delta_real_shuf']['ci95_low']:.4f}, {summary['cell_balanced_delta_real_shuf']['ci95_high']:.4f}]"
        )
    if event_info is not None:
        log(
            f"event enrichment: matched_rows={event_info['matched_rows']} "
            f"duration_rows={event_info['duration_rows']} phoneme_rows={event_info['phoneme_rows']}"
        )


if __name__ == "__main__":
    main()

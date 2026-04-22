#!/usr/bin/env python3
"""Summarize and plot exported decoder cross-attention over brain time."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_shard_path(manifest_path: Path, shard_path: str) -> Path:
    raw = Path(shard_path)
    candidates = [
        raw,
        manifest_path.parent / raw,
        manifest_path.parent / raw.name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not resolve shard path {shard_path!r} from {manifest_path}")


def load_attention_matrix(manifest_path: Path) -> np.ndarray:
    manifest = load_json(manifest_path)
    rows: list[np.ndarray] = []
    max_len = 0
    for shard in manifest["shards"]:
        shard_file = resolve_shard_path(manifest_path, shard["path"])
        arrs = np.load(shard_file, allow_pickle=True)
        attn_last = arrs["attn_last"]
        masks = arrs["brain_mask"]
        for vec, mask in zip(attn_last, masks):
            v = np.asarray(vec, dtype=np.float64)
            m = np.asarray(mask, dtype=np.float64)
            length = min(len(v), len(m))
            v = v[:length] * m[:length]
            denom = float(v.sum())
            if denom <= 0:
                continue
            v = v / denom
            rows.append(v)
            max_len = max(max_len, len(v))
    if not rows:
        raise ValueError(f"No attention rows found in {manifest_path}")
    mat = np.zeros((len(rows), max_len), dtype=np.float64)
    for i, row in enumerate(rows):
        mat[i, : len(row)] = row
    return mat


def summarize_attention(mat: np.ndarray, tmin_sec: float, tmax_sec: float) -> dict[str, Any]:
    n, seq_len = mat.shape
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    sem = std / np.sqrt(max(n, 1))
    peak_bins = mat.argmax(axis=1)
    centers = np.arange(seq_len, dtype=np.float64)
    center_of_mass = (mat * centers[None, :]).sum(axis=1)
    eps = 1e-12
    entropy = -(mat * np.log(mat + eps)).sum(axis=1)
    norm_entropy = entropy / np.log(seq_len)

    edges = np.linspace(0, seq_len, 5, dtype=int)
    windows = []
    for left, right in zip(edges[:-1], edges[1:]):
        mass = float(mean[left:right].sum())
        start_sec = tmin_sec + (tmax_sec - tmin_sec) * (left / seq_len)
        end_sec = tmin_sec + (tmax_sec - tmin_sec) * (right / seq_len)
        windows.append(
            {
                "bin_start": int(left),
                "bin_end": int(right - 1),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "mass": mass,
            }
        )

    top_idx = np.argsort(-mean)[:10]
    dt = (tmax_sec - tmin_sec) / seq_len
    return {
        "n": int(n),
        "seq_len": int(seq_len),
        "tmin_sec": float(tmin_sec),
        "tmax_sec": float(tmax_sec),
        "mean_curve": mean.tolist(),
        "sem_curve": sem.tolist(),
        "peak_bin_mean": float(peak_bins.mean()),
        "peak_bin_median": float(np.median(peak_bins)),
        "peak_bin_q10": float(np.quantile(peak_bins, 0.1)),
        "peak_bin_q90": float(np.quantile(peak_bins, 0.9)),
        "peak_sec_mean": float(tmin_sec + dt * peak_bins.mean()),
        "peak_sec_median": float(tmin_sec + dt * np.median(peak_bins)),
        "center_of_mass_bin_mean": float(center_of_mass.mean()),
        "center_of_mass_sec_mean": float(tmin_sec + dt * center_of_mass.mean()),
        "entropy_mean": float(entropy.mean()),
        "normalized_entropy_mean": float(norm_entropy.mean()),
        "top_bins": [
            {
                "bin": int(i),
                "sec": float(tmin_sec + dt * i),
                "mean": float(mean[i]),
                "sem": float(sem[i]),
            }
            for i in top_idx
        ],
        "window_mass": windows,
    }


def ms_ticks(seq_len: int, tmin_sec: float, tmax_sec: float) -> tuple[np.ndarray, list[str]]:
    tick_bins = np.linspace(0, seq_len - 1, 5)
    tick_labels = []
    for b in tick_bins:
        sec = tmin_sec + (tmax_sec - tmin_sec) * (b / max(seq_len - 1, 1))
        tick_labels.append(f"{sec * 1000:.0f}")
    return tick_bins, tick_labels


def plot_attention(
    summaries: dict[str, dict[str, Any]],
    out_pdf: Path,
    title: str,
) -> None:
    labels = list(summaries.keys())
    seq_len = summaries[labels[0]]["seq_len"]
    tmin_sec = summaries[labels[0]]["tmin_sec"]
    tmax_sec = summaries[labels[0]]["tmax_sec"]
    x = np.arange(seq_len)
    ticks, tick_labels = ms_ticks(seq_len, tmin_sec, tmax_sec)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    colors = {
        "REAL": "#1f77b4",
        "SHUF": "#d62728",
        "ZERO": "#2ca02c",
    }
    for label in labels:
        summary = summaries[label]
        mean = np.asarray(summary["mean_curve"])
        sem = np.asarray(summary["sem_curve"])
        color = colors.get(label, None)
        ax.plot(x, mean, label=label, linewidth=2, color=color)
        ax.fill_between(x, mean - sem, mean + sem, alpha=0.18, color=color)
    ax.set_title("Mean Cross-Attention over Brain Time")
    ax.set_xlabel("Brain Time Bin (ms)")
    ax.set_ylabel("Attention Mass")
    ax.set_xticks(ticks, tick_labels)
    ax.legend(frameon=False)

    ax = axes[1]
    first = summaries[labels[0]]
    window_labels = [
        f"{w['start_sec'] * 1000:.0f}-{w['end_sec'] * 1000:.0f}"
        for w in first["window_mass"]
    ]
    width = 0.8 / max(len(labels), 1)
    base = np.arange(len(window_labels))
    for i, label in enumerate(labels):
        masses = [w["mass"] for w in summaries[label]["window_mass"]]
        ax.bar(base + i * width - 0.4 + width / 2, masses, width=width, label=label, color=colors.get(label, None))
    ax.set_title("Attention Mass by Time Window")
    ax.set_xlabel("Window (ms)")
    ax.set_ylabel("Mean Mass")
    ax.set_xticks(base, window_labels)
    if len(labels) > 1:
        ax.legend(frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--real_manifest", type=Path, required=True)
    ap.add_argument("--shuf_manifest", type=Path, default=None)
    ap.add_argument("--zero_manifest", type=Path, default=None)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--out_pdf", type=Path, default=None)
    ap.add_argument("--title", type=str, default="Cross-Attention over Brain Time")
    ap.add_argument("--tmin_sec", type=float, default=0.0)
    ap.add_argument("--tmax_sec", type=float, default=0.6)
    args = ap.parse_args()

    manifests: list[tuple[str, Path | None]] = [
        ("REAL", args.real_manifest),
        ("SHUF", args.shuf_manifest),
        ("ZERO", args.zero_manifest),
    ]
    summaries: dict[str, dict[str, Any]] = {}
    for label, path in manifests:
        if path is None:
            continue
        mat = load_attention_matrix(path)
        summaries[label] = summarize_attention(mat, args.tmin_sec, args.tmax_sec)

    out = {
        "title": args.title,
        "conditions": summaries,
    }
    if "REAL" in summaries and "SHUF" in summaries:
        out["real_minus_shuf"] = {
            "late_window_mass_diff": summaries["REAL"]["window_mass"][-1]["mass"] - summaries["SHUF"]["window_mass"][-1]["mass"],
            "peak_sec_mean_diff": summaries["REAL"]["peak_sec_mean"] - summaries["SHUF"]["peak_sec_mean"],
            "normalized_entropy_mean_diff": summaries["REAL"]["normalized_entropy_mean"] - summaries["SHUF"]["normalized_entropy_mean"],
        }
    if "REAL" in summaries and "ZERO" in summaries:
        out["real_minus_zero"] = {
            "late_window_mass_diff": summaries["REAL"]["window_mass"][-1]["mass"] - summaries["ZERO"]["window_mass"][-1]["mass"],
            "peak_sec_mean_diff": summaries["REAL"]["peak_sec_mean"] - summaries["ZERO"]["peak_sec_mean"],
            "normalized_entropy_mean_diff": summaries["REAL"]["normalized_entropy_mean"] - summaries["ZERO"]["normalized_entropy_mean"],
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    if args.out_pdf is not None:
        plot_attention(summaries, args.out_pdf, args.title)


if __name__ == "__main__":
    main()

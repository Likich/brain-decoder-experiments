#!/usr/bin/env python3
"""Render appendix-ready LaTeX tables from qualitative REAL/ZERO/SHUF JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
    return rows


def squash_ws(text: str) -> str:
    return " ".join(str(text).split())


def tail_context(text: str, max_chars: int) -> str:
    text = squash_ws(text)
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    if " " in tail:
        tail = tail.split(" ", 1)[1]
    return "... " + tail.lstrip()


def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def stat_cell(stats: dict[str, Any]) -> str:
    rank = stats.get("first_token_rank")
    prob = stats.get("first_token_prob")
    if rank is None or prob is None:
        return "--"
    return f"p={float(prob):.3f}, r={int(rank)}"


def target_text(row: dict[str, Any]) -> str:
    text = row.get("target_text") or row.get("target_decoded") or ""
    return squash_ws(text)


def normalized_target(row: dict[str, Any]) -> str:
    return target_text(row).strip().lower()


def is_stopword_target(row: dict[str, Any]) -> bool:
    return normalized_target(row) in STOPWORD_TARGETS


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (squash_ws(str(row.get("context_text", ""))).lower(), normalized_target(row))


def delta_value(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except Exception:
        return 0.0


def score_row(kind: str, row: dict[str, Any]) -> tuple[float, ...]:
    d_rz = delta_value(row, "delta_real_zero")
    d_rs = delta_value(row, "delta_real_shuf")
    prob_gain = float(row.get("real_target_stats", {}).get("first_token_prob") or 0.0) - float(
        row.get("zero_target_stats", {}).get("first_token_prob") or 0.0
    )
    if kind == "gaussian":
        return (abs(d_rz) + abs(d_rs),)
    if kind in {"bert", "bertplusmeg"}:
        return (d_rz + d_rs,)
    return (1.0 if is_stopword_target(row) else 0.0, -prob_gain, d_rz + d_rs)


def choose_rows(
    kind: str,
    rows: list[dict[str, Any]],
    limit: int,
    max_per_target: int,
) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: score_row(kind, row))
    seen: set[tuple[str, str]] = set()
    target_counts: dict[str, int] = {}
    chosen: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []

    for row in ranked:
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        tgt = normalized_target(row)
        count = target_counts.get(tgt, 0)
        if kind == "meg" and count >= max_per_target:
            deferred.append(row)
            continue
        chosen.append(row)
        target_counts[tgt] = count + 1
        if len(chosen) >= limit:
            return chosen[:limit]

    for row in deferred:
        chosen.append(row)
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def render_table(title: str, rows: list[dict[str, Any]], max_context_chars: int) -> str:
    caption = (
        f"Qualitative paired-control examples for {title}. "
        "Each row shows the preceding context, the single-token target, the first-target "
        "rank/probability under ZERO, REAL, and SHUF, and the absolute target-probability gain from ZERO to REAL."
    )
    lines: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{0.40\linewidth}p{0.07\linewidth}cccc}",
        r"\toprule",
        r"Context & Target & ZERO & REAL & SHUF & $\Delta p$ (R-Z) \\",
        r"\midrule",
    ]
    for row in rows:
        context = latex_escape(tail_context(str(row.get("context_text", "")), max_context_chars))
        target = latex_escape(target_text(row))
        zero = latex_escape(stat_cell(row.get("zero_target_stats", {})))
        real = latex_escape(stat_cell(row.get("real_target_stats", {})))
        shuf = latex_escape(stat_cell(row.get("shuf_target_stats", {})))
        real_prob = float(row.get("real_target_stats", {}).get("first_token_prob") or 0.0)
        zero_prob = float(row.get("zero_target_stats", {}).get("first_token_prob") or 0.0)
        delta_prob = f"{real_prob - zero_prob:+.3f}"
        lines.append(f"{context} & {target} & {zero} & {real} & {shuf} & {delta_prob} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def summarize_rows(title: str, rows: list[dict[str, Any]], max_context_chars: int) -> dict[str, Any]:
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        real_prob = float(row.get("real_target_stats", {}).get("first_token_prob") or 0.0)
        zero_prob = float(row.get("zero_target_stats", {}).get("first_token_prob") or 0.0)
        shuf_prob = float(row.get("shuf_target_stats", {}).get("first_token_prob") or 0.0)
        out_rows.append(
            {
                "context": tail_context(str(row.get("context_text", "")), max_context_chars),
                "target": target_text(row),
                "delta_prob_real_zero": real_prob - zero_prob,
                "delta_real_zero": delta_value(row, "delta_real_zero"),
                "delta_real_shuf": delta_value(row, "delta_real_shuf"),
                "real": row.get("real_target_stats", {}),
                "zero": row.get("zero_target_stats", {}),
                "shuf": row.get("shuf_target_stats", {}),
                "meta": row.get("meta", {}),
                "source_label": row.get("source_label", title),
            }
        )
    return {"source": title, "n": len(out_rows), "rows": out_rows}


def kind_for_title(title: str) -> str:
    if title == "real MEG":
        return "meg"
    if title == "Gaussian null":
        return "gaussian"
    if title == "context-only BERT positive control":
        return "bert"
    if title == "BERT+MEG additive control":
        return "bertplusmeg"
    return "meg"


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out_tex", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--meg_jsonl", type=Path, default=None)
    ap.add_argument("--gaussian_jsonl", type=Path, default=None)
    ap.add_argument("--bert_jsonl", type=Path, default=None)
    ap.add_argument("--bertplusmeg_jsonl", type=Path, default=None)
    ap.add_argument("--meg_n", type=int, default=10)
    ap.add_argument("--gaussian_n", type=int, default=6)
    ap.add_argument("--bert_n", type=int, default=6)
    ap.add_argument("--bertplusmeg_n", type=int, default=6)
    ap.add_argument("--max_context_chars", type=int, default=110)
    ap.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip missing JSONL inputs instead of exiting with an error",
    )
    args = ap.parse_args()

    named_paths = [
        ("real MEG", args.meg_jsonl),
        ("Gaussian null", args.gaussian_jsonl),
        ("context-only BERT positive control", args.bert_jsonl),
        ("BERT+MEG additive control", args.bertplusmeg_jsonl),
    ]
    missing = [(name, path) for name, path in named_paths if path is not None and not path.exists()]
    if missing and not args.skip_missing:
        msg = ["Missing qualitative JSONL input(s):"]
        for name, path in missing:
            msg.append(f"  - {name}: {path}")
        msg.append("Run generate_t5_brain_controls.py first, or pass --skip_missing to render only the sections that exist.")
        raise SystemExit("\n".join(msg))

    chosen_sets: list[tuple[str, list[dict[str, Any]]]] = []
    for title, path, limit in [
        ("real MEG", args.meg_jsonl, args.meg_n),
        ("Gaussian null", args.gaussian_jsonl, args.gaussian_n),
        ("context-only BERT positive control", args.bert_jsonl, args.bert_n),
        ("BERT+MEG additive control", args.bertplusmeg_jsonl, args.bertplusmeg_n),
    ]:
        if path is None or not path.exists():
            continue
        kind = kind_for_title(title)
        rows = load_jsonl(path)
        if not rows:
            continue
        chosen_sets.append((title, choose_rows(kind, rows, limit, max_per_target=2)))

    sections = [render_table(title, rows, args.max_context_chars) for title, rows in chosen_sets]
    body = "\n".join(part for part in sections if part.strip())
    if not body:
        body = "% No qualitative appendix tables were rendered.\n"

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text(body, encoding="utf-8")
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_context_chars": args.max_context_chars,
            "sections": [summarize_rows(title, rows, args.max_context_chars) for title, rows in chosen_sets],
        }
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

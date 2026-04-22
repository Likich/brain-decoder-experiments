#!/usr/bin/env python3
"""Qualitative REAL/ZERO/SHUF inspection for MEG-conditioned T5.

This script is meant for appendix-style examples, not as a replacement for NLL
controls. By default it reports target-token probability/rank summaries under
matched MEG, zeroed MEG, and shuffled MEG, plus top-k alternatives at the first
supervised target position. Optional free generation is available, but it is
off by default because target-only models are not trained for open-ended text.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_text_pipeline.src.data.collators import meg_batch_collator
from brain_text_pipeline.src.data.datasets import ShardedExampleDataset
from brain_text_pipeline.src.models.t5_brain_model import T5BrainModel
from brain_text_pipeline.src.utils.logging import log


def resolve_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def per_example_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab = logits.size(-1)
    losses = F.cross_entropy(
        logits.reshape(-1, vocab),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    valid = labels.ne(-100)
    denom = valid.sum(dim=1).clamp_min(1)
    return (losses * valid).sum(dim=1) / denom


def decode_ids(tokenizer, ids: Any) -> str:
    arr = np.asarray(ids)
    if arr.dtype == object:
        arr = arr.astype(np.int64, copy=False)
    return tokenizer.decode(arr.tolist(), skip_special_tokens=True).strip()


def meta_dict(meta: Any) -> dict[str, Any]:
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, (str, bytes)):
        try:
            return json.loads(meta)
        except Exception:
            return {}
    return {}


def decoder_start_id(model: T5BrainModel, tokenizer) -> int:
    start_id = getattr(model.t5.config, "decoder_start_token_id", None)
    if start_id is None and hasattr(model.t5, "generation_config"):
        start_id = getattr(model.t5.generation_config, "decoder_start_token_id", None)
    if start_id is None:
        start_id = tokenizer.pad_token_id
    if start_id is None:
        start_id = tokenizer.eos_token_id
    if start_id is None:
        raise ValueError("Could not determine decoder_start_token_id, pad_token_id, or eos_token_id")
    return int(start_id)


def token_piece(tokenizer, token_id: int) -> str:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            piece = tokenizer.convert_ids_to_tokens(int(token_id))
            if piece is not None:
                return str(piece)
        except Exception:
            pass
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def rank_of_token(logits_row: torch.Tensor, token_id: int) -> int:
    token_score = logits_row[token_id]
    return int((logits_row > token_score).sum().item()) + 1


def target_token_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    log_probs = logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    summaries: list[dict[str, Any]] = []
    tops: list[list[dict[str, Any]]] = []
    for b in range(labels.size(0)):
        valid_positions = labels[b].ne(-100).nonzero(as_tuple=False).squeeze(-1).tolist()
        token_stats: list[dict[str, Any]] = []
        top_row: list[dict[str, Any]] = []
        for pos in valid_positions:
            tok_id = int(labels[b, pos].item())
            prob = float(probs[b, pos, tok_id].item())
            log_prob = float(log_probs[b, pos, tok_id].item())
            rank = rank_of_token(logits[b, pos], tok_id)
            token_stats.append(
                {
                    "position": int(pos),
                    "token_id": tok_id,
                    "piece": token_piece(tokenizer, tok_id),
                    "text": tokenizer.decode([tok_id], skip_special_tokens=True),
                    "prob": prob,
                    "log_prob": log_prob,
                    "rank": int(rank),
                }
            )

        if valid_positions:
            first_pos = int(valid_positions[0])
            vals, ids = torch.topk(probs[b, first_pos], k=top_k, dim=-1)
            for tok_id, prob in zip(ids.tolist(), vals.tolist()):
                top_row.append(
                    {
                        "token_id": int(tok_id),
                        "piece": token_piece(tokenizer, int(tok_id)),
                        "text": tokenizer.decode([int(tok_id)], skip_special_tokens=True),
                        "prob": float(prob),
                    }
                )

        seq_logprob = float(sum(t["log_prob"] for t in token_stats))
        mean_logprob = float(seq_logprob / len(token_stats)) if token_stats else float("-inf")
        first = token_stats[0] if token_stats else None
        summaries.append(
            {
                "target_len": len(token_stats),
                "target_pieces": [t["piece"] for t in token_stats],
                "target_token_ids": [int(t["token_id"]) for t in token_stats],
                "token_stats": token_stats,
                "first_token_prob": None if first is None else float(first["prob"]),
                "first_token_rank": None if first is None else int(first["rank"]),
                "mean_token_logprob": mean_logprob,
                "mean_token_prob": 0.0 if not token_stats else float(math.exp(mean_logprob)),
                "seq_logprob": seq_logprob,
                "seq_prob": 0.0 if not token_stats else float(math.exp(max(seq_logprob, -700.0))),
            }
        )
        tops.append(top_row)
    return summaries, tops


def generate_for_condition(
    model: T5BrainModel,
    tokenizer,
    brain_seq: torch.Tensor,
    brain_mask: torch.Tensor,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> list[str]:
    enc = model.brain_encoder(brain_seq, brain_mask)
    start_id = decoder_start_id(model, tokenizer)
    kwargs: dict[str, Any] = {
        "inputs_embeds": enc,
        "attention_mask": brain_mask,
        "decoder_start_token_id": start_id,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    out_ids = model.t5.generate(**kwargs)
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--tokenizer_name_or_path", type=str, default=None)
    ap.add_argument("--brain_encoder_ckpt", type=Path, required=True)
    ap.add_argument("--meg_dataset_path", type=Path, required=True)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=256, help="Candidate examples to score")
    ap.add_argument("--show", type=int, default=20, help="Examples to write after selection")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_text_len", type=int, default=8)
    ap.add_argument("--max_brain_len", type=int, default=None)
    ap.add_argument(
        "--decoder_context_mode",
        choices=["context_target", "target_only"],
        default="target_only",
    )
    ap.add_argument(
        "--brain_norm",
        choices=["none", "per_example"],
        default="per_example",
    )
    ap.add_argument(
        "--selection",
        choices=["random", "real_beats_controls", "largest_real_gain"],
        default="real_beats_controls",
        help="How to choose examples for qualitative display",
    )
    ap.add_argument(
        "--single_token_only",
        action="store_true",
        help="Keep only examples whose target consists of a single supervised token",
    )
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument(
        "--include_generation",
        action="store_true",
        help="Also run free generation under REAL/ZERO/SHUF (off by default)",
    )
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    device = resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = ShardedExampleDataset(args.meg_dataset_path)
    tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    sample = ds[0]["brain_seq"]
    brain_dim = sample.shape[1]
    model = T5BrainModel(args.model_name_or_path, brain_dim=brain_dim).to(device)
    model.brain_encoder.load_state_dict(torch.load(args.brain_encoder_ckpt, map_location="cpu"))
    model.eval()

    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[: min(args.samples, len(idxs))]

    candidates: list[dict[str, Any]] = []
    for start in range(0, len(idxs), args.batch_size):
        batch_idxs = idxs[start : start + args.batch_size]
        batch = [ds[i] for i in batch_idxs]
        collated = meg_batch_collator(
            batch,
            pad_id=tokenizer.pad_token_id,
            max_decoder_len=args.max_text_len,
            decoder_context_mode=args.decoder_context_mode,
            brain_norm=args.brain_norm,
        )
        brain_seq = collated["brain_seq"].to(device)
        brain_mask = collated["brain_mask"].to(device)
        dec_in = collated["decoder_input_ids"].to(device)
        dec_attn = collated["decoder_attention_mask"].to(device)
        labels = collated["labels"].to(device)
        if args.max_brain_len is not None and brain_seq.size(1) > args.max_brain_len:
            brain_seq = brain_seq[:, : args.max_brain_len]
            brain_mask = brain_mask[:, : args.max_brain_len]

        brain_zero = torch.zeros_like(brain_seq)
        brain_shuf = torch.roll(brain_seq, shifts=1, dims=0)
        mask_shuf = torch.roll(brain_mask, shifts=1, dims=0)

        with torch.no_grad():
            out_real = model(brain_seq, brain_mask, dec_in, decoder_attention_mask=dec_attn, labels=labels)
            out_zero = model(brain_zero, brain_mask, dec_in, decoder_attention_mask=dec_attn, labels=labels)
            out_shuf = model(brain_shuf, mask_shuf, dec_in, decoder_attention_mask=dec_attn, labels=labels)
            nll_real = per_example_nll(out_real.logits, labels).cpu().numpy()
            nll_zero = per_example_nll(out_zero.logits, labels).cpu().numpy()
            nll_shuf = per_example_nll(out_shuf.logits, labels).cpu().numpy()

            target_real, top_real = target_token_stats(out_real.logits, labels, tokenizer, args.top_k)
            target_zero, top_zero = target_token_stats(out_zero.logits, labels, tokenizer, args.top_k)
            target_shuf, top_shuf = target_token_stats(out_shuf.logits, labels, tokenizer, args.top_k)

            gen_real: list[str] | None = None
            gen_zero: list[str] | None = None
            gen_shuf: list[str] | None = None
            if args.include_generation:
                gen_real = generate_for_condition(
                    model,
                    tokenizer,
                    brain_seq,
                    brain_mask,
                    args.max_new_tokens,
                    args.num_beams,
                    args.do_sample,
                    args.temperature,
                    args.top_p,
                )
                gen_zero = generate_for_condition(
                    model,
                    tokenizer,
                    brain_zero,
                    brain_mask,
                    args.max_new_tokens,
                    args.num_beams,
                    args.do_sample,
                    args.temperature,
                    args.top_p,
                )
                gen_shuf = generate_for_condition(
                    model,
                    tokenizer,
                    brain_shuf,
                    mask_shuf,
                    args.max_new_tokens,
                    args.num_beams,
                    args.do_sample,
                    args.temperature,
                    args.top_p,
                )

        for j, item in enumerate(batch):
            meta = meta_dict(item.get("meta", {}))
            delta_rz = float(nll_real[j] - nll_zero[j])
            delta_rs = float(nll_real[j] - nll_shuf[j])
            candidates.append(
                {
                    "index": int(batch_idxs[j]),
                    "meta": meta,
                    "context_text": decode_ids(tokenizer, item["input_ids_context"]),
                    "target_text": str(meta.get("target_text") or decode_ids(tokenizer, item["decoder_target_ids"])),
                    "target_decoded": decode_ids(tokenizer, item["decoder_target_ids"]),
                    "target_len": int(target_real[j]["target_len"]),
                    "target_pieces": target_real[j]["target_pieces"],
                    "nll_real": float(nll_real[j]),
                    "nll_zero": float(nll_zero[j]),
                    "nll_shuf": float(nll_shuf[j]),
                    "delta_real_zero": delta_rz,
                    "delta_real_shuf": delta_rs,
                    "real_target_stats": target_real[j],
                    "zero_target_stats": target_zero[j],
                    "shuf_target_stats": target_shuf[j],
                    "gen_real": None if gen_real is None else gen_real[j].strip(),
                    "gen_zero": None if gen_zero is None else gen_zero[j].strip(),
                    "gen_shuf": None if gen_shuf is None else gen_shuf[j].strip(),
                    "top_real": top_real[j],
                    "top_zero": top_zero[j],
                    "top_shuf": top_shuf[j],
                }
            )

    if args.single_token_only:
        candidates = [c for c in candidates if c["target_len"] == 1]

    if args.selection == "real_beats_controls":
        selected = [c for c in candidates if c["delta_real_zero"] < 0 and c["delta_real_shuf"] < 0]
        selected.sort(key=lambda c: c["delta_real_zero"] + c["delta_real_shuf"])
    elif args.selection == "largest_real_gain":
        selected = sorted(candidates, key=lambda c: c["delta_real_zero"] + c["delta_real_shuf"])
    else:
        selected = candidates
    selected = selected[: args.show]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log(f"scored {len(candidates)} examples; wrote {len(selected)} to {args.out_jsonl}")
    for row in selected[: min(8, len(selected))]:
        top_real = ", ".join(f"{(x['piece'] or '<blank>').strip() or '<blank>'}:{x['prob']:.3f}" for x in row["top_real"][:5])
        top_zero = ", ".join(f"{(x['piece'] or '<blank>').strip() or '<blank>'}:{x['prob']:.3f}" for x in row["top_zero"][:5])
        top_shuf = ", ".join(f"{(x['piece'] or '<blank>').strip() or '<blank>'}:{x['prob']:.3f}" for x in row["top_shuf"][:5])
        real_stats = row["real_target_stats"]
        zero_stats = row["zero_target_stats"]
        shuf_stats = row["shuf_target_stats"]
        print("\n---")
        print(f"idx={row['index']} target={row['target_text']!r} dRZ={row['delta_real_zero']:.4f} dRS={row['delta_real_shuf']:.4f}")
        print(f"context: {row['context_text'][-240:]}")
        print(f"target pieces: {' '.join(row['target_pieces'])}")
        print(
            f"REAL target: rank1={real_stats['first_token_rank']} prob1={real_stats['first_token_prob']:.4f} "
            f"mean_p={real_stats['mean_token_prob']:.4f} seq_logp={real_stats['seq_logprob']:.4f}"
        )
        print(
            f"ZERO target: rank1={zero_stats['first_token_rank']} prob1={zero_stats['first_token_prob']:.4f} "
            f"mean_p={zero_stats['mean_token_prob']:.4f} seq_logp={zero_stats['seq_logprob']:.4f}"
        )
        print(
            f"SHUF target: rank1={shuf_stats['first_token_rank']} prob1={shuf_stats['first_token_prob']:.4f} "
            f"mean_p={shuf_stats['mean_token_prob']:.4f} seq_logp={shuf_stats['seq_logprob']:.4f}"
        )
        if args.include_generation:
            print(f"gen REAL: {row['gen_real']!r}")
            print(f"gen ZERO: {row['gen_zero']!r}")
            print(f"gen SHUF: {row['gen_shuf']!r}")
        print(f"top REAL: {top_real}")
        print(f"top ZERO: {top_zero}")
        print(f"top SHUF: {top_shuf}")


if __name__ == "__main__":
    main()

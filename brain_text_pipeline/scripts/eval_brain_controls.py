#!/usr/bin/env python3
"""Evaluate REAL vs SHUF vs ZERO on MEG dataset."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_text_pipeline.src.data.datasets import ShardedExampleDataset
from brain_text_pipeline.src.data.collators import meg_batch_collator
from brain_text_pipeline.src.models.t5_brain_model import T5BrainModel
from brain_text_pipeline.src.eval.metrics import js_div
from brain_text_pipeline.src.utils.logging import log, save_json


def resolve_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def per_example_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean token NLL per example, ignoring -100 labels."""
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


def last_valid_token_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels.ne(-100)
    positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
    last_pos = (positions * valid.long()).amax(dim=1)
    batch_idx = torch.arange(labels.size(0), device=labels.device)
    return logits[batch_idx, last_pos, :].softmax(dim=-1)


def paired_summary(diff: np.ndarray) -> dict:
    if diff.size == 0:
        return {"mean": 0.0, "median": 0.0, "se": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "win_rate": 0.0}
    se = float(diff.std(ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else 0.0
    mean = float(diff.mean())
    return {
        "mean": mean,
        "median": float(np.median(diff)),
        "se": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "win_rate": float((diff < 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--brain_encoder_ckpt", type=Path, required=True)
    ap.add_argument("--meg_dataset_path", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_text_len", type=int, default=None)
    ap.add_argument("--max_brain_len", type=int, default=None)
    ap.add_argument(
        "--decoder_context_mode",
        choices=["context_target", "target_only"],
        default="context_target",
    )
    ap.add_argument(
        "--brain_norm",
        choices=["none", "per_example"],
        default="none",
        help="Normalize each brain window before REAL/ZERO/SHUF controls",
    )
    ap.add_argument("--out_json", type=Path, default=Path("eval_controls.json"))
    args = ap.parse_args()

    device = resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ds = ShardedExampleDataset(args.meg_dataset_path)

    sample = ds[0]["brain_seq"]
    brain_dim = sample.shape[1]
    model = T5BrainModel(args.model_name_or_path, brain_dim=brain_dim).to(device)
    model.brain_encoder.load_state_dict(torch.load(args.brain_encoder_ckpt, map_location="cpu"))
    model.eval()

    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[: min(args.samples, len(ds))]

    js_real, js_shuf = [], []
    nll_real, nll_zero, nll_shuf = [], [], []
    top1_real_zero, top1_shuf_zero = [], []

    for i in range(0, len(idxs), args.batch_size):
        batch = [ds[j] for j in idxs[i : i + args.batch_size]]
        collated = meg_batch_collator(
            batch,
            pad_id=0,
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

        with torch.no_grad():
            # REAL
            out_real = model(brain_seq, brain_mask, dec_in, decoder_attention_mask=dec_attn, labels=labels)
            nll_real.extend(per_example_nll(out_real.logits, labels).cpu().tolist())

            # ZERO
            out_zero = model(torch.zeros_like(brain_seq), brain_mask, dec_in, decoder_attention_mask=dec_attn, labels=labels)
            nll_zero.extend(per_example_nll(out_zero.logits, labels).cpu().tolist())

            # SHUF
            perm = torch.randperm(brain_seq.size(0), device=device)
            out_shuf = model(
                brain_seq[perm],
                brain_mask[perm],
                dec_in,
                decoder_attention_mask=dec_attn,
                labels=labels,
            )
            nll_shuf.extend(per_example_nll(out_shuf.logits, labels).cpu().tolist())

            # JS/top-1 on the last supervised target token.
            logits_real = last_valid_token_probs(out_real.logits, labels)
            logits_zero = last_valid_token_probs(out_zero.logits, labels)
            logits_shuf = last_valid_token_probs(out_shuf.logits, labels)
            js_real.extend(js_div(logits_real, logits_zero).cpu().tolist())
            js_shuf.extend(js_div(logits_shuf, logits_zero).cpu().tolist())
            top1_real_zero.extend((logits_real.argmax(dim=-1) == logits_zero.argmax(dim=-1)).float().cpu().tolist())
            top1_shuf_zero.extend((logits_shuf.argmax(dim=-1) == logits_zero.argmax(dim=-1)).float().cpu().tolist())

    nll_real_arr = np.asarray(nll_real, dtype=np.float64)
    nll_zero_arr = np.asarray(nll_zero, dtype=np.float64)
    nll_shuf_arr = np.asarray(nll_shuf, dtype=np.float64)
    delta_real_zero = nll_real_arr - nll_zero_arr
    delta_real_shuf = nll_real_arr - nll_shuf_arr
    result = {
        "n": int(nll_real_arr.size),
        "nll_real": float(nll_real_arr.mean()),
        "nll_real_median": float(np.median(nll_real_arr)),
        "nll_zero": float(nll_zero_arr.mean()),
        "nll_zero_median": float(np.median(nll_zero_arr)),
        "nll_shuf": float(nll_shuf_arr.mean()),
        "nll_shuf_median": float(np.median(nll_shuf_arr)),
        "delta_real_zero": float(delta_real_zero.mean()),
        "delta_real_shuf": float(delta_real_shuf.mean()),
        "delta_real_zero_paired": paired_summary(delta_real_zero),
        "delta_real_shuf_paired": paired_summary(delta_real_shuf),
        "js_real": float(np.mean(js_real)),
        "js_shuf": float(np.mean(js_shuf)),
        "top1_real_zero": float(np.mean(top1_real_zero)),
        "top1_shuf_zero": float(np.mean(top1_shuf_zero)),
    }
    log(str(result))
    save_json(args.out_json, result)


if __name__ == "__main__":
    main()

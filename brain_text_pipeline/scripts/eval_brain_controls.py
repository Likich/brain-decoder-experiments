#!/usr/bin/env python3
"""Evaluate REAL vs SHUF vs ZERO on MEG dataset."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

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


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--brain_encoder_ckpt", type=Path, required=True)
    ap.add_argument("--meg_dataset_path", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_json", type=Path, default=Path("eval_controls.json"))
    args = ap.parse_args()

    device = resolve_device(args.device)
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

    for i in range(0, len(idxs), args.batch_size):
        batch = [ds[j] for j in idxs[i : i + args.batch_size]]
        collated = meg_batch_collator(batch, pad_id=0)
        brain_seq = collated["brain_seq"].to(device)
        brain_mask = collated["brain_mask"].to(device)
        dec_in = collated["decoder_input_ids"].to(device)
        dec_attn = collated["decoder_attention_mask"].to(device)
        labels = collated["labels"].to(device)

        # REAL
        out_real = model(brain_seq, brain_mask, dec_in, decoder_attention_mask=dec_attn, labels=labels)
        nll_real.append(out_real.loss.item())

        # ZERO
        out_zero = model(torch.zeros_like(brain_seq), brain_mask, dec_in, decoder_attention_mask=dec_attn, labels=labels)
        nll_zero.append(out_zero.loss.item())

        # SHUF
        perm = torch.randperm(brain_seq.size(0))
        out_shuf = model(brain_seq[perm], brain_mask, dec_in, decoder_attention_mask=dec_attn, labels=labels)
        nll_shuf.append(out_shuf.loss.item())

        # JS on last token
        with torch.no_grad():
            logits_real = out_real.logits[:, -1, :].softmax(dim=-1)
            logits_zero = out_zero.logits[:, -1, :].softmax(dim=-1)
            logits_shuf = out_shuf.logits[:, -1, :].softmax(dim=-1)
            js_real.append(js_div(logits_real, logits_zero).mean().item())
            js_shuf.append(js_div(logits_shuf, logits_zero).mean().item())

    result = {
        "n": len(idxs),
        "nll_real": float(np.mean(nll_real)),
        "nll_zero": float(np.mean(nll_zero)),
        "nll_shuf": float(np.mean(nll_shuf)),
        "delta_real_zero": float(np.mean(nll_real) - np.mean(nll_zero)),
        "delta_real_shuf": float(np.mean(nll_real) - np.mean(nll_shuf)),
        "js_real": float(np.mean(js_real)),
        "js_shuf": float(np.mean(js_shuf)),
    }
    log(str(result))
    save_json(args.out_json, result)


if __name__ == "__main__":
    main()

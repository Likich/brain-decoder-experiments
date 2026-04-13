#!/usr/bin/env python3
"""Evaluate perturbations on brain sequences."""
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
from brain_text_pipeline.src.eval.perturbations import (
    add_noise,
    channel_dropout,
    temporal_shuffle,
    time_shift,
)
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
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_json", type=Path, default=Path("perturb_eval.json"))
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

    metrics = {"baseline": [], "shuffle": [], "noise": [], "dropout": [], "shift": []}

    for i in range(0, len(idxs), args.batch_size):
        batch = [ds[j] for j in idxs[i : i + args.batch_size]]
        collated = meg_batch_collator(batch, pad_id=0)
        brain_seq = collated["brain_seq"].to(device)
        brain_mask = collated["brain_mask"].to(device)
        dec_in = collated["decoder_input_ids"].to(device)
        labels = collated["labels"].to(device)

        def eval_loss(seq):
            out = model(seq, brain_mask, dec_in, labels=labels)
            return out.loss.item()

        metrics["baseline"].append(eval_loss(brain_seq))
        metrics["shuffle"].append(eval_loss(temporal_shuffle(brain_seq)))
        metrics["noise"].append(eval_loss(add_noise(brain_seq, sigma=0.2)))
        metrics["dropout"].append(eval_loss(channel_dropout(brain_seq, drop_prob=0.2)))
        metrics["shift"].append(eval_loss(time_shift(brain_seq, shift=5)))

    result = {k: float(np.mean(v)) for k, v in metrics.items()}
    log(str(result))
    save_json(args.out_json, result)


if __name__ == "__main__":
    main()

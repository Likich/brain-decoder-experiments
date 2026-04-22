#!/usr/bin/env python3
"""Extract and save decoder cross-attention over brain time.

For each example, we compute mean cross-attention over layers+heads for the
last decoder token (prediction step). Saves sharded NPZ + manifest.
"""
from __future__ import annotations

import argparse
import json
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
from brain_text_pipeline.src.utils.io import ShardWriter, write_manifest
from brain_text_pipeline.src.utils.logging import log


def resolve_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--brain_encoder_ckpt", type=Path, required=True)
    ap.add_argument("--meg_dataset_path", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--shard_size", type=int, default=200)
    ap.add_argument("--max_brain_len", type=int, default=None)
    ap.add_argument("--max_text_len", type=int, default=None)
    ap.add_argument(
        "--decoder_context_mode",
        choices=["context_target", "target_only"],
        default="context_target",
    )
    ap.add_argument(
        "--brain_norm",
        choices=["none", "per_example"],
        default="none",
        help="Normalize each brain window before attention extraction",
    )
    ap.add_argument("--save_full_matrix", action="store_true", help="Save full [tgt_len, src_len] attention")
    ap.add_argument(
        "--condition",
        choices=["real", "zero", "shuf"],
        default="real",
        help="Which brain condition to export attention for",
    )
    args = ap.parse_args()

    device = resolve_device(args.device)
    ds = ShardedExampleDataset(args.meg_dataset_path)
    sample = ds[0]["brain_seq"]
    brain_dim = sample.shape[1]

    model = T5BrainModel(args.model_name_or_path, brain_dim=brain_dim).to(device)
    model.brain_encoder.load_state_dict(torch.load(args.brain_encoder_ckpt, map_location="cpu"))
    model.eval()

    idxs = list(range(len(ds)))
    np.random.shuffle(idxs)
    idxs = idxs[: min(args.samples, len(ds))]

    writer = ShardWriter(args.out_dir, prefix="attn", shard_size=args.shard_size)

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

        if args.max_brain_len is not None and brain_seq.size(1) > args.max_brain_len:
            brain_seq = brain_seq[:, : args.max_brain_len]
            brain_mask = brain_mask[:, : args.max_brain_len]

        if args.condition == "zero":
            brain_cond = torch.zeros_like(brain_seq)
            mask_cond = brain_mask
        elif args.condition == "shuf":
            brain_cond = torch.roll(brain_seq, shifts=1, dims=0)
            mask_cond = torch.roll(brain_mask, shifts=1, dims=0)
        else:
            brain_cond = brain_seq
            mask_cond = brain_mask

        with torch.no_grad():
            out = model(
                brain_cond,
                mask_cond,
                dec_in,
                decoder_attention_mask=dec_attn,
                labels=None,
                output_attentions=True,
                return_dict=True,
            )
            cross = out.cross_attentions  # list of [B, H, tgt_len, src_len]
            attn = torch.stack(cross, dim=0).mean(dim=0).mean(dim=1)  # [B, tgt_len, src_len]
            last = attn[:, -1, :].cpu().numpy()
            if args.save_full_matrix:
                full = attn.cpu().numpy()
            else:
                full = None

        for b_idx in range(last.shape[0]):
            meta = batch[b_idx].get("meta", None)
            if isinstance(meta, (bytes, str)):
                try:
                    meta = json.loads(meta)
                except Exception:
                    pass
            item = {
                "attn_last": last[b_idx].astype(np.float32),
                "brain_mask": mask_cond[b_idx].cpu().numpy().astype(np.int32),
                "meta": json.dumps(meta) if meta is not None else "{}",
            }
            if full is not None:
                item["attn_full"] = full[b_idx].astype(np.float32)
            writer.add(item)

    manifest = writer.finalize()
    manifest.update({
        "samples": len(idxs),
        "model_name_or_path": args.model_name_or_path,
        "brain_encoder_ckpt": str(args.brain_encoder_ckpt),
        "max_brain_len": args.max_brain_len,
        "brain_norm": args.brain_norm,
        "condition": args.condition,
    })
    write_manifest(args.out_dir / "manifest.json", manifest)
    log(f"Saved attention shards to {args.out_dir}")


if __name__ == "__main__":
    main()

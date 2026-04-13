#!/usr/bin/env python3
"""Train T5 brain cross-attention model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_text_pipeline.src.data.datasets import ShardedExampleDataset, TVBSequenceDataset
from brain_text_pipeline.src.data.collators import meg_batch_collator, pad_sequence
from brain_text_pipeline.src.models.t5_brain_model import T5BrainModel
from brain_text_pipeline.src.training.losses import masked_mse
from brain_text_pipeline.src.utils.io import read_manifest, write_manifest
from brain_text_pipeline.src.utils.logging import log
from brain_text_pipeline.src.utils.seed import set_seed


def resolve_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BrainPretrainHead(nn.Module):
    def __init__(self, brain_encoder: nn.Module, d_model: int, brain_dim: int):
        super().__init__()
        self.encoder = brain_encoder
        self.head = nn.Linear(d_model, brain_dim)

    def forward(self, brain_seq, brain_mask):
        enc = self.encoder(brain_seq, brain_mask)
        return self.head(enc)


def tvb_collate(batch):
    brain_seq = [b["brain_seq"] for b in batch]
    brain_tgt = [b["brain_target"] for b in batch]
    brain_seq, brain_mask = pad_sequence(brain_seq, pad_value=0.0, dtype=torch.float32)
    brain_tgt, _ = pad_sequence(brain_tgt, pad_value=0.0, dtype=torch.float32)
    return {"brain_seq": brain_seq, "brain_target": brain_tgt, "brain_mask": brain_mask}


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=["tvb_pretrain", "meg_supervised", "meg_joint_optional"], required=True)
    ap.add_argument("--model_name_or_path", type=str, default="t5-small")
    ap.add_argument("--meg_dataset_path", type=Path, default=None, help="Path to MEG manifest.json")
    ap.add_argument("--tvb_dataset_path", type=Path, default=None, help="Path to TVB pretrain manifest.json")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_brain_len", type=int, default=None)
    ap.add_argument("--freeze_t5", action="store_true")
    ap.add_argument("--unfreeze_last_n", type=int, default=0)
    ap.add_argument("--tvb_aux_weight", type=float, default=0.1)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "tvb_pretrain":
        if args.tvb_dataset_path is None:
            raise SystemExit("tvb_dataset_path required")
        tvb_ds = TVBSequenceDataset(args.tvb_dataset_path)
        loader = DataLoader(tvb_ds, batch_size=args.batch_size, shuffle=True, collate_fn=tvb_collate)
        # infer brain_dim
        sample = tvb_ds[0]["brain_seq"]
        brain_dim = sample.shape[1]
        model = T5BrainModel(args.model_name_or_path, brain_dim=brain_dim)
        pretrain = BrainPretrainHead(model.brain_encoder, model.t5.config.d_model, brain_dim).to(device)
        optimizer = torch.optim.AdamW(pretrain.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            pretrain.train()
            losses = []
            for batch in loader:
                brain_seq = batch["brain_seq"].to(device)
                brain_mask = batch["brain_mask"].to(device)
                brain_tgt = batch["brain_target"].to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = pretrain(brain_seq, brain_mask)
                loss = masked_mse(pred, brain_tgt, brain_mask)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            log(f"epoch {epoch} loss={sum(losses)/len(losses):.4f}")
        torch.save(pretrain.encoder.state_dict(), args.output_dir / "brain_encoder.pt")
        write_manifest(args.output_dir / "config.json", vars(args))
        return

    # MEG supervised / joint
    if args.meg_dataset_path is None:
        raise SystemExit("meg_dataset_path required")
    meg_ds = ShardedExampleDataset(args.meg_dataset_path)
    # infer brain_dim
    sample = meg_ds[0]["brain_seq"]
    brain_dim = sample.shape[1]

    model = T5BrainModel(args.model_name_or_path, brain_dim=brain_dim)
    if args.freeze_t5:
        model.freeze_t5()
    if args.unfreeze_last_n:
        model.unfreeze_last_n(args.unfreeze_last_n)
    if args.gradient_checkpointing:
        model.t5.gradient_checkpointing_enable()
    model = model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    def collate_fn(batch):
        return meg_batch_collator(batch, pad_id=0)

    loader = DataLoader(meg_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    if args.mode == "meg_supervised":
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in loader:
                brain_seq = batch["brain_seq"].to(device)
                brain_mask = batch["brain_mask"].to(device)
                dec_in = batch["decoder_input_ids"].to(device)
                labels = batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(brain_seq, brain_mask, dec_in, labels=labels)
                loss = out.loss
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            log(f"epoch {epoch} loss={sum(losses)/len(losses):.4f}")
        model.t5.save_pretrained(args.output_dir)
        torch.save(model.brain_encoder.state_dict(), args.output_dir / "brain_encoder.pt")
        write_manifest(args.output_dir / "config.json", vars(args))
        return

    # joint
    tvb_ds = TVBSequenceDataset(args.tvb_dataset_path) if args.tvb_dataset_path else None
    tvb_loader = DataLoader(tvb_ds, batch_size=args.batch_size, shuffle=True, collate_fn=tvb_collate)
    tvb_iter = iter(tvb_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            brain_seq = batch["brain_seq"].to(device)
            brain_mask = batch["brain_mask"].to(device)
            dec_in = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(brain_seq, brain_mask, dec_in, labels=labels)
            loss = out.loss
            # tvb aux
            if tvb_ds is not None:
                try:
                    tvb_batch = next(tvb_iter)
                except StopIteration:
                    tvb_iter = iter(tvb_loader)
                    tvb_batch = next(tvb_iter)
                tvb_seq = tvb_batch["brain_seq"].to(device)
                tvb_mask = tvb_batch["brain_mask"].to(device)
                tvb_tgt = tvb_batch["brain_target"].to(device)
                pred = model.brain_encoder(tvb_seq, tvb_mask)
                pred = pred[:, : tvb_tgt.shape[1]]
                aux = masked_mse(pred, tvb_tgt, tvb_mask[:, : tvb_tgt.shape[1]])
                loss = loss + args.tvb_aux_weight * aux
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        log(f"epoch {epoch} loss={sum(losses)/len(losses):.4f}")

    model.t5.save_pretrained(args.output_dir)
    torch.save(model.brain_encoder.state_dict(), args.output_dir / "brain_encoder.pt")
    write_manifest(args.output_dir / "config.json", vars(args))


if __name__ == "__main__":
    main()

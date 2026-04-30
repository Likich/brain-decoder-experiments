#!/usr/bin/env python3
"""Train a strict additive BERT+MEG residual model.

This script freezes a standalone text-aux-conditioned T5 model and learns only
an additive MEG residual branch on top of that fixed auxiliary backbone.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_text_pipeline.src.data.collators import meg_batch_collator
from brain_text_pipeline.src.data.datasets import ShardedExampleDataset
from brain_text_pipeline.src.models.t5_brain_model import T5FixedAuxResidualMEGModel
from brain_text_pipeline.src.utils.io import write_manifest
from brain_text_pipeline.src.utils.logging import log
from brain_text_pipeline.src.utils.seed import set_seed


def resolve_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def jsonable_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def split_combined_dims(manifest: dict) -> tuple[int, int]:
    combined = manifest.get("combined_aux")
    if not isinstance(combined, dict):
        raise ValueError("combined dataset manifest is missing 'combined_aux' metadata")
    if combined.get("feature_order") != "meg_then_aux":
        raise ValueError(
            "only combined_aux.feature_order='meg_then_aux' is supported, "
            f"got {combined.get('feature_order')!r}"
        )
    meg_dim = int(combined.get("meg_dim", 0))
    aux_dim = int(combined.get("aux_dim", 0))
    if meg_dim <= 0 or aux_dim <= 0:
        raise ValueError(f"invalid combined_aux dims: meg_dim={meg_dim}, aux_dim={aux_dim}")
    return meg_dim, aux_dim


def split_combined_brain(brain_seq: torch.Tensor, meg_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    meg_seq = brain_seq[:, :, :meg_dim]
    aux_seq = brain_seq[:, :, meg_dim:]
    return meg_seq, aux_seq


def save_checkpoint(
    model: T5FixedAuxResidualMEGModel,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    meg_dim: int,
    aux_dim: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.t5.save_pretrained(output_dir)
    torch.save(model.aux_encoder.state_dict(), output_dir / "aux_encoder.pt")
    torch.save(model.meg_encoder.state_dict(), output_dir / "meg_encoder.pt")
    config = jsonable_args(args)
    config.update(
        {
            "meg_dim": int(meg_dim),
            "aux_dim": int(aux_dim),
            "residual_mode": "fixed_aux_plus_meg",
        }
    )
    write_manifest(output_dir / "config.json", config)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base_model_name_or_path", type=str, required=True, help="Standalone BERT-conditioned run dir")
    ap.add_argument("--base_aux_encoder_ckpt", type=Path, required=True, help="brain_encoder.pt from the BERT-only run")
    ap.add_argument("--combined_dataset_path", type=Path, required=True, help="Manifest for [MEG ; BERT] dataset")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max_brain_len", type=int, default=None)
    ap.add_argument("--max_text_len", type=int, default=8)
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
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--cpu_threads", type=int, default=None)
    ap.add_argument("--log_interval", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    if args.cpu_threads is not None:
        try:
            torch.set_num_threads(int(args.cpu_threads))
            torch.set_num_interop_threads(int(args.cpu_threads))
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.cpu_threads))

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = ShardedExampleDataset(args.combined_dataset_path)
    meg_dim, aux_dim = split_combined_dims(ds.manifest)
    sample = ds[0]["brain_seq"]
    total_dim = int(sample.shape[1])
    if total_dim != meg_dim + aux_dim:
        raise ValueError(f"dataset feature dim {total_dim} != meg_dim + aux_dim ({meg_dim + aux_dim})")

    model = T5FixedAuxResidualMEGModel(
        base_model_name_or_path=args.base_model_name_or_path,
        aux_dim=aux_dim,
        meg_dim=meg_dim,
    )
    model.load_aux_encoder(str(args.base_aux_encoder_ckpt))
    model.freeze_base()
    if args.gradient_checkpointing:
        model.t5.gradient_checkpointing_enable()
    model = model.to(device)

    total_params, trainable_params = parameter_counts(model)
    log(f"trainable parameters: {trainable_params:,}/{total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    use_autocast = (device.type == "cuda") and (args.fp16 or args.bf16)
    autocast_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.fp16))

    def collate_fn(batch):
        return meg_batch_collator(
            batch,
            pad_id=0,
            max_decoder_len=args.max_text_len,
            decoder_context_mode=args.decoder_context_mode,
            brain_norm=args.brain_norm,
        )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        running = 0.0
        steps = 0
        for step, batch in enumerate(loader, start=1):
            brain_seq = batch["brain_seq"].to(device)
            brain_mask = batch["brain_mask"].to(device)
            dec_in = batch["decoder_input_ids"].to(device)
            dec_attn = batch["decoder_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if args.max_brain_len is not None and brain_seq.size(1) > args.max_brain_len:
                brain_seq = brain_seq[:, : args.max_brain_len]
                brain_mask = brain_mask[:, : args.max_brain_len]
            meg_seq, aux_seq = split_combined_brain(brain_seq, meg_dim)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                out = model(
                    aux_seq,
                    meg_seq,
                    brain_mask,
                    dec_in,
                    decoder_attention_mask=dec_attn,
                    labels=labels,
                    use_meg=True,
                )
                loss = out.loss

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            running += loss.item()
            steps += 1
            global_step += 1
            if args.log_interval and (step % args.log_interval == 0):
                log(f"epoch {epoch} step {step} loss={running/steps:.4f}")
            if args.max_steps is not None and global_step >= args.max_steps:
                log(f"stopping at max_steps={args.max_steps}")
                save_checkpoint(model, args.output_dir, args, meg_dim=meg_dim, aux_dim=aux_dim)
                return

        log(f"epoch {epoch} loss={sum(losses)/len(losses):.4f}")

    save_checkpoint(model, args.output_dir, args, meg_dim=meg_dim, aux_dim=aux_dim)


if __name__ == "__main__":
    main()

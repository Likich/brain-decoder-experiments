#!/usr/bin/env python3
"""Build a context-text auxiliary-control dataset from a MEG manifest.

This keeps the targets, metadata, and original temporal lengths unchanged while
replacing ``brain_seq`` with context-only BERT states. It serves as a positive
control: the same T5+brain-encoder path now receives a strong semantic side
channel derived only from preceding text, without leaking the target word.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_text_pipeline.src.utils.io import ShardWriter, read_manifest, write_manifest
from brain_text_pipeline.src.utils.logging import log


def shard_path(source_manifest: Path, shard: dict[str, Any]) -> Path:
    path = Path(shard["path"])
    if path.is_absolute() or path.exists():
        return path
    candidate = source_manifest.parent / path
    return candidate if candidate.exists() else path


def iter_shard_items(manifest_path: Path, manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for shard in manifest["shards"]:
        path = shard_path(manifest_path, shard)
        with np.load(path, allow_pickle=True) as data:
            keys = list(data.files)
            size = int(shard.get("size", len(data[keys[0]])))
            arrays = {key: data[key] for key in keys}
            for item_idx in range(size):
                yield {key: arrays[key][item_idx] for key in keys}


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    meta = out.get("meta", {})
    if isinstance(meta, dict):
        out["meta"] = json.dumps(meta)
    return out


def coerce_float_matrix(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = arr.astype(np.float32, copy=False)
    else:
        arr = arr.astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"brain_seq must be 2D [T,D], got shape {arr.shape}")
    return arr


def ids_to_lists(batch_ids: list[np.ndarray]) -> list[list[int]]:
    out: list[list[int]] = []
    for ids in batch_ids:
        arr = np.asarray(ids)
        if arr.dtype == object:
            arr = arr.astype(np.int64, copy=False)
        out.append([int(x) for x in arr.tolist()])
    return out


def interpolate_token_states(states: torch.Tensor, target_len: int) -> np.ndarray:
    """Resample [L, D] token states to [target_len, D]."""
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if states.numel() == 0:
        return np.zeros((target_len, 1), dtype=np.float32)
    if states.size(0) == 1:
        return states.repeat(target_len, 1).cpu().numpy().astype(np.float32, copy=False)
    x = states.transpose(0, 1).unsqueeze(0)  # [1, D, L]
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=True)
    return x.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32, copy=False)


def build_context_batch(
    *,
    items: list[dict[str, Any]],
    source_tokenizer: AutoTokenizer,
    text_tokenizer: AutoTokenizer,
    text_model: AutoModel,
    device: torch.device,
    max_text_length: int,
) -> list[np.ndarray]:
    context_ids = ids_to_lists([item["input_ids_context"] for item in items])
    contexts = source_tokenizer.batch_decode(context_ids, skip_special_tokens=True)
    targets = [coerce_float_matrix(item["brain_seq"]).shape[0] for item in items]

    encoded = text_tokenizer(
        contexts,
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = text_model(**encoded)
    hidden = outputs.last_hidden_state
    attn = encoded["attention_mask"]

    aux_seqs: list[np.ndarray] = []
    for idx, target_len in enumerate(targets):
        valid_len = int(attn[idx].sum().item())
        content_start = 1 if valid_len >= 2 else 0
        content_end = max(content_start, valid_len - 1)
        token_states = hidden[idx, content_start:content_end]
        if token_states.numel() == 0:
            dim = int(hidden.size(-1))
            aux_seq = np.zeros((target_len, dim), dtype=np.float32)
        else:
            aux_seq = interpolate_token_states(token_states, target_len)
        aux_seqs.append(aux_seq)
    return aux_seqs


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--source_manifest", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--source_tokenizer_name_or_path", default="t5-small")
    ap.add_argument("--text_model_name_or_path", default="bert-base-uncased")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_text_length", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shard_size", type=int, default=None, help="defaults to source shard size")
    ap.add_argument("--report_every", type=int, default=10000)
    args = ap.parse_args()

    manifest = read_manifest(args.source_manifest)
    shard_size = int(args.shard_size or manifest.get("shard_size", 5000))
    device = torch.device(args.device)

    log(f"loading source tokenizer: {args.source_tokenizer_name_or_path}")
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer_name_or_path)
    log(f"loading text model: {args.text_model_name_or_path}")
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_name_or_path)
    text_model = AutoModel.from_pretrained(args.text_model_name_or_path).to(device)
    text_model.eval()

    writer = ShardWriter(args.out_dir, prefix="text_aux", shard_size=shard_size)
    total = 0
    batch: list[dict[str, Any]] = []

    def flush(cur_batch: list[dict[str, Any]]) -> None:
        nonlocal total
        if not cur_batch:
            return
        aux_seqs = build_context_batch(
            items=cur_batch,
            source_tokenizer=source_tokenizer,
            text_tokenizer=text_tokenizer,
            text_model=text_model,
            device=device,
            max_text_length=args.max_text_length,
        )
        for item, aux_seq in zip(cur_batch, aux_seqs):
            out = normalize_item(item)
            out["brain_seq"] = aux_seq
            writer.add(out)
            total += 1
            if args.report_every and total % args.report_every == 0:
                log(f"processed {total}/{manifest.get('total_examples', '?')} examples")

    for item in iter_shard_items(args.source_manifest, manifest):
        batch.append(item)
        if len(batch) >= args.batch_size:
            flush(batch)
            batch = []
    flush(batch)

    out_manifest = writer.finalize()
    source_metadata = {
        key: value
        for key, value in manifest.items()
        if key not in {"shards", "num_shards", "total_examples", "prefix", "shard_size"}
    }
    out_manifest.update(source_metadata)
    out_manifest.update(
        {
            "source_manifest": str(args.source_manifest),
            "aux_control": {
                "mode": "context_text_model",
                "source_tokenizer_name_or_path": args.source_tokenizer_name_or_path,
                "text_model_name_or_path": args.text_model_name_or_path,
                "max_text_length": args.max_text_length,
            },
        }
    )
    write_manifest(args.out_dir / "manifest.json", out_manifest)
    log(f"saved {total} text-auxiliary examples to {args.out_dir}")


if __name__ == "__main__":
    main()

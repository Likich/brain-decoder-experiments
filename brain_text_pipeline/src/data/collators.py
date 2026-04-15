"""Collators for padding brain-text examples."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


def pad_sequence(seqs: List[np.ndarray], pad_value: float = 0.0, dtype=torch.float32):
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    dim = seqs[0].shape[1]
    out = torch.full((len(seqs), max_len, dim), pad_value, dtype=dtype)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        cur = torch.tensor(seq, dtype=dtype)
        out[i, : len(seq), :] = cur
        mask[i, : len(seq)] = 1
    return out, mask


def pad_tokens(seqs: List[np.ndarray], pad_id: int, dtype=torch.long):
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    out = torch.full((len(seqs), max_len), pad_id, dtype=dtype)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        cur = torch.tensor(seq, dtype=dtype)
        out[i, : len(seq)] = cur
        mask[i, : len(seq)] = 1
    return out, mask


def pad_labels(seqs: List[np.ndarray], pad_value: int = -100, dtype=torch.long) -> torch.Tensor:
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    out = torch.full((len(seqs), max_len), pad_value, dtype=dtype)
    for i, seq in enumerate(seqs):
        cur = torch.tensor(seq, dtype=dtype)
        out[i, : len(seq)] = cur
    return out


def meg_batch_collator(
    batch: List[Dict[str, Any]],
    pad_id: int,
    *,
    decoder_start_id: int | None = None,
    max_decoder_len: int | None = None,
) -> Dict[str, torch.Tensor]:
    brain_seqs = [b["brain_seq"] for b in batch]
    brain_seq, brain_mask = pad_sequence(brain_seqs, pad_value=0.0, dtype=torch.float32)

    context_ids = [b["input_ids_context"] for b in batch]
    decoder_targets = [b["decoder_target_ids"] for b in batch]

    # Build decoder sequences:
    # output_seq = context + target
    # decoder_input_ids = shift_right(output_seq) (T5 uses pad as decoder_start_token)
    # labels = [-100]*len(context) + target (pad labels with -100)
    dec_inputs = []
    dec_labels = []
    start_id = pad_id if decoder_start_id is None else int(decoder_start_id)
    for ctx, tgt in zip(context_ids, decoder_targets):
        ctx = np.asarray(ctx, dtype=np.int32)
        tgt = np.asarray(tgt, dtype=np.int32)
        if tgt.size == 0:
            # Shouldn't happen, but keep shapes consistent.
            tgt = np.asarray([pad_id], dtype=np.int32)
        if max_decoder_len is not None and (ctx.size + tgt.size) > max_decoder_len:
            keep_ctx = max(0, int(max_decoder_len) - int(tgt.size))
            ctx = ctx[-keep_ctx:] if keep_ctx > 0 else ctx[:0]

        out_seq = np.concatenate([ctx, tgt], axis=0)
        # shift-right
        dec_in = np.concatenate([np.array([start_id], dtype=np.int32), out_seq[:-1]], axis=0)
        labels = np.concatenate([np.full(ctx.shape[0], -100, dtype=np.int32), tgt], axis=0)
        dec_inputs.append(dec_in)
        dec_labels.append(labels)

    decoder_input_ids, decoder_attention_mask = pad_tokens(dec_inputs, pad_id)
    labels = pad_labels(dec_labels, pad_value=-100)

    return {
        "brain_seq": brain_seq,
        "brain_mask": brain_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels,
    }

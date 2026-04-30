#!/usr/bin/env python3
"""Evaluate REAL vs SHUF vs ZERO on MEG dataset."""
from __future__ import annotations

import argparse
from collections import Counter
import json
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


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_group_keys(text: str | None) -> list[str]:
    if text is None:
        return []
    keys = [part.strip() for part in text.split(",") if part.strip()]
    if not keys:
        raise ValueError("shuffle group keys must not be empty")
    return keys


def meta_key_tuple(meta: dict, keys: list[str]) -> tuple[str, ...]:
    values = []
    for key in keys:
        value = meta.get(key)
        if value is None or value == "":
            value = "__missing__"
        values.append(str(value))
    return tuple(values)


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


def first_valid_token_stats(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, list[float | int | None]]:
    probs = logits.softmax(dim=-1)
    valid = labels.ne(-100)
    batch = labels.size(0)
    first_prob: list[float | None] = []
    first_rank: list[int | None] = []
    first_token_id: list[int | None] = []
    target_len: list[int] = []
    for b in range(batch):
        valid_pos = valid[b].nonzero(as_tuple=False).squeeze(-1)
        target_len.append(int(valid_pos.numel()))
        if valid_pos.numel() == 0:
            first_prob.append(None)
            first_rank.append(None)
            first_token_id.append(None)
            continue
        pos = int(valid_pos[0].item())
        tok_id = int(labels[b, pos].item())
        row_probs = probs[b, pos]
        prob = float(row_probs[tok_id].item())
        rank = int((row_probs > row_probs[tok_id]).sum().item()) + 1
        first_prob.append(prob)
        first_rank.append(rank)
        first_token_id.append(tok_id)
    return {
        "first_token_prob": first_prob,
        "first_token_rank": first_rank,
        "first_token_id": first_token_id,
        "target_len": target_len,
    }


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


def resolve_control_slice(
    *,
    manifest: dict,
    brain_dim: int,
    control_feature_group: str,
    control_feature_start: int | None,
    control_feature_end: int | None,
) -> slice | None:
    if control_feature_start is not None or control_feature_end is not None:
        if control_feature_start is None or control_feature_end is None:
            raise ValueError("both --control_feature_start and --control_feature_end must be set together")
        start = int(control_feature_start)
        end = int(control_feature_end)
        if not (0 <= start < end <= brain_dim):
            raise ValueError(f"invalid control feature slice [{start}:{end}) for brain_dim={brain_dim}")
        return slice(start, end)

    if control_feature_group == "all":
        return None

    combined = manifest.get("combined_aux")
    if not isinstance(combined, dict):
        raise ValueError(
            f"--control_feature_group={control_feature_group!r} requires manifest metadata 'combined_aux'"
        )
    if combined.get("feature_order") != "meg_then_aux":
        raise ValueError(
            "only combined_aux.feature_order='meg_then_aux' is currently supported, "
            f"got {combined.get('feature_order')!r}"
        )

    meg_dim = int(combined.get("meg_dim", 0))
    aux_dim = int(combined.get("aux_dim", 0))
    total_dim = meg_dim + aux_dim
    if total_dim != brain_dim:
        raise ValueError(f"manifest combined_aux dims sum to {total_dim}, but dataset brain_dim={brain_dim}")

    if control_feature_group == "meg_only":
        return slice(0, meg_dim)
    if control_feature_group == "aux_only":
        return slice(meg_dim, total_dim)
    raise ValueError(f"unknown control_feature_group: {control_feature_group}")


def zero_control(brain_seq: torch.Tensor, control_slice: slice | None) -> torch.Tensor:
    if control_slice is None:
        return torch.zeros_like(brain_seq)
    out = brain_seq.clone()
    out[:, :, control_slice] = 0.0
    return out


def shuffled_control(brain_seq: torch.Tensor, perm: torch.Tensor, control_slice: slice | None) -> torch.Tensor:
    if control_slice is None:
        return brain_seq[perm]
    out = brain_seq.clone()
    out[:, :, control_slice] = brain_seq[perm][:, :, control_slice]
    return out


def splice_control(
    base_brain_seq: torch.Tensor,
    donor_brain_seq: torch.Tensor,
    control_slice: slice | None,
) -> torch.Tensor:
    if donor_brain_seq.shape != base_brain_seq.shape:
        raise ValueError(
            f"donor brain shape {tuple(donor_brain_seq.shape)} does not match base shape {tuple(base_brain_seq.shape)}"
        )
    if control_slice is None:
        return donor_brain_seq
    out = base_brain_seq.clone()
    out[:, :, control_slice] = donor_brain_seq[:, :, control_slice]
    return out


def match_time_dim(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.size(1) == target_len:
        return x
    if x.size(1) > target_len:
        return x[:, :target_len]
    pad_shape = (x.size(0), target_len - x.size(1), x.size(2))
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


def match_mask_dim(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.size(1) == target_len:
        return x
    if x.size(1) > target_len:
        return x[:, :target_len]
    pad_shape = (x.size(0), target_len - x.size(1))
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


def random_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    base = np.arange(n, dtype=np.int64)
    for _ in range(128):
        perm = rng.permutation(n)
        if np.all(perm != base):
            return perm.astype(np.int64, copy=False)
    return np.roll(base, 1)


def build_shuffle_plan(
    *,
    metas: list[dict],
    shuf_mode: str,
    shuf_group_keys: list[str],
    seed: int,
) -> dict | None:
    if shuf_mode in {"batch_global", "circular_time_shift", "block_permute", "phase_randomized"}:
        return None

    rng = np.random.default_rng(seed)
    n = len(metas)
    donor_positions = np.arange(n, dtype=np.int64)
    singleton_groups = 0
    groups_summary: Counter[tuple[str, ...]] | None = None

    if shuf_mode == "global_sample":
        donor_positions = random_derangement(n, rng)
    elif shuf_mode == "within_group":
        if not shuf_group_keys:
            raise ValueError("--shuf_mode=within_group requires --shuf_group_keys")
        groups: dict[tuple[str, ...], list[int]] = {}
        for pos, meta in enumerate(metas):
            groups.setdefault(meta_key_tuple(meta, shuf_group_keys), []).append(pos)
        groups_summary = Counter({k: len(v) for k, v in groups.items()})
        for positions in groups.values():
            if len(positions) < 2:
                singleton_groups += 1
                continue
            local_perm = random_derangement(len(positions), rng)
            pos_arr = np.asarray(positions, dtype=np.int64)
            donor_positions[pos_arr] = pos_arr[local_perm]
    else:
        raise ValueError(f"shuffle plan does not apply to mode {shuf_mode!r}")

    identity_count = int((donor_positions == np.arange(n, dtype=np.int64)).sum())
    plan = {
        "mode": shuf_mode,
        "donor_positions": donor_positions.tolist(),
        "identity_count": identity_count,
        "singleton_groups": int(singleton_groups),
    }
    if shuf_group_keys:
        plan["group_keys"] = shuf_group_keys
    if groups_summary is not None:
        plan["n_groups"] = int(len(groups_summary))
        plan["largest_groups"] = [
            {"group": list(group), "size": int(size)}
            for group, size in groups_summary.most_common(5)
        ]
    return plan


def circular_time_shift_control(
    brain_seq: torch.Tensor,
    brain_mask: torch.Tensor,
    control_slice: slice | None,
    rng: np.random.Generator,
) -> torch.Tensor:
    out = brain_seq.clone()
    lengths = brain_mask.sum(dim=1).tolist()
    for b, length in enumerate(lengths):
        length = int(length)
        if length <= 1:
            continue
        shift = int(rng.integers(1, length))
        if control_slice is None:
            out[b, :length, :] = torch.roll(brain_seq[b, :length, :], shifts=shift, dims=0)
        else:
            out[b, :length, control_slice] = torch.roll(
                brain_seq[b, :length, control_slice], shifts=shift, dims=0
            )
    return out


def block_permute_control(
    brain_seq: torch.Tensor,
    brain_mask: torch.Tensor,
    control_slice: slice | None,
    rng: np.random.Generator,
    block_size: int,
) -> torch.Tensor:
    out = brain_seq.clone()
    lengths = brain_mask.sum(dim=1).tolist()
    for b, length in enumerate(lengths):
        length = int(length)
        if length <= 1:
            continue
        block_size_cur = max(1, min(block_size, length))
        starts = list(range(0, length, block_size_cur))
        if len(starts) < 2:
            continue
        blocks = [slice(s, min(s + block_size_cur, length)) for s in starts]
        perm = random_derangement(len(blocks), rng)
        if control_slice is None:
            source = brain_seq[b, :length, :]
            target = out[b, :length, :]
        else:
            source = brain_seq[b, :length, control_slice]
            target = out[b, :length, control_slice]
        cursor = 0
        for block_idx in perm.tolist():
            seg = source[blocks[block_idx]]
            seg_len = seg.size(0)
            target[cursor : cursor + seg_len] = seg
            cursor += seg_len
    return out


def phase_randomize_segment(seg: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    if seg.size(0) <= 2:
        return seg
    x = seg.transpose(0, 1)  # [D, T]
    spec = torch.fft.rfft(x, dim=-1)
    n_freq = spec.size(-1)
    if n_freq <= 1:
        return seg
    randomized = spec.clone()
    if n_freq > 2:
        phase = torch.tensor(
            rng.uniform(0.0, 2.0 * np.pi, size=(spec.size(0), n_freq - 2)),
            dtype=x.dtype,
            device=x.device,
        )
        randomized[:, 1:-1] = torch.abs(spec[:, 1:-1]) * torch.exp(1j * phase)
    randomized[:, 0] = spec[:, 0]
    if seg.size(0) % 2 == 0 and n_freq > 1:
        randomized[:, -1] = spec[:, -1]
    out = torch.fft.irfft(randomized, n=seg.size(0), dim=-1)
    return out.transpose(0, 1)


def phase_randomized_control(
    brain_seq: torch.Tensor,
    brain_mask: torch.Tensor,
    control_slice: slice | None,
    rng: np.random.Generator,
) -> torch.Tensor:
    out = brain_seq.clone()
    lengths = brain_mask.sum(dim=1).tolist()
    for b, length in enumerate(lengths):
        length = int(length)
        if length <= 2:
            continue
        if control_slice is None:
            out[b, :length, :] = phase_randomize_segment(brain_seq[b, :length, :], rng)
        else:
            out[b, :length, control_slice] = phase_randomize_segment(
                brain_seq[b, :length, control_slice], rng
            )
    return out


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
    ap.add_argument(
        "--control_feature_group",
        choices=["all", "meg_only", "aux_only"],
        default="all",
        help=(
            "Which feature block ZERO/SHUF should perturb. "
            "'all' reproduces the original evaluation. "
            "'meg_only' and 'aux_only' require combined_aux metadata in the manifest."
        ),
    )
    ap.add_argument("--control_feature_start", type=int, default=None)
    ap.add_argument("--control_feature_end", type=int, default=None)
    ap.add_argument(
        "--shuf_mode",
        choices=["batch_global", "global_sample", "within_group", "circular_time_shift", "block_permute", "phase_randomized"],
        default="batch_global",
        help=(
            "How to construct SHUF. "
            "'batch_global' reproduces the original within-batch permutation. "
            "'global_sample' permutes across the entire sampled set. "
            "'within_group' permutes within metadata groups given by --shuf_group_keys. "
            "Temporal modes operate within each example."
        ),
    )
    ap.add_argument(
        "--shuf_group_keys",
        type=str,
        default=None,
        help="Comma-separated metadata keys for --shuf_mode=within_group, e.g. subject or subject,session.",
    )
    ap.add_argument(
        "--shuf_block_size",
        type=int,
        default=10,
        help="Block size in time bins for --shuf_mode=block_permute.",
    )
    ap.add_argument(
        "--out_examples_jsonl",
        type=Path,
        default=None,
        help="Optional per-example export with metadata and REAL/ZERO/SHUF deltas for clustered statistics.",
    )
    args = ap.parse_args()

    device = resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ds = ShardedExampleDataset(args.meg_dataset_path)
    shuf_group_keys = parse_group_keys(args.shuf_group_keys)

    sample = ds[0]["brain_seq"]
    brain_dim = sample.shape[1]
    control_slice = resolve_control_slice(
        manifest=ds.manifest,
        brain_dim=brain_dim,
        control_feature_group=args.control_feature_group,
        control_feature_start=args.control_feature_start,
        control_feature_end=args.control_feature_end,
    )
    model = T5BrainModel(args.model_name_or_path, brain_dim=brain_dim).to(device)
    model.brain_encoder.load_state_dict(torch.load(args.brain_encoder_ckpt, map_location="cpu"))
    model.eval()

    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[: min(args.samples, len(ds))]
    sampled_metas = [ds[j].get("meta") or {} for j in idxs]
    shuffle_plan = build_shuffle_plan(
        metas=sampled_metas,
        shuf_mode=args.shuf_mode,
        shuf_group_keys=shuf_group_keys,
        seed=args.seed,
    )
    temporal_rng = np.random.default_rng(args.seed)

    js_real, js_shuf = [], []
    nll_real, nll_zero, nll_shuf = [], [], []
    top1_real_zero, top1_shuf_zero = [], []
    example_rows: list[dict] = []

    for i in range(0, len(idxs), args.batch_size):
        batch_indices = idxs[i : i + args.batch_size]
        batch = [ds[j] for j in batch_indices]
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
            out_zero = model(
                zero_control(brain_seq, control_slice),
                brain_mask,
                dec_in,
                decoder_attention_mask=dec_attn,
                labels=labels,
            )
            nll_zero.extend(per_example_nll(out_zero.logits, labels).cpu().tolist())

            # SHUF
            if args.shuf_mode == "batch_global":
                perm = torch.randperm(brain_seq.size(0), device=device)
                shuf_brain_seq = shuffled_control(brain_seq, perm, control_slice)
                shuf_mask = brain_mask if control_slice is not None else brain_mask[perm]
            elif args.shuf_mode in {"global_sample", "within_group"}:
                assert shuffle_plan is not None
                donor_positions = shuffle_plan["donor_positions"][i : i + len(batch_indices)]
                donor_dataset_indices = [idxs[int(pos)] for pos in donor_positions]
                donor_batch = [ds[j] for j in donor_dataset_indices]
                donor_collated = meg_batch_collator(
                    donor_batch,
                    pad_id=0,
                    max_decoder_len=args.max_text_len,
                    decoder_context_mode=args.decoder_context_mode,
                    brain_norm=args.brain_norm,
                )
                donor_brain_seq = donor_collated["brain_seq"].to(device)
                donor_brain_mask = donor_collated["brain_mask"].to(device)
                if args.max_brain_len is not None and donor_brain_seq.size(1) > args.max_brain_len:
                    donor_brain_seq = donor_brain_seq[:, : args.max_brain_len]
                    donor_brain_mask = donor_brain_mask[:, : args.max_brain_len]
                donor_brain_seq = match_time_dim(donor_brain_seq, brain_seq.size(1))
                donor_brain_mask = match_mask_dim(donor_brain_mask, brain_mask.size(1))
                shuf_brain_seq = splice_control(brain_seq, donor_brain_seq, control_slice)
                shuf_mask = donor_brain_mask if control_slice is None else brain_mask
            elif args.shuf_mode == "circular_time_shift":
                shuf_brain_seq = circular_time_shift_control(brain_seq, brain_mask, control_slice, temporal_rng)
                shuf_mask = brain_mask
            elif args.shuf_mode == "block_permute":
                shuf_brain_seq = block_permute_control(
                    brain_seq, brain_mask, control_slice, temporal_rng, args.shuf_block_size
                )
                shuf_mask = brain_mask
            elif args.shuf_mode == "phase_randomized":
                shuf_brain_seq = phase_randomized_control(brain_seq, brain_mask, control_slice, temporal_rng)
                shuf_mask = brain_mask
            else:
                raise ValueError(f"unknown shuf_mode: {args.shuf_mode}")
            out_shuf = model(
                shuf_brain_seq,
                shuf_mask,
                dec_in,
                decoder_attention_mask=dec_attn,
                labels=labels,
            )
            nll_shuf.extend(per_example_nll(out_shuf.logits, labels).cpu().tolist())

            # JS/top-1 on the last supervised target token.
            logits_real = last_valid_token_probs(out_real.logits, labels)
            logits_zero = last_valid_token_probs(out_zero.logits, labels)
            logits_shuf = last_valid_token_probs(out_shuf.logits, labels)
            batch_js_real = js_div(logits_real, logits_zero).cpu().tolist()
            batch_js_shuf = js_div(logits_shuf, logits_zero).cpu().tolist()
            batch_top1_real_zero = (
                (logits_real.argmax(dim=-1) == logits_zero.argmax(dim=-1)).float().cpu().tolist()
            )
            batch_top1_shuf_zero = (
                (logits_shuf.argmax(dim=-1) == logits_zero.argmax(dim=-1)).float().cpu().tolist()
            )
            first_real = first_valid_token_stats(out_real.logits, labels)
            first_zero = first_valid_token_stats(out_zero.logits, labels)
            first_shuf = first_valid_token_stats(out_shuf.logits, labels)
            js_real.extend(batch_js_real)
            js_shuf.extend(batch_js_shuf)
            top1_real_zero.extend(batch_top1_real_zero)
            top1_shuf_zero.extend(batch_top1_shuf_zero)

        if args.out_examples_jsonl is not None:
            batch_nll_real = per_example_nll(out_real.logits, labels).cpu().tolist()
            batch_nll_zero = per_example_nll(out_zero.logits, labels).cpu().tolist()
            batch_nll_shuf = per_example_nll(out_shuf.logits, labels).cpu().tolist()
            for local_idx, (dataset_idx, item) in enumerate(zip(batch_indices, batch)):
                meta = item.get("meta") or {}
                nll_r = float(batch_nll_real[local_idx])
                nll_z = float(batch_nll_zero[local_idx])
                nll_s = float(batch_nll_shuf[local_idx])
                row = {
                    "index": int(dataset_idx),
                    "subject": meta.get("subject"),
                    "story": meta.get("story"),
                    "session": meta.get("session"),
                    "task": meta.get("task"),
                    "sound": meta.get("sound"),
                    "sequence_id": meta.get("sequence_id"),
                    "word_index": meta.get("word_index"),
                    "target_text": meta.get("target_text"),
                    "onset_sec": meta.get("onset_sec"),
                    "first_token_id": first_real["first_token_id"][local_idx],
                    "target_len": int(first_real["target_len"][local_idx]),
                    "nll_real": nll_r,
                    "nll_zero": nll_z,
                    "nll_shuf": nll_s,
                    "delta_real_zero": nll_r - nll_z,
                    "delta_real_shuf": nll_r - nll_s,
                    "first_token_prob_real": first_real["first_token_prob"][local_idx],
                    "first_token_prob_zero": first_zero["first_token_prob"][local_idx],
                    "first_token_prob_shuf": first_shuf["first_token_prob"][local_idx],
                    "first_token_rank_real": first_real["first_token_rank"][local_idx],
                    "first_token_rank_zero": first_zero["first_token_rank"][local_idx],
                    "first_token_rank_shuf": first_shuf["first_token_rank"][local_idx],
                    "js_real": float(batch_js_real[local_idx]),
                    "js_shuf": float(batch_js_shuf[local_idx]),
                    "top1_real_zero": float(batch_top1_real_zero[local_idx]),
                    "top1_shuf_zero": float(batch_top1_shuf_zero[local_idx]),
                }
                example_rows.append(row)

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
        "control_feature_group": args.control_feature_group,
        "control_feature_start": None if control_slice is None else int(control_slice.start),
        "control_feature_end": None if control_slice is None else int(control_slice.stop),
        "shuf_mode": args.shuf_mode,
        "shuf_group_keys": shuf_group_keys,
        "shuf_block_size": int(args.shuf_block_size),
    }
    if shuffle_plan is not None:
        result["shuf_plan"] = {
            k: v
            for k, v in shuffle_plan.items()
            if k != "donor_positions"
        }
    log(str(result))
    save_json(args.out_json, result)
    if args.out_examples_jsonl is not None:
        save_jsonl(args.out_examples_jsonl, example_rows)
        log(f"wrote {len(example_rows)} per-example rows to {args.out_examples_jsonl}")


if __name__ == "__main__":
    main()

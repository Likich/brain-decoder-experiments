#!/usr/bin/env python3
"""Evaluate sensor-group ablations for a held-out MEG-conditioned T5 model.

This script keeps the existing REAL/ZERO/SHUF evaluation logic, but zeros a
specified subset of MEG channels across all three conditions before scoring.
It is intended as a lightweight interpretability diagnostic for the story-
blocked MEG-MASC result: if ablating one region weakens the paired-control
gain more than ablating another or a random matched-size subset, that suggests
the model is reading non-uniform sensor structure rather than arbitrary input
energy.

Channel groups are derived from a single raw MEG header using coarse spatial
heuristics in head coordinates:
  - left/right hemispheres from x sign
  - frontal / occipital from anterior-posterior y quartiles
  - left/right temporal from lateral, lower, mid-y sensors with relaxed
    thresholds if needed

The output JSON records the derived channel groups, the baseline metrics, and
the ablated metrics for each group.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_text_pipeline.src.data.collators import meg_batch_collator
from brain_text_pipeline.src.data.datasets import ShardedExampleDataset
from brain_text_pipeline.src.data.meg_masc import headshape_paths, markers_path, meg_con_path
from brain_text_pipeline.src.eval.metrics import js_div
from brain_text_pipeline.src.models.t5_brain_model import T5BrainModel
from brain_text_pipeline.src.utils.logging import log, save_json

try:
    import mne
except ImportError:  # pragma: no cover
    mne = None


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


def last_valid_token_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels.ne(-100)
    positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
    last_pos = (positions * valid.long()).amax(dim=1)
    batch_idx = torch.arange(labels.size(0), device=labels.device)
    return logits[batch_idx, last_pos, :].softmax(dim=-1)


def paired_summary(diff: np.ndarray) -> dict[str, float]:
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


def resolve_decoder_start_id(model: T5BrainModel, pad_id: int) -> int:
    start_id = getattr(model.t5.config, "decoder_start_token_id", None)
    if start_id is None and hasattr(model.t5, "generation_config"):
        start_id = getattr(model.t5.generation_config, "decoder_start_token_id", None)
    if start_id is None:
        start_id = pad_id
    if start_id is None:
        start_id = getattr(model.t5.config, "eos_token_id", None)
    if start_id is None and hasattr(model.t5, "generation_config"):
        start_id = getattr(model.t5.generation_config, "eos_token_id", None)
    if start_id is None:
        raise ValueError("Could not determine decoder start token id")
    return int(start_id)


def ensure_txt(path: Path) -> Path:
    if path.suffix in {".hsp", ".elp", ".txt", ".mat"}:
        return path
    if path.suffix == ".pos":
        txt_path = path.with_suffix(".txt")
        if not txt_path.exists():
            txt_path.write_bytes(path.read_bytes())
        return txt_path
    return path


def parse_groups(text: str) -> list[str]:
    groups = [part.strip() for part in text.split(",") if part.strip()]
    if not groups:
        raise ValueError("--groups must contain at least one group name")
    return groups


def infer_layout_triplet(ds: ShardedExampleDataset, indices: list[int]) -> tuple[str, str, str]:
    probe = ds[indices[0] if indices else 0]
    meta = probe.get("meta") or {}
    subject = meta.get("subject")
    session = meta.get("session")
    task = meta.get("task")
    if not subject or not session or not task:
        raise ValueError("dataset examples are missing subject/session/task metadata")
    return str(subject), str(session), str(task)


def load_meg_layout(meg_root: Path, subject: str, session: str, task: str) -> tuple[list[str], np.ndarray, list[str]]:
    if mne is None:
        raise RuntimeError("mne is required for sensor-group ablations")
    con_path = meg_con_path(meg_root, subject, session, task)
    mrk_path = markers_path(meg_root, subject, session, task)
    hsp_path, elp_path = headshape_paths(meg_root, subject, session)
    if not con_path.exists():
        raise FileNotFoundError(f"missing raw MEG file: {con_path}")
    raw = mne.io.read_raw_kit(
        con_path,
        mrk=mrk_path,
        elp=ensure_txt(elp_path),
        hsp=ensure_txt(hsp_path),
        preload=False,
        verbose="ERROR",
    )
    raw.pick("meg")
    ch_names = list(raw.info["ch_names"])
    ch_pos = []
    ch_types = []
    for idx, ch in enumerate(raw.info["chs"]):
        ch_pos.append(np.asarray(ch["loc"][:3], dtype=np.float64))
        ch_types.append(mne.channel_type(raw.info, idx))
    pos = np.stack(ch_pos, axis=0)
    if not np.isfinite(pos).all():
        raise ValueError("non-finite channel positions encountered in raw MEG header")
    return ch_names, pos, ch_types


def build_temporal_mask(x: np.ndarray, y: np.ndarray, z: np.ndarray, side: str, min_group_size: int) -> tuple[np.ndarray, dict[str, float]]:
    abs_x = np.abs(x)
    side_mask = x < 0 if side == "left" else x > 0
    plans = [
        {"abs_x_q": 0.60, "z_q": 0.50, "y_low_q": 0.25, "y_high_q": 0.75},
        {"abs_x_q": 0.55, "z_q": 0.60, "y_low_q": 0.20, "y_high_q": 0.80},
        {"abs_x_q": 0.50, "z_q": 0.65, "y_low_q": 0.15, "y_high_q": 0.85},
        {"abs_x_q": 0.45, "z_q": 0.70, "y_low_q": 0.10, "y_high_q": 0.90},
    ]
    last_mask = side_mask.copy()
    last_plan = plans[-1]
    for plan in plans:
        abs_thr = float(np.quantile(abs_x, plan["abs_x_q"]))
        z_thr = float(np.quantile(z, plan["z_q"]))
        y_low = float(np.quantile(y, plan["y_low_q"]))
        y_high = float(np.quantile(y, plan["y_high_q"]))
        mask = side_mask & (abs_x >= abs_thr) & (z <= z_thr) & (y >= y_low) & (y <= y_high)
        last_mask = mask
        last_plan = {
            "abs_x_threshold": abs_thr,
            "z_threshold": z_thr,
            "y_low_threshold": y_low,
            "y_high_threshold": y_high,
        }
        if int(mask.sum()) >= min_group_size:
            return mask, last_plan
    return last_mask, last_plan


def derive_sensor_groups(
    *,
    ch_names: list[str],
    ch_pos: np.ndarray,
    ch_types: list[str],
    requested_groups: list[str],
    min_group_size: int,
    random_match_group: str | None,
    random_match_repeats: int,
    seed: int,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    x = ch_pos[:, 0]
    y = ch_pos[:, 1]
    z = ch_pos[:, 2]
    groups: dict[str, list[int]] = {}
    metadata: dict[str, Any] = {
        "n_channels": int(len(ch_names)),
        "sensor_type_counts": dict(Counter(ch_types)),
        "channel_names": ch_names,
        "heuristics": {},
    }

    frontal_mask = y >= np.quantile(y, 0.75)
    occipital_mask = y <= np.quantile(y, 0.25)
    left_mask = x < 0
    right_mask = x > 0
    left_temporal_mask, left_temporal_meta = build_temporal_mask(x, y, z, side="left", min_group_size=min_group_size)
    right_temporal_mask, right_temporal_meta = build_temporal_mask(x, y, z, side="right", min_group_size=min_group_size)
    metadata["heuristics"]["left_temporal"] = left_temporal_meta
    metadata["heuristics"]["right_temporal"] = right_temporal_meta
    metadata["heuristics"]["frontal_y_threshold"] = float(np.quantile(y, 0.75))
    metadata["heuristics"]["occipital_y_threshold"] = float(np.quantile(y, 0.25))

    mask_map: dict[str, np.ndarray] = {
        "left": left_mask,
        "right": right_mask,
        "frontal": frontal_mask,
        "occipital": occipital_mask,
        "left_temporal": left_temporal_mask,
        "right_temporal": right_temporal_mask,
    }
    if int(left_temporal_mask.sum()) >= 2 and int(right_temporal_mask.sum()) >= 2:
        mask_map["bilateral_temporal"] = left_temporal_mask | right_temporal_mask

    type_to_indices: dict[str, list[int]] = {}
    for idx, typ in enumerate(ch_types):
        type_to_indices.setdefault(str(typ), []).append(idx)
    if len(type_to_indices) > 1:
        for typ, idxs in type_to_indices.items():
            mask_map[f"type_{typ}"] = np.isin(np.arange(len(ch_names)), np.asarray(idxs))

    missing = [name for name in requested_groups if name not in mask_map]
    if missing:
        raise ValueError(f"unknown sensor group(s): {missing}. Available groups: {sorted(mask_map)}")

    for name in requested_groups:
        idxs = np.flatnonzero(mask_map[name]).tolist()
        if len(idxs) < 2:
            log(f"skip group {name}: only {len(idxs)} channel(s)")
            continue
        groups[name] = idxs

    if random_match_group:
        if random_match_group not in groups:
            raise ValueError(f"--random_match_group={random_match_group!r} is not available after group derivation")
        rng = np.random.default_rng(seed)
        size = len(groups[random_match_group])
        universe = np.arange(len(ch_names))
        base = np.asarray(groups[random_match_group], dtype=np.int64)
        complement = np.setdiff1d(universe, base, assume_unique=False)
        if complement.size < size:
            complement = universe
        for rep in range(random_match_repeats):
            choice = rng.choice(complement, size=size, replace=False)
            groups[f"random_match_{random_match_group}_{rep + 1}"] = choice.tolist()

    metadata["available_groups"] = {name: len(idxs) for name, idxs in groups.items()}
    return groups, metadata


def apply_channel_ablation(brain_seq: torch.Tensor, channel_indices: list[int]) -> torch.Tensor:
    if not channel_indices:
        return brain_seq
    out = brain_seq.clone()
    out[:, :, channel_indices] = 0.0
    return out


def evaluate_group(
    *,
    model: T5BrainModel,
    loader: DataLoader,
    device: torch.device,
    max_brain_len: int | None,
    channel_indices: list[int],
    batch_perm_rng: np.random.Generator,
) -> dict[str, Any]:
    nll_real: list[float] = []
    nll_zero: list[float] = []
    nll_shuf: list[float] = []
    js_real: list[float] = []
    js_shuf: list[float] = []
    top1_real_zero = 0
    top1_shuf_zero = 0
    total = 0

    for collated in loader:
        brain_seq = collated["brain_seq"].to(device)
        brain_mask = collated["brain_mask"].to(device)
        dec_in = collated["decoder_input_ids"].to(device)
        dec_attn = collated["decoder_attention_mask"].to(device)
        labels = collated["labels"].to(device)
        if max_brain_len is not None and brain_seq.size(1) > max_brain_len:
            brain_seq = brain_seq[:, :max_brain_len]
            brain_mask = brain_mask[:, :max_brain_len]

        brain_real = apply_channel_ablation(brain_seq, channel_indices)
        brain_zero = torch.zeros_like(brain_real)
        perm = np.arange(brain_real.size(0), dtype=np.int64)
        if perm.size > 1:
            perm = batch_perm_rng.permutation(perm.size)
        perm_t = torch.tensor(perm, device=device, dtype=torch.long)
        brain_shuf = brain_real[perm_t]

        with torch.no_grad():
            out_real = model(brain_real, brain_mask, dec_in, decoder_attention_mask=dec_attn, labels=labels)
            out_zero = model(brain_zero, brain_mask, dec_in, decoder_attention_mask=dec_attn, labels=labels)
            out_shuf = model(brain_shuf, brain_mask, dec_in, decoder_attention_mask=dec_attn, labels=labels)

        nll_r = per_example_nll(out_real.logits, labels).detach().cpu().numpy()
        nll_z = per_example_nll(out_zero.logits, labels).detach().cpu().numpy()
        nll_s = per_example_nll(out_shuf.logits, labels).detach().cpu().numpy()
        nll_real.extend(nll_r.tolist())
        nll_zero.extend(nll_z.tolist())
        nll_shuf.extend(nll_s.tolist())

        p_real = last_valid_token_probs(out_real.logits, labels)
        p_zero = last_valid_token_probs(out_zero.logits, labels)
        p_shuf = last_valid_token_probs(out_shuf.logits, labels)
        js_real.extend(js_div(p_real, p_zero).detach().cpu().numpy().tolist())
        js_shuf.extend(js_div(p_shuf, p_zero).detach().cpu().numpy().tolist())
        top1_real_zero += int((p_real.argmax(dim=-1) == p_zero.argmax(dim=-1)).sum().item())
        top1_shuf_zero += int((p_shuf.argmax(dim=-1) == p_zero.argmax(dim=-1)).sum().item())
        total += int(labels.size(0))

    arr_real = np.asarray(nll_real, dtype=np.float64)
    arr_zero = np.asarray(nll_zero, dtype=np.float64)
    arr_shuf = np.asarray(nll_shuf, dtype=np.float64)
    diff_rz = arr_real - arr_zero
    diff_rs = arr_real - arr_shuf
    return {
        "n": int(arr_real.size),
        "nll_real": float(arr_real.mean()),
        "nll_zero": float(arr_zero.mean()),
        "nll_shuf": float(arr_shuf.mean()),
        "delta_real_zero": float(diff_rz.mean()),
        "delta_real_shuf": float(diff_rs.mean()),
        "delta_real_zero_paired": paired_summary(diff_rz),
        "delta_real_shuf_paired": paired_summary(diff_rs),
        "js_real": float(np.mean(js_real)) if js_real else 0.0,
        "js_shuf": float(np.mean(js_shuf)) if js_shuf else 0.0,
        "top1_real_zero": float(top1_real_zero / max(total, 1)),
        "top1_shuf_zero": float(top1_shuf_zero / max(total, 1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--brain_encoder_ckpt", type=Path, required=True)
    ap.add_argument("--meg_dataset_path", type=Path, required=True)
    ap.add_argument("--meg_root", type=Path, required=True, help="MEG-MASC root used to load one raw header")
    ap.add_argument("--layout_subject", type=str, default=None)
    ap.add_argument("--layout_session", type=str, default=None)
    ap.add_argument("--layout_task", type=str, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_text_len", type=int, default=8)
    ap.add_argument("--max_brain_len", type=int, default=120)
    ap.add_argument("--decoder_context_mode", choices=["context_target", "target_only"], default="target_only")
    ap.add_argument("--brain_norm", choices=["none", "per_example"], default="per_example")
    ap.add_argument(
        "--groups",
        type=str,
        default="left_temporal,right_temporal,frontal,occipital,left,right",
        help="Comma-separated sensor groups to ablate",
    )
    ap.add_argument("--min_group_size", type=int, default=8)
    ap.add_argument(
        "--random_match_group",
        type=str,
        default="left_temporal",
        help="Add matched-size random ablations for this group; set empty string to disable",
    )
    ap.add_argument("--random_match_repeats", type=int, default=3)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    device = resolve_device(args.device)
    ds = ShardedExampleDataset(args.meg_dataset_path)
    indices = list(range(len(ds)))
    if args.samples is not None and args.samples < len(indices):
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(indices, size=args.samples, replace=False).tolist()
    subset = Subset(ds, indices)

    sample = ds[indices[0] if indices else 0]["brain_seq"]
    brain_dim = int(np.asarray(sample).shape[1])

    subject = args.layout_subject
    session = args.layout_session
    task = args.layout_task
    if not (subject and session and task):
        infer_sub, infer_ses, infer_task = infer_layout_triplet(ds, indices)
        subject = subject or infer_sub
        session = session or infer_ses
        task = task or infer_task

    ch_names, ch_pos, ch_types = load_meg_layout(args.meg_root, subject, session, task)
    if len(ch_names) != brain_dim:
        raise ValueError(
            f"raw MEG header has {len(ch_names)} channels, but dataset brain dim is {brain_dim}; "
            "make sure the dataset was built from the same preprocessed channel set"
        )

    requested_groups = parse_groups(args.groups)
    random_match_group = args.random_match_group.strip() if args.random_match_group else None
    groups, group_meta = derive_sensor_groups(
        ch_names=ch_names,
        ch_pos=ch_pos,
        ch_types=ch_types,
        requested_groups=requested_groups,
        min_group_size=args.min_group_size,
        random_match_group=random_match_group,
        random_match_repeats=args.random_match_repeats,
        seed=args.seed,
    )
    if not groups:
        raise ValueError("no usable sensor groups were derived")

    model = T5BrainModel(model_name_or_path=args.model_name_or_path, brain_dim=brain_dim)
    state = torch.load(args.brain_encoder_ckpt, map_location="cpu")
    model.brain_encoder.load_state_dict(state)
    model.to(device)
    model.eval()

    pad_id = model.t5.config.pad_token_id
    if pad_id is None:
        pad_id = 0
    decoder_start_id = resolve_decoder_start_id(model, pad_id)
    collate = lambda batch: meg_batch_collator(  # noqa: E731
        batch,
        pad_id,
        decoder_start_id=decoder_start_id,
        max_decoder_len=args.max_text_len,
        decoder_context_mode=args.decoder_context_mode,
        brain_norm=args.brain_norm,
    )
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    baseline_rng = np.random.default_rng(args.seed)
    baseline = evaluate_group(
        model=model,
        loader=loader,
        device=device,
        max_brain_len=args.max_brain_len,
        channel_indices=[],
        batch_perm_rng=baseline_rng,
    )
    log(
        "baseline: "
        f"dRZ={baseline['delta_real_zero']:.4f} "
        f"dRS={baseline['delta_real_shuf']:.4f}"
    )

    rows: list[dict[str, Any]] = []
    for row_idx, (name, channel_indices) in enumerate(groups.items()):
        metrics = evaluate_group(
            model=model,
            loader=loader,
            device=device,
            max_brain_len=args.max_brain_len,
            channel_indices=channel_indices,
            batch_perm_rng=np.random.default_rng(args.seed + 1000 + row_idx),
        )
        metrics.update(
            {
                "group": name,
                "n_channels": int(len(channel_indices)),
                "channel_fraction": float(len(channel_indices) / max(brain_dim, 1)),
                "channels": [ch_names[i] for i in channel_indices],
                "effect_loss_real_zero": float(metrics["delta_real_zero"] - baseline["delta_real_zero"]),
                "effect_loss_real_shuf": float(metrics["delta_real_shuf"] - baseline["delta_real_shuf"]),
            }
        )
        rows.append(metrics)
        log(
            f"{name}: n_ch={len(channel_indices)} "
            f"dRZ={metrics['delta_real_zero']:.4f} "
            f"dRS={metrics['delta_real_shuf']:.4f} "
            f"loss_vs_base=({metrics['effect_loss_real_zero']:+.4f}, {metrics['effect_loss_real_shuf']:+.4f})"
        )

    payload = {
        "meg_dataset_path": str(args.meg_dataset_path),
        "meg_root": str(args.meg_root),
        "layout_subject": subject,
        "layout_session": session,
        "layout_task": task,
        "seed": int(args.seed),
        "samples": int(len(indices)),
        "decoder_context_mode": args.decoder_context_mode,
        "brain_norm": args.brain_norm,
        "groups_requested": requested_groups,
        "group_metadata": group_meta,
        "baseline": baseline,
        "groups": rows,
    }
    save_json(args.out_json, payload)
    log(f"wrote sensor ablations to {args.out_json}")


if __name__ == "__main__":
    main()

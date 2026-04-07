#!/usr/bin/env python3
"""
Filtered Logit PCA: focus on contexts where brain helps (or shuffled hurts).

Pipeline:
 1) For each sample, compute delta logits vs zero and delta NLL:
      delta_real = logits(x, z_real) - logits(x, z_zero)
      delta_shuf = logits(x, z_shuf) - logits(x, z_zero)
      dNLL_real = NLL(real) - NLL(zero)
      dNLL_shuf = NLL(shuf) - NLL(zero)
 2) Select "helpful" indices where dNLL_real < -real_thresh.
    (Optional) select "shuf_harm" where dNLL_shuf > shuf_thresh.
 3) Run PCA on delta matrices restricted to those indices and report:
      cumulative variance @1/2/5/10, effective rank.

Supports vocab subsampling to keep matrices small.

Example:
  python3 scripts/logit_pca_filtered.py \
    --ckpt models/lm_brain272/language_model.pt \
    --tokenizer models/wiki_tokenizer.json \
    --brain_dataset data/brain_ctx_pairs_272.npz \
    --hidden_dim 384 --num_layers 2 --attn_heads 8 --dropout 0.11049 \
    --block_size 96 --samples 5000 --device cuda \
    --vocab_topk 2000 --vocab_seed 42 \
    --real_thresh 0.1 --shuf_thresh 0.1 \
    --out_csv logit_pca_filtered_272.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

try:
    from sklearn.decomposition import PCA
except ImportError as e:  # pragma: no cover
    raise SystemExit("scikit-learn is required for logit PCA. Please install sklearn.") from e

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_language_model import LanguageModel, BrainCrossAttentionLM  # noqa: E402


def resolve_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crop_pad(tokens, block_size: int, pad_id: int = 0):
    tokens = tokens[-block_size:]
    if len(tokens) < block_size:
        tokens = [pad_id] * (block_size - len(tokens)) + tokens
    return tokens


def effective_rank(eigvals: np.ndarray) -> float:
    eigvals = eigvals.astype(np.float64)
    p = eigvals / (eigvals.sum() + 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def summarize(name: str, pca_obj: PCA, v_used: int, n_samples: int) -> dict:
    evr = pca_obj.explained_variance_ratio_
    eigvals = pca_obj.explained_variance_
    cum = np.cumsum(evr)
    erank = effective_rank(eigvals)

    def cv(k: int) -> float:
        return cum[min(k - 1, len(cum) - 1)] if len(cum) else 0.0

    print(
        f"{name}: n={n_samples}, V_used={v_used}, comps={len(evr)}, "
        f"cumvar@1={cv(1):.4f} @2={cv(2):.4f} @5={cv(5):.4f} @10={cv(10):.4f} "
        f"| eff_rank={erank:.2f}"
    )
    return {"cum1": cv(1), "cum2": cv(2), "cum5": cv(5), "cum10": cv(10), "eff_rank": erank}


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Trained language_model.pt")
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    ap.add_argument("--brain_dataset", required=True, help="NPZ with contexts/brain/targets")
    ap.add_argument("--hidden_dim", type=int, default=384)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--attn_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.11)
    ap.add_argument("--block_size", type=int, default=96)
    ap.add_argument("--pad_token_id", type=int, default=0)
    ap.add_argument("--samples", type=int, default=5000, help="Number of samples to evaluate")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--vocab_topk", type=int, default=None, help="Optional vocab subsample size")
    ap.add_argument("--vocab_seed", type=int, default=123, help="Seed for vocab subsampling")
    ap.add_argument("--real_thresh", type=float, default=0.1, help="ΔNLL_real < -real_thresh => helpful")
    ap.add_argument("--shuf_thresh", type=float, default=0.1, help="ΔNLL_shuf > shuf_thresh => harmful")
    ap.add_argument(
        "--brain_fusion",
        type=str,
        default="add",
        choices=["add", "cross_attn"],
        help="Brain conditioning mechanism",
    )
    ap.add_argument("--brain_tokens", type=int, default=4, help="Number of brain memory tokens (cross_attn)")
    ap.add_argument("--out_csv", type=str, default=None, help="Optional CSV summary output")
    args = ap.parse_args()

    device = resolve_device(args.device)

    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    data = np.load(args.brain_dataset, allow_pickle=True)
    contexts = data["contexts"].tolist()
    brain = torch.tensor(data["brain"], dtype=torch.float32)
    targets = torch.tensor(data["targets"], dtype=torch.long)
    brain_dim = brain.shape[1]

    if args.brain_fusion == "cross_attn":
        model = BrainCrossAttentionLM(
            vocab_size=vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            attn_heads=args.attn_heads,
            dropout=args.dropout,
            brain_dim=brain_dim,
            brain_tokens=args.brain_tokens,
            pad_token_id=args.pad_token_id,
        ).to(device)
    else:
        model = LanguageModel(
            vocab_size=vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            attn_heads=args.attn_heads,
            dropout=args.dropout,
            brain_dim=brain_dim,
            pad_token_id=args.pad_token_id,
        ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Vocab mask
    if args.vocab_topk is not None and args.vocab_topk < vocab_size:
        rng = np.random.default_rng(args.vocab_seed)
        vocab_mask = np.sort(rng.choice(vocab_size, size=args.vocab_topk, replace=False))
    else:
        vocab_mask = None
    v_used = len(vocab_mask) if vocab_mask is not None else vocab_size

    subset = min(args.samples, len(contexts))
    idxs = torch.randperm(len(contexts))[:subset]
    shuffle_map = torch.randperm(len(contexts))

    iterator = idxs.tolist()
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Collecting deltas", leave=False)

    delta_real = []
    delta_shuf = []
    dNLL_real = []
    dNLL_shuf = []

    with torch.no_grad():
        for i in iterator:
            ctx_tokens = crop_pad(contexts[i], args.block_size, args.pad_token_id)
            x = torch.tensor(ctx_tokens, dtype=torch.long, device=device).unsqueeze(0)
            y = targets[i].view(1, 1).to(device)

            z_zero = torch.zeros((1, brain_dim), device=device)
            z_real = brain[i].unsqueeze(0).to(device)
            z_shuf = brain[shuffle_map[i]].unsqueeze(0).to(device)

            logits_zero = model(x, z_zero)[:, -1, :]
            logits_real = model(x, z_real)[:, -1, :]
            logits_shuf = model(x, z_shuf)[:, -1, :]

            logp_zero = torch.log_softmax(logits_zero, dim=-1)
            logp_real = torch.log_softmax(logits_real, dim=-1)
            logp_shuf = torch.log_softmax(logits_shuf, dim=-1)

            nll_zero = -logp_zero.gather(-1, y).squeeze().item()
            nll_real = -logp_real.gather(-1, y).squeeze().item()
            nll_shuf = -logp_shuf.gather(-1, y).squeeze().item()

            dNLL_real.append(nll_real - nll_zero)
            dNLL_shuf.append(nll_shuf - nll_zero)

            dr = (logits_real - logits_zero).squeeze(0)
            ds = (logits_shuf - logits_zero).squeeze(0)
            if vocab_mask is not None:
                dr = dr[vocab_mask]
                ds = ds[vocab_mask]
            delta_real.append(dr.cpu().numpy())
            delta_shuf.append(ds.cpu().numpy())

    dNLL_real = np.array(dNLL_real)
    dNLL_shuf = np.array(dNLL_shuf)
    delta_real = np.stack(delta_real, axis=0)
    delta_shuf = np.stack(delta_shuf, axis=0)

    # Filter indices
    help_idx = np.where(dNLL_real < -args.real_thresh)[0]
    shuf_harm_idx = np.where(dNLL_shuf > args.shuf_thresh)[0]

    def run_pca(mat: np.ndarray, name: str):
        if mat.shape[0] < 2:
            print(f"{name}: not enough samples ({mat.shape[0]}) to run PCA.")
            return None
        mat_centered = mat - mat.mean(axis=0, keepdims=True)
        n_comp = min(20, mat_centered.shape[0], mat_centered.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(mat_centered)
        return summarize(name, pca, v_used, mat.shape[0])

    print(f"Samples total: {subset}, vocab used: {v_used}")
    print(f"Helpful (real ΔNLL < -{args.real_thresh}): {len(help_idx)}")
    print(f"Harmful shuf (ΔNLL_shuf > {args.shuf_thresh}): {len(shuf_harm_idx)}")

    stats_real_help = run_pca(delta_real[help_idx], "REAL_helpful")
    stats_shuf_on_help = run_pca(delta_shuf[help_idx], "SHUF_on_helpful")
    stats_shuf_harm = run_pca(delta_shuf[shuf_harm_idx], "SHUF_harmful") if len(shuf_harm_idx) else None

    if args.out_csv:
        import csv

        fieldnames = [
            "condition",
            "subset",
            "vocab_used",
            "samples",
            "cumvar1",
            "cumvar2",
            "cumvar5",
            "cumvar10",
            "eff_rank",
            "real_thresh",
            "shuf_thresh",
        ]
        rows = []
        if stats_real_help:
            rows.append(
                {
                    "condition": "real",
                    "subset": "helpful",
                    "vocab_used": v_used,
                    "samples": len(help_idx),
                    "cumvar1": stats_real_help["cum1"],
                    "cumvar2": stats_real_help["cum2"],
                    "cumvar5": stats_real_help["cum5"],
                    "cumvar10": stats_real_help["cum10"],
                    "eff_rank": stats_real_help["eff_rank"],
                    "real_thresh": args.real_thresh,
                    "shuf_thresh": args.shuf_thresh,
                }
            )
        if stats_shuf_on_help:
            rows.append(
                {
                    "condition": "shuf",
                    "subset": "on_helpful",
                    "vocab_used": v_used,
                    "samples": len(help_idx),
                    "cumvar1": stats_shuf_on_help["cum1"],
                    "cumvar2": stats_shuf_on_help["cum2"],
                    "cumvar5": stats_shuf_on_help["cum5"],
                    "cumvar10": stats_shuf_on_help["cum10"],
                    "eff_rank": stats_shuf_on_help["eff_rank"],
                    "real_thresh": args.real_thresh,
                    "shuf_thresh": args.shuf_thresh,
                }
            )
        if stats_shuf_harm:
            rows.append(
                {
                    "condition": "shuf",
                    "subset": "harmful",
                    "vocab_used": v_used,
                    "samples": len(shuf_harm_idx),
                    "cumvar1": stats_shuf_harm["cum1"],
                    "cumvar2": stats_shuf_harm["cum2"],
                    "cumvar5": stats_shuf_harm["cum5"],
                    "cumvar10": stats_shuf_harm["cum10"],
                    "eff_rank": stats_shuf_harm["eff_rank"],
                    "real_thresh": args.real_thresh,
                    "shuf_thresh": args.shuf_thresh,
                }
            )

        if rows:
            with open(args.out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote summary to {args.out_csv}")


if __name__ == "__main__":
    main()

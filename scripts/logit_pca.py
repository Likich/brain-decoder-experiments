#!/usr/bin/env python3
"""
Logit PCA analysis: does brain conditioning act through a low-dimensional subspace?

For each sample, compute delta logits at the final position:
    delta = logits(z) - logits(0)
for REAL (paired brain) and SHUF (shuffled brain), stack across samples, and run PCA.

Reports explained variance, cumulative variance, and effective rank for REAL vs SHUF.
Supports optional vocab subsampling to keep matrices manageable.

Example:
  python3 scripts/logit_pca.py \
    --ckpt models/lm_brain272/language_model.pt \
    --tokenizer models/wiki_tokenizer.json \
    --brain_dataset data/brain_ctx_pairs_272.npz \
    --hidden_dim 384 --num_layers 2 --attn_heads 8 --dropout 0.11049 \
    --block_size 96 --samples 5000 --device cuda \
    --vocab_topk 2000 --vocab_seed 42 \
    --out_csv logit_pca_272.csv
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

from scripts.train_language_model import LanguageModel  # noqa: E402


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
    ap.add_argument("--vocab_topk", type=int, default=None, help="Optional vocab subsample size (fixed random)")
    ap.add_argument("--vocab_seed", type=int, default=123, help="Seed for vocab subsampling")
    ap.add_argument("--out_csv", type=str, default=None, help="Optional CSV summary output")
    args = ap.parse_args()

    device = resolve_device(args.device)

    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    data = np.load(args.brain_dataset, allow_pickle=True)
    contexts = data["contexts"].tolist()
    brain = torch.tensor(data["brain"], dtype=torch.float32)
    brain_dim = brain.shape[1]

    model = LanguageModel(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        brain_dim=brain_dim,
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

    deltas_real = []
    deltas_shuf = []

    with torch.no_grad():
        for i in iterator:
            ctx_tokens = crop_pad(contexts[i], args.block_size, args.pad_token_id)
            x = torch.tensor(ctx_tokens, dtype=torch.long, device=device).unsqueeze(0)

            z_zero = torch.zeros((1, brain_dim), device=device)
            z_real = brain[i].unsqueeze(0).to(device)
            z_shuf = brain[shuffle_map[i]].unsqueeze(0).to(device)

            logits_zero = model(x, z_zero)[:, -1, :]  # [1, V]
            logits_real = model(x, z_real)[:, -1, :]
            logits_shuf = model(x, z_shuf)[:, -1, :]

            delta_real = (logits_real - logits_zero).squeeze(0)
            delta_shuf = (logits_shuf - logits_zero).squeeze(0)

            if vocab_mask is not None:
                delta_real = delta_real[vocab_mask]
                delta_shuf = delta_shuf[vocab_mask]

            deltas_real.append(delta_real.cpu().numpy())
            deltas_shuf.append(delta_shuf.cpu().numpy())

    D_real = np.stack(deltas_real, axis=0)  # [N, V_used]
    D_shuf = np.stack(deltas_shuf, axis=0)

    # Mean-center columns
    D_real -= D_real.mean(axis=0, keepdims=True)
    D_shuf -= D_shuf.mean(axis=0, keepdims=True)

    n_comp = min(20, v_used, subset)
    pca_real = PCA(n_components=n_comp)
    pca_shuf = PCA(n_components=n_comp)
    pca_real.fit(D_real)
    pca_shuf.fit(D_shuf)

    def summarize(name: str, pca_obj: PCA):
        evr = pca_obj.explained_variance_ratio_
        eigvals = pca_obj.explained_variance_
        cum = np.cumsum(evr)
        erank = effective_rank(eigvals)
        def cv(k): return cum[min(k-1, len(cum)-1)]
        print(
            f"{name}: V_used={v_used}, comps={len(evr)}, "
            f"cumvar@1={cv(1):.4f} @2={cv(2):.4f} @5={cv(5):.4f} @10={cv(10):.4f} "
            f"| eff_rank={erank:.2f}"
        )
        return {
            "cum1": cv(1),
            "cum2": cv(2),
            "cum5": cv(5),
            "cum10": cv(10),
            "eff_rank": erank,
        }

    print(f"Samples: {subset} | Vocab used: {v_used}")
    real_stats = summarize("REAL", pca_real)
    shuf_stats = summarize("SHUF", pca_shuf)

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "condition",
                    "vocab_used",
                    "samples",
                    "cumvar1",
                    "cumvar2",
                    "cumvar5",
                    "cumvar10",
                    "eff_rank",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "condition": "real",
                    "vocab_used": v_used,
                    "samples": subset,
                    "cumvar1": real_stats["cum1"],
                    "cumvar2": real_stats["cum2"],
                    "cumvar5": real_stats["cum5"],
                    "cumvar10": real_stats["cum10"],
                    "eff_rank": real_stats["eff_rank"],
                }
            )
            writer.writerow(
                {
                    "condition": "shuf",
                    "vocab_used": v_used,
                    "samples": subset,
                    "cumvar1": shuf_stats["cum1"],
                    "cumvar2": shuf_stats["cum2"],
                    "cumvar5": shuf_stats["cum5"],
                    "cumvar10": shuf_stats["cum10"],
                    "eff_rank": shuf_stats["eff_rank"],
                }
            )
        print(f"Wrote summary to {args.out_csv}")


if __name__ == "__main__":
    main()

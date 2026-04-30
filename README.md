# Brain Decoder Experiments

This repository now has two distinct tracks:

1. `brain_text_pipeline/`
   The active MEG/T5 pipeline used for paired-control evaluation of brain-conditioned language modeling.

2. `src/lefty_brain_sim/` + `scripts/`
   An older simulator-oriented code path for synthetic TVB-style experiments and local brain-conditioned language-model prototypes.

If you are starting fresh, use `brain_text_pipeline/` first. The legacy simulator code is still useful for synthetic experiments, tokenizer training, and earlier local decoder baselines, but it is no longer the main entry point for the held-out MEG results.

## Main Entry Points

- Active pipeline overview: [`brain_text_pipeline/README.md`](brain_text_pipeline/README.md)
- Full experiment commands: [`brain_text_pipeline/MEG_EXPERIMENT_STEPS.md`](brain_text_pipeline/MEG_EXPERIMENT_STEPS.md)

## Repository Layout

- `brain_text_pipeline/`
  - MEG preprocessing, word-aligned sharded datasets, T5 cross-attention training, paired-control evaluation, stricter SHUF controls, clustered bootstrap, qualitative tables, BERT positive controls, and sensor ablations.
- `scripts/`
  - legacy synthetic/TVB and local LM training utilities.
- `src/lefty_brain_sim/`
  - legacy simulator components for cortical dynamics, gating, memory, and simple report generation.
- `configs/`
  - legacy simulator configuration.
- `data/`, `runs/`, `outputs/`, `models/`
  - local experiment artifacts; these are intentionally ignored by Git.
- `paper/`, `paper_position_neurips2026/`
  - local-only paper directories. These stay on disk but are intentionally not tracked in Git.

## Active MEG Workflow

The current paper-facing workflow is:

1. preprocess MEG-MASC
2. build word-aligned sharded datasets
3. split by random/story/subject/LOSO as needed
4. train a T5 brain cross-attention model with:
   - `decoder_context_mode=target_only`
   - `brain_norm=per_example`
5. evaluate REAL vs ZERO vs SHUF with held-out paired controls
6. run stricter SHUF variants, clustered bootstrap, characterization, and sensor ablations

Use [`brain_text_pipeline/MEG_EXPERIMENT_STEPS.md`](brain_text_pipeline/MEG_EXPERIMENT_STEPS.md) for the exact commands.

## Legacy Simulator Quick Start

If you need the older simulator path:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_experiment.py --config configs/default.yaml
```

Useful legacy components:

- `src/lefty_brain_sim/tvb_iface.py`
- `src/lefty_brain_sim/decision.py`
- `src/lefty_brain_sim/gating.py`
- `src/lefty_brain_sim/memory.py`
- `scripts/build_brain_conditioned_dataset.py`
- `scripts/train_language_model.py`
- `scripts/train_brain_decoder.py`

## Notes

- The paper directories are intentionally excluded from GitHub now. If you need to rebuild the paper locally, use the local files in `paper/`.
- The MEG pipeline assumes local access to MEG-MASC and writes large sharded datasets; keep those artifacts out of version control.
- The most up-to-date experiment instructions live in `brain_text_pipeline/MEG_EXPERIMENT_STEPS.md`, not in old shell history or notebook snippets.

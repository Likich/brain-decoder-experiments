# Brain Decoder Experiments

This repository now has two distinct tracks:

1. `brain_text_pipeline/`
   The active MEG/T5 pipeline used for paired-control evaluation of brain-conditioned language modeling.

2. `src/lefty_brain_sim/` + `scripts/`
   An older simulator-oriented code path for synthetic TVB-style experiments and local brain-conditioned language-model prototypes.

If you are starting fresh, use `brain_text_pipeline/` first. The legacy simulator code is still useful for synthetic experiments, tokenizer training, and earlier local decoder baselines, but it is no longer the main entry point for the held-out MEG results.

## Main Entry Points

- Active pipeline overview: [`brain_text_pipeline/README.md`](brain_text_pipeline/README.md)
- Paired-control toolkit / benchmark protocol: [`brain_text_pipeline/PAIRED_CONTROL_BENCH.md`](brain_text_pipeline/PAIRED_CONTROL_BENCH.md)
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
- local paper draft folders such as `paper/`
  - these stay on disk but are intentionally not tracked in Git.

## Local-Only Artifacts

The GitHub repo intentionally excludes large or generated experiment state. In a working local checkout you may still have:

- `data/`
  - local datasets, preprocessed MEG, simulator outputs, and cached JSONL/NPZ artifacts
- `runs/`
  - model checkpoints, eval JSON, qualitative pools, and logs
- `outputs/`, `models/`
  - other generated artifacts from older code paths

These directories are part of the expected local workflow, but they are intentionally ignored by Git and should not be treated as canonical source files.

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

## PairedControlBench

This repo now also exposes the main evaluation logic as a reusable paired-control
toolkit:

- `eval_brain_controls.py`
  - main `REAL / ZERO / SHUF` evaluation plus stricter SHUF variants
- `clustered_paired_controls.py`
  - clustered bootstrap uncertainty estimates
- `analyze_meg_effect_characterization.py`
  - probability/rank movement and surprisal-stratified summaries
- `eval_meg_sensor_ablation.py`
  - coarse sensor-group ablations
- `run_meg_story_multiseed.sh`
  - multi-seed robustness and normalization sensitivity
- `run_meg_subject_multiseed.sh`
  - multi-seed robustness for the subject-blocked MEG split
- `brain_text_pipeline/scripts/run_paired_control_bench_story.sh`
  - one-command wrapper for the core story-blocked benchmark outputs
- `analyze_meg_covariate_controls.py`
  - nuisance-balanced post-hoc evaluation over subject, sound, frequency, length, and position bins

The protocol and expected outputs are documented in
[`brain_text_pipeline/PAIRED_CONTROL_BENCH.md`](brain_text_pipeline/PAIRED_CONTROL_BENCH.md).
The stricter mismatch family now also includes a same-subject+same-sound
local-time SHUF mode for nuisance-preserving evaluation.

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
- The MEG pipeline assumes local access to MEG-MASC and writes large sharded datasets and checkpoints under local `data/` and `runs/` directories; keep those artifacts out of version control.
- The most up-to-date experiment instructions live in `brain_text_pipeline/MEG_EXPERIMENT_STEPS.md`, not in old shell history or notebook snippets.

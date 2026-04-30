# Brain-Text Pipeline

`brain_text_pipeline/` is the active code path for the held-out MEG experiments.
It implements paired-control evaluation for brain-conditioned language modeling with T5 cross-attention and temporal MEG inputs.

## What This Pipeline Covers

- MEG-MASC preprocessing
- word-aligned sharded MEG datasets
- random / story-blocked / subject-blocked / LOSO splits
- T5 brain cross-attention training
- REAL vs ZERO vs SHUF evaluation
- stricter SHUF controls
- clustered bootstrap statistics
- qualitative example generation
- BERT positive controls and BERT+MEG combined controls
- characterization analyses
- sensor-group ablations
- multi-seed story-blocked robustness checks

## Recommended Starting Point

For the exact paper-facing commands, use:

- [`MEG_EXPERIMENT_STEPS.md`](MEG_EXPERIMENT_STEPS.md)

That file is the canonical experiment logbook. The README here is only a shorter map.

## Core Design

The main real-data setup is:

- model: `t5-small`
- temporal MEG encoder + decoder cross-attention
- decoder context mode: `target_only`
- brain normalization: `per_example`
- main successful window: post-word `0.0` to `0.6` s

The main evaluation principle is paired control:

- `REAL`: matched MEG window
- `ZERO`: zeroed MEG input
- `SHUF`: mismatched MEG input

The central claim is not standard autoregressive LM improvement; it is controlled target-word likelihood modulation under paired controls.

## Minimal Story-Blocked Workflow

### 1. Preprocess MEG-MASC

```bash
python3 brain_text_pipeline/scripts/preprocess_meg_masc.py \
  --root data/meg_masc \
  --out_dir brain_text_pipeline/data/meg_preprocessed \
  --sfreq 200 \
  --l_freq 0.5 \
  --h_freq 40 \
  --zscore
```

### 2. Build word-aligned dataset shards

```bash
python3 brain_text_pipeline/scripts/build_meg_word_aligned_dataset.py \
  --meg_root data/meg_masc \
  --preprocessed_root brain_text_pipeline/data/meg_preprocessed \
  --tokenizer t5-small \
  --out_dir brain_text_pipeline/data/meg_aligned_postword \
  --shard_size 5000 \
  --tmin 0.0 \
  --tmax 0.6
```

### 3. Split into story-blocked train/test

```bash
python3 brain_text_pipeline/scripts/split_sharded_meg_dataset.py \
  --input_dir brain_text_pipeline/data/meg_aligned_postword \
  --train_out brain_text_pipeline/data/meg_aligned_postword_story_train \
  --test_out brain_text_pipeline/data/meg_aligned_postword_story_test \
  --split story \
  --seed 42
```

### 4. Train the main story-blocked model

```bash
python3 brain_text_pipeline/scripts/train_t5_brain_crossattn.py \
  --mode meg_supervised \
  --model_name_or_path t5-small \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_story_train/manifest.json \
  --output_dir brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1 \
  --batch_size 32 \
  --lr 5e-5 \
  --epochs 6 \
  --bf16 \
  --freeze_t5 \
  --unfreeze_cross_attn \
  --cross_attn_last_n 6 \
  --unfreeze_last_n 1 \
  --decoder_context_mode target_only \
  --brain_norm per_example \
  --max_text_len 8 \
  --max_brain_len 120 \
  --device cuda \
  --seed 42
```

### 5. Evaluate paired controls

```bash
python3 brain_text_pipeline/scripts/eval_brain_controls.py \
  --model_name_or_path brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1 \
  --brain_encoder_ckpt brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/brain_encoder.pt \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_story_test/manifest.json \
  --samples 50000 \
  --batch_size 32 \
  --device cuda \
  --decoder_context_mode target_only \
  --brain_norm per_example \
  --max_text_len 8 \
  --max_brain_len 120 \
  --seed 42 \
  --out_json brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/eval_story_test_50k.json
```

## Important Follow-Up Analyses

After the main story-blocked run, the most important scripts are:

- `eval_brain_controls.py`
  - main REAL/ZERO/SHUF eval
  - stricter SHUF variants
  - per-example export
- `clustered_paired_controls.py`
  - clustered bootstrap over subject / sound / sequence-level groups
- `analyze_meg_effect_characterization.py`
  - surprisal, rank/probability movement, top-k changes
- `eval_meg_sensor_ablation.py`
  - coarse sensor-group ablations
- `generate_t5_brain_controls.py`
  - qualitative examples
- `render_qualitative_appendix.py`
  - appendix table rendering
- `run_meg_story_multiseed.sh`
  - 3-seed story-blocked robustness + normalization sensitivity
- `summarize_meg_story_multiseed.py`
  - aggregates the resulting eval JSONs

## Auxiliary and Control Pipelines

The pipeline also supports:

- Gaussian auxiliary streams
  - `build_aux_control_dataset.py`
- context-only BERT positive controls
  - `build_text_aux_dataset.py`
- combined MEG+BERT feature streams
  - `combine_meg_text_aux_dataset.py`
- joint BERT+MEG evaluation
  - `run_megplusbert_story_pipeline.sh`
- frozen-BERT residual-MEG analysis
  - `train_t5_bert_meg_residual.py`
  - `eval_t5_bert_meg_residual_controls.py`

## Data Format

All datasets are stored as sharded `.npz` files with a `manifest.json`.
Examples typically include:

- decoder inputs / labels
- `brain_seq` with shape `[T, D]`
- metadata such as subject, story, session, task, sound, and sequence IDs

## Notes

- The main held-out MEG result depends critically on `brain_norm=per_example`.
- The target-only decoder setup is intentional and is used to prevent decoder-context leakage.
- The paper directories are local-only and not tracked in Git; rebuild the paper from your local `paper/` folder if needed.

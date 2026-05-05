# PairedControlBench

`PairedControlBench` is a lightweight evaluation protocol for testing whether an
auxiliary conditioning signal is **useful**, not merely **different**.

The motivating use case in this repository is post-word MEG conditioning of a
T5 language model, but the protocol is intentionally broader. It can be applied
to any setting where a base text model is conditioned on an aligned side signal:

- neural recordings
- physiological state
- affect or user-state vectors
- retrieved vectors
- multimodal side channels
- learned steering vectors

The central question is:

> Does the matched auxiliary signal improve prediction over both removal and
> mismatch controls, or does it only perturb the output distribution?

## Core Definitions

For an example with text context `x`, target token `y`, and auxiliary signal
`a`, define:

- `REAL`
  - the model evaluated with the matched auxiliary input
- `ZERO`
  - the same model evaluated with the auxiliary pathway removed or zeroed
- `SHUF`
  - the same model evaluated with a mismatched auxiliary input that preserves
    as much nuisance structure as possible while breaking the exact pairing

The core paired-control criteria are:

- `delta_real_zero < 0`
- `delta_real_shuf < 0`

where each delta is a paired target-token NLL difference and lower is better.

`JSD` and top-1 disagreement measure **deformation**. Paired NLL differences
measure **utility**.

## Failure Modes the Protocol Detects

Paired controls are designed to detect common failure cases that can otherwise
look like “conditioning works”:

- no effective use of the auxiliary input
  - `REAL`, `ZERO`, and `SHUF` are all nearly identical
- generic perturbation
  - `REAL` changes logits strongly, but does not improve target likelihood
- nuisance-structure dependence
  - `REAL` beats `ZERO`, but not a stricter `SHUF` preserving subject/story/time
- collapse due to preprocessing or scaling
  - the auxiliary input is numerically present but effectively ignored

## Minimum Reporting Standard

At minimum, report:

1. `REAL`, `ZERO`, and `SHUF` paired target-token NLL differences
2. a deformation metric such as `JSD`
3. top-1 agreement with `ZERO`
4. at least one held-out split
5. uncertainty estimates on the paired differences
6. at least one stricter mismatch control beyond a single global shuffle

Recommended additions:

- clustered bootstrap or mixed-effects uncertainty
- multi-seed robustness
- characterization of where probability mass moves
- nuisance-preserving `SHUF` variants
- ablations showing where the signal matters

## Benchmark API

A benchmark run in `PairedControlBench` is defined by four ingredients:

1. a fixed model checkpoint
2. a held-out evaluation manifest
3. one matched auxiliary stream (`REAL`)
4. two control streams:
   - a removal control (`ZERO`)
   - at least one mismatch control (`SHUF`)

The canonical JSON output should expose:

- `delta_real_zero`
- `delta_real_shuf`
- `js_real`
- `js_shuf`
- `top1_real_zero`
- `top1_shuf_zero`
- paired confidence intervals
- control-construction metadata such as `shuf_mode` and grouping keys

This is the smallest stable API that lets different auxiliary-conditioning
methods be compared under the same protocol.

## Core Story-Blocked Benchmark

In this repository, the canonical benchmark instance is the story-blocked
post-word MEG/T5 setup.

Use the one-command wrapper:

```bash
bash brain_text_pipeline/scripts/run_paired_control_bench_story.sh
```

This runs the core benchmark outputs for the default story-blocked model:

- main `REAL/ZERO/SHUF` evaluation
- per-example export
- clustered bootstrap
- characterization analysis
- covariate-balanced post-hoc analysis
- stricter SHUF family
- same-subject+same-sound local-time SHUF
- optional sensor ablation if `MEG_ROOT` is provided

## Example Commands

Main evaluation only:

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
  --out_json brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/eval_story_test_50k.json \
  --out_examples_jsonl brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/eval_story_test_50k_examples.jsonl
```

Clustered uncertainty:

```bash
python3 brain_text_pipeline/scripts/clustered_paired_controls.py \
  --examples_jsonl brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/eval_story_test_50k_examples.jsonl \
  --cluster_spec subject \
  --cluster_spec sound \
  --cluster_spec sequence_id \
  --bootstrap_samples 10000 \
  --seed 42 \
  --out_json brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/eval_story_test_50k_clustered.json
```

Characterization:

```bash
python3 brain_text_pipeline/scripts/analyze_meg_effect_characterization.py \
  --examples_jsonl brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/eval_story_test_50k_examples.jsonl \
  --train_manifest brain_text_pipeline/data/meg_aligned_postword_story_train/manifest.json \
  --out_json brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/eval_story_test_50k_characterization.json
```

## Stricter SHUF Family

The benchmark supports several mismatch controls:

- `batch_global`
- `global_sample`
- `within_group` with keys like `subject`, `sound`, or `subject,session`
- `within_group_local` with keys like `subject,sound` plus a local key such as
  `word_index` or `onset_sec`
- `circular_time_shift`
- `block_permute`
- `phase_randomized`

Example:

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
  --shuf_mode within_group \
  --shuf_group_keys subject \
  --out_json brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/eval_story_test_50k_shuf_subject.json
```

Stricter local-time nuisance-preserving mismatch:

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
  --shuf_mode within_group_local \
  --shuf_group_keys subject,sound \
  --shuf_local_key word_index \
  --shuf_local_radius 32 \
  --out_json brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/eval_story_test_50k_shuf_subject_sound_local.json
```

## Expected Outputs

The benchmark produces JSON summaries with stable fields that are intended to be
paper- and leaderboard-friendly.

Main evaluation JSON:

- `n`
- `nll_real`, `nll_zero`, `nll_shuf`
- `delta_real_zero`, `delta_real_shuf`
- paired confidence intervals
- `js_real`, `js_shuf`
- `top1_real_zero`, `top1_shuf_zero`

Clustered summary JSON:

- grouped `delta_real_zero` / `delta_real_shuf` means
- clustered 95% confidence intervals
- bootstrap standard errors
- nonnegative tail probabilities

Characterization JSON:

- probability/rank movement
- MRR / top-k changes
- surprisal bins
- word-type bins
- length/frequency/story-position summaries

Sensor ablation JSON:

- baseline deltas
- ablated deltas by coarse sensor group
- loss relative to baseline

## Generic Use Beyond MEG

To apply the same protocol to a different side signal:

1. keep the model fixed across `REAL`, `ZERO`, and `SHUF`
2. define a valid removal control
3. define at least one mismatch control that preserves nuisance structure
4. report both deformation and utility
5. use held-out splits and uncertainty estimates

The protocol is deliberately agnostic to whether the side signal is neural,
physiological, multimodal, retrieved, or learned.

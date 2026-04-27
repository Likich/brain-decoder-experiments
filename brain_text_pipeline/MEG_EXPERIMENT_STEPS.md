# MEG-MASC T5 Experiment Steps

These are the next experiments needed for the paper. The priority is held-out
REAL/ZERO/SHUF evidence. Qualitative generation comes after the held-out
control metrics.

All commands assume you are in the repository root and already have:

```bash
brain_text_pipeline/data/meg_aligned_postword/manifest.json
```

Use the same thread caps for every GPU run:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
```

## Step 1: Random Held-Out Sanity Split

This checks whether the model generalizes beyond the exact examples it trained
on. It is not the final paper split, but it is the fastest sanity test.

```bash
python3 brain_text_pipeline/scripts/split_sharded_meg_dataset.py \
  --manifest brain_text_pipeline/data/meg_aligned_postword/manifest.json \
  --train_out brain_text_pipeline/data/meg_aligned_postword_random_train \
  --test_out brain_text_pipeline/data/meg_aligned_postword_random_test \
  --split random \
  --test_fraction 0.1 \
  --shard_size 5000 \
  --seed 42
```

Train the current strongest post-word model:

```bash
python3 brain_text_pipeline/scripts/train_t5_brain_crossattn.py \
  --mode meg_supervised \
  --model_name_or_path t5-small \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_random_train/manifest.json \
  --output_dir brain_text_pipeline/runs/t5_meg_postword_random_hybrid_last1 \
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
  --log_interval 100 \
  --cpu_threads 8 \
  --num_workers 0 \
  --device cuda
```

Evaluate only on the random held-out test split:

```bash
python3 brain_text_pipeline/scripts/eval_brain_controls.py \
  --model_name_or_path brain_text_pipeline/runs/t5_meg_postword_random_hybrid_last1 \
  --brain_encoder_ckpt brain_text_pipeline/runs/t5_meg_postword_random_hybrid_last1/brain_encoder.pt \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_random_test/manifest.json \
  --samples 50000 \
  --batch_size 32 \
  --device cuda \
  --decoder_context_mode target_only \
  --brain_norm per_example \
  --max_text_len 8 \
  --max_brain_len 120 \
  --seed 42 \
  --out_json brain_text_pipeline/runs/t5_meg_postword_random_hybrid_last1/eval_random_test_50k.json
```

Success pattern:

```text
delta_real_zero < 0
delta_real_shuf < 0
both paired 95% CIs exclude 0
```

## Step 2: Qualitative Target-Probability Appendix

Run this only after Step 1 produces a valid REAL advantage. These examples are
for illustration, not proof. The goal is to show target-word probability/rank
under REAL, ZERO, and SHUF, not open-ended generation.

```bash
python3 brain_text_pipeline/scripts/generate_t5_brain_controls.py \
  --model_name_or_path brain_text_pipeline/runs/t5_meg_postword_random_hybrid_last1 \
  --tokenizer_name_or_path t5-small \
  --brain_encoder_ckpt brain_text_pipeline/runs/t5_meg_postword_random_hybrid_last1/brain_encoder.pt \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_random_test/manifest.json \
  --out_jsonl brain_text_pipeline/runs/t5_meg_postword_random_hybrid_last1/generation_random_test.jsonl \
  --samples 2000 \
  --show 40 \
  --batch_size 32 \
  --device cuda \
  --decoder_context_mode target_only \
  --brain_norm per_example \
  --max_text_len 8 \
  --max_brain_len 120 \
  --selection real_beats_controls \
  --single_token_only \
  --top_k 8
```

Use examples from the JSONL where:

- REAL beats both ZERO and SHUF
- the target is a single token
- REAL has a better first-token rank and/or probability than both controls

If you still want free generation for debugging only, add `--include_generation`.

## Step 3: Story-Blocked Split

This is the more important paper split. It asks whether the effect survives
held-out story content.

```bash
python3 brain_text_pipeline/scripts/split_sharded_meg_dataset.py \
  --manifest brain_text_pipeline/data/meg_aligned_postword/manifest.json \
  --train_out brain_text_pipeline/data/meg_aligned_postword_story_train \
  --test_out brain_text_pipeline/data/meg_aligned_postword_story_test \
  --split story \
  --test_fraction 0.2 \
  --shard_size 5000 \
  --seed 42
```

Train:

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
  --log_interval 100 \
  --cpu_threads 8 \
  --num_workers 0 \
  --device cuda
```

Evaluate:

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

If this succeeds, it becomes the main MEG result. If random succeeds but
story-blocked fails, the paper should say the signal exists under random
held-out examples but robust story generalization remains unresolved.

## Step 4: Subject-Blocked Split

This is the next generalization test after story-blocked. It asks whether the
effect survives held-out subjects rather than just held-out stories.

```bash
python3 brain_text_pipeline/scripts/split_sharded_meg_dataset.py \
  --manifest brain_text_pipeline/data/meg_aligned_postword/manifest.json \
  --train_out brain_text_pipeline/data/meg_aligned_postword_subject_train \
  --test_out brain_text_pipeline/data/meg_aligned_postword_subject_test \
  --split subject \
  --test_fraction 0.2 \
  --shard_size 5000 \
  --seed 42
```

```bash
python3 brain_text_pipeline/scripts/train_t5_brain_crossattn.py \
  --mode meg_supervised \
  --model_name_or_path t5-small \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_subject_train/manifest.json \
  --output_dir brain_text_pipeline/runs/t5_meg_postword_subject_hybrid_last1 \
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
  --log_interval 100 \
  --cpu_threads 8 \
  --num_workers 0 \
  --device cuda
```

```bash
python3 brain_text_pipeline/scripts/eval_brain_controls.py \
  --model_name_or_path brain_text_pipeline/runs/t5_meg_postword_subject_hybrid_last1 \
  --brain_encoder_ckpt brain_text_pipeline/runs/t5_meg_postword_subject_hybrid_last1/brain_encoder.pt \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_subject_test/manifest.json \
  --samples 50000 \
  --batch_size 32 \
  --device cuda \
  --decoder_context_mode target_only \
  --brain_norm per_example \
  --max_text_len 8 \
  --max_brain_len 120 \
  --seed 42 \
  --out_json brain_text_pipeline/runs/t5_meg_postword_subject_hybrid_last1/eval_subject_test_50k.json
```

## Step 5: Leave-One-Subject-Out (Optional, Slower)

Use this only if the subject-blocked split is promising and you want the
strongest subject-level generalization check.

List the available subjects:

```bash
python3 - <<'PY'
import json
from pathlib import Path
import numpy as np

manifest_path = Path('brain_text_pipeline/data/meg_aligned_postword/manifest.json')
with manifest_path.open() as f:
    manifest = json.load(f)

subjects = set()
for shard in manifest["shards"]:
    path = Path(shard["path"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    with np.load(path, allow_pickle=True) as data:
        for raw in data["meta"]:
            meta = raw
            if isinstance(meta, (bytes, str)):
                meta = json.loads(meta)
            if isinstance(meta, dict) and meta.get("subject"):
                subjects.add(str(meta["subject"]))
print(" ".join(sorted(subjects)))
PY
```

Example for one held-out subject:

```bash
python3 brain_text_pipeline/scripts/split_sharded_meg_dataset.py \
  --manifest brain_text_pipeline/data/meg_aligned_postword/manifest.json \
  --train_out brain_text_pipeline/data/meg_aligned_postword_sub17_train \
  --test_out brain_text_pipeline/data/meg_aligned_postword_sub17_test \
  --split subject \
  --test_subjects sub-17 \
  --shard_size 5000 \
  --seed 42
```

Train/eval then matches the subject-blocked commands above, with paths renamed
from `subject_*` to `sub17_*`.

## Step 6: Temporal Window Ablation

This tests whether the held-out effect is concentrated in the later part of
the post-word window. Reuse the exact same held-out stories as the successful
post-word story split so the comparison is apples-to-apples.

Extract the held-out story names once:

```bash
TEST_STORIES=$(python3 - <<'PY'
import json
with open('brain_text_pipeline/data/meg_aligned_postword_story_test/manifest.json') as f:
    m = json.load(f)
print(" ".join(m["split"]["test_values"]))
PY
)
echo "$TEST_STORIES"
```

Build aligned datasets:

```bash
python3 brain_text_pipeline/scripts/build_meg_word_aligned_dataset.py \
  --meg_root data/meg_masc \
  --preprocessed_root brain_text_pipeline/data/meg_preprocessed \
  --tokenizer t5-small \
  --out_dir brain_text_pipeline/data/meg_aligned_postword_early \
  --shard_size 5000 \
  --tmin 0.0 \
  --tmax 0.3 \
  --max_context_tokens 128 \
  --max_target_tokens 8 \
  --max_target_chars 80 \
  --max_examples 200000
```

```bash
python3 brain_text_pipeline/scripts/build_meg_word_aligned_dataset.py \
  --meg_root data/meg_masc \
  --preprocessed_root brain_text_pipeline/data/meg_preprocessed \
  --tokenizer t5-small \
  --out_dir brain_text_pipeline/data/meg_aligned_postword_late \
  --shard_size 5000 \
  --tmin 0.3 \
  --tmax 0.6 \
  --max_context_tokens 128 \
  --max_target_tokens 8 \
  --max_target_chars 80 \
  --max_examples 200000
```

```bash
python3 brain_text_pipeline/scripts/build_meg_word_aligned_dataset.py \
  --meg_root data/meg_masc \
  --preprocessed_root brain_text_pipeline/data/meg_preprocessed \
  --tokenizer t5-small \
  --out_dir brain_text_pipeline/data/meg_aligned_postword_tightlate \
  --shard_size 5000 \
  --tmin 0.45 \
  --tmax 0.6 \
  --max_context_tokens 128 \
  --max_target_tokens 8 \
  --max_target_chars 80 \
  --max_examples 200000
```

Split each dataset with the same held-out stories:

```bash
python3 brain_text_pipeline/scripts/split_sharded_meg_dataset.py \
  --manifest brain_text_pipeline/data/meg_aligned_postword_early/manifest.json \
  --train_out brain_text_pipeline/data/meg_aligned_postword_early_story_train \
  --test_out brain_text_pipeline/data/meg_aligned_postword_early_story_test \
  --split story \
  --test_values $TEST_STORIES \
  --shard_size 5000 \
  --seed 42
```

```bash
python3 brain_text_pipeline/scripts/split_sharded_meg_dataset.py \
  --manifest brain_text_pipeline/data/meg_aligned_postword_late/manifest.json \
  --train_out brain_text_pipeline/data/meg_aligned_postword_late_story_train \
  --test_out brain_text_pipeline/data/meg_aligned_postword_late_story_test \
  --split story \
  --test_values $TEST_STORIES \
  --shard_size 5000 \
  --seed 42
```

```bash
python3 brain_text_pipeline/scripts/split_sharded_meg_dataset.py \
  --manifest brain_text_pipeline/data/meg_aligned_postword_tightlate/manifest.json \
  --train_out brain_text_pipeline/data/meg_aligned_postword_tightlate_story_train \
  --test_out brain_text_pipeline/data/meg_aligned_postword_tightlate_story_test \
  --split story \
  --test_values $TEST_STORIES \
  --shard_size 5000 \
  --seed 42
```

Train and evaluate:

```bash
python3 brain_text_pipeline/scripts/train_t5_brain_crossattn.py \
  --mode meg_supervised \
  --model_name_or_path t5-small \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_early_story_train/manifest.json \
  --output_dir brain_text_pipeline/runs/t5_meg_postword_early_story_hybrid_last1 \
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
  --max_brain_len 60 \
  --log_interval 100 \
  --cpu_threads 8 \
  --num_workers 0 \
  --device cuda
```

```bash
python3 brain_text_pipeline/scripts/eval_brain_controls.py \
  --model_name_or_path brain_text_pipeline/runs/t5_meg_postword_early_story_hybrid_last1 \
  --brain_encoder_ckpt brain_text_pipeline/runs/t5_meg_postword_early_story_hybrid_last1/brain_encoder.pt \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_early_story_test/manifest.json \
  --samples 50000 \
  --batch_size 32 \
  --device cuda \
  --seed 42 \
  --decoder_context_mode target_only \
  --brain_norm per_example \
  --max_text_len 8 \
  --max_brain_len 60 \
  --out_json brain_text_pipeline/runs/t5_meg_postword_early_story_hybrid_last1/eval_story_test_50k.json
```

```bash
python3 brain_text_pipeline/scripts/train_t5_brain_crossattn.py \
  --mode meg_supervised \
  --model_name_or_path t5-small \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_late_story_train/manifest.json \
  --output_dir brain_text_pipeline/runs/t5_meg_postword_late_story_hybrid_last1 \
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
  --max_brain_len 60 \
  --log_interval 100 \
  --cpu_threads 8 \
  --num_workers 0 \
  --device cuda
```

```bash
python3 brain_text_pipeline/scripts/eval_brain_controls.py \
  --model_name_or_path brain_text_pipeline/runs/t5_meg_postword_late_story_hybrid_last1 \
  --brain_encoder_ckpt brain_text_pipeline/runs/t5_meg_postword_late_story_hybrid_last1/brain_encoder.pt \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_late_story_test/manifest.json \
  --samples 50000 \
  --batch_size 32 \
  --device cuda \
  --seed 42 \
  --decoder_context_mode target_only \
  --brain_norm per_example \
  --max_text_len 8 \
  --max_brain_len 60 \
  --out_json brain_text_pipeline/runs/t5_meg_postword_late_story_hybrid_last1/eval_story_test_50k.json
```

```bash
python3 brain_text_pipeline/scripts/train_t5_brain_crossattn.py \
  --mode meg_supervised \
  --model_name_or_path t5-small \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_tightlate_story_train/manifest.json \
  --output_dir brain_text_pipeline/runs/t5_meg_postword_tightlate_story_hybrid_last1 \
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
  --max_brain_len 30 \
  --log_interval 100 \
  --cpu_threads 8 \
  --num_workers 0 \
  --device cuda
```

```bash
python3 brain_text_pipeline/scripts/eval_brain_controls.py \
  --model_name_or_path brain_text_pipeline/runs/t5_meg_postword_tightlate_story_hybrid_last1 \
  --brain_encoder_ckpt brain_text_pipeline/runs/t5_meg_postword_tightlate_story_hybrid_last1/brain_encoder.pt \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_tightlate_story_test/manifest.json \
  --samples 50000 \
  --batch_size 32 \
  --device cuda \
  --seed 42 \
  --decoder_context_mode target_only \
  --brain_norm per_example \
  --max_text_len 8 \
  --max_brain_len 30 \
  --out_json brain_text_pipeline/runs/t5_meg_postword_tightlate_story_hybrid_last1/eval_story_test_50k.json
```

Interpretation target:

```text
late-only ≈ baseline, early-only weaker or near zero -> late post-word signal matters
tight-late > late-only -> even narrower late window is sufficient
all windows similar -> attention timing likely reflects positional preference more than informative timing
```

## Step 7: Auxiliary-Stream Null Baseline

This is the cleanest answer to the reviewer concern that any continuous side
channel might help through the same architecture. Keep the text, metadata,
sequence lengths, and model path fixed, but replace every MEG window with a
matched-shape random auxiliary stream.

For the paper, the best first comparison is against the successful
story-blocked post-word run.

Build matched-shape random auxiliary datasets:

```bash
python3 brain_text_pipeline/scripts/build_aux_control_dataset.py \
  --source_manifest brain_text_pipeline/data/meg_aligned_postword_story_train/manifest.json \
  --out_dir brain_text_pipeline/data/aux_random_postword_story_train \
  --mode gaussian_iid \
  --seed 42
```

```bash
python3 brain_text_pipeline/scripts/build_aux_control_dataset.py \
  --source_manifest brain_text_pipeline/data/meg_aligned_postword_story_test/manifest.json \
  --out_dir brain_text_pipeline/data/aux_random_postword_story_test \
  --mode gaussian_iid \
  --seed 42
```

Train the exact same T5 configuration on the auxiliary stream:

```bash
python3 brain_text_pipeline/scripts/train_t5_brain_crossattn.py \
  --mode meg_supervised \
  --model_name_or_path t5-small \
  --meg_dataset_path brain_text_pipeline/data/aux_random_postword_story_train/manifest.json \
  --output_dir brain_text_pipeline/runs/t5_aux_random_postword_story_hybrid_last1 \
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
  --log_interval 100 \
  --cpu_threads 8 \
  --num_workers 0 \
  --device cuda
```

Evaluate with the same REAL/ZERO/SHUF script:

```bash
python3 brain_text_pipeline/scripts/eval_brain_controls.py \
  --model_name_or_path brain_text_pipeline/runs/t5_aux_random_postword_story_hybrid_last1 \
  --brain_encoder_ckpt brain_text_pipeline/runs/t5_aux_random_postword_story_hybrid_last1/brain_encoder.pt \
  --meg_dataset_path brain_text_pipeline/data/aux_random_postword_story_test/manifest.json \
  --samples 50000 \
  --batch_size 32 \
  --device cuda \
  --decoder_context_mode target_only \
  --brain_norm per_example \
  --max_text_len 8 \
  --max_brain_len 120 \
  --seed 42 \
  --out_json brain_text_pipeline/runs/t5_aux_random_postword_story_hybrid_last1/eval_story_test_50k.json
```

Expected interpretation:

```text
If arbitrary side channels are not enough, the random auxiliary stream should
fail to show a stable REAL advantage over both ZERO and SHUF.
```

Optional variant:

```bash
python3 brain_text_pipeline/scripts/build_aux_control_dataset.py \
  --source_manifest brain_text_pipeline/data/meg_aligned_postword_story_train/manifest.json \
  --out_dir brain_text_pipeline/data/aux_globalgauss_postword_story_train \
  --mode gaussian_global \
  --seed 42
```

`gaussian_global` matches the source dataset's per-channel mean/std before
training. With per-example brain normalization enabled, `gaussian_iid` is
usually the cleaner null.

## Step 8: Attention Extraction

Run this after the best held-out model is selected.

```bash
python3 brain_text_pipeline/scripts/extract_cross_attention.py \
  --model_name_or_path brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1 \
  --brain_encoder_ckpt brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/brain_encoder.pt \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_story_test/manifest.json \
  --out_dir brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/attention_story_test \
  --samples 1000 \
  --batch_size 16 \
  --device cuda \
  --decoder_context_mode target_only \
  --brain_norm per_example \
  --max_text_len 8 \
  --max_brain_len 120 \
  --condition real \
  --save_full_matrix
```

Export the shuffled control with the same examples:

```bash
python3 brain_text_pipeline/scripts/extract_cross_attention.py \
  --model_name_or_path brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1 \
  --brain_encoder_ckpt brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/brain_encoder.pt \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned_postword_story_test/manifest.json \
  --out_dir brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/attention_story_test_shuf \
  --samples 5000 \
  --batch_size 16 \
  --device cuda \
  --decoder_context_mode target_only \
  --brain_norm per_example \
  --max_text_len 8 \
  --max_brain_len 120 \
  --condition shuf \
  --save_full_matrix
```

Summarize and plot the comparison:

```bash
python3 brain_text_pipeline/scripts/summarize_cross_attention.py \
  --real_manifest brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/attention_story_test/manifest.json \
  --shuf_manifest brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/attention_story_test_shuf/manifest.json \
  --out_json brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/attention_story_summary.json \
  --out_pdf brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1/attention_story_summary.pdf \
  --title "Story-Blocked Post-Word Cross-Attention" \
  --tmin_sec 0.0 \
  --tmax_sec 0.6
```

Attention is appendix evidence only. The paper should still treat REAL/ZERO/SHUF
NLL as the main proof.

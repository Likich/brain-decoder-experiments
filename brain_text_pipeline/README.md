# Brain-Text Pipeline

Minimal end-to-end pipeline for brain-conditioned text generation with T5 cross-attention.

## Quick Start

1) Preprocess MEG-MASC
```bash
python3 brain_text_pipeline/scripts/preprocess_meg_masc.py \
  --root data/meg_masc \
  --out_dir brain_text_pipeline/data/meg_preprocessed \
  --sfreq 200 --l_freq 0.5 --h_freq 40 --zscore
```

2) Build word-aligned dataset shards
```bash
python3 brain_text_pipeline/scripts/build_meg_word_aligned_dataset.py \
  --meg_root data/meg_masc \
  --preprocessed_root brain_text_pipeline/data/meg_preprocessed \
  --tokenizer t5-small \
  --out_dir brain_text_pipeline/data/meg_aligned \
  --shard_size 5000 --tmin -0.5 --tmax 0.0
```

3) Train T5 brain cross-attention
```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
python3 brain_text_pipeline/scripts/train_t5_brain_crossattn.py \
  --mode meg_supervised \
  --model_name_or_path t5-small \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned/manifest.json \
  --output_dir brain_text_pipeline/runs/t5_meg \
  --batch_size 8 --lr 5e-5 --epochs 2 --bf16 \
  --freeze_t5 --unfreeze_last_n 2 \
  --max_text_len 512 --log_interval 200 --cpu_threads 8
```

4) Evaluate REAL vs SHUF vs ZERO
```bash
python3 brain_text_pipeline/scripts/eval_brain_controls.py \
  --model_name_or_path brain_text_pipeline/runs/t5_meg \
  --brain_encoder_ckpt brain_text_pipeline/runs/t5_meg/brain_encoder.pt \
  --meg_dataset_path brain_text_pipeline/data/meg_aligned/manifest.json \
  --out_json brain_text_pipeline/runs/t5_meg/eval_controls.json
```

## Notes
- All datasets are sharded `.npz` with a `manifest.json`.
- Brain input is always temporal: shape `[B, T, D]`.

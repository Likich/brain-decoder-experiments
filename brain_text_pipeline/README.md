# Brain-Text Pipeline

Minimal end-to-end pipeline for brain-conditioned text generation with T5 cross-attention.

## Quick Start

1) Preprocess MEG-MASC
```bash
python3 scripts/preprocess_meg_masc.py \
  --root /path/to/meg_masc \
  --out_dir data/meg_preprocessed \
  --sfreq 200 --l_freq 0.5 --h_freq 40 --zscore
```

2) Build word-aligned dataset shards
```bash
python3 scripts/build_meg_word_aligned_dataset.py \
  --meg_root /path/to/meg_masc \
  --preprocessed_root data/meg_preprocessed \
  --tokenizer t5-small \
  --out_dir data/meg_aligned \
  --shard_size 5000 --tmin -0.5 --tmax 0.0
```

3) Train T5 brain cross-attention
```bash
python3 scripts/train_t5_brain_crossattn.py \
  --mode meg_supervised \
  --model_name_or_path t5-small \
  --meg_dataset_path data/meg_aligned/manifest.json \
  --output_dir runs/t5_meg \
  --batch_size 8 --lr 5e-5 --epochs 2 \
  --freeze_t5 --unfreeze_last_n 2
```

4) Evaluate REAL vs SHUF vs ZERO
```bash
python3 scripts/eval_brain_controls.py \
  --model_name_or_path runs/t5_meg \
  --brain_encoder_ckpt runs/t5_meg/brain_encoder.pt \
  --meg_dataset_path data/meg_aligned/manifest.json \
  --out_json runs/t5_meg/eval_controls.json
```

## Notes
- All datasets are sharded `.npz` with a `manifest.json`.
- Brain input is always temporal: shape `[B, T, D]`.

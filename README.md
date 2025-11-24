# Brain Simulator


A minimal, research-ready scaffold for simulating left-hemisphere ignition,
choices, and textual report. TVB provides region-level dynamics; an RL-based
basal-ganglia proxy makes decisions; a thalamic gate polls an LLM module
for report only when a workspace ignition threshold is met. A simple
vector-store acts as hippocampal memory.


> This scaffold compiles to runnable Python components without external APIs.
> TVB integration points are stubbed behind an interface so you can swap in
> real TVB calls later.


## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/run_experiment.py --config configs/default.yaml
```


## Layout
- `src/lefty_brain_sim/tvb_iface.py` — interface & mock TVB engine
- `src/lefty_brain_sim/decision.py` — Wong–Wang-like 2-well decision node
- `src/lefty_brain_sim/gating.py` — thalamic gate & ignition metrics
- `src/lefty_brain_sim/llm_iface.py` — LLM evidence provider interface (mock)
- `src/lefty_brain_sim/memory.py` — simple FAISS vector memory (hippocampus)
- `src/lefty_brain_sim/encdec.py` — encoders/decoders between TVB state and LLM
- `src/lefty_brain_sim/experiment.py` — trial orchestration
- `scripts/run_experiment.py` — CLI entry point
- `configs/default.yaml` — experiment config


## Notes
- Replace the mock TVB engine with `tvb-library` coupling + monitors.
- Replace `MockLLM` with your preferred LLM backend or an API wrapper.
- All components are pure Python and unit-testable.

### Corpus helper
Run `python scripts/download_wiki40b.py --lang en --split train[:0.05%] --max_articles 500`
to grab a small Wiki40B shard locally (requires the Hugging Face `datasets` package).
The resulting JSONL in `data/` can seed your token-level stimulus pipeline.

### Tokenizer + tokens
1. `python scripts/train_tokenizer.py --input data/wiki40b_en.jsonl --vocab_size 2048`
   writes `models/wiki_tokenizer.json`.
2. `python scripts/encode_corpus_tokens.py --tokenizer models/wiki_tokenizer.json`
   emits token-id sequences to `data/wiki40b_tokens.jsonl`, ready for ingestion.

### Brain-conditioned next-token pipeline (recommended)
This ties cortex snapshots directly into a transformer LM and works with 136-region TVB.
1) Train tokenizer + get token schedule (see above).
2) Generate paired data (context, brain snapshot, next token):
   ```
   python scripts/build_brain_conditioned_dataset.py \
     --config configs/default.yaml \
     --token_file data/wiki40b_tokens.jsonl \
     --out data/brain_ctx_pairs_100k.npz \
     --max_samples 100000 --snr high
   ```
3) Train the brain-conditioned LM:
   ```
   python scripts/train_language_model.py \
     --data_file dummy.txt \
     --tokenizer_file models/wiki_tokenizer.json \
     --brain_dataset data/brain_ctx_pairs_100k.npz \
     --epochs 10 --batch_size 32 --block_size 96 \
     --hidden_dim 384 --num_layers 2 --attn_heads 8 \
     --lr 3.9e-4 --dropout 0.11 --device cuda
   ```
4) Chat with the trained model:
   ```
   python scripts/brain_chat.py \
     --tokenizer models/wiki_tokenizer.json \
     --ckpt models/language_model.pt \
     --brain_dataset data/brain_ctx_pairs_100k.npz \
     --brain_index 0 --block_size 96 \
     --hidden_dim 384 --num_layers 2 --attn_heads 8 --dropout 0.11
   ```

### Legacy local decoder pipeline (fruits/5-way)
If you still want the simple classifier path:
1. Run simulator to produce `outputs/experiment.jsonl`.
2. Build dataset:
   ```
   python scripts/build_dataset.py --input outputs/experiment.jsonl \
       --tokenizer models/wiki_tokenizer.json --use-target \
       --out data/brain_next_token.npz
   ```
3. Train decoder:
   ```
   python scripts/train_brain_decoder.py --data data/brain_next_token.npz \
       --tokenizer models/wiki_tokenizer.json --use_attention
   ```
4. Set `llm_provider: "local_decoder"` in `configs/default.yaml` to use it.

### Interactive chat (brain-conditioned)
See step 4 above; `scripts/brain_chat.py` now wraps the brain-conditioned LM. Type `quit` to exit.

## Generative loop
Set `generation.enabled: true` (default) to have each trial roll into an
autoregressive loop. Once the cortex/decoder produces a categorical decision,
that token is fed back as the next stimulus and the loop continues until
`generation.max_tokens` are emitted. Generated sequences are stored on each
trial line under `generated_tokens`.

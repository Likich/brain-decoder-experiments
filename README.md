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

### Token stimulus pipeline
1. Download corpus + train tokenizer (see above).
2. Ensure `configs/default.yaml` has `stimuli.mode: "tokens"` plus the tokenizer
   and `data/wiki40b_tokens.jsonl` schedule paths.
3. Run the simulator: `python scripts/run_experiment.py --config configs/default.yaml`
   to fill `outputs/experiment.jsonl`.
4. Build a training set: `python scripts/build_dataset.py --input outputs/experiment.jsonl --tokenizer models/wiki_tokenizer.json --out data/brain_tokens.npz`.
5. To train for *next-token* prediction (needed for generation), re-run dataset building with the `--use-target` flag once you've enabled `stimuli.predict_next: true` in the config:
   ```
   python scripts/build_dataset.py --input outputs/experiment.jsonl \
       --tokenizer models/wiki_tokenizer.json --use-target --out data/brain_next_token.npz
   ```
6. Train the local decoder on that dataset:
   ```
   python scripts/train_brain_decoder.py --data data/brain_next_token.npz \
       --tokenizer models/wiki_tokenizer.json
   ```
6. Repeat steps 3–5 as you collect new cortex activity (the decoder consumes the metadata written in step 5).

### Interactive chat
With `stimuli.predict_next: true` and the next-token decoder in place, launch a simple REPL:
```
python scripts/brain_chat.py --config configs/default.yaml --max_response 32
```
Type a prompt (e.g., `Hi`) and the simulator will tokenize it, run the cortex dynamics, and autoregressively emit a short response decoded via the tokenizer.

## Generative loop
Set `generation.enabled: true` (default) to have each trial roll into an
autoregressive loop. Once the cortex/decoder produces a categorical decision,
that token is fed back as the next stimulus and the loop continues until
`generation.max_tokens` are emitted. Generated sequences are stored on each
trial line under `generated_tokens`.

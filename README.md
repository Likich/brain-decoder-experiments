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

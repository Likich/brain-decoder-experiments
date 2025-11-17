from __future__ import annotations
import yaml
import numpy as np
from dataclasses import dataclass
import sys, json
from .utils import set_seed, TrialResult
from .tvb_iface import TVBEngine, TVBConfig
from .gating import GateConfig, IgnitionDetector, ThalamicGate
from .decision import DecisionConfig, DecisionNode
from .encdec import StateEncoder, EvidenceDecoder
from .memory import HippocampalMemory
from .stimuli import TokenSchedule
from .llm_iface import MockLLM, LocalDecoderLLM
from tokenizers import Tokenizer
try:
    from .llm_iface import OpenAICompatLLM  # optional backend for vLLM/OpenAI-compatible servers
except Exception:
    OpenAICompatLLM = None
    



@dataclass
class ExpConfig:
    seed: int
    trials: int
    snr_levels: list[str]
    ignition_threshold: float
    ignition_min_ms: float
    llm_policy: str
    speed_accuracy: str
    tvb: dict
    decision: dict
    gate: dict
    memory: dict
    generation: dict | None = None
    stimuli: dict | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_max_tokens: int | None = None
    llm_temperature: float | None = None
    local_decoder_ckpt: str | None = None 
    


class Experiment:
    def __init__(self, cfg: ExpConfig):
        set_seed(cfg.seed)
        self.cfg = cfg

        tvbc = TVBConfig(**cfg.tvb)
        self.tvb = TVBEngine(tvbc)

        gcfg = GateConfig(
            ignition_threshold=cfg.ignition_threshold,
            ignition_min_ms=cfg.ignition_min_ms,
            dt_ms=tvbc.dt_ms,
            cool_down_ms=cfg.gate["cool_down_ms"],
        )
        self.detector = IgnitionDetector(gcfg, tvbc.workspace_nodes)
        self.gate = ThalamicGate(cfg.gate["cool_down_ms"])

        self.decision = DecisionNode(DecisionConfig(**cfg.decision))

        self.encoder = StateEncoder(in_dim=len(tvbc.language_nodes))
        self.decoder = EvidenceDecoder()
        self.mem = HippocampalMemory(dim=cfg.memory.get("dim", 32))
        self.gen_cfg = cfg.generation or {}
        self.gen_enabled = bool(self.gen_cfg.get("enabled", False))
        self.gen_max_tokens = max(1, int(self.gen_cfg.get("max_tokens", 1)))

        provider = getattr(cfg, "llm_provider", "mock")

        if provider == "openai_compat" and OpenAICompatLLM is not None:
            self.llm = OpenAICompatLLM(
                model=getattr(cfg, "llm_model", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
                max_tokens=getattr(cfg, "llm_max_tokens", 64),
                temperature=getattr(cfg, "llm_temperature", 0.2),
                base_url=getattr(cfg, "llm_base_url", "http://10.180.132.23:8188/v1"),
                api_key=getattr(cfg, "llm_api_key", "EMPTY"),
            )
        elif provider == "local_decoder":
            ckpt = getattr(cfg, "local_decoder_ckpt", "models/brain_decoder.pt")
            self.llm = LocalDecoderLLM(ckpt_path=ckpt)
        else:
            self.llm = MockLLM()


        # --- Handy aliases ---
        self.lang_nodes = tvbc.language_nodes
        self.workspace_nodes = tvbc.workspace_nodes
        self.dt_ms = tvbc.dt_ms

        # --- Stimulus configuration ---
        self.stimuli_cfg = cfg.stimuli or {}
        self.stimulus_mode = self.stimuli_cfg.get("mode", "fruits").lower()
        self.tokenizer = None
        self.token_schedule = None
        self.predict_next = bool(self.stimuli_cfg.get("predict_next", False))
        self.class_names, self.stim_patterns = self._init_stimuli()


    def run_trial(
        self,
        snr: str,
        stim_idx: int | None = None,
        allow_generation: bool = True,
        log_debug: bool = True,
    ) -> TrialResult:
        self.tvb.reset()
        self.decision.reset()
        self.detector.ignited = False

        target_token_id = None
        # --- pick a stimulus ---
        if self.stimulus_mode == "tokens":
            if stim_idx is None:
                if self.token_schedule is None:
                    raise ValueError("Token stimulus mode requires a loaded schedule")
                stim_idx = self.token_schedule.next_token()
                if self.predict_next:
                    target_token_id = self.token_schedule.peek()
        else:
            if stim_idx is None:
                stim_idx = np.random.randint(len(self.class_names))

        if stim_idx is None or stim_idx < 0 or stim_idx >= len(self.class_names):
            raise ValueError(f"Stimulus index {stim_idx} out of range for vocab size {len(self.class_names)}")

        stim_label = self.class_names[stim_idx]
        stim_token = stim_label
        stimulus_id = int(stim_idx)

        # --- stimulus timing & amplitude ---
        stim_ms = 200 if snr == "high" else 120 if snr == "med" else 60
        amp = {"low": 0.30, "med": 0.70, "high": 1.20}[snr]
        total_ms = 800
        t = 0.0

        ign_latency = None
        reported = None
        choice = None
        rt = None
        conf = None
        llm_queried = False
        activity_snapshot = None
        all_nodes = np.arange(self.tvb.cfg.regions)

        # --- debug bookkeeping ---
        llm_choice_idx = None
        llm_choice_label = None
        ev_at_decision = None

        policy = str(self.cfg.llm_policy).lower()

        while t < total_ms:
            # 1) build external input for this tick
            ext = np.zeros((self.tvb.cfg.regions,))
            if 20.0 <= t < 20.0 + stim_ms:
                # use class-specific pattern on language nodes
                ext[self.lang_nodes] = amp * self.stim_patterns[stim_idx]

            # 2) step TVB
            state = self.tvb.step(ext)

            # 3) capture a snapshot once (after stimulus)
            if activity_snapshot is None and t >= 200.0:
                full_act = self.tvb.readout(all_nodes)
                activity_snapshot = full_act.tolist()

            # 4) ignition detection
            ignited, _ = self.detector.update(state)
            if ignited and ign_latency is None:
                ign_latency = t
                # full-brain snapshot at ignition
                all_nodes = np.arange(self.tvb.cfg.regions)
                full_act = self.tvb.readout(all_nodes)
                activity_snapshot = full_act.tolist()

            # 5) evidence from LLM (possibly none)
            ev_scalar = 0.0  # default every tick

            # gate dynamics (for "gated" policy)
            gate_open = False
            if policy == "gated":
                gate_open = self.gate.step(self.dt_ms, ignited)

            if policy not in ("always", "gated", "none"):
                raise ValueError(f"Unknown llm_policy: {self.cfg.llm_policy!r}")

            # Decide whether to poll the LLM on this tick
            if policy == "none":
                poll = False
            elif policy == "always":
                poll = ignited and not llm_queried
            else:  # "gated"
                poll = ignited and gate_open and not llm_queried

            if poll:
                # --- LOCAL DECODER BRANCH (5-way classifier) ---
                if isinstance(self.llm, LocalDecoderLLM):
                    # same kind of vector we trained on: full-brain readout
                    all_nodes = np.arange(self.tvb.cfg.regions)
                    z = self.tvb.readout(all_nodes)
                    hits = []  # not using hippocampal memory here

                    out = self.llm.infer(z, hits)
                    probs = np.asarray(out.evidence, dtype=float)

                    llm_choice_idx = int(np.argmax(probs))
                    llm_choice_label = self.class_names[llm_choice_idx]
                    reported = out.report
                    llm_queried = True

                    # For local decoder we treat this as the actual 5-way choice.
                    choice = llm_choice_idx
                    rt = t
                    conf = float(probs[llm_choice_idx])
                    ev_at_decision = conf

                    # end trial once the local decoder has spoken
                    break

                # --- ORIGINAL REMOTE-LLM BRANCH ---
                else:
                    # encoded language workspace + hippocampal memory
                    z = self.encoder(self.tvb.readout(self.lang_nodes))
                    hits = self.mem.search(z, k=self.cfg.memory.get("k", 5))

                    out = self.llm.infer(z, hits)
                    ev_scalar = self.decoder(out.evidence)
                    reported = out.report

                    # store in memory only for non-local decoders
                    self.mem.add(z, {"bias": ev_scalar})

                    llm_queried = True

            # 6) decision dynamics ALWAYS advance in time
            # (this is only meaningful for the non-local-LLM branch now)
            if not isinstance(self.llm, LocalDecoderLLM):
                gain = {
                    "fast": 1.2,
                    "balanced": 1.0,
                    "cautious": 0.8
                }[self.cfg.speed_accuracy]

                done, ch, t_dec, c = self.decision.step(
                    self.dt_ms, gain * ev_scalar
                )
                if done and choice is None:
                    choice, rt, conf = ch, t_dec, c
                    ev_at_decision = ev_scalar
                    break

            # 7) advance simulated time
            t += self.dt_ms

        debug_row = {
            "snr": snr,
            "stimulus": stim_label,
            "stimulus_id": stimulus_id,
            "ignited": bool(self.detector.ignited),
            "ignition_latency_ms": ign_latency,
            "llm_queried": llm_queried,
            "llm_choice_idx": llm_choice_idx,
            "llm_choice_label": llm_choice_label,
            "report": reported,
            "decision_choice": choice,
            "decision_rt_ms": rt,
            "decision_conf": conf,
            "ev_at_decision": ev_at_decision,
        }
        if log_debug:
            print("DEBUG_TRIAL", json.dumps(debug_row), file=sys.stderr)

        generated_tokens = None
        generated_token_ids = None
        if allow_generation:
            gen_out = self._run_generative_loop(snr, choice)
            if gen_out is not None:
                generated_tokens, generated_token_ids = gen_out

        return TrialResult(
            snr=snr,
            ignited=bool(self.detector.ignited),
            ignition_latency_ms=ign_latency,
            choice=choice,           # 0–4 when LocalDecoderLLM is used
            rt_ms=rt,
            confidence=conf,
            report=reported,
            stimulus=stim_label,
            stimulus_id=stimulus_id,
            stimulus_token=stim_token,
            activity_snapshot=activity_snapshot,
            generated_tokens=generated_tokens,
            generated_token_ids=generated_token_ids,
            target_token_id=target_token_id,
        )

    def _init_stimuli(self) -> tuple[list[str], np.ndarray]:
        if self.stimulus_mode == "tokens":
            tok_path = self.stimuli_cfg.get("tokenizer")
            sched_path = self.stimuli_cfg.get("schedule")
            if not tok_path or not sched_path:
                raise ValueError("Token stimuli require 'tokenizer' and 'schedule' paths")
            self.tokenizer = Tokenizer.from_file(tok_path)
            vocab = self.tokenizer.get_vocab()
            if not vocab:
                raise ValueError(f"Tokenizer at {tok_path} returned empty vocab")
            max_id = max(vocab.values())
            class_names = ["<unk>"] * (max_id + 1)
            for token, idx in vocab.items():
                if 0 <= idx < len(class_names):
                    class_names[idx] = token
            self.token_schedule = TokenSchedule(sched_path)
        else:
            class_names = ["APPLE", "BANANA", "GRAPE", "ORANGE", "PEAR"]

        rng = np.random.RandomState(self.cfg.seed)
        stim_patterns = rng.choice(
            [-1.0, 1.0],
            size=(len(class_names), len(self.lang_nodes)),
        )
        return class_names, stim_patterns

    def _run_generative_loop(self, snr: str, seed_choice: int | None) -> tuple[list[str], list[int]] | None:
        """
        After the first perceptual decision, optionally continue in an
        autoregressive mode by feeding the chosen token back as the next
        thalamic stimulus. Each rollout reuses the standard run_trial
        dynamics but skips nested generation and debug logging.
        """
        if not self.gen_enabled:
            return None
        if seed_choice is None or seed_choice < 0 or seed_choice >= len(self.class_names):
            return [], []

        tokens = [self.class_names[seed_choice]]
        token_ids = [seed_choice]
        prev_choice = seed_choice

        for _ in range(1, self.gen_max_tokens):
            follow_up = self.run_trial(
                snr=snr,
                stim_idx=prev_choice,
                allow_generation=False,
                log_debug=False,
            )
            if follow_up.choice is None:
                break
            prev_choice = follow_up.choice
            tokens.append(self.class_names[prev_choice])
            token_ids.append(prev_choice)
        return tokens, token_ids




    @staticmethod
    def from_yaml(path: str) -> "Experiment":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        extras = {}
        for k in ("llm_base_url", "llm_api_key"):
            if k in raw:
                extras[k] = raw.pop(k)

        cfg = ExpConfig(**raw)
        exp = Experiment(cfg)

        for k, v in extras.items():
            setattr(exp.cfg, k, v)

        return exp

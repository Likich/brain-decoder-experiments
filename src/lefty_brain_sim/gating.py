from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class GateConfig:
    ignition_threshold: float
    ignition_min_ms: float
    dt_ms: float
    cool_down_ms: float

class IgnitionDetector:
    def __init__(self, cfg: GateConfig, workspace_idx: list[int]):
        self.cfg = cfg
        self.workspace_idx = workspace_idx
        self.window_ms = 0.0
        self.ignited = False

    def update(self, state: np.ndarray) -> tuple[bool, float | None]:
        ws = np.mean(np.abs(state[self.workspace_idx]))
        if ws >= self.cfg.ignition_threshold:
            self.window_ms += self.cfg.dt_ms
            if not self.ignited and self.window_ms >= self.cfg.ignition_min_ms:
                self.ignited = True
                return True, 0.0  # Latency calculated externally
        else:
            self.window_ms = 0.0
        return self.ignited, None

class ThalamicGate:
    """Decides whether to poll the LLM module based on ignition state."""
    def __init__(self, cool_down_ms: float):
        self.cool_down_ms = cool_down_ms
        self.cool = 0.0

    def step(self, dt_ms: float, ignited: bool) -> bool:
        if self.cool > 0:
            self.cool = max(0.0, self.cool - dt_ms)
            return False
        if ignited:
            self.cool = self.cool_down_ms
            return True
        return False
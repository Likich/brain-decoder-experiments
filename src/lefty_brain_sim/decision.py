from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class DecisionConfig:
    tau_ms: float
    gamma: float
    noise: float
    bound: float

class DecisionNode:
    """
    Minimal 2-well decision dynamic akin to a reduced Wong–Wang.
    State y integrates signed evidence until |y| crosses 'bound'.
    """
    def __init__(self, cfg: DecisionConfig):
        self.cfg = cfg
        self.y = 0.0
        self.t_ms = 0.0
        self.done = False

    def reset(self):
        self.y = 0.0
        self.t_ms = 0.0
        self.done = False

    def step(self, dt_ms: float, evidence: float) -> tuple[bool, int | None, float | None, float | None]:
        if self.done:
            return True, int(self.y > 0), self.t_ms, self.confidence()
        cfg = self.cfg
        noise = np.random.normal(0, cfg.noise)
        dy = (cfg.gamma * evidence - 0.05 * self.y + noise) * (dt_ms / cfg.tau_ms)
        self.y += dy
        self.t_ms += dt_ms
        if abs(self.y) >= cfg.bound:
            self.done = True
            choice = 1 if self.y > 0 else 0
            return True, choice, self.t_ms, self.confidence()
        return False, None, None, None

    def confidence(self) -> float:
        return min(0.99, abs(self.y) / (self.cfg.bound + 1e-6))

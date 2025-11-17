from __future__ import annotations
import json
import numpy as np
import random
from dataclasses import dataclass, asdict
from typing import List, Optional


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class TrialResult:
    snr: str
    ignited: bool
    ignition_latency_ms: float | None
    choice: int | None  # 0/1
    rt_ms: float | None
    confidence: float | None
    report: str | None
    stimulus: str | None = None
    stimulus_id: int | None = None
    stimulus_token: str | None = None
    target_token_id: int | None = None
    decoder_snapshot: Optional[List[float]] = None
    activity_snapshot: Optional[List[float]] = None
    generated_tokens: Optional[List[str]] = None
    generated_token_ids: Optional[List[int]] = None

    def to_json(self) -> str:
        """
        Serialize to a JSON line, exactly what run_experiment.py expects.
        """
        return json.dumps(asdict(self))

from __future__ import annotations
import numpy as np

class StateEncoder:
    """Compress TVB language/workspace state → small feature vector."""
    def __init__(self, in_dim: int, out_dim: int = 32):
        self.W = np.random.randn(out_dim, in_dim) / np.sqrt(in_dim)

    def __call__(self, state_slice: np.ndarray) -> np.ndarray:
        z = self.W @ state_slice
        z = np.tanh(z)
        return z


class EvidenceDecoder:
    def __call__(self, evidence):
        """
        Backwards compatible:

        - old path: evidence is a dict like {"B": 0.3}
        - new local-decoder path: evidence is already a scalar float in [-1, 1]
        """
        # old API: dict from channel -> evidence
        if isinstance(evidence, dict):
            return float(evidence.get("B", 0.0))

        # new API: just a scalar
        return float(evidence)

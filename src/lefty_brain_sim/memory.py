from __future__ import annotations
import numpy as np

class HippocampalMemory:
    """Tiny in-memory approximate memory (placeholder for FAISS)."""
    def __init__(self, dim: int):
        self.keys = np.empty((0, dim))
        self.vals = []

    def add(self, key: np.ndarray, value: dict):
        key = key.reshape(1, -1)
        self.keys = np.vstack([self.keys, key])
        self.vals.append(value)

    def search(self, q: np.ndarray, k: int = 5) -> list[dict]:
        if len(self.vals) == 0:
            return []
        sims = (self.keys @ q) / (np.linalg.norm(self.keys, axis=1) * (np.linalg.norm(q) + 1e-6))
        idx = np.argsort(-sims)[:k]
        return [self.vals[i] for i in idx]

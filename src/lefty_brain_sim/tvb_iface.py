# src/lefty_brain_sim/tvb_iface.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from tvb.simulator.lab import connectivity, models, coupling, integrators, monitors, simulator
from tvb.simulator import noise as tvb_noise
import numpy as np
from tvb.simulator import integrators, coupling, models, simulator, noise as tvb_noise
from tvb.datatypes import connectivity
from tvb.simulator.lab import patterns, equations




@dataclass
class TVBConfig:
    regions: int
    dt_ms: float
    coupling: float
    noise: float
    workspace_nodes: list[int]
    language_nodes: list[int]

def _make_synthetic_connectivity(n: int) -> connectivity.Connectivity:
    """Create a small, symmetric, distance-based Connectivity for modern TVB."""
    C = connectivity.Connectivity()

    rng = np.random.default_rng(42)
    centres = rng.uniform(low=-50, high=50, size=(n, 3)).astype(np.float64)

    # Euclidean distances
    dmat = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    tract_lengths = dmat.astype(np.float64)

    # Sparse symmetric weights
    W = rng.random((n, n))
    W = (W > 0.95).astype(np.float64) * rng.uniform(0.2, 1.0, size=(n, n))
    W = np.triu(W, 1)
    W = W + W.T
    np.fill_diagonal(W, 0.0)

    # --- Proper dtypes ---
    C.weights = W
    C.tract_lengths = tract_lengths
    C.region_labels = np.array([f"R{i}" for i in range(n)], dtype="<U128")
    C.centres = centres
    C.cortical = np.ones((n,), dtype=bool)
    # Boolean left/right hemispheres (TVB requires bool, not int)
    hemi = np.zeros((n,), dtype=bool)
    hemi[n // 2 :] = True
    C.hemispheres = hemi
    C.speed = np.array([3.0], dtype=np.float64)

    # Make sure it finalizes cleanly
    C.configure()
    return C



class TVBEngine:
    def __init__(self, cfg: TVBConfig):
        self.cfg = cfg
        self.C = _make_synthetic_connectivity(cfg.regions)
        self.model = models.DecoBalancedExcInh()
        self.cpl = coupling.Linear(a=np.array([cfg.coupling], dtype=np.float64))
        if cfg.noise > 0:
            self.integrator = integrators.HeunStochastic(
                dt=cfg.dt_ms,
                noise=tvb_noise.Additive(nsig=np.array([cfg.noise], dtype=np.float64)),
            )
        else:
            self.integrator = integrators.HeunDeterministic(dt=cfg.dt_ms)

        self.monitors = (monitors.Raw(period=cfg.dt_ms),)
        self.sim = simulator.Simulator(
            connectivity=self.C,
            model=self.model,
            coupling=self.cpl,
            integrator=self.integrator,
            monitors=self.monitors,
        ).configure()

        self._latest = None
        self._time = 0.0
        # simple persistent drive buffer to emulate stimulus integration
        self._drive = np.zeros((self.C.number_of_regions,), dtype=np.float64)

    def reset(self):
        self.sim = self.sim.configure()
        self._latest = None
        self._time = 0.0
        self._drive[:] = 0.0

    def step(self, external_current: np.ndarray | None = None) -> np.ndarray:
        # run one dt
        t, raw = self.sim.run(simulation_length=self.cfg.dt_ms)[0]
        while raw is None:
            t, raw = self.sim.run(simulation_length=self.cfg.dt_ms)[0]

        x = raw[-1, 0, :, 0]  # first state var per region (float array)

        # persistent additive drive (decays, then adds new input)
        # decay factor ~0.9 gives short-term accumulation across a stimulus window
        self._drive *= 0.9
        if external_current is not None:
            self._drive += external_current.astype(np.float64)

        x = x + self._drive

        self._latest = x
        self._time = float(t[-1]) if hasattr(t, "__len__") else float(t)
        return x.copy()

    def readout(self, nodes: list[int]) -> np.ndarray:
        if self._latest is None:
            return np.zeros((len(nodes),))
        return self._latest[nodes]

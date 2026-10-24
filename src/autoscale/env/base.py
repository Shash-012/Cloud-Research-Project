"""Env interface. Policies only ever see this, so SimPy can later be swapped for a CloudSim bridge."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Observation:
    t: float                      # simulated time at end of the interval (s)
    w: int                        # attached VMs (what the metrics describe)
    pending: int                  # VMs still booting
    metrics: np.ndarray           # cluster-mean metric vector (length metric_count)
    metric_names: list
    rt_mean_ms: float             # mean RT of requests completed in the interval (nan if none)
    n_completed: int
    cpu_pct: float                # cluster-mean CPU utilisation (%)
    rr: float                     # observed arrival rate (req/s)
    hidden_complexity: float = 1.0   # diagnostics only; never given to models


class Env(ABC):
    @abstractmethod
    def reset(self, init_vms: int | None = None) -> Observation: ...

    @abstractmethod
    def step(self, delta_total: int) -> Observation:
        """Change the TOTAL pool (attached + booting) by delta_total, then advance one interval."""

    @property
    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def finalize(self): ...

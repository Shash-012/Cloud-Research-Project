"""Application profiles: how a request's work splits between CPU, disk and network.

The profile is what makes DIFFERENT metrics matter for different apps (paper Fig 1-3):
db_io is disk-bound, cpu_bound is CPU-bound, ml_batch has big payloads and memory per request.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class AppProfile:
    name: str
    vcpu: int
    slots: int
    io_cap: int
    cpu_ms: float
    io_ms: float
    net_in_kb: float
    net_out_kb: float
    mem_mb_per_req: float
    sigma: float
    qos_ms: float
    capacity_per_vm: float
    dist: str = "lognormal"      # "lognormal" | "exp" (exp is used to validate against M/M/c theory)

    @classmethod
    def from_cfg(cls, d: dict) -> "AppProfile":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: d[k] for k in keys if k in d})

    def _scale(self, rng: np.random.Generator, mean: float) -> float:
        if mean <= 0:
            return 0.0
        if self.dist == "exp":
            return float(rng.exponential(mean))
        if self.dist == "det":
            return mean
        return float(mean * np.exp(rng.normal(-0.5 * self.sigma ** 2, self.sigma)))

    def sample(self, rng: np.random.Generator, cx: float):
        """Return (cpu_s, io_s, kb_in, kb_out) for one request of hidden complexity cx."""
        cpu_s = self._scale(rng, self.cpu_ms / 1000.0 * cx)
        io_s = self._scale(rng, self.io_ms / 1000.0 * cx)
        return cpu_s, io_s, self.net_in_kb, self.net_out_kb

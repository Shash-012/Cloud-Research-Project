"""CPU-threshold baseline: +1 VM if CPU > CPUmax, -1 VM if CPU < CPUmin."""
from __future__ import annotations
from .base import Policy


class CPUThreshold(Policy):
    name = "cpu"

    def __init__(self, cpu_max: float, cpu_min: float, cooldown_s: float):
        self.hi, self.lo, self.cool = cpu_max, cpu_min, cooldown_s
        self.last = -1e9

    def reset(self):
        self.last = -1e9

    def decide(self, obs) -> int:
        if obs.t - self.last < self.cool:
            return 0
        if obs.cpu_pct > self.hi:
            self.last = obs.t; return 1
        if obs.cpu_pct < self.lo:
            self.last = obs.t; return -1
        return 0

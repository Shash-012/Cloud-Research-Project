"""RT-threshold baseline: +1 VM if RT > RTmax, -1 VM if RT < RTmin (the paper's text has '>' for the second
condition; that is a typo, '<' is meant). A cooldown stops it re-triggering while VMs boot."""
from __future__ import annotations
import math
from .base import Policy


class RTThreshold(Policy):
    name = "rt"

    def __init__(self, rt_max_ms: float, rt_min_ms: float, cooldown_s: float):
        self.hi, self.lo, self.cool = rt_max_ms, rt_min_ms, cooldown_s
        self.last = -1e9

    def reset(self):
        self.last = -1e9

    def decide(self, obs) -> int:
        if obs.t - self.last < self.cool or math.isnan(obs.rt_mean_ms):
            return 0
        if obs.rt_mean_ms > self.hi:
            self.last = obs.t; return 1
        if obs.rt_mean_ms < self.lo:
            self.last = obs.t; return -1
        return 0

"""OPT reactive (the paper's 'OPT baseline', defined as in the decision log): same trained models as the proactive
version, but the model is only consulted when measured RT leaves the Eq. 10 band; k is chosen by Eq. 8-9
(estimated RT closest to, and not above, RTmax)."""
from __future__ import annotations
import math
import numpy as np
from .base import Policy
from .opt_proactive import candidate_ks


def argmin_cost(bundle, obs, qos_ms, k_min, k_max, max_vms):
    ks = candidate_ks(obs.w, k_min, k_max, max_vms)
    mu, _ = bundle.estimate_rt(obs.metrics, obs.w, ks, obs.rr)
    cost = np.where(mu <= qos_ms, qos_ms - mu, np.inf)          # Eq. 8
    if np.isinf(cost).all():
        return int(ks[-1])                                       # nothing safe: Kmax
    return int(ks[int(np.argmin(cost))])                         # Eq. 9


class OPTReactive(Policy):
    name = "opt_reactive"

    def __init__(self, bundle, cfg: dict, target_frac: float = 1.0):
        self.b, self.cfg, self.target = bundle, cfg, target_frac * bundle.qos_ms   # see OPTProactive for target_frac
        self.up, self.lo = cfg["reactive_upper_frac"], cfg["reactive_lower_frac"]

    def decide(self, obs) -> int:
        q = self.target
        if math.isnan(obs.rt_mean_ms):
            return 0
        if obs.rt_mean_ms > self.up * q or obs.rt_mean_ms < self.lo * q:        # Eq. 10 trigger
            k = argmin_cost(self.b, obs, q, self.cfg["k_min"], self.cfg["k_max"], self.cfg["max_vms"])
            return k - obs.pending
        return 0

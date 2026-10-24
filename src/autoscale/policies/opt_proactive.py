"""OPT proactive = paper Algorithm 1, evaluated EVERY monitoring interval (no trigger, no lower threshold).

Scan k from Kmin upward, keep the first k with RT_est < QoSmax (= fewest VMs that are safe);
if none is predicted safe, fall back to Kmax. Fixes over the printed algorithm: loop is [Kmin, Kmax],
and candidates with w + k < 1 (division by zero / empty pool) or above the pool cap are skipped.
"""
from __future__ import annotations
import numpy as np
from .base import Policy


def candidate_ks(w: int, k_min: int, k_max: int, max_vms: int) -> np.ndarray:
    ks = np.arange(k_min, k_max + 1)
    return ks[(w + ks >= 1) & (w + ks <= max_vms)]


def algorithm1(bundle, obs, qos_ms, k_min, k_max, max_vms):
    """Returns (k_op, found_safe)."""
    ks = candidate_ks(obs.w, k_min, k_max, max_vms)
    mu, _ = bundle.estimate_rt(obs.metrics, obs.w, ks, obs.rr)
    for k, r in zip(ks, mu):                      # ks ascending: first safe k is the smallest safe pool
        if r < qos_ms:
            return int(k), True
    return int(ks[-1]), False                     # fallback: largest permitted scale-out (Kmax)


class OPTProactive(Policy):
    name = "opt_proactive"

    def __init__(self, bundle, cfg: dict, target_frac: float = 1.0):
        # target_frac < 1 keeps head-room: the network predicts MEAN RT per interval, while a violation is a
        # single request above the SLO, so the operator's RT target is a fraction of the SLO (tuned like the
        # baselines' thresholds, on a separate seed).
        self.b, self.cfg, self.target = bundle, cfg, target_frac * bundle.qos_ms

    def decide(self, obs) -> int:
        k, _ = algorithm1(self.b, obs, self.target, self.cfg["k_min"], self.cfg["k_max"], self.cfg["max_vms"])
        return k - obs.pending          # k is relative to attached VMs; convert to a change of the total pool

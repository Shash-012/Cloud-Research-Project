"""Synthetic workload generator and ground-truth application model.

Two things live here:

1. `generate_workload` — the arrival-rate timeline the control loop steps
   through. Shaped to contain a daily cycle, one gradual ramp, and one burst,
   so that the result plots have something to show.

2. `SyntheticApp` — the "real system" the autoscaler is trying to control. It
   maps (VM count, request rate) to a response time and a small vector of
   telemetry metrics. The autoscaler never sees this function; it only sees
   the metrics it emits. This is the stand-in for a CloudSim Plus simulation.

The response-time law is an M/M/1-flavoured queueing curve, RT = L0 / (1 - u),
which is deliberately non-linear in utilisation. The base paper predicts RT
from a linear-regression-estimated metric vector fed to a neural net; a
non-linear ground truth is what makes the non-linear predictor worth having.
"""

from __future__ import annotations

import numpy as np

from . import config as C


def generate_workload(n_steps: int = C.N_STEPS, seed: int = C.WORKLOAD_SEED) -> np.ndarray:
    """Return an array of request rates (req/s), one per control step."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=float)

    daily = C.BASE_RATE + C.DAILY_AMPLITUDE * np.sin(2.0 * np.pi * t / n_steps)

    ramp = np.clip((t - C.RAMP_START) / (C.RAMP_END - C.RAMP_START), 0.0, 1.0)
    ramp = ramp * C.RAMP_HEIGHT

    spike = C.SPIKE_HEIGHT * np.exp(-0.5 * ((t - C.SPIKE_CENTER) / C.SPIKE_WIDTH) ** 2)

    noise = rng.normal(0.0, C.RATE_NOISE_STD, size=n_steps)

    return np.clip(daily + ramp + spike + noise, C.RATE_MIN, C.RATE_MAX)


def complexity_at(step: float, rng: np.random.Generator) -> float:
    """Unobserved per-request complexity multiplier on service demand.

    Values above 1.0 mean each request costs more work than nominal, so the
    effective per-VM service rate drops. Nothing in the feature vector reveals
    this, which is the point: it is the irreducible part of the prediction
    error.
    """
    slow = 1.0 + C.COMPLEXITY_AMPLITUDE * np.sin(2.0 * np.pi * step / C.COMPLEXITY_PERIOD)
    return float(max(0.5, slow + rng.normal(0.0, C.COMPLEXITY_NOISE_STD)))


class SyntheticApp:
    """Ground-truth system under control. Not visible to the autoscaler."""

    def __init__(self, seed: int = C.SIM_SEED):
        self.rng = np.random.default_rng(seed)

    def utilisation(self, vm_count: int, request_rate: float, complexity: float) -> float:
        capacity = vm_count * C.SERVICE_RATE_PER_VM / complexity
        return float(np.clip(request_rate / max(capacity, 1e-9), 0.0, 0.995))

    def observe(self, vm_count: int, request_rate: float, step: float = 0.0) -> dict:
        """Run the app for one control interval and emit telemetry + RT.

        Returns a flat dict containing every name in C.FEATURES plus C.TARGET.
        """
        vm_count = int(np.clip(vm_count, C.VM_MIN, C.VM_MAX))
        complexity = complexity_at(step, self.rng)
        u = self.utilisation(vm_count, request_rate, complexity)

        rt = C.BASE_LATENCY_MS / (1.0 - u)
        rt *= 1.0 + self.rng.normal(0.0, C.RT_NOISE_FRAC)
        rt = float(max(C.BASE_LATENCY_MS * 0.5, rt))

        per_vm_rate = request_rate / vm_count

        return {
            "vm_count": float(vm_count),
            "request_rate": float(request_rate),
            "cpu_util": float(np.clip(100.0 * u + self.rng.normal(0.0, 1.5), 0.0, 100.0)),
            "mem_util": float(np.clip(30.0 + 45.0 * u + self.rng.normal(0.0, 2.0), 0.0, 100.0)),
            "net_in_kbps": float(max(0.0, 8.0 * request_rate + self.rng.normal(0.0, 20.0))),
            "disk_iops": float(max(0.0, 12.0 * per_vm_rate + self.rng.normal(0.0, 3.0))),
            "queue_len": float(max(0.0, u / (1.0 - u) + self.rng.normal(0.0, 0.15))),
            C.TARGET: rt,
        }

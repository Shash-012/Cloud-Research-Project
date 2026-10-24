"""128 synthetic collectl-like metrics with a PLANTED ground truth.

Each metric = f(driver) + noise, where the drivers (CPU busy, disk busy, request rate, ...) come from
the simulator. Metric kinds:
  linear     proportional to a per-VM driver           -> reacts to scaling, predictable by Eq. 1
  sat        convex/hyperbolic in a driver             -> reacts to scaling, NOT predictable by Eq. 1
  total      cluster-wide total (driver x w)           -> does NOT react to scaling
  slow       very slow EWMA of a driver (load avg)     -> barely reacts
  const/drift/noise                                    -> irrelevant distractors
Because noise is scaled to a fixed nominal magnitude per driver, a driver that is tiny for one app
(e.g. disk ops for cpu_bound) makes its metrics indistinguishable from noise there. That is what
produces application-specific metric subsets.

LIMITATION (state it in the report): this validates the selection PROCEDURE, not that these specific
collectl metrics matter on real systems.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

NOMINAL = dict(busy=1.0, io=1.0, rate=5.0, cmpl=5.0, inflight=4.0, runq=2.0, wait=2.0,
               netin=50.0, netout=100.0, mem=100.0, iops=10.0, pkts=100.0, total_rate=40.0)

# (group, count) -> list of (kind, driver, n)
PLAN = {
    "CPU": [("linear", "busy", 3), ("linear", "io", 2), ("linear", "cmpl", 2), ("linear", "inflight", 2),
            ("linear", "pkts", 1), ("sat", "runq", 2), ("sat", "busy", 1), ("slow", "busy", 2),
            ("const", None, 2), ("noise", None, 2), ("drift", None, 2)],
    "MEM": [("linear", "mem", 4), ("linear", "inflight", 2), ("linear", "cmpl", 2), ("const", None, 8),
            ("drift", None, 8), ("noise", None, 3), ("slow", "mem", 3)],
    "BUD": [("drift", None, 6), ("const", None, 4), ("noise", None, 4)],
    "DSK": [("linear", "iops", 4), ("linear", "io", 2), ("sat", "io", 1), ("const", None, 1), ("noise", None, 1)],
    "INODE": [("const", None, 2), ("drift", None, 2)],
    "NET": [("linear", "netin", 4), ("linear", "netout", 4), ("linear", "pkts", 3), ("linear", "rate", 2),
            ("total", "total_rate", 2), ("noise", None, 1), ("const", None, 1), ("drift", None, 1)],
    "TCP": [("linear", "rate", 4), ("linear", "cmpl", 3), ("linear", "netin", 2), ("linear", "netout", 2),
            ("linear", "inflight", 3), ("sat", "inflight", 3), ("sat", "wait", 2), ("total", "total_rate", 3),
            ("const", None, 5), ("drift", None, 3), ("noise", None, 2)],
}
HEADLINE = {("CPU", "linear", "busy", 0): "[CPU]User%", ("CPU", "linear", "cmpl", 0): "[CPU]Intrpt/sec",
            ("CPU", "linear", "inflight", 0): "[CPU]CtxSwitch/sec", ("CPU", "sat", "runq", 0): "[CPU]ProcRun",
            ("CPU", "slow", "busy", 0): "[CPU]L-Avg15", ("TCP", "linear", "rate", 0): "[TCPD]PassiveOpens",
            ("MEM", "linear", "mem", 0): "[MEM]Used", ("DSK", "linear", "iops", 0): "[DSK]WriteTot",
            ("INODE", "const", None, 0): "[INODE]OpenFiles"}


def metric_names() -> tuple[str, ...]:
    """Return the stable names used by the synthetic metric catalog."""
    return tuple(build_catalog()["name"])


def build_catalog(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for grp, items in PLAN.items():
        for kind, drv, n in items:
            for i in range(n):
                name = HEADLINE.get((grp, kind, drv, i), f"[{grp}]{kind}_{drv or 'x'}_{i}")
                a = float(rng.uniform(0.6, 2.0)) * (1 if rng.random() > 0.2 else -1)
                b = float(100.0 if a < 0 else rng.uniform(0, 10))
                rows.append(dict(name=name, group=grp, kind=kind, driver=drv, a=a, b=b))
    cat = pd.DataFrame(rows)
    assert len(cat) == 128 and cat["name"].is_unique, (len(cat), cat["name"].duplicated().sum())
    return cat


class MetricGenerator:
    def __init__(self, seed: int, interval_s: float = 10.0):
        self.catalog = build_catalog(0)               # same metric definitions for every run
        self.names = self.catalog["name"].tolist()
        self.rng = np.random.default_rng(seed + 991)
        self.dt = interval_s
        n = len(self.catalog)
        self.drift = np.zeros(n)
        self.slow = np.zeros(n)
        self._init_slow = True

    def _x(self, drv: str, d: dict, w: int) -> float:
        return d["rate"] * w if drv == "total_rate" else d[drv]

    def generate(self, d: dict, w: int) -> np.ndarray:
        cat, rng = self.catalog, self.rng
        out = np.zeros(len(cat))
        for i, r in enumerate(cat.itertuples()):
            kind, drv, a, b = r.kind, r.driver, r.a, r.b
            nom = NOMINAL.get(drv, 10.0) if drv else 10.0
            noise_abs = 0.015 * abs(a) * nom
            if kind in ("linear", "total"):
                x = self._x(drv, d, w)
                out[i] = a * x * (1 + rng.normal(0, 0.015)) + b + rng.normal(0, noise_abs)
            elif kind == "sat":
                x = d[drv]
                u = min(x, 0.99) if drv == "busy" else x / (x + nom)
                u = min(u, 0.99)
                out[i] = a * (u / (1.05 - u)) * (1 + rng.normal(0, 0.015)) + b + rng.normal(0, 0.04 * abs(a))
            elif kind == "slow":
                x = a * d[drv] + b
                alpha = self.dt / 1800.0
                self.slow[i] = x if self._init_slow else (1 - alpha) * self.slow[i] + alpha * x
                out[i] = self.slow[i] + rng.normal(0, noise_abs)
            elif kind == "const":
                out[i] = b * 5 + rng.normal(0, 0.002 * (abs(b) * 5 + 1))
            elif kind == "drift":
                self.drift[i] += rng.normal(0, 0.01 * 100)
                out[i] = 100 + b + self.drift[i]
            else:  # noise
                out[i] = b + rng.normal(0, 5.0)
        self._init_slow = False
        return out

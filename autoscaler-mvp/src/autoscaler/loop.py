"""Proactive control loop.

The structure follows Algorithm 1 of the base paper: on a periodic tick --
not only when the SLO is already breached -- estimate the post-scaling state
for every candidate action, predict response time, and apply the winning
delta. Everything the decision saw is logged, so the run can be replayed and
plotted afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .decision import decide, energy_of
from .environment import SyntheticApp
from .metric_estimator import MetricEstimator
from .predictor import EnsembleRTPredictor


def run(
    workload: np.ndarray,
    predictor: EnsembleRTPredictor,
    estimator: MetricEstimator,
    vm_start: int = 4,
    control_every: int = 5,
    seed: int = C.SIM_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Step the synthetic app through `workload`, autoscaling as it goes.

    `control_every` sets how many simulation steps pass between decisions --
    the loop observes at every step but only acts on a tick, which is how a
    real orchestrator behaves and which keeps the VM trajectory from thrashing.

    Returns (per-step log, list of every decision taken). Each decision records
    the full candidate set with its feasibility and Pareto flags, so any
    timestep can be re-plotted afterwards without re-running the loop.
    """
    app = SyntheticApp(seed=seed)
    vm = int(np.clip(vm_start, C.VM_MIN, C.VM_MAX))

    rows = []
    decisions = []

    for t, rate in enumerate(workload):
        obs = app.observe(vm, rate, step=t)

        chosen_k = 0
        used_fallback = False
        decision_made = False

        if t % control_every == 0:
            # Estimate the post-scaling feature row for each candidate, then
            # score them all in a single batched forward pass.
            ks = list(C.K_CANDIDATES)
            rows_X = np.vstack(
                [estimator.feature_vector(obs, vm, k, rate) for k in ks]
            )
            pred = predictor.predict(rows_X)
            predictions = {k: (pred.mean[i], pred.std[i]) for i, k in enumerate(ks)}

            chosen, candidates, used_fallback = decide(vm, rate, predictions)
            chosen_k = chosen.vm_after - vm
            decision_made = True

            decisions.append(
                {
                    "step": t,
                    "request_rate": float(rate),
                    "vm_current": vm,
                    "chosen_vm": chosen.vm_after,
                    "used_fallback": used_fallback,
                    "candidates": [
                        {
                            "k": c.k,
                            "vm_after": c.vm_after,
                            "mean_rt": c.mean_rt,
                            "std_rt": c.std_rt,
                            "upper_rt": c.upper_rt,
                            "cost": c.cost,
                            "energy": c.energy,
                            "feasible": c.feasible,
                            "pareto": c.pareto,
                        }
                        for c in candidates
                    ],
                }
            )

        rows.append(
            {
                "step": t,
                "request_rate": float(rate),
                "vm_count": vm,
                "response_time_ms": obs[C.TARGET],
                "slo_violated": obs[C.TARGET] > C.RT_SLO_MS,
                "cpu_util": obs["cpu_util"],
                "queue_len": obs["queue_len"],
                "chosen_k": chosen_k,
                "decision_made": decision_made,
                "used_fallback": used_fallback,
                "cost": vm * C.DT_HOURS * C.PRICE_PER_VM_HOUR,
                "energy_wh": energy_of(vm, rate),
            }
        )

        vm = int(np.clip(vm + chosen_k, C.VM_MIN, C.VM_MAX))

    return pd.DataFrame(rows), decisions

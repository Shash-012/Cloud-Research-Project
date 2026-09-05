"""Pareto-aware, uncertainty-filtered scaling decision.

This is the second extension point and the heart of the contribution. The base
paper picks the smallest |k| whose predicted response time stays under the QoS
maximum -- one objective, a point estimate, first acceptable answer wins. Here
the decision is three stages:

  1. Chance-constrained feasibility.  Keep only actions where
         mean + z * std <= RT_SLO
     The z term is what turns a point estimate into a risk-aware one: an
     action whose prediction is uncertain must have a better mean to survive.

  2. Pareto filter over (cost, energy).  Among feasible actions, discard any
     action that some other feasible action beats on both objectives. What
     survives is the non-dominated set -- the actual trade-off frontier.

  3. Tie-break by minimum cost.

Fallback: if no action is feasible, take the largest allowed scale-out. This
mirrors Algorithm 1, lines 14-15 of the base paper rather than leaving the
behaviour undefined.

This module depends on nothing but config and numpy. It takes (mean, std) per
candidate as plain numbers, so it can be unit-tested with fabricated
predictions -- see tests/test_decision.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C


@dataclass
class Candidate:
    k: int
    vm_after: int
    mean_rt: float
    std_rt: float
    cost: float
    energy: float
    feasible: bool = False
    pareto: bool = False
    extras: dict = field(default_factory=dict)

    @property
    def upper_rt(self) -> float:
        """Risk-adjusted response time: mean + z * std."""
        return self.mean_rt + C.Z_SCORE * self.std_rt


def cost_of(vm_count: int, dt_hours: float = C.DT_HOURS) -> float:
    """Billed VM-hours at a flat per-VM price. Monotone increasing in vm_count."""
    return vm_count * dt_hours * C.PRICE_PER_VM_HOUR


def energy_of(vm_count: int, request_rate: float, dt_hours: float = C.DT_HOURS) -> float:
    """Watt-hours over one control interval, from a utilisation-aware power model.

    Per VM:  P(u) = P_IDLE + (P_MAX - P_IDLE) * u**ALPHA,  u = lambda / (n * mu)

    STAND-IN, NOT CLOUDSIM. CloudSim Plus supplies real PowerModel classes that
    integrate over host utilisation; this is a closed-form imitation of
    PowerModelSquare so the MVP has a defensible second objective. Replacing it
    with CloudSim's model is part of the next milestone.

    Because ALPHA > 1, total pool power falls and then rises as VMs are added:
    idle draw grows linearly while the dynamic term shrinks super-linearly. That
    is what makes (cost, energy) genuinely conflict over part of the range --
    with a flat per-VM power draw the two objectives would be the same number
    twice and the Pareto front would collapse to a single point.
    """
    vm_count = max(vm_count, C.VM_MIN)
    u = np.clip(request_rate / (vm_count * C.SERVICE_RATE_PER_VM), 0.0, 1.0)
    per_vm_w = C.P_IDLE_W + (C.P_MAX_W - C.P_IDLE_W) * (u ** C.POWER_ALPHA)
    return float(vm_count * per_vm_w * dt_hours)


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows. Both columns are minimised.

    Row i is dominated if some row j is <= on every objective and < on at least
    one. Duplicate points are all kept (neither strictly beats the other).
    """
    n = len(points)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                mask[i] = False
                break
    return mask


def build_candidates(
    vm_current: int,
    request_rate: float,
    predictions: dict[int, tuple[float, float]],
    dt_hours: float = C.DT_HOURS,
) -> list[Candidate]:
    """Turn {k: (mean_rt, std_rt)} into scored Candidate objects.

    Actions that clamp to the same VM count (e.g. k=-1 and k=-2 at the floor)
    are collapsed, so the candidate list holds distinct pool sizes only.
    """
    seen: set[int] = set()
    candidates: list[Candidate] = []
    for k in sorted(predictions):
        vm_after = int(np.clip(vm_current + k, C.VM_MIN, C.VM_MAX))
        if vm_after in seen:
            continue
        seen.add(vm_after)
        mean_rt, std_rt = predictions[k]
        candidates.append(
            Candidate(
                k=k,
                vm_after=vm_after,
                mean_rt=float(mean_rt),
                std_rt=float(std_rt),
                cost=cost_of(vm_after, dt_hours),
                energy=energy_of(vm_after, request_rate, dt_hours),
            )
        )
    return candidates


def decide(
    vm_current: int,
    request_rate: float,
    predictions: dict[int, tuple[float, float]],
    rt_slo: float = C.RT_SLO_MS,
    dt_hours: float = C.DT_HOURS,
) -> tuple[Candidate, list[Candidate], bool]:
    """Run the full three-stage decision.

    Returns (chosen, all_candidates, used_fallback). Every candidate is
    annotated in place with its `feasible` and `pareto` flags so the caller can
    log or plot the whole decision, not just the winner.
    """
    candidates = build_candidates(vm_current, request_rate, predictions, dt_hours)

    for c in candidates:
        c.feasible = c.upper_rt <= rt_slo

    feasible = [c for c in candidates if c.feasible]

    if not feasible:
        # Base paper Algorithm 1, lines 14-15: nothing satisfies the SLO, so
        # take the most aggressive scale-out available and accept the breach.
        fallback = max(candidates, key=lambda c: c.vm_after)
        return fallback, candidates, True

    pts = np.array([[c.cost, c.energy] for c in feasible], dtype=float)
    mask = pareto_front(pts)
    for c, keep in zip(feasible, mask):
        c.pareto = bool(keep)

    front = [c for c in feasible if c.pareto]
    chosen = min(front, key=lambda c: (c.cost, c.energy, abs(c.k)))
    return chosen, candidates, False

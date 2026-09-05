"""Unit tests for the decision layer, using fabricated predictions.

The decision module never touches the ensemble or the environment, so it can be
exercised against hand-written (mean, std) pairs. That is the point of testing
it in isolation: if the loop later misbehaves, these tests say whether the
decision rule or the predictor is at fault.

Run:  python -m pytest tests -q      (from the autoscaler-mvp directory)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autoscaler import config as C  # noqa: E402
from autoscaler.decision import (  # noqa: E402
    build_candidates,
    cost_of,
    decide,
    energy_of,
    pareto_front,
)


def test_pareto_front_simple():
    pts = np.array([[1.0, 5.0], [2.0, 2.0], [3.0, 9.0], [5.0, 1.0]])
    mask = pareto_front(pts)
    # [3, 9] is beaten by [2, 2] on both objectives; nothing else is dominated.
    assert list(mask) == [True, True, False, True]


def test_pareto_front_keeps_duplicates():
    pts = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert pareto_front(pts).all()


def test_cost_is_monotone_in_vm_count():
    costs = [cost_of(n) for n in range(1, 11)]
    assert costs == sorted(costs)
    assert costs[0] < costs[-1]


def _min_feasible_vms(rate):
    """Smallest pool whose nominal response time clears the SLO."""
    u_slo = 1.0 - C.BASE_LATENCY_MS / C.RT_SLO_MS
    return int(np.ceil(rate / (C.SERVICE_RATE_PER_VM * u_slo)))


def test_energy_falls_before_it_rises():
    """The whole (cost, energy) trade-off rests on energy being non-monotone.

    Checked away from saturation: below the SLO-feasible pool size the power
    model clips at P_MAX and stops being meaningful, which is fine because the
    feasibility filter never lets the Pareto stage see those actions.
    """
    rate = 150.0
    lo = _min_feasible_vms(rate)
    energies = [energy_of(n, rate) for n in range(lo, C.VM_MAX + 1)]
    assert energies[1] < energies[0], "adding a VM should first reduce total energy"
    assert min(energies) < energies[-1], "and the curve should turn back up"


def test_energy_optimum_sits_above_the_slo_boundary():
    """If it did not, every feasible front would collapse to the cheapest action."""
    for rate in (100.0, 150.0, 200.0):
        n_star = (rate / C.SERVICE_RATE_PER_VM) * np.sqrt(
            (C.P_MAX_W - C.P_IDLE_W) / C.P_IDLE_W
        )
        assert n_star > _min_feasible_vms(rate)


def test_front_has_more_than_one_point_in_the_conflict_region():
    rate = 150.0
    vm = _min_feasible_vms(rate) + 1
    preds = {k: (120.0, 3.0) for k in C.K_CANDIDATES}  # all feasible by construction
    _, cands, fb = decide(vm, rate, preds)
    assert not fb
    assert sum(c.pareto for c in cands) > 1


def test_uncertainty_makes_an_action_infeasible():
    """Same mean, more spread -> filtered out by the chance constraint."""
    slo = C.RT_SLO_MS
    confident = {0: (slo - 20.0, 1.0)}
    uncertain = {0: (slo - 20.0, 60.0)}

    c1, _, fb1 = decide(5, 200.0, confident)
    assert not fb1 and c1.feasible

    _, cands2, fb2 = decide(5, 200.0, uncertain)
    assert fb2, "high variance should push the action out of the feasible set"
    assert not cands2[0].feasible


def test_prefers_cheaper_of_two_feasible_actions():
    preds = {k: (100.0, 5.0) for k in C.K_CANDIDATES}  # all comfortably feasible
    chosen, cands, fb = decide(10, 150.0, preds)
    assert not fb
    feasible = [c for c in cands if c.feasible]
    assert chosen.cost == min(c.cost for c in feasible)


def test_fallback_scales_out_hard_when_nothing_is_feasible():
    preds = {k: (C.RT_SLO_MS * 3, 10.0) for k in C.K_CANDIDATES}
    chosen, cands, fb = decide(8, 400.0, preds)
    assert fb
    assert chosen.vm_after == max(c.vm_after for c in cands)


def test_candidates_collapse_at_the_vm_floor():
    preds = {k: (100.0, 1.0) for k in C.K_CANDIDATES}
    cands = build_candidates(C.VM_MIN, 100.0, preds)
    vm_counts = [c.vm_after for c in cands]
    assert len(vm_counts) == len(set(vm_counts))
    assert min(vm_counts) == C.VM_MIN


def test_chosen_action_is_on_the_front():
    preds = {k: (120.0 - 4.0 * k, 6.0) for k in C.K_CANDIDATES}
    chosen, _, fb = decide(6, 200.0, preds)
    assert not fb
    assert chosen.pareto


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

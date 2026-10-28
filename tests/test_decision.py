import numpy as np
import pytest

from autoscaler.config import default_prototype_config
from autoscaler.decision.objectives import candidate_cost, candidate_energy, objective_matrix
from autoscaler.decision.pareto import pareto_mask
from autoscaler.decision.power import idle_for_ratio, power
from autoscaler.decision.selection import select


@pytest.fixture
def cfg():
    c = default_prototype_config()
    c["power"] = {"model": "square", "p_idle_w": 80.0, "p_max_w": 400.0}
    return c


def worked_example(cfg, lam=150.0):
    """Notes example: lambda = 150, n = 8..12, square power, f_L = 40 / (1 - u)."""
    n = np.arange(8, 13)
    u = lam / (25.0 * n)
    U = 40.0 / (1.0 - u)
    w = 10
    return n - w, n, objective_matrix(U, n, u, cfg)


def test_worked_example_all_on_front(cfg):
    ks, n, F = worked_example(cfg)
    assert pareto_mask(F).all()
    out = select(ks, n, F, slo=250.0)
    assert out["on_front"].all() and out["feasible"].all()


def test_worked_example_aug_tchebycheff_picks_10(cfg):
    ks, n, F = worked_example(cfg)
    assert n[select(ks, n, F, 250.0, "aug_tchebycheff", (1, 1, 1), 0.01)["index"]] == 10


def test_worked_example_weighted_sum_picks_11(cfg):
    ks, n, F = worked_example(cfg)
    assert n[select(ks, n, F, 250.0, "weighted_sum", (1, 1, 1))["index"]] == 11


def test_worked_example_min_cost_picks_8(cfg):
    ks, n, F = worked_example(cfg)
    assert n[select(ks, n, F, 250.0, "min_cost")["index"]] == 8


def test_min_cost_smallest_feasible_pool_1000_random_sets(cfg):
    rng = np.random.default_rng(0)
    for _ in range(1000):
        w = int(rng.integers(1, 31))
        ks = np.array([k for k in range(-7, 8) if 1 <= w + k <= 30])
        n = w + ks
        U = rng.uniform(50, 400, len(ks))
        u_hat = rng.uniform(0, 1, len(ks))
        F = objective_matrix(U, n, u_hat, cfg)
        out = select(ks, n, F, 250.0, "min_cost")
        feas = U <= 250.0
        if feas.any():
            assert not out["fallback"]
            assert n[out["index"]] == n[feas].min()
        else:
            assert out["fallback"]


def test_linear_power_energy_increases_with_n(cfg):
    lam_over_mu = 6.0                       # total work in VM units
    n = np.arange(7, 31)
    e = candidate_energy(n, lam_over_mu / n, cfg, {"model": "linear", "p_idle_w": 80.0, "p_max_w": 400.0})
    assert np.all(np.diff(e) > 0)


def test_fallback_when_all_infeasible(cfg):
    ks = np.arange(-2, 3)
    n = 10 + ks
    F = objective_matrix(np.full(5, 900.0), n, np.full(5, 0.9), cfg)
    out = select(ks, n, F, 250.0)
    assert out["fallback"] and n[out["index"]] == 12
    assert not out["on_front"].any()


def test_dominated_candidate_never_chosen(cfg):
    rng = np.random.default_rng(1)
    for _ in range(500):
        K = int(rng.integers(2, 15))
        ks = np.arange(K)
        n = 5 + ks
        F = rng.uniform(0, 1, (K, 3)) * [200, 1, 1]
        out = select(ks, n, F, 250.0, "aug_tchebycheff")
        assert pareto_mask(F)[out["index"]]


def test_pareto_mask_basic():
    F = np.array([[1, 1], [2, 2], [1, 2], [0, 3]])
    assert pareto_mask(F).tolist() == [True, False, False, True]
    assert pareto_mask(np.array([[1, 1], [1, 1]])).tolist() == [True, True]   # equal rows: neither dominates


def test_power_models_and_ratio():
    assert power(0.5, "linear", 80, 400) == pytest.approx(240)
    assert power(0.5, "square", 80, 400) == pytest.approx(160)
    assert power(0.5, "cubic", 80, 400) == pytest.approx(120)
    assert power(0.25, "sqrt", 80, 400) == pytest.approx(240)
    assert idle_for_ratio(400, 4) == pytest.approx(80)


def test_cost_scales_with_n(cfg):
    assert candidate_cost([1, 2], cfg).tolist() == pytest.approx([0.1 / 60, 0.2 / 60])


def test_two_objective_and_min_energy(cfg):
    ks, n, F = worked_example(cfg)
    out = select(ks, n, F, 250.0, "aug_tchebycheff", objectives=("cost", "energy"))
    assert not out["fallback"] and out["on_front"].sum() == 5
    assert n[select(ks, n, F, 250.0, "min_energy")["index"]] == 12


def test_ties_prefer_smaller_abs_k(cfg):
    ks = np.array([-1, 0, 1])
    F = np.array([[100.0, 1.0, 1.0]] * 3)
    assert ks[select(ks, 10 + ks, F, 250.0)["index"]] == 0

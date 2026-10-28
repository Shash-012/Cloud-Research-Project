import numpy as np
import pytest

from autoscaler.uncertainty.propagation import (bounds_from_means, candidate_bounds,
                                                 quantile_candidate_bounds)

Z = 1.6448536269514722
METRICS = ["cpu_util", "disk_iops"]


class FakeStage1:
    def __init__(self, cov, metrics=METRICS):
        self.metrics = list(metrics)
        self.residual_cov = np.asarray(cov, float)
        rng = np.random.default_rng(0)
        self.residuals = rng.multivariate_normal(np.zeros(len(metrics)), self.residual_cov + 1e-12, 500)

    def predict(self, before, w, ks, metrics=None):
        ks = np.asarray(ks, float)
        n = np.maximum(w + ks, 1)
        return np.column_stack([before[m] * w / n for m in self.metrics])


class FakeEnsemble:
    """Linear members: mu_j = X @ a_j + b_j, constant variance."""

    columns = ["cpu_util_after", "disk_iops_after", "n_after"]

    def __init__(self, slopes, biases, var):
        self.a = np.asarray(slopes, float)
        self.b = np.asarray(biases, float)
        self.v = float(var)
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        mu = np.stack([X @ a + b for a, b in zip(self.a, self.b)])
        return mu, np.full_like(mu, self.v)


BEFORE = {"cpu_util": 60.0, "disk_iops": 300.0}
COV = [[4.0, 1.0], [1.0, 9.0]]
KS = np.arange(-3, 4)


def _ens(M=5, shared_slope=False, var=0.02):
    rng = np.random.default_rng(1)
    base = np.array([0.01, 0.001, -0.02])
    slopes = np.tile(base, (M, 1)) if shared_slope else base + rng.normal(0, 0.003, (M, 3))
    return FakeEnsemble(slopes, rng.normal(4.0, 0.05, M), var)


def test_zero_cov_full_equals_ale_epi():
    s1, ens = FakeStage1(np.zeros((2, 2))), _ens()
    full = candidate_bounds(s1, ens, BEFORE, 8, KS, Z, 32, np.random.default_rng(0), mode="full")
    ae = candidate_bounds(s1, ens, BEFORE, 8, KS, Z, 32, np.random.default_rng(0), mode="ale_epi")
    for key in ("mean_log", "sigma", "U", "sigma_ale", "sigma_epi", "sigma_stage1"):
        assert np.allclose(full[key], ae[key])


def test_one_member_zero_aleatoric_gives_zero_sigma():
    s1, ens = FakeStage1(COV), _ens(M=1, var=0.0)
    for mode in ("ale_epi", "epistemic", "none"):
        out = candidate_bounds(s1, ens, BEFORE, 8, KS, Z, 32, np.random.default_rng(0), mode=mode)
        assert np.allclose(out["sigma"], 0.0)
        assert np.allclose(out["U"], np.exp(out["mean_log"]))
    full0 = candidate_bounds(FakeStage1(np.zeros((2, 2))), ens, BEFORE, 8, KS, Z, 32,
                             np.random.default_rng(0), mode="full")
    assert np.allclose(full0["sigma"], 0.0)


def test_increasing_cov_never_decreases_sigma():
    ens = _ens(shared_slope=True)
    prev = None
    for c in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0]:
        out = candidate_bounds(FakeStage1(np.asarray(COV) * c), ens, BEFORE, 8, KS, Z, 32,
                               np.random.default_rng(3), mode="full")
        if prev is not None:
            assert np.all(out["sigma"] >= prev - 1e-12)
        prev = out["sigma"]


def test_decomposition_matches_population_variance():
    ens = _ens(M=5)
    rng = np.random.default_rng(7)
    means = np.array([[60.0, 300.0], [40.0, 100.0]])
    n = np.array([8.0, 12.0])
    cov = np.asarray(COV)
    S = 32
    out = bounds_from_means(ens, means, n, METRICS, cov, None, Z, S, rng, mode="full")
    # recompute the M*S means with the same draws
    rng = np.random.default_rng(7)
    draws = rng.standard_normal((2, S, 2)) @ np.linalg.cholesky(cov).T
    vals = (means[:, None, :] + draws)
    X = np.column_stack([vals.reshape(-1, 2), np.repeat(n, S)])
    mu = np.stack([X @ a + b for a, b in zip(ens.a, ens.b)]).reshape(5, 2, S)
    pop = mu.reshape(5, 2, S).transpose(1, 0, 2).reshape(2, -1).var(axis=1)
    assert np.allclose(out["sigma_epi"] ** 2 + out["sigma_stage1"] ** 2, pop, rtol=1e-10, atol=1e-14)
    # and the total is aleatoric variance plus that spread (population divisor, see propagation.py)
    assert np.allclose(out["sigma"] ** 2, ens.v + pop)


def test_one_batched_pass_and_reproducible():
    s1, ens = FakeStage1(COV), _ens()
    a = candidate_bounds(s1, ens, BEFORE, 8, KS, Z, 32, np.random.default_rng(5))
    assert ens.calls == 1
    b = candidate_bounds(s1, ens, BEFORE, 8, KS, Z, 32, np.random.default_rng(5))
    assert np.array_equal(a["U"], b["U"])
    assert set(a) >= {"n", "mean_log", "sigma", "sigma_ale", "sigma_epi", "sigma_stage1", "U"}


def test_full_wider_than_ale_epi_and_epistemic_narrowest():
    s1, ens = FakeStage1(COV), _ens()
    rng = np.random.default_rng(0)
    full = candidate_bounds(s1, ens, BEFORE, 8, KS, Z, 64, rng, mode="full")
    ae = candidate_bounds(s1, ens, BEFORE, 8, KS, Z, 64, rng, mode="ale_epi")
    ep = candidate_bounds(s1, ens, BEFORE, 8, KS, Z, 64, rng, mode="epistemic")
    none = candidate_bounds(s1, ens, BEFORE, 8, KS, Z, 64, rng, mode="none")
    assert np.all(ep["sigma"] < ae["sigma"])
    assert np.all(none["sigma"] == 0)
    assert np.mean(full["sigma"]) > np.mean(ae["sigma"])


def test_resample_mode_runs():
    s1, ens = FakeStage1(COV), _ens()
    out = candidate_bounds(s1, ens, BEFORE, 8, KS, Z, 16, np.random.default_rng(0), resample=True)
    assert np.all(out["sigma"] > 0)


def test_quantile_bounds():
    class Q:
        columns = ["cpu_util_after", "disk_iops_after", "n_after"]

        def predict(self, X):
            return np.log(100.0) + 0 * X[:, 0]
    out = quantile_candidate_bounds(FakeStage1(COV), Q(), BEFORE, 8, KS)
    assert np.allclose(out["U"], 100.0) and np.all(out["sigma"] == 0)


def test_unknown_mode():
    with pytest.raises(ValueError):
        candidate_bounds(FakeStage1(COV), _ens(), BEFORE, 8, KS, Z, 4, np.random.default_rng(0), mode="x")

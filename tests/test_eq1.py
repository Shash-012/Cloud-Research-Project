import numpy as np
from autoscale.models.metric_lr import MetricLR
from autoscale.policies.opt_proactive import candidate_ks


def test_eq1_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    N = 500
    m = rng.uniform(10, 90, (N, 2)); w = rng.integers(2, 12, N).astype(float)
    k = rng.integers(-1, 4, N).astype(float); k[k == 0] = 1
    c = np.array([[2.0, 0.95, 0.05], [-1.0, 1.0, 0.3]])
    mp = np.stack([c[j, 0] + c[j, 1] * m[:, j] * w / (w + k) + c[j, 2] * m[:, j] * k / (w + k) for j in range(2)], 1)
    lr = MetricLR().fit(m, w, k, mp)
    assert np.allclose(lr.coef, c, atol=1e-6)
    assert (lr.r2(m, w, k, mp) > 0.999999).all()


def test_candidate_guard_blocks_empty_pool():
    ks = candidate_ks(w=3, k_min=-7, k_max=7, max_vms=14)
    assert ks.min() == -2 and (3 + ks >= 1).all()      # k=-3 would give w+k=0 (division by zero)
    assert (3 + ks <= 14).all()
    assert candidate_ks(w=14, k_min=-7, k_max=7, max_vms=14).max() == 0


def test_saturation_limitation_is_visible():
    """Eq. 1 is linear, so it can predict impossible CPU values (150%): documented limitation / Part-2 motivation."""
    lr = MetricLR(); lr.coef = np.array([[2.0, 0.95, 0.05]]); lr.lo = np.array([-1e9]); lr.hi = np.array([1e9])
    p = lr.predict(np.array([80.0]), 4, [-2])[0, 0]
    assert p > 100


def test_sgd_update_moves_toward_observation():
    lr = MetricLR(); lr.coef = np.array([[0.0, 1.0, 0.0]]); lr.lo = np.array([0.0]); lr.hi = np.array([100.0])
    m = np.array([50.0]); obs = np.array([80.0])
    before = abs(lr.predict(m, 4, [2])[0, 0] - 80)
    assert lr.sgd_update(m, 4, 2, obs, lr=0.5, rel_thr=0.05)
    assert abs(lr.predict(m, 4, [2])[0, 0] - 80) < before

import numpy as np
import pytest

from autoscaler.uncertainty.calibration_metrics import (conditional_coverage, coverage, gaussian_nll,
                                                         mean_margin, pinball_loss, reliability)
from autoscaler.uncertainty.conformal import bound_z, conformal_z


def _gauss(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.uniform(3, 6, n)
    s = rng.uniform(0.05, 0.3, n)
    y = np.exp(m + s * rng.standard_normal(n))
    return y, m, s


def test_calibrated_gaussian_has_nominal_coverage():
    y, m, s = _gauss()
    U = np.exp(m + 1.6448536 * s)
    assert coverage(y, U) == pytest.approx(0.95, abs=0.005)
    rel = reliability(y, m, s, [0.5, 0.8, 0.95])
    assert np.allclose(rel, [0.5, 0.8, 0.95], atol=0.01)


def test_margin_and_nll():
    y, m, s = _gauss(1000)
    assert mean_margin(np.exp(m), m) == 0.0
    assert gaussian_nll(y, m, s) < gaussian_nll(y, m, s * 3)
    assert np.isnan(gaussian_nll(y, m, np.zeros_like(s)))


def test_conditional_coverage_bins():
    y, m, s = _gauss(5000)
    t = conditional_coverage(y, np.exp(m + 1.645 * s), m, bins=5, slo=250.0)
    assert len(t) == 5 and t.rows.sum() == 5000
    assert t.contains_slo.sum() == 1
    assert (t.coverage > 0.9).all()


def test_pinball_minimised_at_true_quantile():
    rng = np.random.default_rng(1)
    y = np.exp(rng.standard_normal(20000))
    losses = {q: pinball_loss(y, np.full(len(y), q), 0.95) for q in (1.0, 1.645, 2.3)}
    assert min(losses, key=losses.get) == 1.645


def test_conformal_z_recovers_gaussian_quantile():
    y, m, s = _gauss(20000, seed=3)
    assert conformal_z(y, m, s, 0.05) == pytest.approx(1.645, abs=0.05)
    # inflated sigma -> smaller z
    assert conformal_z(y, m, 2 * s, 0.05) == pytest.approx(0.82, abs=0.05)


def test_conformal_rank_formula():
    # scores 1..19 -> rank ceil(20 * 0.95) = 19 -> 19th smallest
    y = np.exp(np.arange(1, 20, dtype=float))
    assert conformal_z(y, np.zeros(19), np.ones(19), 0.05) == 19.0
    with pytest.raises(ValueError):
        conformal_z(y, np.zeros(19), np.zeros(19), 0.05)


def test_bound_z_switch(tmp_path):
    import json
    cfg = {"slo": {"z": 1.6449, "use_conformal": False}}
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"full": 2.1}))
    assert bound_z(cfg, "full", p) == 1.6449
    cfg["slo"]["use_conformal"] = True
    assert bound_z(cfg, "full", p) == 2.1

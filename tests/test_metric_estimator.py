import numpy as np
import pytest

from autoscaler.config import default_prototype_config
from autoscaler.data.calibration import collect_synthetic
from autoscaler.models.features import basis_features, paper_features
from autoscaler.models.metric_estimator import MetricEstimator, MLPEnsembleRegressor


@pytest.fixture(scope="module")
def setup():
    cfg = default_prototype_config()
    cfg["calibration"]["rows"] = 1500
    cfg["stage1"]["hgb"]["max_iter"] = 40     # keep the test fast
    df = collect_synthetic(cfg, np.random.default_rng(5))
    est = MetricEstimator(cfg, candidates=["paper_linear", "basis_linear", "hgb"])
    est.fit(df, ["cpu_util", "disk_iops", "mem_util"])
    return cfg, df, est


def test_features():
    p = paper_features([10.0], [4], [1])
    assert np.allclose(p, [[8.0, 2.0]])
    b = basis_features([10.0], [4], [-4])     # n = max(0, 1) = 1
    assert np.allclose(b, [[10.0, 4.0, -4.0, 40.0]])


def test_disk_iops_nearly_perfect_with_linear(setup):
    _, _, est = setup
    assert est.scores["disk_iops"]["basis_linear"] > 0.99
    assert est.chosen["disk_iops"] in ("paper_linear", "basis_linear")


def test_predict_shapes_and_clipping(setup):
    _, _, est = setup
    before = {"cpu_util": 99.0, "disk_iops": 500.0, "mem_util": 70.0}
    out = est.predict(before, w=3, ks=np.arange(-2, 8))
    assert out.shape == (10, 3)
    assert out[:, 0].min() >= 0 and out[:, 0].max() <= 100
    assert out[:, 1].min() >= 0


def test_predict_frame_matches_predict(setup):
    _, df, est = setup
    row = df.iloc[0]
    k_eff = int(row.n_after - row.n_before)
    single = est.predict({m: row[f"{m}_before"] for m in est.metrics}, int(row.n_before), [k_eff])
    assert np.allclose(est.predict_frame(df.iloc[:1]), single)


def test_oof_residuals_bigger_than_in_sample(setup):
    _, df, est = setup
    in_sample = df[[f"{m}_after" for m in est.metrics]].to_numpy() - est.predict_frame(df)
    assert est.residuals.shape == (len(df), 3)
    assert np.mean(est.residuals ** 2) >= np.mean(in_sample ** 2)


def test_restrict_and_cov(setup):
    _, _, est = setup
    est.restrict(["cpu_util", "disk_iops"])
    assert est.residual_cov.shape == (2, 2)
    est.restrict(["cpu_util", "disk_iops", "mem_util"])
    with pytest.raises(ValueError):
        est.restrict(["queue_len"])


def test_save_load_roundtrip(setup, tmp_path):
    _, df, est = setup
    est.save(tmp_path / "s1.joblib")
    est2 = MetricEstimator.load(tmp_path / "s1.joblib")
    assert np.array_equal(est.predict_frame(df.iloc[:20]), est2.predict_frame(df.iloc[:20]))


def test_mlp_ensemble_deterministic_and_averages():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, 300)
    a = MLPEnsembleRegressor(members=2, hidden=(8,), max_iter=50, seed=1).fit(X, y)
    b = MLPEnsembleRegressor(members=2, hidden=(8,), max_iter=50, seed=1).fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))
    assert np.allclose(a.predict(X), np.mean([m.predict(X) for m in a.models_], axis=0))

import numpy as np
import pytest

from autoscaler.config import default_prototype_config
from autoscaler.env.synthetic import SyntheticSystem


@pytest.fixture
def cfg():
    return default_prototype_config()


def test_rt_monotone_in_rate(cfg):
    sys_ = SyntheticSystem(cfg, np.random.default_rng(0))
    means = [np.mean([sys_.observe(rate, 10, 0)["rt_ms"] for _ in range(200)])
             for rate in np.linspace(50, 240, 12)]
    assert all(b > a for a, b in zip(means, means[1:]))


def test_noiseless_rt_at_u084_is_slo(cfg):
    sys_ = SyntheticSystem(cfg, np.random.default_rng(0))
    assert sys_.noiseless_rt(0.84) == pytest.approx(250.0)


def test_noise_free_observation_matches_formula(cfg):
    cfg["system"]["rt_noise_frac"] = 0.0
    sys_ = SyntheticSystem(cfg, np.random.default_rng(0))
    # 84% utilisation with c = 1: rate = 0.84 * n * mu
    obs = sys_.observe(0.84 * 4 * 25.0, 4, 0, c=1.0)
    assert obs["u"] == pytest.approx(0.84)
    assert obs["rt_ms"] == pytest.approx(250.0)


def test_vm_added_at_t_serves_from_t_plus_1(cfg):
    sys_ = SyntheticSystem(cfg, np.random.default_rng(0))
    n0 = sys_.n_running
    sys_.advance(5)
    sys_.apply_action(+2, 5)
    assert sys_.n_running == n0 and sys_.n_booting == 2 and sys_.n_total == n0 + 2
    sys_.advance(6)
    assert sys_.n_running == n0 + 2 and sys_.n_booting == 0


def test_scale_in_never_below_vm_min(cfg):
    sys_ = SyntheticSystem(cfg, np.random.default_rng(0))
    k = sys_.apply_action(-7, 0)
    assert sys_.n_total == cfg["system"]["vm_min"]
    assert k == cfg["system"]["vm_min"] - cfg["system"]["initial_vms"]
    sys_.apply_action(-7, 1)
    assert sys_.n_running == cfg["system"]["vm_min"]


def test_scale_out_clamped_at_vm_max(cfg):
    sys_ = SyntheticSystem(cfg, np.random.default_rng(0))
    for t in range(10):
        sys_.advance(t)
        sys_.apply_action(+7, t)
    assert sys_.n_total == cfg["system"]["vm_max"]


def test_scale_in_removes_booting_first(cfg):
    sys_ = SyntheticSystem(cfg, np.random.default_rng(0))
    n0 = sys_.n_running
    sys_.apply_action(+3, 0)
    sys_.apply_action(-2, 0)
    assert sys_.n_running == n0 and sys_.n_booting == 1


def test_observe_reproducible(cfg):
    a = SyntheticSystem(cfg, np.random.default_rng(9)).observe(200, 8, 3)
    b = SyntheticSystem(cfg, np.random.default_rng(9)).observe(200, 8, 3)
    assert a == b


def test_metrics_in_valid_ranges(cfg):
    sys_ = SyntheticSystem(cfg, np.random.default_rng(1))
    for rate in (30, 200, 460):
        for n in (1, 5, 30):
            o = sys_.observe(rate, n, 7)
            assert 0 <= o["cpu_util"] <= 100 and 0 <= o["mem_util"] <= 100
            assert min(o["net_in_kbps"], o["disk_iops"], o["queue_len"]) >= 0
            assert o["rt_ms"] >= 20

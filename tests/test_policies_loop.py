import hashlib

import numpy as np
import pandas as pd
import pytest

from autoscaler.config import default_experiments_config, default_prototype_config
from autoscaler.decision.policies import (CpuThreshold, Opt, Proposed, RtThreshold, make_policy,
                                          valid_actions)
from autoscaler.env.synthetic import SyntheticSystem
from autoscaler.env.workloads import make_workload
from autoscaler.schema import RUN_LOG_COLUMNS
from autoscaler.sim.loop import load_decisions, run_episode, save_logs


class FakeStage1:
    metrics = ["cpu_util"]
    models = {"cpu_util": None}
    residual_cov = np.array([[4.0]])
    residuals = np.random.default_rng(0).normal(0, 2, (200, 1))

    def predict(self, before, w, ks, metrics=None):
        n = np.maximum(w + np.asarray(ks, float), 1)
        return np.clip(before["cpu_util"] * w / n, 0, 100)[:, None]


class FakeEnsemble:
    """log RT of an M/M/1-like queue at the given utilisation, two slightly different members."""
    columns = ["cpu_util_after", "n_after"]

    def predict(self, X):
        u = np.clip(X[:, 0] / 100, 0, 0.995)
        base = np.log(40 / (1 - u))
        return np.stack([base, base + 0.02]), np.full((2, len(X)), 0.01)


@pytest.fixture
def cfg():
    return default_prototype_config()


def obs(cpu=50.0, rt=100.0, n_running=10, n_booting=0):
    return {"cpu_util": cpu, "mem_util": 50, "net_in_kbps": 1000, "disk_iops": 100, "queue_len": 1,
            "rate": 100.0, "n_running": n_running, "n_booting": n_booting, "rt_mean_ms": rt, "step": 0}


def test_valid_actions_respect_pool_limits(cfg):
    assert valid_actions(1, cfg).min() == 0
    assert valid_actions(30, cfg).max() == 0
    assert len(valid_actions(15, cfg)) == 15


def test_cpu_threshold_and_cooldown(cfg):
    p = CpuThreshold(cfg, 80, 30, 3)
    st = {}
    assert p.decide(obs(cpu=90), st)[0] == 1
    for _ in range(3):
        assert p.decide(obs(cpu=90), st)[0] == 0     # cooldown
    assert p.decide(obs(cpu=90), st)[0] == 1
    assert CpuThreshold(cfg, 80, 30, 3).decide(obs(cpu=10), {})[0] == -1
    assert CpuThreshold(cfg, 80, 30, 3).decide(obs(cpu=50), {})[0] == 0
    assert CpuThreshold(cfg, 80, 30, 3).decide(obs(cpu=10, n_running=1), {})[0] == 0   # vm_min


def test_rt_threshold(cfg):
    p = RtThreshold(cfg, 0.8, 0.4, 3)
    assert p.decide(obs(rt=230), {})[0] == 1
    assert p.decide(obs(rt=50), {})[0] == -1


def test_opt_first_feasible_from_kmin(cfg):
    p = Opt(cfg, FakeStage1(), FakeEnsemble())
    k, info = p.decide(obs(cpu=90), {})
    # need u <= 0.84 -> n >= 90*10/84 = 10.7 -> 11 (first feasible scanning from k_min)
    assert k == 1 and not info["fallback"]
    k, info = p.decide(obs(cpu=100, n_running=30), {})
    assert k == 0


def test_proposed_feasible_choice(cfg):
    p = Proposed(cfg, FakeStage1(), FakeEnsemble(), "full")
    p.reset(np.random.default_rng(0))
    k, info = p.decide(obs(cpu=80), {})
    c = {r["k"]: r for r in info["candidates"]}
    assert c[k]["feasible"] and c[k]["on_front"] and c[k]["U"] <= 250
    assert not info["fallback"]
    # min_cost picks the smallest feasible pool
    pm = Proposed(cfg, FakeStage1(), FakeEnsemble(), "full", rule="min_cost")
    pm.reset(np.random.default_rng(0))
    k2, info2 = pm.decide(obs(cpu=80), {})
    feas = [r["n"] for r in info2["candidates"] if r["feasible"]]
    assert 10 + k2 == min(feas)


def test_threshold_factory(cfg):
    exp = default_experiments_config()
    assert make_policy("cpu_threshold", cfg, exp).name == "cpu_threshold"
    with pytest.raises(ValueError):
        make_policy("opt", cfg, exp)


def _episode(cfg, seed, steps=300):
    from autoscaler.seeds import episode_rngs
    wl_rng, sys_rng, _ = episode_rngs(seed)
    wl = make_workload("daily_ramp_burst", steps, wl_rng, cfg["workloads"])
    p = Proposed(cfg, FakeStage1(), FakeEnsemble(), "full")
    return run_episode(p, SyntheticSystem(cfg, sys_rng), wl, cfg, seed, "r", "daily_ramp_burst")


def test_episode_logs_and_determinism(cfg, tmp_path):
    a_log, a_dec = _episode(cfg, 3)
    b_log, b_dec = _episode(cfg, 3)
    assert list(a_log.columns) == RUN_LOG_COLUMNS and len(a_log) == 300
    save_logs(a_log, a_dec, tmp_path / "a")
    save_logs(b_log, b_dec, tmp_path / "b")
    for f in ("run_log.parquet", "decisions.jsonl.gz"):
        ha = hashlib.sha256((tmp_path / "a" / f).read_bytes()).hexdigest()
        hb = hashlib.sha256((tmp_path / "b" / f).read_bytes()).hexdigest()
        assert ha == hb
    assert len(load_decisions(tmp_path / "a" / "decisions.jsonl.gz")) == 300
    c_log, _ = _episode(cfg, 4)
    assert not a_log.rate.equals(c_log.rate)


def test_episode_follows_load_and_bills_booting(cfg):
    log, dec = _episode(cfg, 1, steps=600)
    # pool grows with the load
    assert np.corrcoef(log.rate, log.n_running)[0, 1] > 0.8
    dt_h = cfg["time"]["interval_s"] / 3600
    assert np.allclose(log.cost_usd, (log.n_running + log.n_booting) * 0.10 * dt_h)
    # a VM requested at step t serves at t+1
    up = log.index[log.k > 0][0]
    assert log.n_running[up + 1] == log.n_running[up] + log.k[up] or log.k[up + 1] < 0
    assert (log.n_booting == 0).all()      # boot delay of one step: nothing left booting at observe time

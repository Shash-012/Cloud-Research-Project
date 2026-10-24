import numpy as np
from types import SimpleNamespace
from autoscale.policies.opt_proactive import algorithm1, OPTProactive
from autoscale.policies.opt_reactive import argmin_cost, OPTReactive
from autoscale.policies.rt_threshold import RTThreshold
from autoscale.policies.cpu_threshold import CPUThreshold


class FakeBundle:
    """RT_est(k) from a hand-made table indexed by k."""
    qos_ms = 700.0

    def __init__(self, table):
        self.table = table

    def estimate_rt(self, m, w, ks, rr=0.0):
        mu = np.array([self.table[int(k)] for k in ks], dtype=float)
        return mu, np.zeros_like(mu)


def obs(w=5, pending=0, rt=500.0, cpu=50.0, t=100.0):
    return SimpleNamespace(w=w, pending=pending, metrics=np.zeros(3), rr=10.0, rt_mean_ms=rt, cpu_pct=cpu, t=t)


def test_algorithm1_picks_smallest_safe_k():
    # paper's worked example: w=5, RTmax=700; RT'(-1)=820, RT'(0)=640, RT'(+1)=510, RT'(+2)=430
    tbl = {k: 2000 for k in range(-7, 8)}; tbl.update({-1: 820, 0: 640, 1: 510, 2: 430, 3: 400})
    tbl.update({k: 300 for k in range(4, 8)})
    k, ok = algorithm1(FakeBundle(tbl), obs(w=5), 700, -7, 7, 14)
    assert ok and k == 0
    assert argmin_cost(FakeBundle(tbl), obs(w=5), 700, -7, 7, 14) == 0       # Eq. 8-9 gives the same answer


def test_algorithm1_fallback_is_kmax_and_respects_cap():
    tbl = {k: 5000 for k in range(-7, 8)}
    k, ok = algorithm1(FakeBundle(tbl), obs(w=5), 700, -7, 7, 14)
    assert not ok and k == 7
    k, ok = algorithm1(FakeBundle(tbl), obs(w=12), 700, -7, 7, 14)
    assert not ok and k == 2                                                   # cannot exceed the 14-VM cap


def test_algorithm1_never_proposes_empty_pool():
    tbl = {k: 100 for k in range(-7, 8)}                                      # everything is "safe"
    k, _ = algorithm1(FakeBundle(tbl), obs(w=3), 700, -7, 7, 14)
    assert 3 + k >= 1 and k == -2


def test_proactive_accounts_for_booting_vms():
    tbl = {k: (5000 if k < 3 else 100) for k in range(-7, 8)}
    p = OPTProactive(FakeBundle(tbl), dict(k_min=-7, k_max=7, max_vms=14))
    assert p.decide(obs(w=5, pending=0)) == 3
    assert p.decide(obs(w=5, pending=3)) == 0                                  # already booting what is needed


def test_reactive_only_acts_outside_the_band():
    tbl = {k: 100 for k in range(-7, 8)}
    cfg = dict(k_min=-7, k_max=7, max_vms=14, reactive_upper_frac=0.9, reactive_lower_frac=0.6)
    r = OPTReactive(FakeBundle(tbl), cfg)
    assert r.decide(obs(rt=500.0)) == 0                                        # inside band 420..630: do nothing
    assert r.decide(obs(rt=300.0)) != 0                                        # below 60% of QoS: consult model


def test_threshold_policies_and_cooldown():
    rt = RTThreshold(700, 300, cooldown_s=30)
    assert rt.decide(obs(rt=800, t=100)) == 1
    assert rt.decide(obs(rt=800, t=110)) == 0                                  # cooling down
    assert rt.decide(obs(rt=200, t=140)) == -1
    cpu = CPUThreshold(75, 35, cooldown_s=30)
    assert cpu.decide(obs(cpu=90, t=100)) == 1 and cpu.decide(obs(cpu=20, t=200)) == -1

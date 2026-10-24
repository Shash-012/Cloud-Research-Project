"""Simulator validation against queueing theory (E0)."""
from __future__ import annotations
import math
import numpy as np
from ..config import load_config
from ..env.simpy_env import SimpyEnv
from ..env.workloads import Workload


def erlang_c_rt(lam: float, mu: float, c: int) -> float:
    """Mean response time of an M/M/c queue."""
    a = lam / mu; rho = a / c
    assert rho < 1
    s = sum(a ** n / math.factorial(n) for n in range(c))
    top = a ** c / math.factorial(c) / (1 - rho)
    pw = top / (s + top)
    return pw / (c * mu - lam) + 1.0 / mu


def mmc_check(c: int = 2, mean_service_s: float = 0.25, util: float = 0.8, duration_s: float = 20000, seed: int = 1,
              warmup_s: float = 1000):
    cfg = load_config(); cfg["min_vms"] = cfg["max_vms"] = 1; cfg["init_vms"] = 1; cfg["boot_delay_s"] = 0
    app = dict(cfg["apps"]["cpu_bound"]); app.update(vcpu=c, slots=1000, cpu_ms=mean_service_s * 1000, dist="exp")
    mu = 1.0 / mean_service_s; lam = util * c * mu
    n = int(duration_s) + 700
    wl = Workload("const", np.full(n, lam), np.ones(n))
    env = SimpyEnv(app, wl, cfg, seed, duration_s)
    env.reset(1)
    while not env.done:
        env.step(0)
    iv, rq = env.finalize()
    rq = rq[rq["t0"] > warmup_s]; iv2 = iv[iv["t"] > warmup_s]
    W_sim = rq["rt_ms"].mean() / 1000.0
    L_sim = iv2["inflight"].mean()                         # time-averaged number in system (1 s sampler)
    lam_eff = len(rq) / (rq["t0"].max() - rq["t0"].min())
    return dict(c=c, util=util, lam=lam, rt_sim_s=W_sim, rt_theory_s=erlang_c_rt(lam, mu, c),
                L_sim=L_sim, L_littles=lam_eff * W_sim)

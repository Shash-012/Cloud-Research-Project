import numpy as np
from autoscale.env.simpy_env import make_env
from autoscale.runner.validate import mmc_check


def _run(cfg, seed, delta_fn=lambda o: 0, dur=300):
    env = make_env(cfg, "cpu_bound", "webshop", seed, dur)
    obs = env.reset(6)
    while not env.done:
        obs = env.step(delta_fn(obs))
    return env.finalize()


def test_determinism(cfg):
    a_iv, a_rq = _run(cfg, 3); b_iv, b_rq = _run(cfg, 3)
    assert a_iv.equals(b_iv) and a_rq.equals(b_rq)


def test_same_load_for_every_policy(cfg):
    """Paired design: the arrival process must not depend on the pool decisions."""
    iv1, _ = _run(cfg, 4, lambda o: 0)
    iv2, _ = _run(cfg, 4, lambda o: 1 if o.t < 100 else -1)
    assert np.allclose(iv1["rr"], iv2["rr"])


def test_mmc_and_littles_law():
    r = mmc_check(c=2, util=0.7, duration_s=8000, warmup_s=800)
    assert abs(r["rt_sim_s"] - r["rt_theory_s"]) / r["rt_theory_s"] < 0.06
    assert abs(r["L_sim"] - r["L_littles"]) / r["L_littles"] < 0.05


def test_pool_constraints(cfg):
    env = make_env(cfg, "cpu_bound", "webshop", 0, 200)
    env.reset(3)
    env.step(100)                                   # ask for far too many
    assert env.w + env.pending == cfg["max_vms"]
    env.step(-100)                                  # ask for far too few
    assert env.w >= cfg["min_vms"] and env.w + env.pending >= cfg["min_vms"]


def test_boot_delay_and_cancel(cfg):
    env = make_env(cfg, "cpu_bound", "webshop", 0, 400)
    env.reset(2)
    obs = env.step(2)
    assert obs.w == 2 and obs.pending == 2          # still booting (30 s)
    obs = env.step(-1)                              # cancels one booting VM first
    assert obs.pending == 1
    for _ in range(3):
        obs = env.step(0)
    assert obs.w == 3 and obs.pending == 0

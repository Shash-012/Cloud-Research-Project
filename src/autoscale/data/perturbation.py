"""Free-run / perturbation data collection (paper Sec 4.1).

Load changes gradually (ramps up and down) while the pool is systematically resized. Each observation is a
`hold`-interval period: the VM change is applied at its start, the last `perturb_avg_intervals` (after boot) are averaged
as the 'after' state; the same number of intervals at the end of the previous period are the 'before' state.
"""
from __future__ import annotations
import numpy as np
from ..env.simpy_env import make_env
from .dataset import TransitionDataset


def _avg_state(buf):
    """buf: list of Observation (settled intervals). Metrics averaged; RT weighted by completions."""
    M = np.mean([o.metrics for o in buf], axis=0)
    n = sum(o.n_completed for o in buf)
    rts = [(o.rt_mean_ms, o.n_completed) for o in buf if o.n_completed > 0]
    rt = float(sum(r * c for r, c in rts) / n) if n else float("nan")
    return M, rt, float(np.mean([o.rr for o in buf])), float(np.mean([o.hidden_complexity for o in buf]))


def collect_perturbation(cfg: dict, app: str, n_obs: int, seed: int, workload: str = "ramp",
                         pool_mode: str = "load_following", pool_range=(1, 8)) -> TransitionDataset:
    """pool_mode: 'load_following' = random multiple of the capacity needed (training for the scaler);
                  'random' = uniform pool size independent of load (constant-rate experiment, VM decorrelated from RT)."""
    hold = int(cfg["perturb_hold_intervals"]); navg = int(cfg["perturb_avg_intervals"])
    lo_m, hi_m = cfg["perturb_mult_range"]
    dt, boot = cfg["interval_s"], cfg["boot_delay_s"]
    assert hold * dt >= boot + (navg + 1) * dt, "hold period too short for boot delay"
    dur = (n_obs + 2) * hold * dt
    env = make_env(cfg, app, workload, seed, duration_s=dur)
    rng = np.random.default_rng(seed + 4242)
    cap = cfg["apps"][app]["capacity_per_vm"]
    names = None
    rows = []
    obs = env.reset(init_vms=cfg["init_vms"])
    names = obs.metric_names
    prev_buf = [obs]
    for _ in range(n_obs + 1):
        lam = env.wl.rate_at(env.t + hold * dt / 2)
        if pool_mode == "random":
            target = int(rng.integers(pool_range[0], pool_range[1] + 1))
        else:
            needed = max(1, int(np.ceil(lam / (cap * 0.7))))
            target = int(np.clip(round(needed * rng.uniform(lo_m, hi_m)), 1, cfg["max_vms"]))
        delta = int(np.clip(target - (obs.w + obs.pending), cfg["k_min"], cfg["k_max"]))
        if delta == 0 and rng.random() < 0.85:
            delta = int(rng.choice([-1, 1]))
        w_before = obs.w
        buf = []
        for j in range(hold):
            obs = env.step(delta if j == 0 else 0)
            buf.append(obs)
        after = buf[-navg:]
        mb, rtb, rrb, _ = _avg_state(prev_buf)
        ma, rta, rra, cx = _avg_state(after)
        k_real = after[-1].w - w_before
        if not (np.isnan(rtb) or np.isnan(rta)):
            rows.append((mb, w_before, k_real, ma, rtb, rta, rrb, rra, cx))
        prev_buf = after
    env.finalize()
    mb, wb, k, ma, rtb, rta, rrb, rra, cx = zip(*rows)
    return TransitionDataset(list(names), np.array(mb), np.array(ma), np.array(wb, dtype=float), np.array(k, dtype=float),
                             np.array(rtb), np.array(rta), np.array(rrb), np.array(rra), np.array(cx))

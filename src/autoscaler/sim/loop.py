"""Closed-loop episode in the synthetic environment (Step 2.12)."""
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..decision.power import power_from_cfg
from ..schema import METRICS, RUN_LOG_COLUMNS
from ..seeds import STREAM_CONTROLLER, child_rngs

# The synthetic system reports only the interval mean. For the p95 column we assume exponential
# response times within an interval (M/M/1 sojourn): p95 = mean * ln(20).
P95_FACTOR = math.log(20.0)


def run_episode(policy, system, workload, cfg: dict, seed: int, run_id: str = "",
                workload_name: str = "", policy_name: str | None = None):
    """Run one episode; returns (run_log DataFrame, decision_log list of dicts).

    Cost and energy use realised (true) utilisation, not predictions.
    """
    policy.reset(child_rngs(seed, 3)[STREAM_CONTROLLER])
    every = cfg["time"].get("decision_every", 1)
    dt_h = cfg["time"]["interval_s"] / 3600.0
    price = cfg["cost"]["price_per_vm_hour"]
    slo = cfg["slo"]["rt_ms"]
    p_idle = float(power_from_cfg(0.0, cfg))
    pname = policy_name or policy.name
    state: dict = {}
    rows, decisions = [], []
    for t, rate in enumerate(np.asarray(workload, float)):
        system.advance(t)
        o = system.observe(rate, system.n_running, t)
        n_run, n_boot = system.n_running, system.n_booting
        cost = (n_run + n_boot) * price * dt_h
        energy = (n_run * float(power_from_cfg(o["u"], cfg)) + n_boot * p_idle) * dt_h
        obs = {m: o[m] for m in METRICS} | {"rate": float(rate), "n_running": n_run, "n_booting": n_boot,
                                           "rt_mean_ms": o["rt_ms"], "step": t}
        k, fallback = 0, False
        if t % every == 0:
            k_req, info = policy.decide(obs, state)
            k = system.apply_action(k_req, t)
            fallback = bool(info.get("fallback", False))
            decisions.append({"step": t, "w": n_run + n_boot, "k": int(k), "fallback": fallback,
                              "chosen_U": info.get("chosen_U"), "chosen_mean_log": info.get("chosen_mean_log"),
                              "candidates": info.get("candidates", [])})
        rows.append({"run_id": run_id, "policy": pname, "workload": workload_name, "seed": seed, "step": t,
                     "rate": float(rate), "n_running": n_run, "n_booting": n_boot, "k": int(k),
                     "fallback": fallback, "rt_mean_ms": o["rt_ms"], "rt_p95_ms": o["rt_ms"] * P95_FACTOR,
                     "slo_violated": bool(o["rt_ms"] > slo), "cost_usd": cost, "energy_wh": energy,
                     **{m: o[m] for m in METRICS}})
    return pd.DataFrame(rows, columns=RUN_LOG_COLUMNS), decisions


def episode(policy_name: str, workload_name: str, seed: int, cfg: dict, exp_cfg: dict, models=None):
    """Build workload, system and policy from independent seed streams and run one episode."""
    from ..decision.policies import make_policy
    from ..env.synthetic import SyntheticSystem
    from ..env.workloads import make_workload
    from ..seeds import episode_rngs
    wl_rng, sys_rng, _ = episode_rngs(seed)
    wl = make_workload(workload_name, cfg["time"]["steps_per_run"], wl_rng, cfg["workloads"])
    policy = make_policy(policy_name, cfg, exp_cfg, models)
    run_id = f"synthetic__{policy_name}__{workload_name}__s{seed}"
    return run_episode(policy, SyntheticSystem(cfg, sys_rng), wl, cfg, seed, run_id, workload_name, policy_name)


def save_logs(run_log: pd.DataFrame, decisions: list, folder) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    run_log.to_parquet(folder / "run_log.parquet", index=False)
    # mtime=0 keeps the gzip bytes identical for identical content
    with open(folder / "decisions.jsonl.gz", "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
            for d in decisions:
                f.write((json.dumps(d, sort_keys=True) + "\n").encode())


def load_decisions(path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]

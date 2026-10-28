"""Calibration transitions: synthetic (Step 2.3) and extraction from CloudSim Plus run logs (Step 3.14)."""
import numpy as np
import pandas as pd

from ..env.synthetic import SyntheticSystem
from ..schema import METRICS, METRICS_CLOUDSIM, TRANSITION_COLUMNS, transition_columns


def collect_synthetic(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """One row per random scaling transition; rate is drawn uniformly, never from a workload."""
    c = cfg["calibration"]
    s = cfg["system"]
    a = cfg["actions"]
    steps = cfg["time"]["steps_per_run"]
    system = SyntheticSystem(cfg, rng)
    rows = []
    for i in range(c["rows"]):
        lam = rng.uniform(c["rate_min"], c["rate_max"])
        t = int(rng.integers(0, steps))
        n_before = int(rng.integers(s["vm_min"], s["vm_max"] + 1))
        k = int(rng.integers(a["k_min"], a["k_max"] + 1))
        n_after = int(np.clip(n_before + k, s["vm_min"], s["vm_max"]))
        cx = system.complexity(t)           # same interval, same complexity on both sides
        before = system.observe(lam, n_before, t, c=cx)
        after = system.observe(lam, n_after, t, c=cx)
        row = {"source": "synthetic", "run_id": f"synthetic_calib_seed{c['seed']}", "step": t,
               "n_before": n_before, "k": k, "n_after": n_after,
               "rate_before": lam, "rate_after": lam}
        for m in METRICS:
            row[f"{m}_before"] = before[m]
            row[f"{m}_after"] = after[m]
        row["rt_after_ms"] = after["rt_ms"]
        rows.append(row)
    return pd.DataFrame(rows, columns=TRANSITION_COLUMNS)


def extract_cloudsim_transitions(log: pd.DataFrame, offset: int, drift_flag: float) -> pd.DataFrame:
    """Transitions from one CloudSim Plus run log (Step 3.14).

    An action taken at the end of interval t: before-metrics from interval t, after-metrics and target from
    t + offset (offset 2: the first interval in which booted VMs serve for the whole interval). The pool must not
    change in between, every VM must have booted, and both intervals must have completions.
    """
    log = log.sort_values("step").reset_index(drop=True)
    rows = []
    for t in range(len(log) - offset):
        b, a = log.iloc[t], log.iloc[t + offset]
        # the pool must not change between the action and the after-interval, and booted VMs must serve
        if any(log.iloc[t + j]["k"] != 0 for j in range(1, offset)) or a["n_booting"] != 0:
            continue
        if not np.isfinite(a["rt_mean_ms"]) or a["completions"] == 0 or b["completions"] == 0:
            continue
        n_before = int(b["n_running"] + b["n_booting"])
        n_after = int(a["n_running"])
        if n_after != n_before + int(b["k"]):
            continue
        row = {"source": "cloudsim", "run_id": b["run_id"], "step": int(b["step"]), "n_before": n_before,
               "k": int(b["k"]), "n_after": n_after, "rate_before": float(b["rate"]), "rate_after": float(a["rate"])}
        for m in METRICS_CLOUDSIM:
            row[f"{m}_before"] = float(b[m])
            row[f"{m}_after"] = float(a[m])
        row["rt_after_ms"] = float(a["rt_mean_ms"])
        row["rate_drift"] = abs(row["rate_after"] - row["rate_before"]) / max(row["rate_before"], 1e-9)
        row["rate_drift_flag"] = bool(row["rate_drift"] > drift_flag)
        rows.append(row)
    return pd.DataFrame(rows, columns=transition_columns("cloudsim") + ["rate_drift", "rate_drift_flag"])

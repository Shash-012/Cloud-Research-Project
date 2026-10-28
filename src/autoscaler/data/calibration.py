"""Synthetic calibration transitions (Step 2.3)."""
import numpy as np
import pandas as pd

from ..env.synthetic import SyntheticSystem
from ..schema import METRICS, TRANSITION_COLUMNS


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

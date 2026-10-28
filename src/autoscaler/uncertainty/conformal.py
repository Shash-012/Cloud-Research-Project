"""Split-conformal correction of the bound multiplier z (Step 2.9)."""
import json
import math
from pathlib import Path

import numpy as np


def conformal_z(y, mean_log, sigma, delta: float) -> float:
    """z_cal = r_(ceil((n+1)(1-delta))) with scores r = (log y - mean) / sigma on the calibration split."""
    sigma = np.asarray(sigma, float)
    if np.any(sigma <= 0):
        raise ValueError("conformal z needs sigma > 0 for every row")
    r = np.sort((np.log(np.asarray(y, float)) - np.asarray(mean_log, float)) / sigma)
    n = len(r)
    rank = math.ceil((n + 1) * (1 - delta))
    return float(r[min(rank, n) - 1])


def save_conformal(values: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(values, indent=2))


def load_conformal(path) -> dict:
    return json.loads(Path(path).read_text())


def bound_z(cfg: dict, mode: str = "full", conformal_path=None) -> float:
    """z used by the controller: Phi^-1(1-delta) or, if slo.use_conformal, the calibrated value."""
    if cfg["slo"].get("use_conformal") and conformal_path is not None and Path(conformal_path).exists():
        return float(load_conformal(conformal_path)[mode])
    return float(cfg["slo"]["z"])

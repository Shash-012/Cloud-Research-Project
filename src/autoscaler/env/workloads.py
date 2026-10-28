"""Request-rate workloads (Step 2.1), defined on the run fraction x = t/N."""
import numpy as np

from ..config import default_prototype_config

SHAPES = ["daily_ramp_burst", "step", "flash_crowd", "sawtooth", "random_walk"]


def make_workload(shape: str, steps: int, rng: np.random.Generator,
                  params: dict | None = None) -> np.ndarray:
    """Request rates (req/s) for `steps` intervals, clipped to the configured range.

    `params` is the `workloads` section of a config; defaults to configs/prototype.yaml.
    """
    p = params if params is not None else default_prototype_config()["workloads"]
    lo, hi = p["clip"]
    n = steps
    t = np.arange(n, dtype=float)
    x = t / n

    if shape == "random_walk":
        q = p["random_walk"]
        lam = np.empty(n)
        lam[0] = q["start"]
        steps_ = rng.normal(0.0, q["step_std"], size=n)
        for i in range(1, n):
            v = lam[i - 1] + steps_[i]
            # reflect at the bounds
            while v < lo or v > hi:
                v = 2 * lo - v if v < lo else 2 * hi - v
            lam[i] = v
        return np.clip(lam, lo, hi)

    if shape == "daily_ramp_burst":
        q = p["daily_ramp_burst"]
        base = (q["base"] + q["sin_amp"] * np.sin(2 * np.pi * x)
                + q["ramp_height"] * np.clip((x - q["ramp_start"]) / q["ramp_len"], 0, 1)
                + q["burst_height"] * np.exp(-0.5 * ((x - q["burst_centre"]) / q["burst_width"]) ** 2))
    elif shape == "step":
        q = p["step"]
        base = np.where(x < q["at"], q["low"], q["high"])
    elif shape == "flash_crowd":
        q = p["flash_crowd"]
        centres = rng.uniform(q["x_min"], q["x_max"], size=q["spikes"]) * n
        width = q["width_frac"] * n
        base = np.full(n, q["base"])
        for c in centres:
            base = base + q["height"] * np.exp(-0.5 * ((t - c) / width) ** 2)
    elif shape == "sawtooth":
        q = p["sawtooth"]
        period = q["period_frac"] * n
        phase = (t % period) / period
        base = q["low"] + (q["high"] - q["low"]) * phase
    else:
        raise ValueError(f"unknown workload shape: {shape}")

    eps = rng.normal(0.0, p["noise_std"], size=n)
    return np.clip(base + eps, lo, hi)

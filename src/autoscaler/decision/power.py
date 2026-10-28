"""Server power models (Step 2.10): P(u) = P_idle + (P_max - P_idle) h(u)."""
import numpy as np

SHAPES = {
    "linear": lambda u: u,
    "square": lambda u: u ** 2,
    "cubic": lambda u: u ** 3,
    "sqrt": lambda u: np.sqrt(u),
}


def power(u, model: str, p_idle: float, p_max: float):
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return p_idle + (p_max - p_idle) * SHAPES[model](u)


def power_from_cfg(u, cfg: dict):
    p = cfg["power"]
    return power(u, p["model"], p["p_idle_w"], p["p_max_w"])


def idle_for_ratio(p_max: float, ratio: float) -> float:
    """P_idle such that (P_max - P_idle) / P_idle = ratio."""
    return p_max / (1.0 + ratio)

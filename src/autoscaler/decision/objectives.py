"""Per-candidate objectives (Step 2.10): latency U_k, cost and energy."""
import numpy as np

from .power import power

OBJECTIVES = ("latency", "cost", "energy")


def interval_hours(cfg: dict) -> float:
    return cfg["time"]["interval_s"] / 3600.0


def candidate_cost(n, cfg: dict):
    """n counts running plus booting VMs after the action."""
    return np.asarray(n, float) * cfg["cost"]["price_per_vm_hour"] * interval_hours(cfg)


def candidate_energy(n, u_hat, cfg: dict, power_cfg: dict | None = None):
    """Wh for one interval: n machines at predicted utilisation u_hat (one machine per VM)."""
    p = power_cfg or cfg["power"]
    return np.asarray(n, float) * power(u_hat, p["model"], p["p_idle_w"], p["p_max_w"]) * interval_hours(cfg)


def objective_matrix(U, n, u_hat, cfg: dict, power_cfg: dict | None = None) -> np.ndarray:
    """(candidates, 3) matrix of (f_L, f_C, f_E), all minimised."""
    return np.column_stack([np.asarray(U, float), candidate_cost(n, cfg),
                            candidate_energy(n, u_hat, cfg, power_cfg)])

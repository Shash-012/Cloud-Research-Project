"""Column lists shared by every writer and reader (Step 1.3)."""

METRICS = ["cpu_util", "mem_util", "net_in_kbps", "disk_iops", "queue_len"]
PERCENT_METRICS = ["cpu_util", "mem_util"]     # clipped to [0, 100]
COUNT_METRICS = ["net_in_kbps", "disk_iops", "queue_len"]   # clipped to >= 0

TRANSITION_COLUMNS = (
    ["source", "run_id", "step", "n_before", "k", "n_after", "rate_before", "rate_after"]
    + [f"{m}_before" for m in METRICS]
    + [f"{m}_after" for m in METRICS]
    + ["rt_after_ms"]
)

RUN_LOG_COLUMNS = [
    "run_id", "policy", "workload", "seed", "step", "rate", "n_running", "n_booting", "k",
    "fallback", "rt_mean_ms", "rt_p95_ms", "slo_violated", "cost_usd", "energy_wh",
] + METRICS

DECISION_CANDIDATE_FIELDS = [
    "k", "n", "mean_log", "sigma", "U", "cost", "energy", "u_hat", "feasible", "on_front", "score",
]

DECISION_LOG_FIELDS = ["step", "w", "k", "fallback", "candidates"]


def clip_metric(name: str, values):
    """Clip metric values to their valid range."""
    import numpy as np
    if name in PERCENT_METRICS:
        return np.clip(values, 0.0, 100.0)
    return np.maximum(values, 0.0)

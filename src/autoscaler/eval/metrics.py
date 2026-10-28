"""Per-run summaries (Step 5.1)."""
import numpy as np
import pandas as pd


def summarise(run_log: pd.DataFrame, decision_log: list, cfg: dict, lag: int = 1) -> dict:
    """Summary after warm-up (Step 5.1). `lag` = intervals between a decision and the measured outcome:
    1 in the synthetic environment, 2 in CloudSim Plus. The pool counts running + booting (+ draining) VMs."""
    warm = cfg["time"]["warmup_steps"]
    dt_h = cfg["time"]["interval_s"] / 3600.0
    log = run_log[run_log.step >= warm]
    pool = log.n_running + log.n_booting + (log.n_draining if "n_draining" in log else 0)
    dec = [d for d in decision_log if d["step"] >= warm]
    # Java-side policies (CloudSim Plus thresholds) leave no Python decision log: use the run log's actions
    ks = np.array([d["k"] for d in dec]) if dec else log.k.to_numpy()
    fallbacks = [d["fallback"] for d in dec] if dec else log.fallback.astype(bool).tolist()

    rt_by_step = run_log.set_index("step")["rt_mean_ms"]
    hits, margins = [], []
    for d in dec:
        if d.get("chosen_U") is None:
            continue
        t_out = d["step"] + lag
        if t_out in rt_by_step.index:
            hits.append(rt_by_step[t_out] <= d["chosen_U"])
        margins.append(d["chosen_U"] - np.exp(d["chosen_mean_log"]))

    return {
        "violation_rate": float(log.slo_violated.mean()),
        "rt_mean_ms": float(log.rt_mean_ms.mean()),
        "rt_p95_ms": float(np.nanpercentile(log.rt_mean_ms, 95)),
        "vm_hours": float(pool.sum() * dt_h),
        "cost_usd": float(log.cost_usd.sum()),
        "energy_kwh": float(log.energy_wh.sum() / 1000.0),
        "mean_pool": float(pool.mean()),
        "churn": int(np.count_nonzero(ks)),
        "mean_abs_k": float(np.abs(ks).mean()) if len(ks) else 0.0,
        "fallback_rate": float(np.mean(fallbacks)) if len(fallbacks) else 0.0,
        "realised_coverage": float(np.mean(hits)) if hits else np.nan,
        "mean_margin_ms": float(np.mean(margins)) if margins else np.nan,
    }

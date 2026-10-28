"""End-to-end calibration measures (Step 2.9). Inputs are stage-1 estimates, never true after-metrics."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .propagation import _stage2_inputs, bounds_from_means


def end_to_end(stage1, ensemble, df: pd.DataFrame, z: float, S: int, rng, mode="full",
               resample=False) -> dict:
    """Per-row mean_log, sigma and U for held-out transitions, using stage-1 estimates as inputs."""
    means = stage1.predict_frame(df)
    n = df["n_after"].to_numpy(float)
    rate = df["rate_before"].to_numpy(float)       # the rate known at decision time
    out = bounds_from_means(ensemble, means, n, stage1.metrics, stage1.residual_cov, stage1.residuals,
                            z, S, rng, mode=mode, resample=resample, rate=rate)
    out["y"] = df["rt_after_ms"].to_numpy(float)
    return out


def end_to_end_quantile(stage1, qmodel, df: pd.DataFrame) -> dict:
    means = stage1.predict_frame(df)
    X = _stage2_inputs(qmodel.columns, list(stage1.metrics), means, df["n_after"].to_numpy(float),
                       df["rate_before"].to_numpy(float))
    q = qmodel.predict(X)
    return {"mean_log": q, "sigma": np.zeros_like(q), "U": np.exp(q),
            "y": df["rt_after_ms"].to_numpy(float)}


def coverage(y, U) -> float:
    return float(np.mean(np.asarray(y) <= np.asarray(U)))


def mean_margin(U, mean_log) -> float:
    return float(np.mean(np.asarray(U) - np.exp(mean_log)))


def conditional_coverage(y, U, mean_log, bins: int = 5, slo: float | None = None) -> pd.DataFrame:
    """Coverage in quantile bins of the predicted response time exp(mean_log)."""
    pred = np.exp(np.asarray(mean_log))
    edges = np.quantile(pred, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        rows.append({"bin": b, "pred_lo_ms": edges[b], "pred_hi_ms": edges[b + 1], "rows": int(sel.sum()),
                     "coverage": coverage(np.asarray(y)[sel], np.asarray(U)[sel]) if sel.any() else np.nan,
                     "contains_slo": bool(slo is not None and edges[b] <= slo <= edges[b + 1])})
    return pd.DataFrame(rows)


def gaussian_nll(y, mean_log, sigma) -> float:
    s2 = np.asarray(sigma) ** 2
    if np.any(s2 <= 0):
        return float("nan")
    r = np.log(np.asarray(y)) - np.asarray(mean_log)
    return float(np.mean(0.5 * np.log(2 * np.pi * s2) + r ** 2 / (2 * s2)))


def pinball_loss(y, q_log, tau) -> float:
    r = np.log(np.asarray(y)) - np.asarray(q_log)
    return float(np.mean(np.maximum(tau * r, (tau - 1) * r)))


def reliability(y, mean_log, sigma, levels) -> list[float]:
    """Observed coverage of exp(mean + Phi^-1(tau) sigma) at each nominal level tau."""
    return [coverage(y, np.exp(np.asarray(mean_log) + norm.ppf(t) * np.asarray(sigma))) for t in levels]

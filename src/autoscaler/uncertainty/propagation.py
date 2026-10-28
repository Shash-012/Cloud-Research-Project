"""Monte Carlo propagation of stage-1 error through the deep ensemble (Step 2.8, E2)."""
import numpy as np

from ..schema import clip_metric

MODES = ("none", "epistemic", "ale_epi", "full")


def cholesky_factor(cov: np.ndarray) -> np.ndarray:
    cov = np.atleast_2d(cov)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(cov + 1e-9 * np.eye(len(cov)))


def _stage2_inputs(columns, metric_names, values, n, rate):
    """Assemble the ensemble's input matrix from stage-1 metric values (rows, d), n and rate."""
    cols = []
    for c in columns:
        if c == "n_after":
            cols.append(n)
        elif c == "rate_after":
            if rate is None:
                raise ValueError("ensemble needs rate_after but no rate was given")
            cols.append(np.broadcast_to(np.asarray(rate, float), n.shape))
        else:
            cols.append(values[:, metric_names.index(c[: -len("_after")])])
    return np.column_stack(cols)


def bounds_from_means(ensemble, means, n, metric_names, cov, residuals, z, S, rng,
                      mode="full", resample=False, rate=None) -> dict:
    """Pool ensemble outputs over stage-1 error draws for each row of `means` (rows, d).

    Every member runs on every draw (one batched forward pass per member); inputs are never
    averaged first.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    means = np.atleast_2d(np.asarray(means, float))
    n = np.asarray(n, float)
    R, d = means.shape
    rate_rows = None if rate is None else np.broadcast_to(np.asarray(rate, float), (R,))

    propagate = mode == "full"
    if propagate and not resample and not np.any(np.asarray(cov)):
        propagate = False            # zero stage-1 covariance: nothing to propagate
    if propagate:
        if resample:
            res = np.asarray(residuals, float)
            draws = res[rng.integers(0, len(res), size=(R, S))]
        else:
            L = cholesky_factor(cov)
            draws = rng.standard_normal((R, S, d)) @ L.T
        S_eff = S
    else:
        draws = np.zeros((R, 1, d))
        S_eff = 1

    vals = means[:, None, :] + draws                      # (R, S, d)
    for i, m in enumerate(metric_names):
        vals[..., i] = clip_metric(m, vals[..., i])
    flat = vals.reshape(R * S_eff, d)
    n_rep = np.repeat(n, S_eff)
    rate_rep = None if rate_rows is None else np.repeat(rate_rows, S_eff)
    X = _stage2_inputs(ensemble.columns, list(metric_names), flat, n_rep, rate_rep)

    mu, var = ensemble.predict(X)                         # (M, R*S)
    M = mu.shape[0]
    mu = mu.reshape(M, R, S_eff)
    var = var.reshape(M, R, S_eff)

    mbar = mu.mean(axis=(0, 2))
    ale = var.mean(axis=(0, 2))
    # Spread of the M*S means with the population divisor 1/(MS). The roadmap writes 1/(MS-1);
    # that makes S=1 modes divide by M-1 but full mode by MS-1, which breaks its own checks
    # (Sigma_e = 0 => full == ale_epi; sigma non-decreasing in Sigma_e). With 1/(MS) the spread
    # equals sigma_epi^2 + sigma_stage1^2 exactly (law of total variance); at M=5, S=32 the
    # difference from 1/(MS-1) is 0.6%.
    spread = ((mu - mbar[None, :, None]) ** 2).mean(axis=(0, 2))
    sigma_epi2 = mu.var(axis=0).mean(axis=-1)             # mean over draws of across-member variance
    sigma_s12 = mu.mean(axis=0).var(axis=-1)              # variance over draws of the member average

    if mode == "none":
        sigma2 = np.zeros(R)
    elif mode == "epistemic":
        sigma2 = spread
    else:                                                 # ale_epi, full
        sigma2 = ale + spread
    sigma = np.sqrt(sigma2)
    return {"n": n, "mean_log": mbar, "sigma": sigma, "sigma_ale": np.sqrt(ale),
            "sigma_epi": np.sqrt(sigma_epi2), "sigma_stage1": np.sqrt(sigma_s12),
            "U": np.exp(mbar + z * sigma), "m_hat": means}


def candidate_bounds(stage1, ensemble, before, w, ks, z, S, rng, mode="full", resample=False):
    """Return per-candidate dict arrays: n, mean_log, sigma, sigma_ale, sigma_epi, sigma_stage1, U (ms).

    `before` holds the last interval's metrics (and `rate` if the ensemble uses it).
    """
    ks = np.asarray(ks)
    means = stage1.predict(before, w, ks)
    n = np.maximum(w + ks, 1)
    out = bounds_from_means(ensemble, means, n, stage1.metrics, stage1.residual_cov, stage1.residuals,
                            z, S, rng, mode=mode, resample=resample, rate=before.get("rate"))
    out["k"] = ks
    return out


def quantile_candidate_bounds(stage1, qmodel, before, w, ks) -> dict:
    """Bound from a direct quantile model on stage-1 point estimates: U = exp(q_tau)."""
    ks = np.asarray(ks)
    means = stage1.predict(before, w, ks)
    n = np.maximum(w + ks, 1).astype(float)
    X = _stage2_inputs(qmodel.columns, list(stage1.metrics), means, n, before.get("rate"))
    q = qmodel.predict(X)
    zeros = np.zeros(len(ks))
    return {"k": ks, "n": n, "mean_log": q, "sigma": zeros, "sigma_ale": zeros, "sigma_epi": zeros,
            "sigma_stage1": zeros, "U": np.exp(q), "m_hat": means}


def point_prediction(stage1, net, before, w, ks) -> dict:
    """Mean-only prediction (OPT baseline): stage-1 point estimates through one network."""
    ks = np.asarray(ks)
    means = stage1.predict(before, w, ks)
    n = np.maximum(w + ks, 1).astype(float)
    X = _stage2_inputs(net.columns, list(stage1.metrics), means, n, before.get("rate"))
    mu, _ = net.predict(X)
    m = mu.mean(axis=0)
    zeros = np.zeros(len(ks))
    return {"k": ks, "n": n, "mean_log": m, "sigma": zeros, "sigma_ale": zeros, "sigma_epi": zeros,
            "sigma_stage1": zeros, "U": np.exp(m), "m_hat": means}

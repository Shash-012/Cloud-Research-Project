"""Confidence intervals and paired tests (Step 5.2 subset used by the prototype sweep)."""
import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(x, level: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of the t interval."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), np.nan
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(h)


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[j]))
        adj[j] = running
    return adj


def paired_wilcoxon(x, y) -> tuple[float, float]:
    """(median paired difference x - y, p-value); p = 1 when all differences are zero."""
    d = np.asarray(x, float) - np.asarray(y, float)
    if np.allclose(d, 0):
        return 0.0, 1.0
    return float(np.median(d)), float(stats.wilcoxon(x, y).pvalue)


def summary_table(df: pd.DataFrame, metrics, by=("workload", "policy")) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(list(by)):
        row = dict(zip(by, key))
        for m in metrics:
            row[m], row[f"{m}_ci"] = mean_ci(g[m])
        row["runs"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_table(df: pd.DataFrame, reference: str, metrics, seed_col="seed") -> pd.DataFrame:
    rows = []
    for wl, g in df.groupby("workload"):
        ref = g[g.policy == reference].set_index(seed_col)
        for pol in sorted(set(g.policy) - {reference}):
            other = g[g.policy == pol].set_index(seed_col)
            common = ref.index.intersection(other.index)
            for m in metrics:
                med, p = paired_wilcoxon(ref.loc[common, m], other.loc[common, m])
                rows.append({"workload": wl, "reference": reference, "policy": pol, "metric": m,
                             "median_diff_ref_minus_policy": med, "p": p, "pairs": len(common)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = np.nan
        for m, g in out.groupby("metric"):
            out.loc[g.index, "p_holm"] = holm(g.p.to_numpy())
    return out

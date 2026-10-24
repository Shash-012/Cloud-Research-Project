"""Filter 1: does the metric react to scaling? Welch t-test on delta-m (scale-up vs scale-down), paper Eq. 11-12."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(a, b) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else 0.0


def dynamicity_test(ds, alpha: float = 0.001, cohen_min: float | None = None, equal_var: bool = False) -> pd.DataFrame:
    dm = ds.m_after - ds.m_before
    up, down = ds.k > 0, ds.k < 0
    if up.sum() < 3 or down.sum() < 3:
        raise ValueError("need scale-up and scale-down observations")
    rows = []
    for j, name in enumerate(ds.names):
        a, b = dm[down, j], dm[up, j]
        if np.var(a) == 0 and np.var(b) == 0:
            t = p = np.nan; d = 0.0
        else:
            t, p = stats.ttest_ind(a, b, equal_var=equal_var); d = cohens_d(a, b)
        keep = bool(p < alpha) if not np.isnan(p) else False
        if keep and cohen_min is not None:
            keep = abs(d) >= cohen_min
        rows.append(dict(metric=name, idx=j, t=t, p=p, cohen_d=d, dynamic=keep))
    return pd.DataFrame(rows)

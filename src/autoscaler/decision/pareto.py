"""Pareto dominance (Step 2.10)."""
import numpy as np


def pareto_mask(F) -> np.ndarray:
    """True for non-dominated rows of F (candidates, objectives), all minimised.

    A row is dominated if another row is <= in every column and < in at least one.
    """
    F = np.asarray(F, float)
    if F.ndim != 2:
        raise ValueError("F must be 2-D")
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)    # le[j, i]: row j <= row i everywhere
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dominated = np.any(le & lt, axis=0)
    return ~dominated

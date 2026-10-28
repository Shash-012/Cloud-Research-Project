"""Chance-constrained candidate selection (Step 2.10, E3)."""
import numpy as np

from .objectives import OBJECTIVES
from .pareto import pareto_mask

RULES = ("aug_tchebycheff", "weighted_sum", "min_cost", "min_energy")


def normalise(F: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Min-max normalise each column of F over the rows of `ref` (0 where all equal)."""
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return np.where(hi > lo, (F - lo) / span, 0.0)


def select(ks, n, F, slo: float, rule: str = "aug_tchebycheff", weights=(1.0, 1.0, 1.0),
           rho: float = 0.01, objectives=OBJECTIVES) -> dict:
    """Pick one candidate.

    F holds (latency U, cost, energy) per candidate. Feasibility is U <= slo (chance constraint).
    `objectives` chooses which columns enter the front and score (e.g. ("cost", "energy")).
    Returns index, fallback flag and per-candidate feasible / on_front / score arrays.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule}")
    ks, n, F = np.asarray(ks), np.asarray(n, float), np.asarray(F, float)
    K = len(ks)
    feasible = F[:, 0] <= slo
    on_front = np.zeros(K, bool)
    score = np.full(K, np.nan)

    if not feasible.any():                       # base-paper fallback: the largest pool
        idx = int(np.lexsort((np.abs(ks), -n))[0])
        return {"index": idx, "fallback": True, "feasible": feasible, "on_front": on_front, "score": score}

    cols = [OBJECTIVES.index(o) for o in objectives]
    w = np.asarray(weights, float)
    w = w[cols] if len(w) == len(OBJECTIVES) else w
    feas_idx = np.flatnonzero(feasible)
    Ff = F[np.ix_(feas_idx, cols)]
    front_local = pareto_mask(Ff)
    front_idx = feas_idx[front_local]
    on_front[front_idx] = True
    Fp = Ff[front_local]

    if rule == "min_cost":
        s = Fp[:, objectives.index("cost")] if "cost" in objectives else F[front_idx, 1]
    elif rule == "min_energy":
        s = Fp[:, objectives.index("energy")] if "energy" in objectives else F[front_idx, 2]
    else:
        Fn = normalise(Fp, Fp) * w
        s = Fn.max(axis=1) + rho * Fn.sum(axis=1) if rule == "aug_tchebycheff" else Fn.sum(axis=1)
    score[front_idx] = s
    # lowest score; exact ties by smaller |k|, then smaller pool
    order = np.lexsort((n[front_idx], np.abs(ks[front_idx]), s))
    return {"index": int(front_idx[order[0]]), "fallback": False, "feasible": feasible,
            "on_front": on_front, "score": score}

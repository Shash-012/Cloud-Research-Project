"""Per-metric linear estimator of post-scaling telemetry.

This is the base paper's Eq. 1-2 layer. Before the response-time predictor can
score a candidate action k, something has to answer: "if I had vm_count + k
VMs right now, what would CPU / memory / queue length read?" The paper fits one
linear regression per metric on (current VM count, delta k).

Deviation from the paper, stated plainly: our metrics scale roughly with 1/n,
so a regression that is linear in k alone fits badly at small pool sizes. We
add one interaction feature, metric_before * n / (n + k), which is the exact
inverse-scaling term. The model stays a plain linear regression -- it just has
the right basis function in it.

Metric selection: the paper's first criterion (an independent t-test on
scale-up vs scale-down deltas, p < 0.001) is skipped -- with five hand-built
synthetic metrics there is nothing to prune on significance grounds. The second
criterion is kept, because it turns out to matter. A metric is retained only if
its linear estimator reaches R^2 > METRIC_R2_THRESHOLD on held-out transitions.
Queue length fails it: queue length goes as u/(1-u), which no linear model in
the pool size can track near saturation, and an unreliable estimate of a
candidate action's queue length is worse than no estimate at all. This is
exactly the failure mode the paper's threshold exists to catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from . import config as C


def _design(metric_before: np.ndarray, vm_before: np.ndarray, k: np.ndarray) -> np.ndarray:
    vm_after = np.maximum(vm_before + k, C.VM_MIN)
    ratio = vm_before / vm_after
    return np.column_stack([metric_before, vm_before, k, metric_before * ratio])


class MetricEstimator:
    """One LinearRegression per load metric."""

    def __init__(self, r2_threshold: float = C.METRIC_R2_THRESHOLD):
        self.r2_threshold = r2_threshold
        self.models: dict[str, LinearRegression] = {}
        self.cv_r2: dict[str, float] = {}
        self.retained: tuple = ()
        self.dropped: tuple = ()

    def fit(self, df: pd.DataFrame) -> "MetricEstimator":
        """Fit one LR per metric and keep only those clearing the R^2 threshold.

        Selection uses cross-validated R^2, not the training fit, so a metric is
        not retained on the strength of memorised rows.
        """
        vm_before = df["vm_before"].to_numpy()
        k = df["k"].to_numpy()

        retained, dropped = [], []
        for m in C.LOAD_METRICS:
            X = _design(df[f"before_{m}"].to_numpy(), vm_before, k)
            y = df[f"after_{m}"].to_numpy()
            score = float(cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2").mean())
            self.models[m] = LinearRegression().fit(X, y)
            self.cv_r2[m] = score
            (retained if score > self.r2_threshold else dropped).append(m)

        if not retained:
            raise RuntimeError("no metric cleared the R^2 threshold; check the estimator basis")
        self.retained = tuple(retained)
        self.dropped = tuple(dropped)
        return self

    @property
    def features(self) -> tuple:
        """Feature order the RT predictor must be trained and queried with."""
        return C.features_for(self.retained)

    def predict(self, metrics_before: dict, vm_before: int, k: int) -> dict:
        """Estimate the retained load metrics that would follow from applying k."""
        out = {}
        for m in self.retained:
            X = _design(
                np.array([metrics_before[m]], dtype=float),
                np.array([float(vm_before)]),
                np.array([float(k)]),
            )
            out[m] = float(self.models[m].predict(X)[0])
        return out

    def feature_vector(self, metrics_before: dict, vm_before: int, k: int,
                       request_rate: float) -> np.ndarray:
        """Full post-scaling feature row, in C.FEATURES order.

        vm_count and request_rate are known exactly for a candidate action --
        the first because we chose it, the second because scaling does not
        change how many requests arrive. Only the load metrics are estimated.
        """
        vm_after = int(np.clip(vm_before + k, C.VM_MIN, C.VM_MAX))
        est = self.predict(metrics_before, vm_before, vm_after - vm_before)
        row = {"vm_count": float(vm_after), "request_rate": float(request_rate)}
        row.update(est)
        return np.array([row[f] for f in self.features], dtype=float)

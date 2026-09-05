"""Calibration data collection.

The base paper collects training data by perturbing the VM count and recording
what happens. We do the same against the synthetic app. Each calibration row is
a *transition*:

    (metrics observed at vm_count)  --apply delta k-->  (metrics + RT at vm_count+k)

One run therefore produces both training sets the pipeline needs:

  * metric estimator : (metrics_before, vm_count, k) -> metrics_after
  * RT predictor     : metrics_after                 -> response_time_after

Request rate is held fixed across a transition. Scaling the pool does not
change how many requests arrive, so the arrival rate is exogenous to the
action; only the load-dependent metrics move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .environment import SyntheticApp, generate_workload


def collect(n_rows: int = C.CALIBRATION_ROWS, seed: int = C.CALIBRATION_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    app = SyntheticApp(seed=seed + 1)

    # Sample request rates from the same distribution the control loop will
    # see, so the predictor is not asked to extrapolate at run time.
    workload = generate_workload()

    rows = []
    for i in range(n_rows):
        rate = float(rng.choice(workload))
        vm_before = int(rng.integers(C.VM_MIN, C.VM_MAX + 1))
        k = int(rng.choice(C.K_CANDIDATES))
        vm_after = int(np.clip(vm_before + k, C.VM_MIN, C.VM_MAX))
        k_effective = vm_after - vm_before

        before = app.observe(vm_before, rate, step=i)
        after = app.observe(vm_after, rate, step=i)

        row = {f"before_{m}": before[m] for m in C.LOAD_METRICS}
        row.update({f"after_{m}": after[m] for m in C.LOAD_METRICS})
        row.update(
            {
                "vm_before": float(vm_before),
                "k": float(k_effective),
                "vm_after": float(vm_after),
                "request_rate": rate,
                C.TARGET: after[C.TARGET],
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def rt_training_frame(df: pd.DataFrame, metrics=C.LOAD_METRICS) -> pd.DataFrame:
    """Project the calibration frame onto the RT predictor's feature order.

    `metrics` is the estimator's retained list, so the predictor only ever sees
    features that will actually be available at decision time.
    """
    out = pd.DataFrame(
        {
            "vm_count": df["vm_after"],
            "request_rate": df["request_rate"],
        }
    )
    for m in metrics:
        out[m] = df[f"after_{m}"]
    out[C.TARGET] = df[C.TARGET]
    return out[list(C.features_for(metrics)) + [C.TARGET]]

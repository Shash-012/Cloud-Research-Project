import numpy as np
import pandas as pd

from autoscaler.data.calibration import extract_cloudsim_transitions
from autoscaler.schema import METRICS_CLOUDSIM, transition_columns


def _log(ks, n0=5, rt=None):
    """A run log where the action at the end of step t takes effect from step t+1 (booting during t+1)."""
    rows, pool = [], n0
    for t, k in enumerate(ks):
        row = {"run_id": "r", "step": t, "rate": 100.0 + t, "n_running": pool, "n_booting": 0, "k": k,
               "rt_mean_ms": 50.0 + t if rt is None else rt[t], "completions": 10}
        row.update({m: float(t) for m in METRICS_CLOUDSIM})
        rows.append(row)
        pool += k
    return pd.DataFrame(rows)


def test_before_t_after_t_plus_2_with_hold():
    # RandomTarget pattern: act, hold, act, hold ...
    log = _log([2, 0, -1, 0, 3, 0, 0, 0])
    df = extract_cloudsim_transitions(log, 2, 0.15)
    assert list(df.columns[: len(transition_columns("cloudsim"))]) == transition_columns("cloudsim")
    first = df.iloc[0]
    assert (first.step, first.n_before, first.k, first.n_after) == (0, 5, 2, 7)
    assert first.cpu_util_before == 0.0 and first.cpu_util_after == 2.0 and first.rt_after_ms == 52.0
    assert first.rate_before == 100.0 and first.rate_after == 102.0
    # step 1 is a hold before an action at step 2: the pool changes before t+2, so it is not a transition
    assert 1 not in set(df.step)
    assert set(df.step) == {0, 2, 4, 5}


def test_skips_missing_rt_and_booting():
    log = _log([1, 0, 0, 0, 0], rt=[50, 50, np.nan, 50, 50])
    df = extract_cloudsim_transitions(log, 2, 0.15)
    assert 0 not in set(df.step)                      # after-interval without a response time
    log2 = _log([0, 0, 0, 0])
    log2.loc[2, "n_booting"] = 1
    assert 0 not in set(extract_cloudsim_transitions(log2, 2, 0.15).step)


def test_rate_drift_flag():
    log = _log([0, 0, 0, 0])
    log["rate"] = [100.0, 100.0, 120.0, 100.0]
    df = extract_cloudsim_transitions(log, 2, 0.15).set_index("step")
    assert bool(df.loc[0, "rate_drift_flag"]) and not bool(df.loc[1, "rate_drift_flag"])

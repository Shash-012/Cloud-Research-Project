import numpy as np
import pandas as pd

from autoscaler.models.metric_filters import final_metric_set, r2_filter, ttest_filter


def _frame(rng, n=400):
    k = rng.choice([-3, -2, -1, 1, 2, 3], size=n)
    before = rng.normal(50, 5, size=n)
    return pd.DataFrame({
        "k": k,
        # informative: moves against the action
        "a_before": before, "a_after": before - 4 * k + rng.normal(0, 1, n),
        # uninformative: same noise whatever the action
        "b_before": before, "b_after": before + rng.normal(0, 1, n),
    })


def test_ttest_keeps_informative_drops_noise():
    res = ttest_filter(_frame(np.random.default_rng(0)), ["a", "b"], alpha=0.05)
    assert res["a"]["keep"] and res["a"]["p"] < 1e-10
    assert not res["b"]["keep"]


def test_r2_filter_threshold():
    keep = r2_filter({"a": {"x": 0.95, "y": 0.5}, "b": {"x": 0.89, "y": 0.90}}, 0.90)
    assert keep == {"a": True, "b": False}


def test_final_set_is_intersection():
    t = {"a": {"keep": True}, "b": {"keep": True}, "c": {"keep": False}}
    r = {"a": True, "b": False, "c": True}
    assert final_metric_set(t, r, ["a", "b", "c"]) == ["a"]

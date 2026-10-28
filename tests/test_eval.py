import numpy as np
import pandas as pd
import pytest

from autoscaler.config import default_prototype_config
from autoscaler.eval.metrics import summarise
from autoscaler.eval.stats import holm, mean_ci, paired_table, paired_wilcoxon


def test_mean_ci():
    m, h = mean_ci([1.0, 2.0, 3.0])
    assert m == 2.0 and h == pytest.approx(4.302653 * 1 / np.sqrt(3), rel=1e-5)


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_paired_wilcoxon_zero_diff():
    assert paired_wilcoxon([1, 2], [1, 2]) == (0.0, 1.0)


def test_summarise_and_paired_table():
    cfg = default_prototype_config()
    cfg["time"]["warmup_steps"] = 2
    n = 6
    log = pd.DataFrame({"step": range(n), "rt_mean_ms": [100, 100, 300, 100, 200, 260],
                        "slo_violated": [False, False, True, False, False, True],
                        "n_running": [2] * n, "n_booting": [0] * n, "cost_usd": [1.0] * n,
                        "energy_wh": [500.0] * n})
    dec = [{"step": t, "k": k, "fallback": t == 5, "chosen_U": 250.0, "chosen_mean_log": np.log(200.0)}
           for t, k in zip(range(n), [0, 1, 0, -1, 2, 0])]
    s = summarise(log, dec, cfg)
    assert s["violation_rate"] == 0.5 and s["cost_usd"] == 4.0 and s["energy_kwh"] == 2.0
    assert s["churn"] == 2 and s["fallback_rate"] == 0.25
    # outcomes at steps 3, 4, 5 vs U = 250: 100 ok, 200 ok, 260 miss
    assert s["realised_coverage"] == pytest.approx(2 / 3)
    assert s["mean_margin_ms"] == pytest.approx(50.0)

    df = pd.DataFrame({"workload": ["a"] * 6, "policy": ["p"] * 3 + ["q"] * 3, "seed": [1, 2, 3] * 2,
                       "v": [0.1, 0.2, 0.3, 0.2, 0.3, 0.5]})
    t = paired_table(df, "p", ["v"])
    assert len(t) == 1 and t.median_diff_ref_minus_policy[0] == pytest.approx(-0.1)


def test_hypervolume_hand_computed_two_points():
    from autoscaler.eval.stats import hypervolume
    # boxes [p, 1.1]^3: 1.0*0.6*0.6 + 0.6*1.0*0.6 - 0.6^3 (overlap from (0.5, 0.5, 0.5))
    pts = [[0.1, 0.5, 0.5], [0.5, 0.1, 0.5]]
    assert hypervolume(pts) == pytest.approx(0.36 + 0.36 - 0.216)
    assert hypervolume([[0.0, 0.0, 0.0]]) == pytest.approx(1.1 ** 3)
    # a dominated point adds nothing
    assert hypervolume(pts + [[0.6, 0.6, 0.6]]) == pytest.approx(hypervolume(pts))
    assert hypervolume([[1.2, 0.0, 0.0]]) == 0.0


def test_minmax_normalise():
    from autoscaler.eval.stats import minmax_normalise
    ref = np.array([[0.0, 5.0], [10.0, 5.0]])
    assert np.allclose(minmax_normalise(np.array([[5.0, 5.0]]), ref), [[0.5, 0.0]])

import numpy as np
import pandas as pd
import pytest
import torch

from autoscaler.config import default_prototype_config
from autoscaler.models.quantile import QuantileHGB, QuantileNetModel, pinball

HP = {"hidden": [16, 8], "lr": 3e-3, "dropout": 0.0, "weight_decay": 1e-6, "batch_size": 128, "slope": 0.01}


def test_pinball_values():
    y = torch.tensor([1.0, 1.0])
    assert float(pinball(y, torch.tensor([0.0, 0.0]), 0.95)) == pytest.approx(0.95)
    assert float(pinball(y, torch.tensor([2.0, 2.0]), 0.95)) == pytest.approx(0.05)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 3000
    u = rng.uniform(0.1, 0.9, n)
    df = pd.DataFrame({"cpu_util_after": 100 * u, "n_after": rng.integers(1, 31, n).astype(float)})
    df["rt_after_ms"] = 40 / (1 - u) * np.exp(rng.normal(0, 0.1, n))
    cfg = default_prototype_config()
    cfg["ensemble"].update(max_epochs=60, patience=10)
    return cfg, df


def test_quantile_models_cover_about_tau(data, tmp_path):
    cfg, df = data
    cols = ["cpu_util_after", "n_after"]
    tr, te = df.iloc[:2000], df.iloc[2000:]
    y = np.log(te.rt_after_ms.to_numpy())
    qn = QuantileNetModel(cols, HP, 0.95).fit(tr, cfg, seed=1)
    qh = QuantileHGB(cols, 0.95, 100, 0.05, 1).fit(tr)
    for mdl in (qn, qh):
        assert 0.90 <= np.mean(y <= mdl.predict_frame(te)) <= 0.99
    qn.save(tmp_path / "qn")
    assert np.array_equal(QuantileNetModel.load(tmp_path / "qn").predict_frame(te), qn.predict_frame(te))
    qn2 = QuantileNetModel(cols, HP, 0.95).fit(tr, cfg, seed=1)
    assert np.array_equal(qn2.predict_frame(te), qn.predict_frame(te))

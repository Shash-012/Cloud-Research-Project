import numpy as np
import pandas as pd
import pytest
import torch

from autoscaler.config import default_prototype_config
from autoscaler.models.rt_ensemble import (Member, RTEnsemble, member_data, stage2_columns)
from autoscaler.models.tuning import random_search, sample_hparams

HP = {"hidden": [16, 8], "lr": 3e-3, "dropout": 0.1, "weight_decay": 1e-5, "batch_size": 64, "slope": 0.01}


@pytest.fixture(scope="module")
def data():
    cfg = default_prototype_config()
    cfg["ensemble"].update(max_epochs=15, patience=5)
    rng = np.random.default_rng(0)
    n = 800
    u = rng.uniform(0.1, 0.95, n)
    df = pd.DataFrame({"cpu_util_after": 100 * u + rng.normal(0, 1.5, n),
                       "n_after": rng.integers(1, 31, n).astype(float)})
    df["rt_after_ms"] = 40 / (1 - u) * (1 + rng.normal(0, 0.05, n))
    return cfg, df


def test_member_shapes_and_positive_var():
    m = Member(3, 8, 4, 0.1, 0.01).eval()
    mu, var = m(torch.zeros(5, 3))
    assert mu.shape == (5,) and var.shape == (5,) and (var > 0).all()


def test_stage2_columns():
    assert stage2_columns(["cpu_util"]) == ["cpu_util_after", "n_after"]
    assert stage2_columns(["cpu_util"], True)[-1] == "rate_after"


def test_member_data_val_disjoint_from_bootstrap():
    boot, val = member_data(1000, 0.15, seed=3)
    assert len(val) == 150 and len(boot) == 850
    assert not set(boot) & set(val)
    assert len(set(boot)) < len(boot)        # sampled with replacement


def test_ensemble_predict_shapes_and_determinism(data):
    cfg, df = data
    cols = ["cpu_util_after", "n_after"]
    a = RTEnsemble(cols, HP).fit(df, cfg, members=3, seed=10)
    b = RTEnsemble(cols, HP).fit(df, cfg, members=3, seed=10)
    mu, var = a.predict_frame(df)
    assert mu.shape == (3, len(df)) and var.shape == (3, len(df)) and (var > 0).all()
    assert np.array_equal(mu, b.predict_frame(df)[0])
    # members differ (bootstrap + seeds)
    assert not np.allclose(mu[0], mu[1])
    # learned something: ensemble mean correlates with the target
    assert np.corrcoef(mu.mean(0), np.log(df.rt_after_ms))[0, 1] > 0.6


def test_parallel_fit_equals_serial(data):
    cfg, df = data
    cols = ["cpu_util_after", "n_after"]
    a = RTEnsemble(cols, HP).fit(df, cfg, members=2, seed=4, n_jobs=1)
    b = RTEnsemble(cols, HP).fit(df, cfg, members=2, seed=4, n_jobs=2)
    assert np.array_equal(a.predict_frame(df)[0], b.predict_frame(df)[0])


def test_mean_kind_and_subset_and_save(data, tmp_path):
    cfg, df = data
    cols = ["cpu_util_after", "n_after"]
    mse = RTEnsemble(cols, HP, kind="mean").fit(df, cfg, members=1, seed=1)
    mu, var = mse.predict_frame(df)
    assert mu.shape == (1, len(df)) and np.all(var == 0)
    ens = RTEnsemble(cols, HP).fit(df, cfg, members=2, seed=2)
    assert ens.subset(1).members == 1
    ens.save(tmp_path / "e")
    back = RTEnsemble.load(tmp_path / "e")
    assert np.array_equal(ens.predict_frame(df)[0], back.predict_frame(df)[0])


def test_hparam_sampling_in_range():
    cfg = default_prototype_config()
    rng = np.random.default_rng(0)
    for _ in range(50):
        hp = sample_hparams(cfg["tuning"], rng)
        assert 1e-4 <= hp["lr"] <= 3e-3 and 1e-6 <= hp["weight_decay"] <= 1e-3
        assert hp["hidden"] in cfg["tuning"]["hidden"]


def test_random_search_returns_best(data):
    cfg, df = data
    hp, table = random_search(df, ["cpu_util_after", "n_after"], cfg, seed=1, trials=3)
    assert len(table) == 3
    assert table["val_nll"].min() == pytest.approx(table.loc[table["lr"] == hp["lr"], "val_nll"].iloc[0])

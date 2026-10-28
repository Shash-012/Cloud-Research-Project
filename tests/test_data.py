import numpy as np
import pandas as pd
import pytest

from autoscaler.config import default_prototype_config
from autoscaler.data.calibration import collect_synthetic
from autoscaler.data.splits import group_split, random_split
from autoscaler.schema import TRANSITION_COLUMNS


@pytest.fixture(scope="module")
def small():
    cfg = default_prototype_config()
    cfg["calibration"]["rows"] = 600
    return cfg, collect_synthetic(cfg, np.random.default_rng(cfg["calibration"]["seed"]))


def test_columns_and_no_missing(small):
    cfg, df = small
    assert list(df.columns) == TRANSITION_COLUMNS
    assert len(df) == 600 and not df.isna().any().any()


def test_rate_same_both_sides_and_clamped_pool(small):
    _, df = small
    assert (df.rate_before == df.rate_after).all()
    assert (df.n_after == np.clip(df.n_before + df.k, 1, 30)).all()
    assert df.k.between(-7, 7).all()


def test_collect_reproducible():
    cfg = default_prototype_config()
    cfg["calibration"]["rows"] = 50
    a = collect_synthetic(cfg, np.random.default_rng(1))
    b = collect_synthetic(cfg, np.random.default_rng(1))
    pd.testing.assert_frame_equal(a, b)


def test_random_split_partitions(small):
    _, df = small
    tr, ca, te = random_split(df, [0.6, 0.2, 0.2], seed=11)
    assert (len(tr), len(ca), len(te)) == (360, 120, 120)
    tr2, _, _ = random_split(df, [0.6, 0.2, 0.2], seed=11)
    pd.testing.assert_frame_equal(tr, tr2)


def test_group_split_keeps_groups_together():
    df = pd.DataFrame({"run_id": np.repeat([f"r{i}" for i in range(10)], 5), "x": range(50)})
    parts = group_split(df, "run_id", [0.6, 0.2, 0.2], seed=3)
    sets = [set(p.run_id) for p in parts]
    assert sum(len(p) for p in parts) == 50
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

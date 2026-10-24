import numpy as np
from autoscale.data.dataset import TransitionDataset, blocked_folds
from autoscale.selection.accuracy import select_metrics
from autoscale.env.metrics_gen import build_catalog


def _planted(N=400, seed=0):
    rng = np.random.default_rng(seed)
    load = rng.uniform(20, 60, N); w = rng.integers(2, 12, N).astype(float)
    k = rng.choice([-2, -1, 1, 2], N).astype(float)
    k = np.where(w + k < 1, 1, k)
    u0, u1 = load / w, load / (w + k)
    f = lambda u, s: u + rng.normal(0, s, N)
    cols = {"linear": (f(u0, .3), f(u1, .3)),
            "sat": ((u0 / 10) ** 5 + rng.normal(0, .02, N), (u1 / 10) ** 5 + rng.normal(0, .02, N)),
            "const": (np.full(N, 5.0) + rng.normal(0, .01, N), np.full(N, 5.0) + rng.normal(0, .01, N)),
            "noise": (rng.normal(0, 1, N), rng.normal(0, 1, N))}
    names = list(cols)
    mb = np.stack([cols[n][0] for n in names], 1); ma = np.stack([cols[n][1] for n in names], 1)
    z = np.zeros(N)
    return TransitionDataset(names, mb, ma, w, k, z, z, z, z)


def test_planted_metrics_recovered():
    s = select_metrics(_planted()).set_index("metric")
    assert s.loc["linear", "selected"]
    assert not s.loc["const", "dynamic"] and not s.loc["noise", "dynamic"]
    assert s.loc["sat", "dynamic"] and not s.loc["sat", "predictable"]     # reacts, but Eq. 1 cannot predict it
    assert not s.loc["sat", "selected"]


def test_blocked_folds_contiguous_and_disjoint():
    folds = blocked_folds(100, 3)
    seen = np.concatenate([v for _, v in folds])
    assert sorted(seen) == list(range(100))
    for tr, va in folds:
        assert set(tr).isdisjoint(va)
        assert (np.diff(va) == 1).all()                 # contiguous time block


def test_catalog_matches_paper_group_sizes():
    c = build_catalog()
    assert c.group.value_counts().to_dict() == {"TCP": 32, "MEM": 30, "CPU": 21, "NET": 18, "BUD": 14, "DSK": 9, "INODE": 4}
    assert len(c) == 128

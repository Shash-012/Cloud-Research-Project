"""Filter 2: can the metric's post-scaling value be predicted by Eq. 1? Keep only R2 > threshold."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..models.metric_lr import MetricLR
from ..data.dataset import TransitionDataset
from .dynamicity import dynamicity_test


def accuracy_filter(train: TransitionDataset, test: TransitionDataset, thr: float = 0.9):
    lr = MetricLR().fit(train.m_before, train.w_before, train.k, train.m_after)
    r2 = lr.r2(test.m_before, test.w_before, test.k, test.m_after)
    return lr, pd.DataFrame(dict(metric=train.names, idx=np.arange(len(r2)), r2=r2, predictable=r2 > thr))


def select_metrics(ds: TransitionDataset, alpha=0.001, r2_thr=0.9, cohen_min=None, train_frac=0.7) -> pd.DataFrame:
    """Both filters on a blocked (time-ordered) split: dynamicity + LR fit on the first part, R2 on the held-out tail."""
    n = len(ds); cut = int(n * train_frac)
    tr, te = ds.subset(np.arange(cut)), ds.subset(np.arange(cut, n))
    dyn = dynamicity_test(tr, alpha, cohen_min)
    _, acc = accuracy_filter(tr, te, r2_thr)
    out = dyn.merge(acc[["idx", "r2", "predictable"]], on="idx")
    out["selected"] = out["dynamic"] & out["predictable"]
    return out

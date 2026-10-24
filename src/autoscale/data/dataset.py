"""Transition dataset: one row = (metrics before, w, k, metrics after, RT before/after)."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


def blocked_folds(n: int, k: int = 3):
    """Contiguous, time-ordered folds (Barrow & Crone 2016). Returns [(train_idx, val_idx), ...]."""
    edges = np.linspace(0, n, k + 1).astype(int)
    folds = []
    for i in range(k):
        val = np.arange(edges[i], edges[i + 1])
        tr = np.concatenate([np.arange(0, edges[i]), np.arange(edges[i + 1], n)])
        folds.append((tr, val))
    return folds


@dataclass
class TransitionDataset:
    names: list
    m_before: np.ndarray
    m_after: np.ndarray
    w_before: np.ndarray
    k: np.ndarray
    rt_before: np.ndarray
    rt_after: np.ndarray
    rr_before: np.ndarray
    rr_after: np.ndarray
    cx: np.ndarray = field(default=None)

    def __len__(self):
        return len(self.k)

    def subset(self, idx) -> "TransitionDataset":
        idx = np.asarray(idx)
        return TransitionDataset(self.names, self.m_before[idx], self.m_after[idx], self.w_before[idx], self.k[idx],
                                 self.rt_before[idx], self.rt_after[idx], self.rr_before[idx], self.rr_after[idx],
                                 None if self.cx is None else self.cx[idx])

    def head(self, n: int) -> "TransitionDataset":
        return self.subset(np.arange(min(n, len(self))))

    def states(self, sel: np.ndarray | None = None, use_rr: bool = False, clip: float | None = None):
        """(X, y): both the 'after' and the 'before' state of every row, with the RT measured in that state."""
        sel = np.arange(self.m_after.shape[1]) if sel is None else np.asarray(sel)
        xa, xb = self.m_after[:, sel], self.m_before[:, sel]
        if use_rr:
            xa = np.column_stack([xa, self.rr_after]); xb = np.column_stack([xb, self.rr_before])
        X = np.vstack([xa, xb]); y = np.concatenate([self.rt_after, self.rt_before])
        if clip is not None:
            y = np.minimum(y, clip)
        return X, y

    def save(self, path):
        np.savez_compressed(path, names=np.array(self.names), m_before=self.m_before, m_after=self.m_after,
                            w_before=self.w_before, k=self.k, rt_before=self.rt_before, rt_after=self.rt_after,
                            rr_before=self.rr_before, rr_after=self.rr_after,
                            cx=self.cx if self.cx is not None else np.zeros(len(self.k)))

    @classmethod
    def load(cls, path) -> "TransitionDataset":
        z = np.load(path, allow_pickle=False)
        return cls(list(z["names"]), z["m_before"], z["m_after"], z["w_before"], z["k"], z["rt_before"],
                   z["rt_after"], z["rr_before"], z["rr_after"], z["cx"])

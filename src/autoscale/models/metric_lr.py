"""Per-metric linear regression, paper Eq. 1:
    m' = c0 + c1 * m * w/(w+k) + c2 * m * k/(w+k)
Fitted by least squares for all metrics at once. Includes the SGD online update (Sec 4.4)."""
from __future__ import annotations
import numpy as np


def _feats(M, w, k):
    w = np.asarray(w, dtype=float)[..., None]; k = np.asarray(k, dtype=float)[..., None]
    return M * w / (w + k), M * k / (w + k)


class MetricLR:
    def __init__(self):
        self.coef = None          # (d, 3): c0, c1, c2
        self.lo = self.hi = None  # clipping range (training range +/-10%) keeps predictions physical

    def fit(self, M, w, k, Mp):
        x1, x2 = _feats(M, w, k)
        d = M.shape[1]
        self.coef = np.zeros((d, 3))
        for j in range(d):
            A = np.column_stack([np.ones(len(M)), x1[:, j], x2[:, j]])
            self.coef[j] = np.linalg.lstsq(A, Mp[:, j], rcond=None)[0]
        span = Mp.max(0) - Mp.min(0)
        self.lo, self.hi = Mp.min(0) - 0.1 * span, Mp.max(0) + 0.1 * span
        return self

    def predict_rows(self, M, w, k):
        x1, x2 = _feats(M, w, k)
        P = self.coef[:, 0] + self.coef[:, 1] * x1 + self.coef[:, 2] * x2
        return np.clip(P, self.lo, self.hi)

    def predict(self, m, w: int, ks):
        """m: (d,) current metrics. ks: array of candidate k. Returns (len(ks), d)."""
        ks = np.asarray(ks, dtype=float)
        return self.predict_rows(np.tile(m, (len(ks), 1)), np.full(len(ks), w, dtype=float), ks)

    def r2(self, M, w, k, Mp):
        P = self.predict_rows(M, w, k)
        sse = ((P - Mp) ** 2).sum(0); sst = ((Mp - Mp.mean(0)) ** 2).sum(0)
        return np.where(sst > 1e-12, 1 - sse / np.maximum(sst, 1e-12), 0.0)

    def sgd_update(self, m, w, k, m_obs, lr=0.01, rel_thr=0.15) -> bool:
        """Normalised-LMS step on c0..c2 for metrics whose prediction deviates by > rel_thr (relative to training span)."""
        x1, x2 = _feats(m[None, :], [w], [k]); x1, x2 = x1[0], x2[0]
        pred = self.coef[:, 0] + self.coef[:, 1] * x1 + self.coef[:, 2] * x2
        err = pred - m_obs
        span = np.maximum(self.hi - self.lo, 1e-9)
        mask = np.abs(err) / span > rel_thr
        if not mask.any():
            return False
        phi = np.stack([np.ones_like(x1), x1, x2], axis=1)
        norm = (phi ** 2).sum(1) + 1e-9
        self.coef[mask] -= lr * (err[mask] / norm[mask])[:, None] * phi[mask]
        return True

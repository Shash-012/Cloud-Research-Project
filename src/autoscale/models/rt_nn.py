"""Response-time network (paper Eq. 7, Sec 4.5): two hidden layers, LeakyReLU(0.1), dropout, Adam, early stopping.

Sizes follow the number n of selected metrics: H1 = 2n + 1, H2 = ceil(H1/2) + 1.
Deviation (documented): the TARGET is standardised internally during training and un-standardised in predict().
This is only a reparameterisation of the output layer; without it, Adam at lr 0.01 needs ~1e5 steps to reach ~1000 ms.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

ACTS = {"sigmoid": lambda: nn.Sigmoid(), "tanh": lambda: nn.Tanh(), "relu": lambda: nn.ReLU(),
        "leaky_relu": lambda: nn.LeakyReLU(0.1)}


@dataclass
class NNConfig:
    optimizer: str = "adam"
    lr: float = 0.01
    activation: str = "leaky_relu"
    dropout: float = 0.2
    batch_size: int = 64
    normalization: str = "standard"      # none | minmax | standard
    loss: str = "mse"                    # mse | mae
    max_epochs: int = 500
    patience: int = 100

    @classmethod
    def from_dict(cls, d: dict) -> "NNConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def layer_sizes(n: int):
    h1 = 2 * n + 1
    return h1, math.ceil(h1 / 2) + 1


class RTNet(nn.Module):
    def __init__(self, n_in: int, act: str, dropout: float):
        super().__init__()
        h1, h2 = layer_sizes(n_in)
        self.net = nn.Sequential(nn.Linear(n_in, h1), ACTS[act](), nn.Dropout(dropout),
                                 nn.Linear(h1, h2), ACTS[act](), nn.Dropout(dropout), nn.Linear(h2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


OPTS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop, "adagrad": torch.optim.Adagrad}


class RTModel:
    """Sklearn-like wrapper. predict() works on a batch of candidate metric vectors (one per k)."""

    def __init__(self, n_in: int, cfg: NNConfig | None = None):
        self.n_in = n_in
        self.cfg = cfg or NNConfig()
        self.net = None
        self.xs = (np.zeros(n_in), np.ones(n_in))
        self.ys = (0.0, 1.0)

    def _norm_fit(self, X):
        m = self.cfg.normalization
        if m == "minmax":
            lo, hi = X.min(0), X.max(0); self.xs = (lo, np.where(hi - lo > 1e-12, hi - lo, 1.0))
        elif m == "standard":
            mu, sd = X.mean(0), X.std(0); self.xs = (mu, np.where(sd > 1e-12, sd, 1.0))
        else:
            self.xs = (np.zeros(self.n_in), np.ones(self.n_in))

    def _tx(self, X):
        return torch.tensor((np.asarray(X, dtype=np.float64) - self.xs[0]) / self.xs[1], dtype=torch.float32)

    def fit(self, X, y, seed: int = 0, val_frac: float = 0.1):
        c = self.cfg
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        X = np.asarray(X, float); y = np.asarray(y, float)
        n_val = max(5, int(len(X) * val_frac))
        # contiguous tail as validation (time-ordered data)
        Xtr, ytr, Xva, yva = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
        self._norm_fit(Xtr)
        self.ys = (float(ytr.mean()), float(ytr.std() + 1e-9))
        xt, xv = self._tx(Xtr), self._tx(Xva)
        yt = torch.tensor((ytr - self.ys[0]) / self.ys[1], dtype=torch.float32)
        yv = torch.tensor((yva - self.ys[0]) / self.ys[1], dtype=torch.float32)
        self.net = RTNet(self.n_in, c.activation, c.dropout)
        opt = OPTS[c.optimizer](self.net.parameters(), lr=c.lr)
        lossf = nn.MSELoss() if c.loss == "mse" else nn.L1Loss()
        best, best_state, since = float("inf"), None, 0
        for ep in range(c.max_epochs):
            self.net.train()
            perm = torch.tensor(rng.permutation(len(xt)))
            for i in range(0, len(perm), c.batch_size):
                b = perm[i:i + c.batch_size]
                opt.zero_grad(); loss = lossf(self.net(xt[b]), yt[b]); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(lossf(self.net(xv), yv))
            if not math.isfinite(v):
                break
            if v < best - 1e-9:
                best, since = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                since += 1
                if since >= c.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        self.best_val = best
        return self

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(X)
        with torch.no_grad():
            z = self.net(self._tx(X)).numpy().astype(np.float64)
        return z * self.ys[1] + self.ys[0]

    def predict_dist(self, X):
        """(mu, sigma). Sigma is 0 for the single network of the base paper; the Part-2 ensemble plugs in here."""
        mu = self.predict(X)
        return mu, np.zeros_like(mu)

    # ---- persistence ----
    def save(self, path):
        torch.save(dict(n_in=self.n_in, cfg=asdict(self.cfg), state=self.net.state_dict(),
                        xs=self.xs, ys=self.ys), path)

    @classmethod
    def load(cls, path) -> "RTModel":
        d = torch.load(path, weights_only=False)
        m = cls(d["n_in"], NNConfig.from_dict(d["cfg"]))
        m.net = RTNet(m.n_in, m.cfg.activation, m.cfg.dropout)
        m.net.load_state_dict(d["state"]); m.net.eval()
        m.xs, m.ys = d["xs"], d["ys"]
        return m

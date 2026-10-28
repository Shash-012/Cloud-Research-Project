"""Quantile-regression models for the ablation (Step 2.7, E2). Both predict q_tau of log RT."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn

from .rt_ensemble import Standardiser, member_data, train_net


def pinball(y, q, tau):
    r = y - q
    return torch.mean(torch.maximum(tau * r, (tau - 1) * r))


class QuantileNet(nn.Module):
    """Same body as Member, one output trained with the pinball loss."""

    def __init__(self, d_in, h1, h2, dropout, slope):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, h1), nn.LeakyReLU(slope), nn.Dropout(dropout),
            nn.Linear(h1, h2),   nn.LeakyReLU(slope), nn.Dropout(dropout))
        self.out = nn.Linear(h2, 1)

    def forward(self, x):
        return self.out(self.body(x)).squeeze(-1)


class QuantileNetModel:
    """Standardised inputs + QuantileNet; predict(X) -> q_tau of log RT."""

    def __init__(self, columns, hparams, tau):
        self.columns, self.hparams, self.tau = list(columns), dict(hparams), float(tau)
        self.scaler = Standardiser()
        self.net = None

    def fit(self, df: pd.DataFrame, cfg: dict, seed: int):
        torch.set_num_threads(1)
        e = cfg["ensemble"]
        X = self.scaler.fit(df[self.columns].to_numpy(float)).transform(df[self.columns].to_numpy(float))
        y = np.log(df["rt_after_ms"].to_numpy(float))
        # validation slice drawn first, then training on the remaining rows (no bootstrap: one model)
        boot, val = member_data(len(X), e["val_frac"], seed)
        train_idx = np.setdiff1d(np.arange(len(X)), val)
        h1, h2 = self.hparams["hidden"]
        torch.manual_seed(seed)
        net = QuantileNet(X.shape[1], h1, h2, self.hparams["dropout"], self.hparams["slope"])
        tau = self.tau
        self.net, self.val_loss, self.epochs = train_net(
            net, lambda m, Xb, yb: pinball(yb, m(Xb), tau), X[train_idx], y[train_idx], X[val], y[val],
            self.hparams, e["max_epochs"], e["patience"], seed)
        return self

    def predict(self, X) -> np.ndarray:
        Xt = torch.as_tensor(self.scaler.transform(np.asarray(X, float)), dtype=torch.float32)
        with torch.no_grad():
            return self.net(Xt).numpy().astype(float)

    def predict_frame(self, df):
        return self.predict(df[self.columns].to_numpy(float))

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "weights.pt")
        (path / "meta.json").write_text(json.dumps({
            "columns": self.columns, "hparams": self.hparams, "tau": self.tau,
            "scaler_mean": self.scaler.mean.tolist(), "scaler_std": self.scaler.std.tolist()}, indent=2))

    @staticmethod
    def load(path):
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        q = QuantileNetModel(meta["columns"], meta["hparams"], meta["tau"])
        q.scaler = Standardiser(np.array(meta["scaler_mean"]), np.array(meta["scaler_std"]))
        h1, h2 = q.hparams["hidden"]
        q.net = QuantileNet(len(q.columns), h1, h2, q.hparams["dropout"], q.hparams["slope"])
        q.net.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
        q.net.eval()
        return q


class QuantileHGB:
    """HistGradientBoostingRegressor(loss='quantile') on the same inputs."""

    def __init__(self, columns, tau, max_iter, learning_rate, seed):
        self.columns, self.tau = list(columns), float(tau)
        self.model = HistGradientBoostingRegressor(loss="quantile", quantile=self.tau, max_iter=max_iter,
                                                   learning_rate=learning_rate, random_state=seed)

    def fit(self, df: pd.DataFrame):
        self.model.fit(df[self.columns].to_numpy(float), np.log(df["rt_after_ms"].to_numpy(float)))
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(np.asarray(X, float))

    def predict_frame(self, df):
        return self.predict(df[self.columns].to_numpy(float))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

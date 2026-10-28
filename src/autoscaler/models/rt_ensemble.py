"""Stage 2: deep ensemble for log response time (Step 2.6, E1/E2)."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class Member(nn.Module):
    def __init__(self, d_in, h1, h2, dropout, slope):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, h1), nn.LeakyReLU(slope), nn.Dropout(dropout),
            nn.Linear(h1, h2),   nn.LeakyReLU(slope), nn.Dropout(dropout))
        self.mean = nn.Linear(h2, 1)
        self.var_raw = nn.Linear(h2, 1)

    def forward(self, x):
        h = self.body(x)
        mu = self.mean(h).squeeze(-1)
        var = nn.functional.softplus(self.var_raw(h)).squeeze(-1) + 1e-6
        return mu, var


class MeanNet(nn.Module):
    """Same body as Member, one output: the base paper's MSE-trained predictor."""

    def __init__(self, d_in, h1, h2, dropout, slope):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, h1), nn.LeakyReLU(slope), nn.Dropout(dropout),
            nn.Linear(h1, h2),   nn.LeakyReLU(slope), nn.Dropout(dropout))
        self.out = nn.Linear(h2, 1)

    def forward(self, x):
        return self.out(self.body(x)).squeeze(-1)


def stage2_columns(metrics, include_rate: bool = False) -> list[str]:
    return [f"{m}_after" for m in metrics] + ["n_after"] + (["rate_after"] if include_rate else [])


def build_net(kind: str, d_in: int, hp: dict) -> nn.Module:
    h1, h2 = hp["hidden"]
    cls = {"member": Member, "mean": MeanNet}[kind]
    return cls(d_in, h1, h2, hp["dropout"], hp["slope"])


def nll_loss(model, X, y):
    mu, var = model(X)
    return nn.GaussianNLLLoss()(mu, y, var)


def mse_loss(model, X, y):
    return nn.functional.mse_loss(model(X), y)


def train_net(model, loss_fn, Xtr, ytr, Xval, yval, hp: dict, max_epochs: int, patience: int,
              seed: int):
    """Adam on mini-batches; early stopping on validation loss; best weights restored."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    Xtr, ytr = torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(ytr, dtype=torch.float32)
    Xval, yval = torch.as_tensor(Xval, dtype=torch.float32), torch.as_tensor(yval, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    bs = int(hp["batch_size"])
    best, best_state, bad, epochs = float("inf"), copy.deepcopy(model.state_dict()), 0, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model, Xtr[idx], ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model, Xval, yval))
        epochs = epoch + 1
        if val < best - 1e-7:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, epochs


class Standardiser:
    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean) / self.std


def member_data(n: int, val_frac: float, seed: int):
    """Validation slice first, then a bootstrap sample of the remaining rows."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    val, rest = perm[:n_val], perm[n_val:]
    boot = rng.choice(rest, size=len(rest), replace=True)
    return boot, val


def _train_member(kind, Xs, y, hp, e, seed):
    torch.set_num_threads(1)
    boot, val = member_data(len(Xs), e["val_frac"], seed)
    torch.manual_seed(seed)                  # weight initialisation depends on the seed only
    net = build_net(kind, Xs.shape[1], hp)
    loss = nll_loss if kind == "member" else mse_loss
    net, best, epochs = train_net(net, loss, Xs[boot], y[boot], Xs[val], y[val], hp,
                                  e["max_epochs"], e["patience"], seed)
    return net, {"seed": seed, "val_loss": best, "epochs": epochs}


class RTEnsemble:
    """M members with mean and variance heads on log response time."""

    def __init__(self, columns: list[str], hparams: dict, kind: str = "member"):
        self.columns = list(columns)
        self.hparams = dict(hparams)
        self.kind = kind
        self.scaler = Standardiser()
        self.nets: list[nn.Module] = []
        self.train_info: list[dict] = []

    @property
    def members(self) -> int:
        return len(self.nets)

    def fit(self, df: pd.DataFrame, cfg: dict, members: int | None = None, seed: int | None = None,
            n_jobs: int = 1):
        """Member j trains with seed base + j; members are independent, so n_jobs > 1 is exact."""
        from joblib import Parallel, delayed
        e = cfg["ensemble"]
        M = members if members is not None else e["members"]
        base = seed if seed is not None else e["seed"]
        X = df[self.columns].to_numpy(float)
        y = np.log(df["rt_after_ms"].to_numpy(float))
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        if n_jobs == 1:
            results = [_train_member(self.kind, Xs, y, self.hparams, e, base + j) for j in range(1, M + 1)]
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_train_member)(self.kind, Xs, y, self.hparams, e, base + j) for j in range(1, M + 1))
        self.nets = [r[0] for r in results]
        self.train_info = [r[1] for r in results]
        return self

    def _forward(self, X: np.ndarray):
        Xt = torch.as_tensor(self.scaler.transform(X), dtype=torch.float32)
        with torch.no_grad():
            return [net(Xt) for net in self.nets]

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """(mu, var), each shape (M, rows); dropout off. For kind='mean', var is zero."""
        X = np.asarray(X, float)
        outs = self._forward(X)
        if self.kind == "member":
            mu = np.stack([o[0].numpy() for o in outs]).astype(float)
            var = np.stack([o[1].numpy() for o in outs]).astype(float)
        else:
            mu = np.stack([o.numpy() for o in outs]).astype(float)
            var = np.zeros_like(mu)
        return mu, var

    def predict_frame(self, df: pd.DataFrame):
        return self.predict(df[self.columns].to_numpy(float))

    def subset(self, n: int) -> "RTEnsemble":
        """The first n members (e.g. n=1 for the single-member ablation)."""
        other = RTEnsemble(self.columns, self.hparams, self.kind)
        other.scaler = self.scaler
        other.nets = self.nets[:n]
        other.train_info = self.train_info[:n]
        return other

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save([n.state_dict() for n in self.nets], path / "weights.pt")
        meta = {"columns": self.columns, "hparams": self.hparams, "kind": self.kind,
                "scaler_mean": self.scaler.mean.tolist(), "scaler_std": self.scaler.std.tolist(),
                "train_info": self.train_info}
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @staticmethod
    def load(path) -> "RTEnsemble":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        ens = RTEnsemble(meta["columns"], meta["hparams"], meta["kind"])
        ens.scaler = Standardiser(np.array(meta["scaler_mean"]), np.array(meta["scaler_std"]))
        ens.train_info = meta["train_info"]
        for state in torch.load(path / "weights.pt", weights_only=True):
            net = build_net(ens.kind, len(ens.columns), ens.hparams)
            net.load_state_dict(state)
            net.eval()
            ens.nets.append(net)
        return ens

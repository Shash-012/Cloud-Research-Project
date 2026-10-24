"""Hyper-parameter optimisation (paper Sec 6.3): random search, 20 trials, 3-fold BLOCKED CV,
candidates ranked by RMSE of f(g(k=0)) against measured RT (the only case where ground truth exists)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..data.dataset import blocked_folds
from .metric_lr import MetricLR
from .rt_nn import RTModel, NNConfig

SEARCH_SPACE = dict(
    optimizer=["rmsprop", "sgd", "adam", "adagrad"],
    lr=(0.001, 0.1),                       # log-uniform
    activation=["sigmoid", "tanh", "relu", "leaky_relu"],
    dropout=(0.1, 0.5),
    batch_size=[2 ** i for i in range(4, 11)],   # 16 .. 1024
    normalization=["minmax", "standard"],
    loss=["mse", "mae"],
)


def sample_config(rng: np.random.Generator, max_epochs: int, patience: int) -> NNConfig:
    s = SEARCH_SPACE
    return NNConfig(optimizer=str(rng.choice(s["optimizer"])),
                    lr=float(np.exp(rng.uniform(np.log(s["lr"][0]), np.log(s["lr"][1])))),
                    activation=str(rng.choice(s["activation"])), dropout=float(rng.uniform(*s["dropout"])),
                    batch_size=int(rng.choice(s["batch_size"])), normalization=str(rng.choice(s["normalization"])),
                    loss=str(rng.choice(s["loss"])), max_epochs=max_epochs, patience=patience)


def _fg0_rmse(ds_tr, ds_va, sel, cfg: NNConfig, clip, seed, use_rr=False) -> tuple[float, float]:
    lr = MetricLR().fit(ds_tr.m_before, ds_tr.w_before, ds_tr.k, ds_tr.m_after)
    X, y = ds_tr.states(sel, use_rr, clip)
    model = RTModel(X.shape[1], cfg).fit(X, y, seed=seed)
    g0 = lr.predict_rows(ds_va.m_before, ds_va.w_before, np.zeros(len(ds_va)))[:, sel]   # f(g(k=0))
    if use_rr:
        g0 = np.column_stack([g0, ds_va.rr_before])
    pred = model.predict(g0); y_true = np.minimum(ds_va.rt_before, clip)
    err = pred - y_true
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def random_search(ds, sel, hpo_cfg: dict, clip: float, seed: int = 0, use_rr: bool = False, verbose=False):
    rng = np.random.default_rng(seed)
    folds = blocked_folds(len(ds), hpo_cfg["folds"])
    rows, best, best_rmse = [], None, np.inf
    for t in range(hpo_cfg["trials"]):
        cfg = sample_config(rng, hpo_cfg["max_epochs"], hpo_cfg["patience"])
        rm, ma = zip(*[_fg0_rmse(ds.subset(tr), ds.subset(va), sel, cfg, clip, seed + t, use_rr) for tr, va in folds])
        rmse, mae = float(np.mean(rm)), float(np.mean(ma))
        rows.append(dict(trial=t, rmse=rmse, mae=mae, **{k: getattr(cfg, k) for k in
                         ("optimizer", "lr", "activation", "dropout", "batch_size", "normalization", "loss")}))
        if verbose:
            print(f"  trial {t:2d} rmse={rmse:8.1f} mae={mae:8.1f} {cfg.optimizer}/{cfg.activation}")
        if np.isfinite(rmse) and rmse < best_rmse:
            best, best_rmse = cfg, rmse
    return best, pd.DataFrame(rows)

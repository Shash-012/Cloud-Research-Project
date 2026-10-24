"""Trained artefacts for one application: selected metrics, Eq.1 coefficients, RT network."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .metric_lr import MetricLR
from .rt_nn import RTModel, NNConfig
from .hpo import random_search
from ..data.dataset import TransitionDataset
from ..selection.accuracy import select_metrics


@dataclass
class ModelBundle:
    app: str
    metric_names: list
    sel: np.ndarray
    lr: MetricLR
    rt: RTModel
    use_rr: bool
    qos_ms: float
    meta: dict

    def estimate_rt(self, m: np.ndarray, w: int, ks, rr: float = 0.0):
        """RT_est(k) for each candidate k: f(g(m, w, k)). Returns (mu, sigma) arrays (sigma = 0 in the base paper)."""
        P = self.lr.predict(m, w, ks)[:, self.sel]
        if self.use_rr:
            P = np.column_stack([P, np.full(len(P), rr)])
        return self.rt.predict_dist(P)

    def save(self, d):
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "lr.npz", coef=self.lr.coef, lo=self.lr.lo, hi=self.lr.hi, sel=self.sel)
        self.rt.save(d / "rt.pt")
        json.dump(dict(app=self.app, metric_names=self.metric_names, use_rr=self.use_rr, qos_ms=self.qos_ms,
                       meta=self.meta), open(d / "meta.json", "w"), indent=1, default=float)

    @classmethod
    def load(cls, d) -> "ModelBundle":
        d = Path(d); z = np.load(d / "lr.npz"); meta = json.load(open(d / "meta.json"))
        lr = MetricLR(); lr.coef, lr.lo, lr.hi = z["coef"], z["lo"], z["hi"]
        return cls(meta["app"], meta["metric_names"], z["sel"], lr, RTModel.load(d / "rt.pt"), meta["use_rr"],
                   meta["qos_ms"], meta["meta"])


def train_bundle(ds: TransitionDataset, cfg: dict, app: str, use_rr: bool = False, run_hpo: bool = True,
                 seed: int = 0, verbose: bool = True):
    """Full offline pipeline (Fig 2): filters -> Eq.1 fit -> HPO -> final NN fit."""
    sel_df = select_metrics(ds, cfg["alpha"], cfg["r2_threshold"], cfg["cohen_d_min"], cfg["select_train_frac"])
    sel = sel_df.index[sel_df["selected"]].to_numpy()
    if len(sel) < 2:
        raise RuntimeError(f"only {len(sel)} metrics selected for {app}; check data/thresholds")
    clip = cfg["rt_clip_ms"]
    lr = MetricLR().fit(ds.m_before, ds.w_before, ds.k, ds.m_after)
    nn_cfg = NNConfig.from_dict(cfg["nn"]); trials = None
    if run_hpo:
        best, trials = random_search(ds, sel, cfg["hpo"], clip, seed, use_rr, verbose)
        d = {k: getattr(best, k) for k in ("optimizer", "lr", "activation", "dropout", "batch_size", "normalization", "loss")}
        nn_cfg = NNConfig.from_dict({**cfg["nn"], **d})
    X, y = ds.states(sel, use_rr, clip)
    rt = RTModel(X.shape[1], nn_cfg).fit(X, y, seed=seed)
    meta = dict(n_selected=int(len(sel)), nn=nn_cfg.__dict__, n_rows=len(ds))
    return ModelBundle(app, ds.names, sel, lr, rt, use_rr, cfg["apps"][app]["qos_ms"], meta), sel_df, trials

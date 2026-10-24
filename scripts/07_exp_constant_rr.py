"""E3 (paper Sec 7.2): constant request rate, hidden complexity varies, VM count decorrelated from RT.
Five models (RR, VM, RR+VM, OPT, OPT+RR) are scored on out-of-fold predictions of 3-fold BLOCKED CV."""
import _common  # noqa
import argparse
import numpy as np, pandas as pd
from autoscale.config import load_config, RESULTS
from autoscale.data.perturbation import collect_perturbation
from autoscale.data.dataset import blocked_folds
from autoscale.selection.accuracy import select_metrics
from autoscale.models.rt_nn import RTModel, NNConfig

ap = argparse.ArgumentParser()
ap.add_argument("--app", default="cpu_bound"); ap.add_argument("--obs", type=int, default=250)
a = ap.parse_args()
cfg = load_config(); clip = cfg["rt_clip_ms"]
ds = collect_perturbation(cfg, a.app, a.obs, cfg["train_seed"] + 1, workload="const_rr", pool_mode="random", pool_range=(2, 10))
ds.save(RESULTS / "raw" / f"dataset_constrr_{a.app}.npz")
print(f"{len(ds)} observations; corr(VM count, RT) = {np.corrcoef(ds.w_before + ds.k, ds.rt_after)[0, 1]:.2f}; "
      f"corr(hidden complexity, RT) = {np.corrcoef(ds.cx, ds.rt_after)[0, 1]:.2f}")
sel_df = select_metrics(ds, cfg["alpha"], cfg["r2_threshold"], cfg["cohen_d_min"], cfg["select_train_frac"])
# E3 scores the RT network on OBSERVED metrics (no Eq. 1 prediction involved), so only the dynamicity filter is applied here;
# the R2 filter exists to protect the predicted-metric pipeline and is skipped (decision log).
sel = sel_df.index[sel_df.dynamic].to_numpy(); print("dynamic metrics used as OPT inputs:", len(sel))
w_after = ds.w_before + ds.k
feats = {  # inputs for each model: stack (after state, before state) like TransitionDataset.states()
    "RR": (np.c_[ds.rr_after], np.c_[ds.rr_before]),
    "VM": (np.c_[w_after], np.c_[ds.w_before]),
    "RR+VM": (np.c_[ds.rr_after, w_after], np.c_[ds.rr_before, ds.w_before]),
    "OPT": (ds.m_after[:, sel], ds.m_before[:, sel]),
    "OPT+RR": (np.c_[ds.m_after[:, sel], ds.rr_after], np.c_[ds.m_before[:, sel], ds.rr_before]),
}
y_after, y_before = np.minimum(ds.rt_after, clip), np.minimum(ds.rt_before, clip)
nn = NNConfig(max_epochs=200, patience=30)
recs = []
for name, (xa, xb) in feats.items():
    for f, (tr, va) in enumerate(blocked_folds(len(ds), 3)):
        X = np.vstack([xa[tr], xb[tr]]); y = np.concatenate([y_after[tr], y_before[tr]])
        m = RTModel(X.shape[1], nn).fit(X, y, seed=f)
        p = m.predict(xa[va])
        recs += [dict(model=name, fold=f, y_true=t, y_pred=q) for t, q in zip(y_after[va], p)]
pred = pd.DataFrame(recs); pred.to_csv(RESULTS / "raw" / "e3_predictions.csv", index=False)
tab = []
for m, g in pred.groupby("model", sort=False):
    y, p = g.y_true.to_numpy(), g.y_pred.to_numpy()
    tab.append(dict(model=m, MAE_ms=np.abs(p - y).mean(), r2=1 - ((p - y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)))
tab = pd.DataFrame(tab); tab.to_csv(RESULTS / "tables" / "E3_constant_rr.csv", index=False); print(tab.round(2).to_string(index=False))

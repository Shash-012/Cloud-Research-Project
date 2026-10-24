"""Train the offline pipeline per app (filters -> Eq.1 -> HPO -> final NN) and save bundles to results/models/<app>/"""
import _common  # noqa
import argparse, time
import numpy as np, pandas as pd
from autoscale.config import load_config, RESULTS
from autoscale.data.dataset import TransitionDataset
from autoscale.models.bundle import train_bundle

ap = argparse.ArgumentParser()
ap.add_argument("--apps", nargs="*"); ap.add_argument("--trials", type=int); ap.add_argument("--no-hpo", action="store_true")
ap.add_argument("--use-rr", action="store_true", help="also feed request rate to the NN (paper: optional)")
a = ap.parse_args()
cfg = load_config()
if a.trials:
    cfg["hpo"]["trials"] = a.trials
rows = []
for app in a.apps or list(cfg["apps"]):
    ds = TransitionDataset.load(RESULTS / "raw" / f"dataset_{app}.npz")
    t = time.time()
    bundle, sel_df, trials = train_bundle(ds, cfg, app, use_rr=a.use_rr, run_hpo=not a.no_hpo, seed=0, verbose=False)
    bundle.save(RESULTS / "models" / app)
    q = dict(app=app, n_selected=len(bundle.sel), train_time_s=round(time.time() - t, 1), **{f"nn_{k}": v for k, v in bundle.meta["nn"].items() if k in ("optimizer", "lr", "activation", "dropout", "batch_size", "normalization", "loss")})
    if trials is not None:
        trials.to_csv(RESULTS / "tables" / f"hpo_trials_{app}.csv", index=False)
        b = trials.sort_values("rmse").iloc[0]; q.update(cv_rmse_ms=b.rmse, cv_mae_ms=b.mae)
    q["baseline_rmse_mean_predictor_ms"] = float(np.minimum(ds.rt_before, cfg["rt_clip_ms"]).std())
    rows.append(q); print(q)
pd.DataFrame(rows).to_csv(RESULTS / "tables" / "model_quality.csv", index=False)

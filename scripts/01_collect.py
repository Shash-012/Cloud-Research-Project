"""Perturbation (free-run) data collection per application -> results/raw/dataset_<app>.npz"""
import _common  # noqa
import argparse, time
from autoscale.config import load_config, RESULTS
from autoscale.data.perturbation import collect_perturbation

ap = argparse.ArgumentParser()
ap.add_argument("--apps", nargs="*"); ap.add_argument("--obs", type=int)
a = ap.parse_args()
cfg = load_config()
apps = a.apps or list(cfg["apps"])
n_obs = a.obs or cfg["perturb_obs"]
assert cfg["train_seed"] not in cfg["eval_seeds"], "training seed must differ from evaluation seeds"
(RESULTS / "raw").mkdir(parents=True, exist_ok=True)
for app in apps:
    t = time.time()
    ds = collect_perturbation(cfg, app, n_obs, cfg["train_seed"])
    ds.save(RESULTS / "raw" / f"dataset_{app}.npz")
    print(f"{app}: {len(ds)} observations in {time.time() - t:.0f}s "
          f"(up {int((ds.k > 0).sum())}, down {int((ds.k < 0).sum())})")

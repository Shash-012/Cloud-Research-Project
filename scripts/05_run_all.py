"""E4-E6: policies x apps x workloads x seeds. Writes results/raw/runs_<workload>.csv and seed-0 time series."""
import _common  # noqa
import argparse
import pandas as pd
from autoscale.config import load_config, RESULTS
from autoscale.runner.sweep import run_grid
from autoscale.runner.run_experiment import run_one, load_tuned

ap = argparse.ArgumentParser()
ap.add_argument("--workloads", nargs="*", default=["waves", "webshop", "surge"])
ap.add_argument("--apps", nargs="*"); ap.add_argument("--seeds", type=int); ap.add_argument("--duration", type=float)
ap.add_argument("--workers", type=int)
a = ap.parse_args()
cfg = load_config()
seeds = cfg["eval_seeds"][: a.seeds] if a.seeds else cfg["eval_seeds"]
apps = a.apps or list(cfg["apps"])
raw = RESULTS / "raw"; raw.mkdir(parents=True, exist_ok=True)
tuned = load_tuned()
if not tuned:
    print("WARNING: no tuned baselines found (run scripts/04_tune_baselines.py); using default thresholds.")
for wl in a.workloads:
    print(f"== workload {wl}")
    runs = run_grid(apps, [wl], cfg["policies"], seeds, a.duration or cfg["duration_s"], a.workers)
    runs.to_csv(raw / f"runs_{wl}.csv", index=False)
    print(runs.groupby(["app", "policy"])[["qos_viol_pct", "rt_avg_ms", "vm_used_pct"]].mean().round(2).to_string())
    for app in apps:                                  # seed-0 time series for the Fig 4 / Fig 8 style plots
        for pol in cfg["policies"]:
            _, iv, _ = run_one(app, wl, pol, seeds[0], cfg=cfg, tuned=tuned, duration_s=a.duration or cfg["duration_s"], keep=True)
            iv.to_csv(raw / f"ts_{app}_{wl}_{pol}.csv", index=False)
